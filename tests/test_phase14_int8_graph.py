"""
Phase 14c — the INT8 cache at graph speed.

Deferring a block's quantization changes *when* it happens, never *what*
it computes — so most of this is exact, and testable on CPU:

  * a deferred cache's pool is byte-identical to an immediate cache's,
    block for block, once pending blocks are flushed;
  * the invariant the kernel relies on holds at every step: all pages but
    a sequence's last are quantized, the last is in the residual;
  * decode through the kernel path gives identical logits either way,
    because both read the last page from the residual and every other
    page from the same quantized pool;
  * the gather path stays exact too, because read() flushes first.

The GPU tests then check the Triton kernels' residual read against the
reference, and graph replay against eager.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from cache.int8_paged_cache import Int8PagedKVCache  # noqa: E402
from cache.kv_cache import KVCacheSpec  # noqa: E402
from kernels.gqa.paged_decode import HAS_TRITON, paged_decode_attention, paged_decode_reference  # noqa: E402
from model.latentserve_qwen import LatentServeQwen  # noqa: E402
from model.qwen import ModelShape  # noqa: E402

CFG = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
           num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)
SHAPE = ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32")
requires_gpu = pytest.mark.skipif(not (torch.cuda.is_available() and HAS_TRITON),
                                  reason="needs CUDA with Triton")
BS = 16


# ----------------------------------------------------------------------
# The cache
# ----------------------------------------------------------------------


def _cache(deferred, layers=2, batch=2):
    spec = KVCacheSpec(num_layers=layers, num_kv_heads=2, head_dim=8, max_batch_size=batch,
                       max_seq_len=128, dtype=torch.float32, device="cpu")
    c = Int8PagedKVCache(spec, block_size=BS)
    if deferred:
        c.enable_deferred_finalize()
    return c


def _drive(cache, prompt=10, steps=40, layers=2, seed=0):
    """Prefill then decode, writing identical K/V into any cache."""
    g = torch.Generator().manual_seed(seed)
    cache.advance(prompt, batch_size=1)
    for layer in range(layers):
        cache.write(layer, torch.randn(1, 2, prompt, 8, generator=g),
                    torch.randn(1, 2, prompt, 8, generator=g))
    for _ in range(steps):
        cache.advance(1, slots=[0])
        for layer in range(layers):
            cache.write(layer, torch.randn(1, 2, 1, 8, generator=g),
                        torch.randn(1, 2, 1, 8, generator=g))


def test_deferred_pool_is_byte_identical_to_immediate():
    """10 prompt tokens and 40 decode steps crosses three block boundaries,
    each completed by a decode write — the case deferral changes."""
    a, b = _cache(False), _cache(True)
    _drive(a)
    _drive(b)
    b._finalize_pending([0])                          # flush what is still pending
    done = a.tables[0].length // BS
    assert done == 3
    for layer in range(2):
        for j in range(done):
            ba, bb = a.tables[0].blocks[j], b.tables[0].blocks[j]
            assert torch.equal(a.k_pool[layer][ba], b.k_pool[layer][bb]), (layer, j)
            assert torch.equal(a.k_scale_pool[layer][ba], b.k_scale_pool[layer][bb])
        assert torch.equal(a.v_pool[layer], b.v_pool[layer])


def test_every_page_but_the_last_is_quantized_at_every_step():
    """What the kernel relies on, checked after each decode advance."""
    c = _cache(True)
    g = torch.Generator().manual_seed(1)
    c.advance(5, batch_size=1)
    for layer in range(2):
        c.write(layer, torch.randn(1, 2, 5, 8, generator=g), torch.randn(1, 2, 5, 8, generator=g))
    for _ in range(45):
        c.advance(1, slots=[0])
        length = c.tables[0].length
        assert c._finalized[0] == (length - 1) // BS, length
        for layer in range(2):
            c.write(layer, torch.randn(1, 2, 1, 8, generator=g), torch.randn(1, 2, 1, 8, generator=g))


def test_a_second_pending_block_is_refused_not_lost():
    """The residual holds one block. Writing past two boundaries without an
    advance in between would lose the first — so it raises."""
    c = _cache(True)
    c.advance(1, batch_size=1)
    c.tables[0].append(2 * BS)            # bypass advance: two blocks complete at once
    with pytest.raises(RuntimeError, match="unquantized"):
        c._finalize_pending([0])


def test_freeing_a_slot_resets_its_count():
    c = _cache(True)
    _drive(c, prompt=20, steps=0)
    assert c._finalized[0] == 1
    c.free_sequence(0)
    assert c._finalized[0] == 0


# ----------------------------------------------------------------------
# The kernel's residual read
# ----------------------------------------------------------------------


def _dense(q, k_pages, v_pages, lens):
    out = torch.zeros_like(q)
    for i, length in enumerate(lens):
        k = k_pages[i][:length]
        v = v_pages[i][:length]
        for h in range(q.shape[1]):
            w = torch.softmax(q[i, h] @ k[:, h].T / math.sqrt(q.shape[-1]), dim=-1)
            out[i, h] = w @ v[:, h]
    return out


def test_reference_reads_the_last_page_from_the_residual():
    torch.manual_seed(2)
    nb, h, d, rows = 10, 2, 16, 3
    k_q = torch.randint(-127, 127, (nb, BS, h, d), dtype=torch.int8)
    v_q = torch.randint(-127, 127, (nb, BS, h, d), dtype=torch.int8)
    k_scale = torch.rand(nb, h, d) * 0.01 + 1e-3
    v_scale = torch.rand(nb, BS, h) * 0.01 + 1e-3
    residual = torch.randn(rows, BS, h, d)
    tables = torch.tensor([[3, 7, 1], [5, 0, 2]], dtype=torch.int32)
    lens = [40, 21]                                   # tails of 8 and 5
    res_rows = torch.tensor([2, 0], dtype=torch.int32)
    q = torch.randn(2, h, 4, d)

    k_pages, v_pages = [], []
    for i, length in enumerate(lens):
        pages = (length + BS - 1) // BS
        ks, vs = [], []
        for p in range(pages):
            blk = int(tables[i, p])
            vs.append(v_q[blk].float() * v_scale[blk][:, :, None])
            ks.append(residual[int(res_rows[i])] if p == pages - 1
                      else k_q[blk].float() * k_scale[blk][None])
        k_pages.append(torch.cat(ks))
        v_pages.append(torch.cat(vs))
    want = _dense(q, k_pages, v_pages, lens)
    got = paged_decode_reference(q, k_q, v_q, tables, torch.tensor(lens, dtype=torch.int32),
                                 k_scale=k_scale, v_scale=v_scale, num_splits=2,
                                 k_residual=residual, res_rows=res_rows)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


# ----------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------


def _model(impl, device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    hf = Qwen2ForCausalLM(Qwen2Config(**CFG)).to(dtype).eval().to(device)
    return LatentServeQwen(hf_model=hf, tokenizer=None, shape=SHAPE, device=device,
                           attn_impl=impl, max_seq_len_hint=256)


def _decode(m, deferred, prompt, stream):
    m.allocate_cache(1, 128, paged=True, block_size=BS, kv_dtype="int8")
    m.cache.reset()
    if deferred:
        m.cache.enable_deferred_finalize()
    m.prefill_slot(prompt, slot=0)
    return torch.cat([m.decode_step_ragged(stream[:, t : t + 1],
                                           torch.tensor([[prompt.shape[1] + t]],
                                                        device=prompt.device), [0])
                      for t in range(stream.shape[1])], dim=1)


@pytest.mark.parametrize("impl", ["triton_paged", "sdpa"])
def test_deferral_changes_no_logit(impl):
    """Kernel path: both modes read the last page from the residual and all
    others from the same pool. Gather path: read() flushes pending blocks
    before gathering. Either way, exactly the same numbers."""
    prompt = torch.randint(0, 128, (1, 10))
    stream = torch.randint(0, 128, (1, 40))
    want = _decode(_model(impl), False, prompt, stream)
    got = _decode(_model(impl), True, prompt, stream)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_engine_serves_int8_through_the_kernel_path():
    """Ragged requests and slot recycling, INT8 cache, deferred mode."""
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    def serve(deferred):
        e = ServingEngine(_model("triton_paged"), max_running=3, max_seq_len=256,
                          block_size=BS, kv_dtype="int8")
        if deferred:
            e.cache.enable_deferred_finalize()
        g = torch.Generator().manual_seed(0)
        for i, (p, n) in enumerate([(12, 25), (40, 4), (7, 30), (25, 18)]):
            e.add_request(ServedRequest(request_id=i,
                          prompt_ids=torch.randint(0, 128, (p,), generator=g).tolist(),
                          max_new_tokens=n))
        return {r.request_id: r.output_ids for r in e.run()}

    assert serve(True) == serve(False)


# ----------------------------------------------------------------------
# GPU: Triton residual read, and graph replay
# ----------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("ppi", [1, 4])
def test_triton_residual_read_matches_reference(ppi):
    torch.manual_seed(3)
    nb, h, d, rows = 24, 2, 128, 3
    dev = "cuda"
    k_q = torch.randint(-127, 127, (nb, BS, h, d), dtype=torch.int8, device=dev)
    v_q = torch.randint(-127, 127, (nb, BS, h, d), dtype=torch.int8, device=dev)
    k_scale = (torch.rand(nb, h, d, device=dev) * 0.01 + 1e-3)
    v_scale = (torch.rand(nb, BS, h, device=dev) * 0.01 + 1e-3)
    residual = torch.randn(rows, BS, h, d, device=dev, dtype=torch.float16)
    tables = torch.randperm(nb, device=dev)[:16].reshape(2, 8).to(torch.int32)
    lens = torch.tensor([117, 40], dtype=torch.int32, device=dev)
    res_rows = torch.tensor([1, 2], dtype=torch.int32, device=dev)
    q = torch.randn(2, h, 6, d, device=dev, dtype=torch.float16)
    kw = dict(k_scale=k_scale, v_scale=v_scale, num_splits=4,
              k_residual=residual, res_rows=res_rows)
    want = paged_decode_reference(q, k_q, v_q, tables, lens, **kw)
    got = paged_decode_attention(q, k_q, v_q, tables, lens, pages_per_iter=ppi, **kw)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-3)


@requires_gpu
def test_graphed_int8_decode_matches_eager_int8():
    from runtime.cuda_graph import GraphedDecoder

    prompt = torch.randint(0, 128, (1, 10)).cuda()
    stream = torch.randint(0, 128, (1, 40)).cuda()
    want = _decode(_model("triton_paged", "cuda", torch.float16), False, prompt, stream)

    m = _model("triton_paged", "cuda", torch.float16)
    m.allocate_cache(1, 128, paged=True, block_size=BS, kv_dtype="int8")
    m.cache.reset()
    m.prefill_slot(prompt, slot=0)
    decoder = GraphedDecoder(m)                        # turns deferral on
    assert m.cache.deferred_finalize
    got = [decoder.step(stream[:, t : t + 1], torch.tensor([[10 + t]]).cuda(), [0]).clone()
           for t in range(40)]
    torch.testing.assert_close(torch.cat(got, dim=1), want, rtol=3e-2, atol=3e-2)
