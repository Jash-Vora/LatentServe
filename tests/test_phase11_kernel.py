"""
Phase 11 — paged decode attention.

The kernel replaces a gather plus SDPA with a single in-place read, so
the only thing that makes it trustworthy is equivalence to what it
replaces. These tests check the *algorithm* (block walk, online softmax,
split-and-combine) against dense attention on CPU, so the logic is
verified independently of whether Triton lowers it correctly.

That separation is the point: given a wrong output tensor, "the
algorithm is wrong" and "the Triton lowering is wrong" are otherwise
indistinguishable. The GPU-gated tests at the bottom compare the Triton
kernel against this same reference, so a failure there localises to the
lowering.

    pytest tests/test_phase11_kernel.py -v            # CPU, reference only
    pytest tests/test_phase11_kernel.py -v -m gpu     # adds the Triton path
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from kernels.gqa.paged_decode import (  # noqa: E402
    HAS_TRITON,
    combine_splits,
    paged_decode_attention,
    paged_decode_reference,
)

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON),
    reason="needs a CUDA device with Triton",
)


def build_case(batch=2, h_kv=2, n_rep=3, d=32, page=16, lengths=None, seed=0):
    """A paged cache with deliberately *scattered* block tables.

    Contiguous block ids would let an indexing bug pass: page p landing
    at block p is exactly what a broken block-table lookup produces.
    """
    torch.manual_seed(seed)
    lengths = lengths or [40, 17]
    max_pages = max((l + page - 1) // page for l in lengths)
    num_blocks = batch * max_pages + 5

    k_pool = torch.randn(num_blocks, page, h_kv, d)
    v_pool = torch.randn(num_blocks, page, h_kv, d)
    perm = torch.randperm(num_blocks)[: batch * max_pages].reshape(batch, max_pages)
    q = torch.randn(batch, h_kv, n_rep, d)
    return q, k_pool, v_pool, perm.to(torch.int32), torch.tensor(lengths, dtype=torch.int32)


def dense_attention(q, k_pool, v_pool, block_tables, seq_lens):
    """Ground truth: gather every live token, then one softmax. This is
    what the current runtime does — gather then SDPA — so agreeing with
    it is precisely the property Phase 11 must preserve."""
    b, h_kv, n_rep, d = q.shape
    page = k_pool.shape[1]
    out = torch.zeros_like(q)
    for i in range(b):
        length = int(seq_lens[i])
        idx = []
        for p in range((length + page - 1) // page):
            blk = int(block_tables[i, p])
            for off in range(page):
                if p * page + off < length:
                    idx.append((blk, off))
        for h in range(h_kv):
            k = torch.stack([k_pool[blk, off, h] for blk, off in idx])
            v = torch.stack([v_pool[blk, off, h] for blk, off in idx])
            logits = (q[i, h] @ k.T) / math.sqrt(d)
            out[i, h] = torch.softmax(logits, dim=-1) @ v
    return out


# ----------------------------------------------------------------------
# The algorithm
# ----------------------------------------------------------------------


def test_reference_matches_dense_attention():
    q, k_pool, v_pool, tables, lens = build_case()
    got = paged_decode_reference(q, k_pool, v_pool, tables, lens)
    want = dense_attention(q, k_pool, v_pool, tables, lens)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("num_splits", [1, 2, 3, 4, 8])
def test_split_k_is_exact(num_splits):
    """Splitting the KV length changes how the softmax is accumulated but
    must not change the result. If it does, the partial-softmax combine
    is wrong — and at batch 1 this kernel *needs* splitting to fill the
    GPU, so an approximate combine would be silently wrong everywhere."""
    q, k_pool, v_pool, tables, lens = build_case(lengths=[63, 40])
    want = dense_attention(q, k_pool, v_pool, tables, lens)
    got = paged_decode_reference(q, k_pool, v_pool, tables, lens, num_splits=num_splits)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_more_splits_than_pages_is_harmless():
    """The default split count is chosen to fill the SMs, so it can
    exceed the number of pages on a short sequence. Empty splits must
    contribute nothing rather than NaN — exp(-inf - -inf)."""
    q, k_pool, v_pool, tables, lens = build_case(lengths=[10, 5], page=16)
    want = dense_attention(q, k_pool, v_pool, tables, lens)
    got = paged_decode_reference(q, k_pool, v_pool, tables, lens, num_splits=8)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33])
def test_ragged_final_page(length):
    """Sequence lengths are not multiples of the page size, and tokens
    past the end must be masked rather than attended to."""
    q, k_pool, v_pool, tables, lens = build_case(batch=1, lengths=[length])
    want = dense_attention(q, k_pool, v_pool, tables, lens)
    got = paged_decode_reference(q, k_pool, v_pool, tables, lens, num_splits=2)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_sequences_do_not_leak_into_each_other():
    """Two sequences of very different lengths in one batch: the short
    one must not see the long one's blocks."""
    q, k_pool, v_pool, tables, lens = build_case(batch=2, lengths=[64, 3])
    got = paged_decode_reference(q, k_pool, v_pool, tables, lens)
    solo = paged_decode_reference(
        q[1:2], k_pool, v_pool, tables[1:2], lens[1:2]
    )
    torch.testing.assert_close(got[1:2], solo, rtol=1e-5, atol=1e-6)


def test_int8_path_matches_its_own_dequantized_tensors():
    """The INT8 read must equal dense attention over the values it
    decodes to — separating quantization error (expected) from a wrong
    scale axis (not). K's scale is per channel, V's per token; swapping
    them produces plausible output and wrong numbers."""
    torch.manual_seed(3)
    b, h_kv, n_rep, d, page = 1, 2, 4, 32, 16
    num_blocks, length = 6, 40
    k_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8)
    v_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8)
    k_scale = torch.rand(num_blocks, h_kv, d) * 0.01 + 0.001
    v_scale = torch.rand(num_blocks, page, h_kv) * 0.01 + 0.001
    tables = torch.randperm(num_blocks)[:3].reshape(1, 3).to(torch.int32)
    lens = torch.tensor([length], dtype=torch.int32)
    q = torch.randn(b, h_kv, n_rep, d)

    k_deq = k_q.float() * k_scale[:, None, :, :]
    v_deq = v_q.float() * v_scale[:, :, :, None]
    want = dense_attention(q, k_deq, v_deq, tables, lens)
    got = paged_decode_reference(
        q, k_q, v_q, tables, lens, k_scale=k_scale, v_scale=v_scale, num_splits=2
    )
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_asymmetric_zero_points_are_applied_on_the_right_axis():
    torch.manual_seed(4)
    num_blocks, page, h_kv, d, length = 5, 16, 2, 32, 30
    k_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8)
    v_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8)
    k_scale = torch.rand(num_blocks, h_kv, d) * 0.01 + 0.001
    v_scale = torch.rand(num_blocks, page, h_kv) * 0.01 + 0.001
    k_zero = torch.randn(num_blocks, h_kv, d)
    v_zero = torch.randn(num_blocks, page, h_kv)
    tables = torch.randperm(num_blocks)[:2].reshape(1, 2).to(torch.int32)
    lens = torch.tensor([length], dtype=torch.int32)
    q = torch.randn(1, h_kv, 3, d)

    k_deq = k_q.float() * k_scale[:, None, :, :] + k_zero[:, None, :, :]
    v_deq = v_q.float() * v_scale[:, :, :, None] + v_zero[:, :, :, None]
    want = dense_attention(q, k_deq, v_deq, tables, lens)
    got = paged_decode_reference(
        q, k_q, v_q, tables, lens, k_scale=k_scale, v_scale=v_scale,
        k_zero=k_zero, v_zero=v_zero, num_splits=2,
    )
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_combine_splits_handles_an_all_empty_split():
    acc = torch.zeros(1, 1, 3, 4, 8)
    m = torch.full((1, 1, 3, 4), float("-inf"))
    l = torch.zeros(1, 1, 3, 4)
    m[:, :, 0] = 0.0
    l[:, :, 0] = 1.0
    acc[:, :, 0] = 1.0
    out = combine_splits(acc, m, l, torch.float32)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.ones(1, 1, 4, 8))


# ----------------------------------------------------------------------
# The Triton lowering (GPU only)
# ----------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("num_splits", [1, 4])
def test_triton_matches_reference_fp16(num_splits):
    q, k_pool, v_pool, tables, lens = build_case(n_rep=6, d=128, page=16, lengths=[130, 48])
    args = [t.cuda() for t in (q.half(), k_pool.half(), v_pool.half(), tables, lens)]
    want = paged_decode_reference(*args, num_splits=num_splits)
    got = paged_decode_attention(*args, num_splits=num_splits)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-3)


@requires_gpu
def test_triton_matches_reference_int8():
    torch.manual_seed(5)
    num_blocks, page, h_kv, d = 12, 16, 2, 128
    k_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8).cuda()
    v_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8).cuda()
    k_scale = (torch.rand(num_blocks, h_kv, d) * 0.01 + 0.001).cuda()
    v_scale = (torch.rand(num_blocks, page, h_kv) * 0.01 + 0.001).cuda()
    tables = torch.randperm(num_blocks)[:8].reshape(1, 8).to(torch.int32).cuda()
    lens = torch.tensor([120], dtype=torch.int32).cuda()
    q = torch.randn(1, h_kv, 6, d).half().cuda()

    want = paged_decode_reference(q, k_q, v_q, tables, lens,
                                  k_scale=k_scale, v_scale=v_scale, num_splits=4)
    got = paged_decode_attention(q, k_q, v_q, tables, lens,
                                 k_scale=k_scale, v_scale=v_scale, num_splits=4)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-3)


# ----------------------------------------------------------------------
# End to end: the kernel path must produce the same tokens
# ----------------------------------------------------------------------


def _tiny_model():
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
               num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)
    return Qwen2ForCausalLM(Qwen2Config(**cfg)).eval(), cfg


def test_block_tables_tensor_matches_the_block_tables():
    """The kernel walks pages; the gather walked slots. A table that
    disagrees with `BlockTable.blocks` sends the kernel to another
    sequence's memory while looking entirely healthy."""
    from cache.kv_cache import KVCacheSpec
    from cache.paged_cache import PagedKVCache

    spec = KVCacheSpec(num_layers=1, num_kv_heads=2, head_dim=8, max_batch_size=3,
                       max_seq_len=128, dtype=torch.float32, device="cpu")
    cache = PagedKVCache(spec, block_size=16)
    cache.advance(40, batch_size=3)
    table = cache.block_tables_tensor(3)
    assert table.shape == (3, 3)
    for row, i in enumerate(cache.active_slots):
        assert table[row].tolist()[: len(cache.tables[i].blocks)] == cache.tables[i].blocks
    assert cache.seq_lens_tensor(3).tolist() == [40, 40, 40]


def test_kernel_inputs_are_built_once_per_step_not_per_layer():
    """The tensors are built in `advance()` and handed out unchanged.

    Rebuilding them per call cost ~20 ms per decode step — 28 Python
    loops and 28 host-to-device copies, more than the gather this kernel
    exists to remove. `_read_slots` is built once for the same reason,
    and its docstring warned about exactly this.
    """
    from cache.kv_cache import KVCacheSpec
    from cache.paged_cache import PagedKVCache

    spec = KVCacheSpec(num_layers=28, num_kv_heads=2, head_dim=8, max_batch_size=2,
                       max_seq_len=128, dtype=torch.float32, device="cpu")
    cache = PagedKVCache(spec, block_size=16)
    cache.advance(40, batch_size=2)
    first = cache.block_tables_tensor(2)
    # Every layer asks again during one step. What matters is that no
    # rebuild and no copy happens — same storage, not necessarily the
    # same Python object, since a partial-batch request returns a view.
    for _ in range(28):
        again = cache.block_tables_tensor(2)
        assert again.data_ptr() == first.data_ptr()
    partial = cache.block_tables_tensor(1)
    assert partial.data_ptr() == first.data_ptr()
    assert partial.shape[0] == 1


def test_kernel_inputs_raise_before_advance():
    """Better than returning a stale or empty table: the kernel would
    read block 0 for every page and produce plausible nonsense."""
    from cache.kv_cache import KVCacheSpec
    from cache.paged_cache import PagedKVCache

    spec = KVCacheSpec(num_layers=1, num_kv_heads=2, head_dim=8, max_batch_size=1,
                       max_seq_len=32, dtype=torch.float32, device="cpu")
    cache = PagedKVCache(spec, block_size=16)
    with pytest.raises(RuntimeError, match="advance"):
        cache.block_tables_tensor(1)


def test_block_tables_grow_with_each_decode_step():
    """Rebuilt per advance, not cached: a stale table is a silent
    correctness bug rather than a crash."""
    from cache.kv_cache import KVCacheSpec
    from cache.paged_cache import PagedKVCache

    spec = KVCacheSpec(num_layers=1, num_kv_heads=2, head_dim=8, max_batch_size=1,
                       max_seq_len=128, dtype=torch.float32, device="cpu")
    cache = PagedKVCache(spec, block_size=16)
    cache.advance(16, batch_size=1)
    assert cache.block_tables_tensor(1).shape[1] == 1
    cache.advance(1, batch_size=1)
    assert cache.block_tables_tensor(1).shape[1] == 2
    assert cache.seq_lens_tensor(1).tolist() == [17]


@requires_gpu
def test_kernel_decode_matches_sdpa_end_to_end():
    """Gate 10: the kernel path must generate the same tokens as the
    gather path. Everything else in Phase 11 is a latency claim, and a
    latency claim about a different model is worth nothing."""
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape

    model, cfg = _tiny_model()
    model = model.half().cuda()
    shape = ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float16")
    prompt = torch.randint(0, 128, (1, 100)).cuda()
    stream = torch.randint(0, 128, (1, 24)).cuda()

    def run(impl):
        ls = LatentServeQwen(hf_model=model, tokenizer=None, shape=shape, device="cuda",
                             attn_impl=impl, max_seq_len_hint=256)
        ls.allocate_cache(1, 256, paged=True, block_size=16)
        ls.cache.reset()
        ls.prefill(prompt)
        return torch.cat([ls.decode_step(stream[:, t : t + 1]) for t in range(24)], dim=1)

    torch.testing.assert_close(run("triton_paged"), run("sdpa"), rtol=3e-2, atol=3e-2)


def test_contiguous_cache_accepts_a_contiguous_slot_prefix():
    """`decode_step_ragged` passes `slots` unconditionally, so every
    cache has to accept it. A contiguous cache can represent
    [0, 1, 2] — it means the same thing as batch_size 3 — and cannot
    represent [0, 3, 7]. Silently treating the second as the first
    would write sequence 3's KV into slot 1."""
    from cache.kv_cache import ContiguousKVCache, KVCacheSpec

    spec = KVCacheSpec(num_layers=1, num_kv_heads=2, head_dim=4, max_batch_size=3,
                       max_seq_len=32, dtype=torch.float32, device="cpu")
    cache = ContiguousKVCache(spec)
    cache.advance(4, slots=[0, 1, 2])
    assert cache.length == 4
    with pytest.raises(NotImplementedError, match="non-contiguous slots"):
        cache.advance(1, slots=[0, 3, 7])


def test_split_count_bounds_the_serial_chain_not_just_the_sm_count():
    """Two constraints on `num_splits`, not one.

    Filling the SMs is necessary (Phase 2: achieved bandwidth tracks
    batch x kv_heads) but not sufficient: at batch 4 / 16K the SM rule
    alone gave 5 programs each walking 204 pages through a *serial*
    online-softmax chain, and that row cost 54 ms of unexplained time
    against ~22 ms everywhere else.
    """
    from kernels.gqa.paged_decode import TARGET_PAGES_PER_SPLIT

    def splits(batch, ctx, kv_heads=2, sms=40, page=16):
        pages = ctx // page
        by_sms = -(-sms // (batch * kv_heads))
        by_chain = -(-pages // TARGET_PAGES_PER_SPLIT)
        return max(1, min(pages, max(by_sms, by_chain)))

    # The pathological row: the chain bound must take over.
    assert splits(4, 16384) > -(-40 // 8)
    for batch, ctx in ((1, 4096), (1, 16384), (4, 4096), (4, 8192), (4, 16384)):
        pages_each = (ctx // 16) / splits(batch, ctx)
        assert pages_each <= TARGET_PAGES_PER_SPLIT + 1


def test_scratch_buffers_are_reused_across_calls():
    """28 layers a step means 28 allocation triples otherwise. Same
    shape must hand back the same storage."""
    from kernels.gqa.paged_decode import _scratch

    a = _scratch("t", (2, 3), torch.float32, torch.device("cpu"))
    b = _scratch("t", (2, 3), torch.float32, torch.device("cpu"))
    assert a.data_ptr() == b.data_ptr()
    c = _scratch("t", (4, 3), torch.float32, torch.device("cpu"))
    assert c.shape == (4, 3)


def test_scratch_fill_clears_stale_state():
    """The partial-softmax buffers carry -inf / 0 sentinels. Reusing
    them without clearing would merge the previous layer's splits into
    this one's result."""
    from kernels.gqa.paged_decode import _scratch

    buf = _scratch("f", (2, 2), torch.float32, torch.device("cpu"), fill=float("-inf"))
    buf[0, 0] = 5.0
    again = _scratch("f", (2, 2), torch.float32, torch.device("cpu"), fill=float("-inf"))
    assert torch.isinf(again).all()


def test_tunables_are_per_call_not_module_level():
    """Phase 12 sweeps the design space, so the knobs must be arguments.

    Reading them from module constants would mean re-importing between
    configurations — which also loses Triton's compilation cache, so
    every point would pay a recompile and the timings would measure the
    compiler.
    """
    import inspect

    from kernels.gqa.paged_decode import paged_decode_attention

    params = inspect.signature(paged_decode_attention).parameters
    for name in ("pages_per_iter", "num_warps", "num_stages", "num_splits"):
        assert name in params, f"{name} must be tunable per call"
        assert params[name].default is None, f"{name} should default to the module setting"


@requires_gpu
def test_kernel_path_does_not_gather():
    """The whole point of Phase 11 is to stop staging the cache into an
    fp16 buffer. Placed after `cache.read()`, the kernel branch paid the
    gather *and* the kernel — torch.profiler showed 56 `aten::index`
    calls per decode step, two per layer, on a path that should have
    none. Counting the calls is the only way this stays fixed.
    """
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape

    model, cfg = _tiny_model()
    model = model.half().cuda()
    shape = ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float16")
    prompt = torch.randint(0, 128, (1, 64)).cuda()

    def gather_calls(impl: str) -> int:
        ls = LatentServeQwen(hf_model=model, tokenizer=None, shape=shape, device="cuda",
                             attn_impl=impl, max_seq_len_hint=256)
        ls.allocate_cache(1, 256, paged=True, block_size=16)
        ls.cache.reset()
        ls.prefill(prompt)
        token = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        ls.decode_step(token)                       # warm up
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            ls.decode_step(token)
            torch.cuda.synchronize()
        return sum(
            e.count for e in prof.key_averages() if e.key in ("aten::index", "aten::index_select")
        )

    assert gather_calls("sdpa") > 0, "the SDPA path gathers, by construction"
    assert gather_calls("triton_paged") == 0, "the kernel path must not gather"