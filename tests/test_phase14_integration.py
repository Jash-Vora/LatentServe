"""Phase 14 integration: page bounds in the cache, sparse decode in the model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _model(device="cpu", hidden=128, heads=8, head_dim=16, layers=2, attn="triton_paged"):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=hidden, intermediate_size=256,
                      num_hidden_layers=layers, num_attention_heads=heads,
                      num_key_value_heads=2, max_position_embeddings=4096)
    hf = Qwen2ForCausalLM(cfg).eval()
    dtype = "torch.float32"
    if device == "cuda":
        hf = hf.half().cuda()
        dtype = "torch.float16"
    shape = ModelShape(layers, heads, 2, head_dim, hidden, 128, 4096, dtype)
    return LatentServeQwen(hf_model=hf, tokenizer=None, shape=shape, device=device,
                           attn_impl=attn, max_seq_len_hint=1024)


def _true_bounds(cache, layer, slot=0):
    """Min/max over each block's *written* rows, straight from the pool."""
    table = cache.tables[slot]
    bs = cache.block_size
    out = {}
    for page, blk in enumerate(table.blocks):
        n = min(bs, table.length - page * bs)
        if n <= 0:
            continue
        rows = cache.k_pool[layer][blk, :n]
        out[blk] = (rows.amin(0), rows.amax(0))
    return out


def _check(cache):
    for layer in range(len(cache.k_pool)):
        kmin, kmax = cache.page_bounds(layer)
        for blk, (lo, hi) in _true_bounds(cache, layer).items():
            assert torch.equal(kmin[blk], lo) and torch.equal(kmax[blk], hi), (layer, blk)


def test_bounds_after_prefill_and_decode_cover_exactly_the_written_rows():
    ls = _model()
    ls.set_sparse(0.25)
    ls.allocate_cache(1, 256, paged=True, block_size=16)
    ls.cache.reset()
    with torch.no_grad():
        ls.prefill(torch.randint(0, 128, (1, 70)))        # last page: 6 of 16 rows written
        _check(ls.cache)
        for t in range(13):                                # crosses into a fresh page
            ls.decode_step(torch.tensor([[t]]))
    _check(ls.cache)


def test_enabling_after_a_prefill_rebuilds_from_what_is_written():
    ls = _model()
    ls.allocate_cache(1, 256, paged=True, block_size=16)
    ls.cache.reset()
    with torch.no_grad():
        ls.prefill(torch.randint(0, 128, (1, 53)))
    ls.set_sparse(0.25)
    _check(ls.cache)


def test_rewind_recomputes_the_page_it_lands_in():
    ls = _model()
    ls.set_sparse(0.25)
    ls.allocate_cache(1, 256, paged=True, block_size=16)
    ls.cache.reset()
    with torch.no_grad():
        ls.prefill(torch.randint(0, 128, (1, 70)))
        for t in range(8):
            ls.decode_step(torch.tensor([[t]]))
        ls.cache.rewind(70)
    _check(ls.cache)


def test_int8_cache_is_refused():
    ls = _model()
    ls.allocate_cache(1, 256, paged=True, block_size=16, kv_dtype="int8")
    with pytest.raises(ValueError, match="fp16 paged cache"):
        ls.set_sparse(0.25)


# --------------------------------------------------------------- GPU ---

try:
    from kernels.cuda import paged_sparse as _ps

    _ps.compile_cubin("sm_75")
    _OK = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    _OK = False
requires_gpu = pytest.mark.skipif(not _OK, reason="needs a GPU, Triton and CuPy")


def _gpu_model():
    return _model("cuda", hidden=128 * 12, heads=12, head_dim=128)


@requires_gpu
def test_kernel_maintained_bounds_match_the_written_rows():
    ls = _gpu_model()
    ls.set_sparse(0.25)
    ls.allocate_cache(2, 512, paged=True, block_size=16)
    ls.cache.reset()
    with torch.no_grad():
        ls.prefill(torch.randint(0, 128, (2, 150)).cuda())
        for t in range(20):
            ls.decode_step(torch.full((2, 1), t, device="cuda"))
    torch.cuda.synchronize()
    for slot in (0, 1):
        for layer in range(2):
            kmin, kmax = ls.cache.page_bounds(layer)
            for blk, (lo, hi) in _true_bounds(ls.cache, layer, slot).items():
                assert torch.equal(kmin[blk], lo) and torch.equal(kmax[blk], hi)


@requires_gpu
def test_full_budget_decode_matches_dense():
    from kernels.gqa import paged_decode as pd

    ls = _gpu_model()
    prompt = torch.randint(0, 128, (2, 300)).cuda()

    def decode(ratio):
        ls.set_sparse(ratio)
        ls.allocate_cache(2, 512, paged=True, block_size=16)
        ls.cache.reset()
        with torch.no_grad():
            ls.prefill(prompt)
            return ls.decode_step(torch.zeros(2, 1, dtype=torch.long, device="cuda")).float()

    before = pd.decode_backend()
    try:
        pd.set_decode_backend("cuda")
        dense, sparse = decode(None), decode(1.0)
    finally:
        pd.set_decode_backend(before)
    torch.testing.assert_close(sparse, dense, atol=2e-2, rtol=2e-2)


@requires_gpu
def test_sparse_generation_is_identical_with_and_without_cuda_graphs():
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    ls = _gpu_model()
    ls.set_sparse(0.25)
    prompts = [torch.randint(0, 128, (600,)).tolist() for _ in range(3)]

    def run(graphs):
        engine = ServingEngine(ls, max_running=3, block_size=16, max_seq_len=700,
                               use_cuda_graphs=graphs)
        for i, p in enumerate(prompts):
            engine.add_request(ServedRequest(request_id=i, prompt_ids=list(p), max_new_tokens=24))
        return {r.request_id: r.output_ids for r in engine.run()}

    assert run(True) == run(False)


def test_scoring_persists_when_only_the_ratio_changes():
    """The harness toggles configurations with set_sparse(ratio) alone; the
    scoring chosen once must survive that, or a "mass" run silently becomes
    a "bounds" run."""
    ls = _model()
    ls.set_sparse(None, scoring="mass")
    ls.set_sparse(0.25)
    assert ls.sparse_scoring == "mass"
    assert all(layer.attn.sparse_scoring == "mass" for layer in ls.layers)
    ls.set_sparse(None)
    ls.set_sparse(0.5)
    assert ls.layers[0].attn.sparse_scoring == "mass"
    with pytest.raises(ValueError, match="unknown sparse scoring"):
        ls.set_sparse(0.25, scoring="max")


def test_admission_reserves_room_for_generated_tokens():
    """Prompts that fit but whose outputs do not used to exhaust the pool
    mid-decode (OutOfBlocks). Now the second request waits, and both finish."""
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    ls = _model(attn="sdpa")
    engine = ServingEngine(ls, max_running=2, max_seq_len=200, block_size=16, num_blocks=12)
    for i in range(2):
        engine.add_request(ServedRequest(request_id=i, prompt_ids=list(range(1, 81)),
                                         max_new_tokens=64))
    done = engine.run()
    assert sorted((r.request_id, len(r.output_ids)) for r in done) == [(0, 64), (1, 64)]
