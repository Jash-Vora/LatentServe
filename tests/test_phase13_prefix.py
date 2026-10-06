"""Phase 13: prefix caching — exact, accounted, evictable, collision-safe."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cache.block_allocator import BlockAllocator  # noqa: E402
from cache.prefix_cache import PrefixCache, block_keys  # noqa: E402


def _model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=2048)
    hf = Qwen2ForCausalLM(cfg).eval()
    return LatentServeQwen(hf_model=hf, tokenizer=None,
                           shape=ModelShape(2, 8, 2, 16, 128, 128, 2048, "torch.float32"),
                           device="cpu", attn_impl="sdpa", max_seq_len_hint=1024)


def _serve(ls, prompts, prefix_caching, max_running=4, num_blocks=None, max_new=12, arrive=None):
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    engine = ServingEngine(ls, max_running=max_running, max_seq_len=512, block_size=16,
                           num_blocks=num_blocks, prefix_caching=prefix_caching)
    reqs = [ServedRequest(request_id=i, prompt_ids=list(p), max_new_tokens=max_new)
            for i, p in enumerate(prompts)]
    if arrive == "sequential":
        out = {}
        for r in reqs:
            engine.add_request(r)
            for d in engine.run():
                out[d.request_id] = d
        return engine, out
    for r in reqs:
        engine.add_request(r)
    return engine, {d.request_id: d for d in engine.run()}


SYSTEM = list(range(3, 3 + 70))                         # 70 shared tokens: 4 full blocks


def _prompts(n=4):
    g = torch.Generator().manual_seed(1)
    return [SYSTEM + torch.randint(0, 128, (int(torch.randint(5, 40, (1,), generator=g)),),
                                   generator=g).tolist() for _ in range(n)]


@pytest.mark.parametrize("arrive", ["sequential", "concurrent"])
def test_outputs_are_identical_with_and_without_prefix_caching(arrive):
    prompts = _prompts()
    ls = _model()
    _, off = _serve(ls, prompts, False, arrive=arrive)
    eng, on = _serve(ls, prompts, True, arrive=arrive)
    for i in off:
        assert on[i].output_ids == off[i].output_ids, f"request {i} differs"
    if arrive == "sequential":
        # The first request computes everything; the rest reuse the shared
        # system prompt's 4 full blocks.
        assert [on[i].prefix_hit_tokens for i in range(4)] == [0, 64, 64, 64]
        assert eng.prefix_hit_tokens == 192


def test_an_identical_prompt_still_prefills_its_last_token():
    ls = _model()
    prompt = list(range(5, 5 + 64))                       # exactly 4 blocks
    eng, out = _serve(ls, [prompt, prompt], True, arrive="sequential")
    assert out[1].prefix_hit_tokens == 48                  # 3 blocks: the 4th holds the last token
    assert out[0].output_ids == out[1].output_ids


def test_a_second_turn_reuses_the_first_turns_reply():
    ls = _model()
    first = SYSTEM + list(range(90, 110))
    eng, out = _serve(ls, [first], True, max_new=30)
    reply = out[0].output_ids
    second = first + reply + list(range(60, 70))
    from runtime.request import ServedRequest

    eng.add_request(ServedRequest(request_id=1, prompt_ids=second, max_new_tokens=8))
    r2 = {d.request_id: d for d in eng.run()}[1]
    written = len(first) + len(reply) - 1                   # the last reply token is never fed back
    assert r2.prefix_hit_tokens == (written // 16) * 16
    assert r2.prefix_hit_tokens > len(SYSTEM)               # reached into the generated tokens
    _, ref = _serve(_model(), [second], False, max_new=8)
    assert r2.output_ids == ref[0].output_ids


def test_eviction_under_a_small_pool_stays_exact():
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 128, (60,), generator=g).tolist() for _ in range(8)]
    ls = _model()
    _, off = _serve(ls, prompts, False, max_running=2, num_blocks=12)
    eng, on = _serve(ls, prompts, True, max_running=2, num_blocks=12)
    assert {i: r.output_ids for i, r in on.items()} == {i: r.output_ids for i, r in off.items()}
    assert eng.prefix.evictions > 0


def test_after_everything_finishes_blocks_are_free_or_held_only_by_the_cache():
    ls = _model()
    eng, _ = _serve(ls, _prompts(6), True)
    alloc = eng.cache.allocator
    cached = set(eng.prefix.entry)
    assert all(alloc.ref_count(b) == 1 for b in cached)
    assert alloc.num_free + len(cached) == alloc.num_blocks
    assert eng.prefix.num_reclaimable() == len(cached)


def test_a_match_is_verified_against_the_stored_tokens():
    alloc = BlockAllocator(num_blocks=8, block_size=4)
    pc = PrefixCache(alloc, 4)
    blocks = alloc.allocate(2)
    pc.register([1, 2, 3, 4, 5, 6, 7, 8], blocks)
    assert pc.match([1, 2, 3, 4, 5, 6, 7, 8, 9]) == blocks
    # Simulate a collision: same hash, different tokens stored.
    h, parent, _ = pc.entry[blocks[0]]
    pc.entry[blocks[0]] = (h, parent, (9, 9, 9, 9))
    assert pc.match([1, 2, 3, 4, 5, 6, 7, 8, 9]) == []


def test_block_keys_chain_so_equal_blocks_after_different_prefixes_differ():
    a = block_keys([1, 2, 3, 4, 7, 7, 7, 7], 4)
    b = block_keys([9, 9, 9, 9, 7, 7, 7, 7], 4)
    assert a[1][2] == b[1][2] and a[1][0] != b[1][0]


# ------------------------------------------------- stacking with INT8 ---


def _serve_kv(ls, prompts, prefix_caching, kv_dtype, graphs=False, max_new=12, sequential=True):
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    engine = ServingEngine(ls, max_running=4, max_seq_len=512, block_size=16,
                           prefix_caching=prefix_caching, kv_dtype=kv_dtype,
                           use_cuda_graphs=graphs)
    out = {}
    for i, p in enumerate(prompts):
        engine.add_request(ServedRequest(request_id=i, prompt_ids=list(p), max_new_tokens=max_new))
        if sequential:
            out.update({d.request_id: d for d in engine.run()})
    if not sequential:
        out.update({d.request_id: d for d in engine.run()})
    return engine, out


def test_prefix_caching_stacks_with_int8_exactly():
    """Shared blocks carry their quantized K, per-block scales and per-token
    V along: reuse must be exact for INT8 too."""
    prompts = _prompts()
    ls = _model()
    _, off = _serve_kv(ls, prompts, False, "int8")
    eng, on = _serve_kv(ls, prompts, True, "int8")
    assert {i: r.output_ids for i, r in on.items()} == {i: r.output_ids for i, r in off.items()}
    assert eng.prefix_hit_tokens == 192


try:
    from kernels.cuda import paged_decode_cuda as _pdc

    _pdc.compile_cubin("sm_75")
    _GPU = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    _GPU = False


@pytest.mark.skipif(not _GPU, reason="needs a GPU and CuPy")
def test_int8_under_graphs_never_shares_an_unfinalized_block():
    """Under CUDA graphs INT8 finalizes a block one step late. A conversation's
    second turn, reusing the first turn's reply, must match a run without
    prefix caching — sharing a not-yet-finalized block would not."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from kernels.gqa import paged_decode as pd
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128 * 12, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                      max_position_embeddings=2048)
    hf = Qwen2ForCausalLM(cfg).half().cuda().eval()
    shape = ModelShape(2, 12, 2, 128, 128 * 12, 128, 2048, "torch.float16")

    def fresh():
        return LatentServeQwen(hf_model=hf, tokenizer=None, shape=shape, device="cuda",
                               attn_impl="triton_paged", max_seq_len_hint=1024)

    before = pd.decode_backend()
    pd.set_decode_backend("cuda")
    try:
        first = SYSTEM + list(range(90, 110))
        eng, out = _serve_kv(fresh(), [first], True, "int8", graphs=True, max_new=33)
        second = first + out[0].output_ids + list(range(60, 70))
        from runtime.request import ServedRequest

        eng.add_request(ServedRequest(request_id=1, prompt_ids=second, max_new_tokens=10))
        got = {d.request_id: d for d in eng.run()}[1]
        _, ref = _serve_kv(fresh(), [second], False, "int8", graphs=True, max_new=10)
        assert got.prefix_hit_tokens > len(SYSTEM)
        assert got.output_ids == ref[0].output_ids
    finally:
        pd.set_decode_backend(before)


def test_the_o1_reclaimable_count_always_equals_the_scan():
    """Checked after a workload with sharing, finishing and eviction."""
    g = torch.Generator().manual_seed(4)
    prompts = [SYSTEM + torch.randint(0, 128, (30,), generator=g).tolist() for _ in range(5)]
    prompts += [torch.randint(0, 128, (60,), generator=g).tolist() for _ in range(5)]
    ls = _model()
    eng, _ = _serve(ls, prompts, True, max_running=2, num_blocks=14)
    assert eng.prefix.num_reclaimable() == eng.prefix._count_reclaimable()
    assert eng.prefix.evictions > 0


@pytest.mark.parametrize("hooked", [False, True])
def test_free_returns_a_block_to_the_pool_only_at_zero(hooked):
    """Guards an edit that once captured the pool return inside the new
    callback branch: with a hook, shared blocks went back while still in use
    (double free); without one, nothing went back at all (a leak)."""
    alloc = BlockAllocator(num_blocks=4, block_size=4)
    seen = []
    if hooked:
        alloc.on_ref_change = lambda b, old, new: seen.append((b, old, new))
    [b] = alloc.allocate(1)
    alloc.incref([b])                                     # shared: count 2
    alloc.free([b])
    assert alloc.ref_count(b) == 1 and alloc.num_free == 3
    alloc.free([b])
    assert alloc.ref_count(b) == 0 and alloc.num_free == 4
    if hooked:
        assert seen == [(b, 1, 2), (b, 2, 1), (b, 1, 0)]


def test_benchmark_workloads_have_the_intended_shape():
    from benchmarks.runners.phase13_prefix import workload

    shared = workload("shared", 0)["requests"]
    assert len(shared) == 32 and all(r["prompt"][:2048] == shared[0]["prompt"][:2048] for r in shared)
    assert len({tuple(r["prompt"][2048:]) for r in shared}) == 32      # distinct messages
    none = workload("none", 0)["requests"]
    assert len({tuple(r["prompt"][:16]) for r in none}) == 32          # nothing shared
    chat = workload("chat", 0)
    assert chat["chat"] and chat["turns"] == 5 and len(chat["requests"]) == 8


def test_peak_live_blocks_exclude_what_the_cache_merely_keeps():
    g = torch.Generator().manual_seed(6)
    prompts = [torch.randint(0, 128, (60,), generator=g).tolist() for _ in range(6)]
    ls = _model()
    eng, _ = _serve(ls, prompts, True, max_running=2, arrive="sequential")
    alloc = eng.cache.allocator
    assert len(eng.prefix.entry) > 0
    assert eng.prefix.peak_live <= alloc.peak_used
    assert eng.prefix.peak_live < alloc.peak_used          # cached-only blocks inflate peak_used
