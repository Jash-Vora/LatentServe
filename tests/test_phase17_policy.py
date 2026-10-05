"""Phase 17: the sparsity policy, and its wiring into the engine."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.policy import ANSWER_LOSS, SparsePolicy, StepTable, expected_answer_loss  # noqa: E402

# A small measured table: sparsity buys nothing at batch 1, a lot at batch 16.
ROWS = []
for b in (1, 16):
    for c in (2048, 32768):
        dense = 17.0 + (0.0 if b == 1 else 0.0004 * b * c)
        attn_frac = 0.0 if b == 1 else 1.0
        ROWS += [{"batch": b, "ctx": c, "ratio": None, "ms": dense},
                 {"batch": b, "ctx": c, "ratio": 0.5, "ms": 17.3 + attn_frac * 0.0004 * b * c * 0.55},
                 {"batch": b, "ctx": c, "ratio": 0.375, "ms": 17.4 + attn_frac * 0.0004 * b * c * 0.43}]
TABLE = StepTable(ROWS)


def test_prediction_is_exact_on_measured_points_and_between_them_otherwise():
    assert TABLE.predict(16, 32768, None) == pytest.approx(17.0 + 0.0004 * 16 * 32768)
    mid = TABLE.predict(4, 8192, None)
    lo, hi = TABLE.predict(1, 2048, None), TABLE.predict(16, 32768, None)
    assert lo < mid < hi


def test_missing_corners_fall_back_to_the_nearest_measured_point():
    rows = [r for r in ROWS if not (r["batch"] == 16 and r["ctx"] == 32768)]
    t = StepTable(rows)
    assert t.predict(12, 24000, None) == t._points(None)[(16, 2048)] or \
        t.predict(12, 24000, None) == t._points(None)[(1, 32768)]


def test_dense_where_sparsity_buys_nothing():
    pol = SparsePolicy(TABLE, tier="relaxed")
    assert pol.choose(1, 32768) is None                  # batch 1: weights dominate
    assert pol.choose(16, 32768) == 0.375                # attention dominates: sparsest wins


def test_tiers_bound_what_may_be_chosen():
    assert SparsePolicy(TABLE, tier="strict").choose(16, 32768) is None
    assert SparsePolicy(TABLE, tier="balanced").choose(16, 32768) == 0.5
    assert not SparsePolicy(TABLE, tier="strict").may_sparsify
    with pytest.raises(ValueError):
        SparsePolicy(TABLE, tier="reckless")


def test_min_gain_is_the_price_of_quality():
    """A budget that wins by less than min_gain is not worth its answer cost."""
    small = SparsePolicy(TABLE, tier="relaxed", min_gain=0.0)
    big = SparsePolicy(TABLE, tier="relaxed", min_gain=10.0)
    assert small.choose(16, 2048) is not None
    assert big.choose(16, 32768) is None


def test_expected_answer_loss_weights_rates_by_tokens():
    loss = expected_answer_loss({None: 50, 0.5: 30, 0.375: 20})
    assert loss == pytest.approx((30 * ANSWER_LOSS[0.5] + 20 * ANSWER_LOSS[0.375]) / 100)
    assert expected_answer_loss({}) == 0.0


def test_the_engine_counts_tokens_per_budget_on_the_eager_path():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    class Always:
        may_sparsify = True

        def choose(self, batch, ctx):
            return 0.5 if batch >= 2 else None

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)
    hf = Qwen2ForCausalLM(cfg).eval()
    ls = LatentServeQwen(hf_model=hf, tokenizer=None,
                         shape=ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32"),
                         device="cpu", attn_impl="triton_paged", max_seq_len_hint=512)
    engine = ServingEngine(ls, max_running=2, max_seq_len=256, block_size=16, policy=Always())
    assert ls.cache.page_bounds(0) is not None           # bounds on from the first write
    engine.add_request(ServedRequest(request_id=0, prompt_ids=list(range(1, 60)), max_new_tokens=12))
    engine.add_request(ServedRequest(request_id=1, prompt_ids=list(range(1, 40)), max_new_tokens=6))
    engine.run()
    counts = dict(engine.ratio_tokens)
    assert counts.get(0.5, 0) == 2 * 5                   # both running: 5 decode steps
    assert counts.get(None, 0) == 6                      # request 0 alone: 6 more


# ---------------------------------------------------------------- GPU ---

try:
    from kernels.cuda import paged_sparse as _ps

    _ps.compile_cubin("sm_75")
    _OK = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    _OK = False
requires_gpu = pytest.mark.skipif(not _OK, reason="needs a GPU, Triton and CuPy")


@requires_gpu
def test_per_sequence_budgets_match_the_reference_in_a_ragged_batch():
    """Each sequence attends to its own budget, not one sized for the longest:
    with selection sized for 4096 tokens, a 700-token sequence must attend
    to exactly the best ceil(0.375 x 44) = 17 of its pages."""
    import math

    from kernels.cuda import paged_sparse as ps

    g = torch.Generator(device="cuda").manual_seed(21)
    lens, page, h, d = [4096, 700], 16, 2, 128
    pages = 256
    nb = 2 * pages + 3
    k = torch.randn(nb, page, h, d, dtype=torch.float16, device="cuda", generator=g)
    v = torch.randn(nb, page, h, d, dtype=torch.float16, device="cuda", generator=g)
    tables = torch.randperm(nb, device="cuda", generator=g)[: 2 * pages].reshape(2, pages).to(torch.int32)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q = torch.randn(2, h, 6, d, dtype=torch.float16, device="cuda", generator=g)
    kmin = torch.empty(nb, h, d, dtype=torch.float16, device="cuda")
    kmax = torch.empty_like(kmin)
    ps.rebuild_bounds(k, kmin, kmax)
    got = ps.sparse_attention(q, k, v, tables, seq, kmin, kmax, ratio=0.375, max_seq_len=4096)
    scores = ps.page_scores(q, kmin, kmax, tables, seq)
    sel = ps.select(scores, ps.budget(0.375, pages), sorted=True)
    n = ps.budgets(seq, 0.375).tolist()
    assert n == [96, 17]
    for b, L in enumerate(lens):
        rows_k = torch.stack([k[int(tables[b, p // page]), p % page] for p in range(L)], 1).float()
        rows_v = torch.stack([v[int(tables[b, p // page]), p % page] for p in range(L)], 1).float()
        for hh in range(h):
            keep = torch.zeros(L, dtype=torch.bool, device="cuda")
            for p in sel[b, hh, :n[b]].tolist():
                keep[p * page:min(L, (p + 1) * page)] = True
            s_ = (q[b, hh].float() @ rows_k[hh].T) / math.sqrt(d)
            want = torch.softmax(s_.masked_fill(~keep[None, :], float("-inf")), -1) @ rows_v[hh]
            torch.testing.assert_close(got[b, hh].float(), want, atol=5e-3, rtol=1e-2)


@requires_gpu
def test_a_graph_keeps_its_budget_as_the_sequence_grows_through_the_bucket():
    """Captured early in a bucket and replayed later, a graph must attend to
    the pages an eager step at the *current* length would — the budget fixed
    at capture used to decay toward half its ratio through a doubling bucket."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from kernels.gqa import paged_decode as pd
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape
    from runtime.cuda_graph import GraphedDecoder

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128 * 12, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                      max_position_embeddings=8192)
    hf = Qwen2ForCausalLM(cfg).half().cuda().eval()
    ls = LatentServeQwen(hf_model=hf, tokenizer=None,
                         shape=ModelShape(2, 12, 2, 128, 128 * 12, 128, 8192, "torch.float16"),
                         device="cuda", attn_impl="triton_paged", max_seq_len_hint=4200)
    before = pd.decode_backend()
    pd.set_decode_backend("cuda")
    try:
        ls.set_sparse(0.375)
        ls.allocate_cache(1, 4200, paged=True, block_size=16)
        ls.cache.reset()
        prompt = torch.randint(0, 128, (1, 2100)).cuda()
        with torch.no_grad():
            ls.prefill(prompt)
            decoder = GraphedDecoder(ls)
            tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
            for i in range(1500):                      # 2100 -> 3600, inside the 4096 bucket
                pos = torch.full((1, 1), 2100 + i, dtype=torch.long, device="cuda")
                graphed = decoder.step(tok, pos, [0]).float().clone()
            # the same step, eager, at the current length
            ls.cache.rewind(3599)
            eager = ls.decode_step(tok).float()
        assert decoder.captures == 1
        torch.testing.assert_close(graphed[0, -1], eager[0, -1], atol=3e-2, rtol=3e-2)
    finally:
        pd.set_decode_backend(before)


def test_the_workload_mix_is_deterministic_and_mixed():
    from benchmarks.runners.phase17_workload import workload

    a, b = workload(40, 0), workload(40, 0)
    assert a == b
    lengths = {r["prompt_len"] for r in a}
    assert {2048, 30000} <= lengths and len(lengths) >= 4


def test_traffic_has_quiet_bursty_and_medium_load():
    from benchmarks.runners.phase17_workload import traffic

    reqs = traffic(0)
    assert reqs == traffic(0) and len(reqs) == 48
    burst = [r for r in reqs if r["phase"] == "burst"]
    assert len({r["arrival"] for r in burst}) == 1                 # all at once
    quiet = [r for r in reqs if r["phase"] == "quiet"][:12]
    gaps = {b["arrival"] - a["arrival"] for a, b in zip(quiet, quiet[1:])}
    assert gaps == {140} and max(r["prompt_len"] for r in quiet) <= 4096
    assert [r["arrival"] for r in reqs] == sorted(r["arrival"] for r in reqs)
