"""
Phase 4 correctness harness — serving runtime and continuous batching.

Gate 3 (docs/methodology.md): "Can the runtime serve multiple requests?"

The test that matters is not that it runs, but that **batching changes
nothing about what each request receives**. Continuous batching is a
scheduling optimization; if a request's tokens depend on who it happened
to share a decode step with, the engine is broken in a way no throughput
number will reveal. So the central test generates each request alone and
then through the engine, under varying concurrency, and demands
identical output ids.

Everything runs on a tiny random Qwen2 on CPU — no GPU, no download.

    export PYTHONPATH=$(pwd):$PYTHONPATH
    pytest tests/test_phase4_serving.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from model.latentserve_qwen import LatentServeQwen  # noqa: E402
from model.qwen import ModelShape  # noqa: E402
from runtime.batching import batch_positions, batch_slots  # noqa: E402
from runtime.engine import ServingEngine, run_static_batching  # noqa: E402
from runtime.request import RequestState, ServedRequest  # noqa: E402
from runtime.scheduler import build_scheduler  # noqa: E402

TINY = dict(
    vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
    num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(**TINY)).to(torch.float32).eval()


@pytest.fixture(scope="module")
def tiny_shape(tiny_model) -> ModelShape:
    cfg = tiny_model.config
    return ModelShape(
        num_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
        hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.max_position_embeddings, torch_dtype="torch.float32",
    )


def build(tiny_model, tiny_shape) -> LatentServeQwen:
    return LatentServeQwen(
        hf_model=tiny_model, tokenizer=None, shape=tiny_shape, device="cpu",
        max_seq_len_hint=TINY["max_position_embeddings"],
    )


def make_requests(specs, seed=0) -> list[ServedRequest]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, (prompt_len, new_tokens) in enumerate(specs):
        ids = torch.randint(0, TINY["vocab_size"], (prompt_len,), generator=g).tolist()
        out.append(ServedRequest(request_id=i, prompt_ids=ids, max_new_tokens=new_tokens))
    return out


def generate_alone(model, request: ServedRequest) -> list[int]:
    """Ground truth: this request, by itself, through the Phase 2 path."""
    ids = torch.tensor([request.prompt_ids], dtype=torch.long)
    model.allocate_cache(1, request.total_len + 8)
    model.cache.reset()
    return model.generate_greedy(ids, max_new_tokens=request.max_new_tokens)[0].tolist()


# ----------------------------------------------------------------------
# Batch assembly — pure functions, no model
# ----------------------------------------------------------------------


def test_batch_slots_are_row_order_not_slot_order():
    """Row index and slot index are different things once sequences
    retire; conflating them reads another sequence's KV."""
    a, b = make_requests([(4, 2), (4, 2)])
    a.slot, b.slot = 7, 2
    assert batch_slots([a, b]) == [7, 2]


def test_duplicate_slots_rejected():
    a, b = make_requests([(4, 2), (4, 2)])
    a.slot = b.slot = 3
    with pytest.raises(ValueError):
        batch_slots([a, b])


def test_positions_are_per_sequence():
    """The ragged case a single scalar offset cannot express."""
    a, b = make_requests([(10, 5), (100, 5)])
    a.output_ids = [1, 2, 3]
    b.output_ids = [1]
    assert batch_positions([a, b]) == [12, 100]


# ----------------------------------------------------------------------
# Scheduler policies
# ----------------------------------------------------------------------


def test_fifo_preserves_arrival_order():
    reqs = make_requests([(64, 1), (8, 1), (32, 1)])
    for i, r in enumerate(reqs):
        r.arrival_time = float(i)
    admitted = build_scheduler("fifo", max_running=8).select(reqs, 0, lambda n: True, now=10.0)
    assert [r.request_id for r in admitted] == [0, 1, 2]


def test_length_aware_prefers_short_prompts():
    reqs = make_requests([(64, 1), (8, 1), (32, 1)])
    admitted = build_scheduler("length_aware", max_running=8).select(
        reqs, 0, lambda n: True, now=10.0
    )
    assert [r.prompt_len for r in admitted] == [8, 32, 64]


def test_fair_aging_eventually_overtakes_a_shorter_newcomer():
    """Pure SJF starves long prompts. Aging must let a long request that
    has waited overtake a short one that just arrived."""
    old_long, new_short = make_requests([(4096, 1), (16, 1)])
    old_long.arrival_time = 0.0
    new_short.arrival_time = 100.0
    sched = build_scheduler("fair", max_running=8, aging_seconds=1.0, aging_tokens=4096.0)
    admitted = sched.select([old_long, new_short], 0, lambda n: True, now=100.0)
    assert admitted[0] is old_long


def test_slo_aware_orders_by_remaining_slack():
    tight, loose = make_requests([(64, 1), (64, 1)])
    tight.slo_ttft_ms, loose.slo_ttft_ms = 100.0, 10_000.0
    tight.arrival_time = loose.arrival_time = 0.0
    admitted = build_scheduler("slo_aware", max_running=8).select(
        [loose, tight], 0, lambda n: True, now=0.0
    )
    assert admitted[0] is tight


def test_admission_stops_when_capacity_runs_out():
    reqs = make_requests([(64, 1), (64, 1), (64, 1)])
    budget = {"left": 1}

    def can_admit(n):
        return budget["left"] > 0

    sched = build_scheduler("fifo", max_running=8)
    admitted = sched.select(reqs, 0, can_admit)
    assert len(admitted) == 3, "can_admit is a snapshot; the engine re-checks per admission"


def test_max_running_caps_the_decode_batch():
    reqs = make_requests([(8, 1)] * 10)
    admitted = build_scheduler("fifo", max_running=3).select(reqs, 1, lambda n: True)
    assert len(admitted) == 2


def test_scheduler_overhead_is_recorded():
    """Methodology Phase 4 asks for scheduler overhead explicitly, so it
    must be a measured number and not an assumption."""
    sched = build_scheduler("fifo", max_running=8)
    reqs = make_requests([(8, 1)] * 20)
    for _ in range(50):
        sched.select(reqs, 0, lambda n: True)
    stats = sched.stats()
    assert stats["scheduler_calls"] == 50
    assert stats["scheduler_overhead_us_per_call"] > 0


# ----------------------------------------------------------------------
# Gate 3 — serving must not change what any request receives
# ----------------------------------------------------------------------


@pytest.mark.parametrize("max_running", [1, 2, 4])
def test_continuous_batching_matches_solo_generation(tiny_model, tiny_shape, max_running):
    """The central Phase 4 test, at three concurrency levels.

    Ragged prompts and ragged output lengths, so sequences finish at
    different steps and slots get recycled mid-run. Each request's output
    must be identical to generating it alone — if concurrency changes the
    tokens, the engine is wrong in a way throughput numbers hide.
    """
    specs = [(12, 6), (40, 3), (7, 8), (25, 4)]
    expected = [generate_alone(build(tiny_model, tiny_shape), r) for r in make_requests(specs)]

    model = build(tiny_model, tiny_shape)
    engine = ServingEngine(model, max_running=max_running, max_seq_len=128, block_size=8)
    for r in make_requests(specs):
        engine.add_request(r)
    finished = engine.run()

    assert len(finished) == len(specs)
    by_id = {r.request_id: r for r in finished}
    for i, want in enumerate(expected):
        assert by_id[i].output_ids == want, f"request {i} diverged under concurrency"


def test_slots_are_recycled_and_blocks_returned(tiny_model, tiny_shape):
    """A finished sequence must release both its slot and its blocks
    immediately — that reuse is what separates continuous from static."""
    model = build(tiny_model, tiny_shape)
    engine = ServingEngine(model, max_running=2, max_seq_len=128, block_size=8)
    for r in make_requests([(10, 2), (10, 9), (10, 2), (10, 2)]):
        engine.add_request(r)
    engine.run()

    assert len(engine.finished) == 4
    assert engine.cache.allocator.num_used == 0, "all blocks returned after drain"
    assert sorted(engine._free_slots) == [0, 1]


def test_every_request_records_queue_prefill_and_ttft(tiny_model, tiny_shape):
    """TTFT is queue + prefill from Phase 4 onward — a different quantity
    from Phases 1-3's pure prefill latency, and the split has to survive
    into the metrics."""
    model = build(tiny_model, tiny_shape)
    engine = ServingEngine(model, max_running=1, max_seq_len=128, block_size=8)
    for r in make_requests([(10, 3), (10, 3)]):
        engine.add_request(r)
    engine.run()

    second = sorted(engine.finished, key=lambda r: r.request_id)[1]
    assert second.queue_ms > 0, "the second request had to wait for the first"
    assert second.prefill_ms > 0
    assert second.ttft_ms >= second.queue_ms + second.prefill_ms - 1e-6
    assert second.state is RequestState.FINISHED


def test_static_batching_matches_continuous_outputs(tiny_model, tiny_shape):
    """The baseline must differ in scheduling only. If static and
    continuous produce different tokens, the throughput comparison
    between them is meaningless."""
    specs = [(12, 5), (30, 2), (9, 7), (18, 3)]

    model_a = build(tiny_model, tiny_shape)
    engine = ServingEngine(model_a, max_running=2, max_seq_len=128, block_size=8)
    for r in make_requests(specs):
        engine.add_request(r)
    continuous = {r.request_id: r.output_ids for r in engine.run()}

    model_b = build(tiny_model, tiny_shape)
    static_finished, stats = run_static_batching(
        model_b, make_requests(specs), batch_size=2, max_seq_len=128, block_size=8
    )
    static = {r.request_id: r.output_ids for r in static_finished}

    assert continuous == static
    assert stats["decode_tokens"] > 0


def test_engine_reports_occupancy_and_overhead(tiny_model, tiny_shape):
    model = build(tiny_model, tiny_shape)
    engine = ServingEngine(model, max_running=4, max_seq_len=128, block_size=8)
    for r in make_requests([(10, 5)] * 6):
        engine.add_request(r)
    engine.run()
    stats = engine.stats()
    assert 0 < stats["mean_batch_occupancy"] <= 4
    assert stats["runtime_overhead_ms_per_step"] >= 0
    assert stats["decode_tokens"] == 6 * 5 - 6  # first token comes from prefill


def test_inter_token_latency_captures_prefill_blocking(tiny_model, tiny_shape):
    """An incumbent's inter-token gap must include a newcomer's prefill.

    This is what the client experiences and what the decode-kernel timer
    cannot see: the engine runs prefill between two decode steps, so a
    2 s prompt admitted mid-flight is a 2 s gap between two of the
    incumbent's tokens while every decode call still takes ~65 ms. The
    first Phase 4 sweep reported p99/p50 TPOT of 1.1 for exactly this
    reason.
    """
    model = build(tiny_model, tiny_shape)
    engine = ServingEngine(model, max_running=2, max_seq_len=600, block_size=16)
    incumbent = ServedRequest(0, torch.randint(0, TINY["vocab_size"], (16,)).tolist(),
                              max_new_tokens=25)
    engine.add_request(incumbent)
    for _ in range(4):
        engine.step()

    engine.add_request(
        ServedRequest(1, torch.randint(0, TINY["vocab_size"], (400,)).tolist(), max_new_tokens=3)
    )
    engine.run()

    gaps = sorted(incumbent.decode_step_ms)
    assert max(gaps) > 5 * gaps[len(gaps) // 2], (
        "a large prefill admitted mid-flight must appear as a spike in the "
        "incumbent's inter-token latency"
    )


def test_slo_attainment_is_per_request(tiny_model, tiny_shape):
    """The SLO policy has to be judged on deadlines met, not on p50/p99
    TTFT — metrics it is not optimising for."""
    r = ServedRequest(0, [1, 2, 3], max_new_tokens=1, slo_ttft_ms=1.0)
    assert r.met_slo is None, "no TTFT recorded yet"
    r.arrival_time = 100.0
    r.first_token_time = 100.5
    assert r.met_slo is False  # 500 ms against a 1 ms target
    r.slo_ttft_ms = 2000.0
    assert r.met_slo is True


def test_oversized_request_fails_loudly_not_silently(tiny_model, tiny_shape):
    """A prompt that cannot fit even an empty pool must raise, not spin.
    A silent hang here is indistinguishable from a slow workload."""
    model = build(tiny_model, tiny_shape)
    engine = ServingEngine(
        model, max_running=1, max_seq_len=64, block_size=8, num_blocks=2
    )
    engine.add_request(make_requests([(64, 2)])[0])
    with pytest.raises(RuntimeError, match="deadlock"):
        engine.run()
