"""Phase 18: the router, percentiles, and the replica harness's orchestration."""

from __future__ import annotations

import random

import pytest

from runtime.router import Router, percentiles

SYSTEM = list(range(100, 100 + 512))


def test_round_robin_alternates_and_least_loaded_balances():
    rr = Router(2, policy="round_robin")
    assert [rr.route([1] * 40) for _ in range(4)] == [0, 1, 0, 1]
    ll = Router(2, policy="least_loaded")
    a, b = ll.route([1]), ll.route([1])
    assert {a, b} == {0, 1}
    ll.done(a)
    assert ll.route([1]) == a


def test_prefix_aware_sends_a_conversation_home():
    r = Router(2, policy="prefix_aware")
    turn1 = SYSTEM + list(range(1, 129))
    home = r.route(turn1)
    r.done(home)
    other = r.route(SYSTEM + list(range(500, 628)))          # another conversation
    r.done(other)
    turn2 = turn1 + list(range(900, 1156))                   # same conversation, deeper
    assert r.route(turn2) == home
    assert r.affinity_hits >= 1


def test_prefix_aware_balances_new_conversations_despite_a_shared_system_prompt():
    """Pure affinity would pile every conversation onto the replica that first
    saw the system prompt; the slack rule spreads them."""
    r = Router(2, policy="prefix_aware", slack=2)
    rng = random.Random(0)
    for _ in range(16):
        r.route(SYSTEM + [rng.randrange(1000, 9999) for _ in range(128)])
    assert abs(r.routed[0] - r.routed[1]) <= 3


def test_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        Router(2, policy="random")


def test_percentiles_are_nearest_rank():
    xs = list(range(1, 101))
    assert percentiles(xs) == {"p50": 50, "p95": 95, "p99": 99}
    assert percentiles([7.0]) == {"p50": 7.0, "p95": 7.0, "p99": 7.0}


class FakeCluster:
    """In-process stand-in for the worker processes: a request finishes when
    received, with synthetic but consistent times and a reply of its own."""

    def __init__(self):
        import time

        # From the real clock: the harness stamps submissions with
        # time.perf_counter() (system uptime on Linux). A fixed 100.0 made the
        # makespan negative on any machine up for more than ~100 s, and the
        # test passed only because two negatives multiplied back positive.
        self.inbox, self.clock = [], time.perf_counter()

    def submit(self, gpu, rid, prompt, max_new):
        self.clock += 0.01
        self.inbox.append(("done", {"rid": rid, "gpu": gpu, "n_out": max_new,
                                    "output_ids": [rid % 97] * max_new,
                                    "t_first": self.clock + 0.05, "t_finish": self.clock + 0.5,
                                    "prompt_len": len(prompt), "hit": len(prompt) // 2}))

    def recv(self, timeout=None):
        return self.inbox.pop(0)


def test_burst_and_chat_run_through_the_harness():
    from benchmarks.runners import phase18_replicas as pr

    res = pr.run_workload(FakeCluster(), Router(2, policy="least_loaded"), pr.burst(0, n=10), 2)
    assert res["makespan_s"] > 0 and res["out_tok_s"] > 0
    assert res["requests"] == 10
    assert sum(res["tokens_per_gpu"]) == res["out_tokens"]
    assert res["out_tok_s"] * res["makespan_s"] == pytest.approx(res["out_tokens"])
    assert set(res["ttft_ms"]) == {"p50", "p95", "p99"}

    wl = pr.chat(0, convs=3, turns=4)
    res = pr.run_workload(FakeCluster(), Router(2, policy="prefix_aware"), wl, 2)
    assert res["requests"] == 3 * 4                         # every turn of every conversation
    assert res["hit_rate"] == pytest.approx(0.5, abs=0.01)



def test_capture_is_excluded_as_the_longest_parallel_capture():
    from benchmarks.runners.phase18_replicas import ex_capture

    res = {"makespan_s": 50.0, "out_tokens": 4600, "capture_s": [3.0, 4.0]}
    assert ex_capture(res) == pytest.approx(4600 / 46.0)        # GPUs capture in parallel
    assert ex_capture({"makespan_s": 50.0, "out_tokens": 5000, "capture_s": []}) == 100.0


class _FakeOut:
    def __init__(self, rid, toks, finished, cached=0):
        class _C:
            token_ids = toks
        self.request_id, self.outputs, self.finished, self.num_cached_tokens = rid, [_C()], finished, cached


def test_the_vllm_adapter_reports_first_tokens_and_completions():
    """vLLM behind the worker's interface, with a stand-in for its engine."""
    from benchmarks.runners.phase18_replicas import _VLLMEngine
    from runtime.request import ServedRequest

    script = [[_FakeOut("7", [1], False)], [_FakeOut("7", [1, 2], False)],
              [_FakeOut("7", [1, 2, 3], True, cached=32)]]

    class Eng:
        def has_unfinished_requests(self):
            return bool(script)

        def step(self):
            return script.pop(0)

    class LLM:
        llm_engine = Eng()

    eng = _VLLMEngine(LLM())
    req = ServedRequest(request_id=7, prompt_ids=list(range(40)), max_new_tokens=3)
    eng.reqs["7"] = req                                   # as add_request would
    done = []
    eng.on_retire = lambda r, e: done.append(r)
    eng.step()
    t_first = req.first_token_time
    assert t_first is not None and not done
    eng.step()
    assert req.first_token_time == t_first                # first token stamped once
    eng.step()
    assert done == [req] and req.output_ids == [1, 2, 3]
    assert req.finish_time >= t_first and req.prefix_hit_tokens == 32
    assert not eng.has_work


def test_aggregate_takes_medians_and_reports_the_spread():
    from benchmarks.runners.phase18_replicas import aggregate

    def run(tps, ttft):
        return {"out_tok_s_ex_capture": tps, "out_tok_s": tps, "hit_rate": 0.5,
                "ttft_ms": {"p50": ttft, "p95": ttft, "p99": ttft},
                "tpot_ms": {"p50": 1.0, "p95": 1.0, "p99": 1.0},
                "e2e_ms": {"p50": 2.0, "p95": 2.0, "p99": 2.0}, "busy_frac": [0.9, 0.8]}

    agg = aggregate([run(100.0, 10.0), run(110.0, 30.0), run(90.0, 20.0)])
    assert agg["out_tok_s_ex_capture"] == 100.0 and agg["ttft_ms"]["p50"] == 20.0
    assert agg["tok_s_spread"] == pytest.approx(0.2)


def test_a_worker_failure_surfaces_with_its_traceback():
    """A worker that fails must fail the run quickly and say why — the first
    vLLM run's workers died silently and the parent waited 15 minutes."""
    import queue as q

    from benchmarks.runners import phase18_replicas as pr

    import os

    # worker_main sets CUDA_VISIBLE_DEVICES for its process — here, the test
    # process: restore it, or later tests could lose the second GPU.
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    out = q.Queue()
    try:
        pr.worker_main(0, "no-such-backend", q.Queue(), out, {})
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    kind, gpu, msg = out.get_nowait()
    assert kind == "error" and gpu == 0 and "no-such-backend" in msg


# ------------------------------------------------- sweep harness pieces ---


def test_measure_steps_times_a_full_batch_and_restores_the_hook():
    """On a tiny CPU engine: every request decoding before timing starts, the
    requested number of steps timed, and the worker's own hook back after."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import benchmarks.runners.phase18_replicas as pr
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)
    ls = LatentServeQwen(hf_model=Qwen2ForCausalLM(cfg).eval(), tokenizer=None,
                         shape=ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32"),
                         device="cpu", attn_impl="sdpa", max_seq_len_hint=512)
    engine = ServingEngine(ls, max_running=4, max_seq_len=256, block_size=16)
    mine = lambda r, e: None  # noqa: E731
    engine.on_retire = mine
    res = pr._measure_steps(engine, ServedRequest, batch=3, ctx=40, steps=5, warmup=2, seed=1,
                            vocab=(0, 128))
    assert res["fits"] and len(res["step_ms"]) == 5
    assert engine.on_retire is mine
    assert not engine.has_work


def test_measure_steps_refuses_a_batch_that_cannot_fit():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import benchmarks.runners.phase18_replicas as pr
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)
    ls = LatentServeQwen(hf_model=Qwen2ForCausalLM(cfg).eval(), tokenizer=None,
                         shape=ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32"),
                         device="cpu", attn_impl="sdpa", max_seq_len_hint=512)
    engine = ServingEngine(ls, max_running=4, max_seq_len=256, block_size=16, num_blocks=10)
    res = pr._measure_steps(engine, ServedRequest, batch=3, ctx=100, steps=5, warmup=2)
    assert not res["fits"] and "blocks" in res["reason"]


def test_timed_arrivals_are_released_on_schedule():
    import time as _t

    from benchmarks.runners import phase18_replicas as pr

    class Clock(FakeCluster):
        def __init__(self):
            super().__init__()
            self.sent_at = []

        def submit(self, gpu, rid, prompt, max_new):
            self.sent_at.append(_t.perf_counter())
            super().submit(gpu, rid, prompt, max_new)

        def recv(self, timeout=None):
            if not self.inbox:
                import queue

                _t.sleep(timeout or 0)
                raise queue.Empty
            return self.inbox.pop(0)

    reqs = [{"rid": i, "prompt": [1] * 32, "max_new": 4, "at": 0.05 * i} for i in range(4)]
    c = Clock()
    res = pr.run_workload(c, Router(1), {"kind": "open", "requests": reqs}, 1)
    gaps = [b - a for a, b in zip(c.sent_at, c.sent_at[1:])]
    assert res["requests"] == 4 and all(0.035 < g < 0.2 for g in gaps)


def test_a_sequential_workload_sends_one_request_at_a_time():
    from benchmarks.runners import phase18_replicas as pr

    class Seq(FakeCluster):
        outstanding = 0
        peak = 0

        def submit(self, gpu, rid, prompt, max_new):
            Seq.outstanding += 1
            Seq.peak = max(Seq.peak, Seq.outstanding)
            super().submit(gpu, rid, prompt, max_new)

        def recv(self, timeout=None):
            Seq.outstanding -= 1
            return self.inbox.pop(0)

    reqs = [{"rid": i, "prompt": [1] * 32, "max_new": 1} for i in range(5)]
    res = pr.run_workload(Seq(), Router(1), {"kind": "sequential", "requests": reqs}, 1)
    assert res["requests"] == 5 and Seq.peak == 1
