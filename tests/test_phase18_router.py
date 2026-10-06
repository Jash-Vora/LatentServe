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
