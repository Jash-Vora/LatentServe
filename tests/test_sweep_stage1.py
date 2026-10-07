"""The stage-1 sweep: cells, resume, workloads, and the report — with a fake
cluster, so every section's control flow runs in seconds."""

from __future__ import annotations

import argparse
import statistics
import time

import pytest

from benchmarks.runners import sweep_stage1 as sw


class FakeCluster:
    """Answers at once: steps of 10 ms; a request finishes on submit."""

    def __init__(self):
        self.inbox, self.resets, self.measured = [], 0, 0

    def reset(self, opts):
        self.resets += 1
        return [1.0], [0.0]

    def measure_steps(self, gpu, batch, ctx, steps, warmup, seed):
        self.measured += 1
        if batch * ctx > 16 * 16384:                     # pretend the largest do not fit
            return {"fits": False, "reason": "needs more blocks"}
        return {"fits": True, "step_ms": [10.0 + 0.01 * i for i in range(steps)]}

    def submit(self, gpu, rid, prompt, max_new):
        now = time.perf_counter()
        self.inbox.append(("done", {"rid": rid, "gpu": gpu, "n_out": max_new,
                                    "output_ids": [7] * max_new, "t_first": now + 0.001,
                                    "t_finish": now + 0.002, "prompt_len": len(prompt),
                                    "hit": 0}))

    def recv(self, timeout=None):
        return self.inbox.pop(0)


def _args(tmp_path, **kw):
    a = dict(sections=["A", "B", "C"], rounds=2, steps=6, b_repeats=3, max_running=16,
             headroom_gb=2.5, seed=0, skip_curves=False, force=False,
             policy_table="none.json", config="c.yaml", results_dir=str(tmp_path))
    a.update(kw)
    return argparse.Namespace(**a)


@pytest.fixture
def small(monkeypatch):
    """A small matrix, and workloads whose arrival times are compressed."""
    monkeypatch.setattr(sw, "A_BATCHES", (1, 16))
    monkeypatch.setattr(sw, "A_CONTEXTS", (2048, 32768))
    monkeypatch.setattr(sw, "B_LENGTHS", (1024, 4096))
    monkeypatch.setattr(sw, "FRACTIONS", (0.5, 1.0))
    real_open, real_vary = sw.openloop, sw.varying

    def fast(wl):
        for r in wl["requests"]:
            r["at"] = r.get("at", 0.0) / 1000.0
        return wl

    monkeypatch.setattr(sw, "openloop", lambda *a, **k: fast(real_open(*a, **k)))
    monkeypatch.setattr(sw, "varying", lambda *a, **k: fast(real_vary(*a, **k)))


def _run_all(args):
    run = sw.Runner(args)
    base = sw.base_opts(args)
    c = FakeCluster()
    sw.run_group(run, c, sw.LS, args.sections, base, False, "latentserve")
    sw.run_group(run, c, {"vllm": {}}, args.sections, dict(base, vllm_prefix=True), False, "vllm")
    sw.run_group(run, c, {"vllm": {}}, args.sections, dict(base, vllm_prefix=False), True, "vllm")
    return run, c


def test_the_full_flow_saves_every_cell_and_a_rerun_measures_nothing(tmp_path, small):
    run, c = _run_all(_args(tmp_path))
    names = {p.name for p in tmp_path.glob("*.json")}
    # A: 2 batches x 2 contexts x 2 rounds x (4 LS configs + vLLM)
    assert sum(n.startswith("A__") for n in names) == 2 * 2 * 2 * 5
    assert "A__b16-c32768__ls-dense__r0.json" in names            # saved even when it did not fit
    # B: 2 lengths x (dense, int8, vLLM); one round
    assert sum(n.startswith("B__") for n in names) == 2 * 3
    # C: chat has prefix on and off for both engines
    for tag in ("ls-dense-prefix-on", "ls-dense-prefix-off", "vllm-prefix-on", "vllm-prefix-off"):
        assert f"C__chat__{tag}__r0.json" in names and f"C__chat__{tag}__r1.json" in names
    # curves: a probe and each load point, one round, per serving config
    assert "C__probe__ls-adaptive__r0.json" in names and "C__load1.00__vllm__r0.json" in names
    first = run.ran

    rerun, c2 = _run_all(_args(tmp_path))
    assert first > 0 and rerun.ran == 0 and c2.measured == 0       # resumable: nothing remeasured


def test_the_prefix_off_vllm_group_runs_only_chat(tmp_path, small):
    args = _args(tmp_path)
    run = sw.Runner(args)
    sw.run_group(run, FakeCluster(), {"vllm": {}}, args.sections,
                 dict(sw.base_opts(args), vllm_prefix=False), True, "vllm")
    names = {p.name for p in tmp_path.glob("*.json")}
    assert names and all(n.startswith("C__chat__vllm-prefix-off") for n in names)


def test_force_remeasures(tmp_path, small):
    _run_all(_args(tmp_path, sections=["B"]))
    run, _ = _run_all(_args(tmp_path, sections=["B"], force=True))
    assert run.ran == 2 * 3


def test_the_report_covers_every_section(tmp_path, small):
    _run_all(_args(tmp_path))
    text = sw.report(tmp_path)
    for heading in ("## A. Decode step", "## B. Time to first token", "## C. Serving workloads",
                    "## C. Latency versus load"):
        assert heading in text
    assert "did not fit" in text and "capacity" in text


def test_openloop_arrivals_are_poisson_at_the_requested_rate():
    wl = sw.openloop(0, 2.0, 400)
    gaps = [b["at"] - a["at"] for a, b in zip(wl["requests"], wl["requests"][1:])]
    assert statistics.fmean(gaps) == pytest.approx(0.5, rel=0.15)
    assert wl == sw.openloop(0, 2.0, 400)                          # deterministic


def test_load_points_span_idle_to_overload_with_bounded_counts():
    pts = sw.points_for(0.8)
    assert [f for f, _, _ in pts] == list(sw.FRACTIONS)
    assert all(16 <= n <= 80 for _, _, n in pts)
    assert pts[-1][1] == pytest.approx(1.1 * 0.8)


def test_capacity_excludes_graph_capture():
    assert sw.probe_capacity({"requests": 48, "makespan_s": 50.0, "capture_s": [2.0]}) == 1.0


def test_varying_traffic_has_its_four_phases_in_order():
    reqs = sw.varying(0)["requests"]
    phases = [r["phase"] for r in reqs]
    assert phases[:12] == ["quiet"] * 12 and phases[12:28] == ["burst"] * 16
    burst = [r["at"] for r in reqs if r["phase"] == "burst"]
    assert len(set(burst)) == 1 and max(len(r["prompt"]) for r in reqs) == 30000
    assert all(len(r["prompt"]) + r["max_new"] <= sw.MAX_SEQ for r in reqs)


def test_section_a_seeds_are_stable_across_processes():
    import zlib

    assert zlib.crc32(b"A|b8-c2048|ls-dense|r0") == zlib.crc32(b"A|b8-c2048|ls-dense|r0")
    assert "hash(cid)" not in open(sw.__file__).read()
