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
        self.opts = opts
        return [1.0], [0.0]

    def measure_steps(self, gpu, batch, ctx, steps, warmup, seed):
        self.measured += 1
        # Section A must run at its own concurrency limit, or batch 32 cannot run.
        assert self.opts["max_running"] >= batch, "section A ran under the serving limit"
        if batch * ctx > 16 * 16384:                     # pretend the largest do not fit
            return {"fits": False, "reason": "needs more blocks"}
        return {"fits": True, "step_ms": [10.0 + 0.01 * i for i in range(steps)]}

    def submit(self, gpu, rid, prompt, max_new):
        # Section B (one-token requests) at the A/B limit on both engines;
        # serving cells at 16.
        want = sw.A_MAX_RUNNING if max_new == 1 else 16
        assert self.opts["max_running"] == want, f"limit {self.opts['max_running']}, want {want}"
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
    monkeypatch.setattr(sw, "A_BATCHES", (1, 32))   # 32: the batch the first version could never run
    monkeypatch.setattr(sw, "A_CONTEXTS", (2048, 32000))
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
    assert "A__b32-c32000__ls-dense__r0.json" in names            # saved even when it did not fit
    assert "A__b32-c2048__ls-dense__r0.json" in names
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



def test_main_gives_vllm_separate_engines_for_sections_a_b_and_for_serving(tmp_path, small,
                                                                           monkeypatch):
    """vLLM fixes max_num_seqs at startup: A/B need 32, serving 16 (prefix on,
    then off)."""
    import sys

    from benchmarks.runners import phase18_replicas as pr

    built = []

    class Fake(FakeCluster):
        def __init__(self, gpus, backend, opts):
            super().__init__()
            built.append((backend, opts.get("max_running"), opts.get("vllm_prefix")))

        def warmup(self, opts):
            pass

        def stop(self):
            pass

    table = tmp_path / "table.json"
    table.write_text('{"rows": []}')
    monkeypatch.setattr(pr, "Cluster", Fake)
    monkeypatch.setattr(sys, "argv", ["sweep", "--results-dir", str(tmp_path / "r"),
                                      "--policy-table", str(table)])
    assert sw.main() == 0
    vllm = [b for b in built if b[0] == "vllm"]
    assert vllm == [("vllm", 32, None), ("vllm", 16, True), ("vllm", 16, False)]
    assert (tmp_path / "r" / "A__b32-c2048__vllm__r0.json").exists()


def test_a_batch_over_the_limit_is_reported_as_concurrency_not_memory():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from benchmarks.runners import phase18_replicas as pr
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
    engine = ServingEngine(ls, max_running=2, max_seq_len=256, block_size=16)
    res = pr._measure_steps(engine, ServedRequest, batch=3, ctx=40, steps=4, warmup=1,
                            vocab=(0, 128))
    assert not res["fits"] and "concurrency limit" in res["reason"]



def test_every_cell_stays_within_the_models_position_limit():
    """Qwen2.5-1.5B has 32,768 positions. vLLM refuses anything longer; LatentServe
    would run past the trained range. Every cell must fit, on both engines."""
    assert sw.MAX_SEQ == 32768
    for ctx in sw.A_CONTEXTS:                                    # context + every decoded token
        assert ctx + 48 + sw.A_STEPS_HEADROOM <= sw.MAX_SEQ
    assert all(L + 1 <= sw.MAX_SEQ for L in sw.B_LENGTHS)
    from benchmarks.runners import phase18_replicas as pr

    for wl in (sw.varying(0), sw.probe(0), sw.openloop(0, 1.0, 40), pr.burst(0), pr.chat(0)):
        for r in wl["requests"]:
            assert len(r["prompt"]) + r["max_new"] <= sw.MAX_SEQ