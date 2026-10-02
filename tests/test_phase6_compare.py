"""
The vLLM comparison must never pair rows from different machines.

A results file accumulates rows from every session that wrote to it, and
Kaggle hands out a different host per session. Pairing by (system, batch,
context, length) alone kept whichever row came last, so a table could put
vLLM on one T4 beside LatentServe on another and call the ratio a system
comparison. The runs that motivated this test came from four hosts.
"""

from __future__ import annotations

import copy
import io
import json
from contextlib import redirect_stdout

import pytest

pytest.importorskip("pydantic")

from benchmarks.runners.phase6_vllm import compare  # noqa: E402
from config import load_config  # noqa: E402

BASE = {"model": "m", "dtype": "float16", "num_gpus": 1, "gpu_info": [{"name": "T4"}],
        "workload": "uniform_64", "batch_size": 1, "context_length": 64, "seed": 0,
        "git_commit": "abc1234", "lib_versions": {"torch": "2.13"}, "cuda_version": "13.0",
        "attention": "gqa"}


def _rows(system, host, ts, wall_short, wall_long, tput):
    out = []
    for length, wall in ((8, wall_short), (16, wall_long)):
        r = copy.deepcopy(BASE)
        r.update(system=system, hostname=host, timestamp_utc=ts, output_length=length,
                 throughput_tokens_sec=tput,
                 extra={"status": "ok", "num_requests": 4, "wall_s": wall,
                        "sampling": "greedy", "arrival_rate": None})
        out.append(r)
    return out


def _run(tmp_path, rows, host=None):
    cfg = load_config("configs/phase6_vllm.yaml")
    (tmp_path / f"{cfg.tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    buf = io.StringIO()
    with redirect_stdout(buf):
        compare(cfg, str(tmp_path), host=host)
    return buf.getvalue()


def test_each_host_gets_its_own_table(tmp_path):
    rows = (_rows("latentserve_kernel_graphed", "hostA", "2026-10-01T10:00", 1.0, 1.8, 50)
            + _rows("vllm", "hostA", "2026-10-01T10:30", 1.2, 2.2, 40)
            + _rows("latentserve_kernel_graphed", "hostB", "2026-10-02T09:00", 2.0, 3.6, 25)
            + _rows("vllm", "hostB", "2026-10-02T09:30", 1.2, 2.2, 40))
    out = _run(tmp_path, rows)
    assert "host hostA" in out and "host hostB" in out
    # hostA: LatentServe decode 0.8s/... vs vLLM 1.0 -> faster; hostB: slower.
    a, b = out.split("host hostB")
    assert "faster" in a.split("decode latency")[1].splitlines()[0]
    assert "slower" in b.split("decode latency")[1].splitlines()[0]


def test_a_host_with_one_system_is_not_paired_with_another_hosts_rows(tmp_path):
    """The exact shape of the real file: LatentServe ran on a host where
    vLLM never did. Its rows must be listed and left out, not matched to
    vLLM rows from a different machine."""
    rows = (_rows("latentserve_kernel_graphed", "hostA", "2026-10-01T10:00", 1.0, 1.8, 50)
            + _rows("vllm", "hostA", "2026-10-01T10:30", 1.2, 2.2, 40)
            + _rows("latentserve_kernel_graphed", "lonely", "2026-10-03T12:00", 9.0, 9.9, 5))
    out = _run(tmp_path, rows)
    assert "only one system, not compared" in out and "lonely" in out
    assert "host lonely" not in out


def test_host_flag_restricts_to_one_machine(tmp_path):
    rows = (_rows("latentserve_kernel_graphed", "hostA", "2026-10-01T10:00", 1.0, 1.8, 50)
            + _rows("vllm", "hostA", "2026-10-01T10:30", 1.2, 2.2, 40)
            + _rows("latentserve_kernel_graphed", "hostB", "2026-10-02T09:00", 2.0, 3.6, 25)
            + _rows("vllm", "hostB", "2026-10-02T09:30", 1.2, 2.2, 40))
    out = _run(tmp_path, rows, host="hostB")
    assert "host hostB" in out and "host hostA" not in out


def test_one_host_with_two_stacks_is_never_paired_across_them(tmp_path):
    """Installing vLLM replaces torch in place, so one session — one
    hostname — can hold LatentServe rows from before the install and vLLM
    rows from after. Exactly what happened on a real host."""
    before = _rows("latentserve_kernel_graphed", "same", "2026-10-02T10:00", 2.0, 3.6, 25)
    after = _rows("vllm", "same", "2026-10-02T12:00", 1.2, 2.2, 40)
    for r in before:
        r["lib_versions"] = {"torch": "2.10.0"}
        r["cuda_version"] = "12.8"
    for r in after:
        r["lib_versions"] = {"torch": "2.13.0", "vllm": "0.30.0"}
        r["cuda_version"] = "13.0"
    out = _run(tmp_path, before + after)
    assert "No host has rows for both systems" in out
    assert "decode latency" not in out


def test_drift_between_runs_is_flagged_and_the_median_gap_reported(tmp_path):
    """The real failure, reproduced: the short run measured at 19.1 ms per
    step, the long one at 20.1, so differencing reports 2b - a = 21.1 —
    above both. The compare must show the median gap beside it and say
    which to trust."""
    n, batch, short, long, prefill_s = 4, 1, 8, 16, 1.0
    a, b = 0.0191, 0.0201
    rows = []
    for system in ("latentserve_kernel_graphed", "vllm"):
        for length, per_step in ((short, a), (long, b)):
            wall = prefill_s + length * n / batch * per_step
            r = _rows(system, "h", "2026-10-02T10:00", wall, wall, 40)[0]
            r["output_length"] = length
            r["extra"]["wall_s"] = wall
            r["tpot_p50_ms"] = per_step * 1000
            rows.append(r)
    out = _run(tmp_path, rows)
    assert "median gap" in out
    assert "runs drifted" in out and "[WARN] the short and long runs drifted" in out
    assert "differenced 21.1 discarded" in out    # 2b - a, named and set aside
    assert "20.1" in out                          # the long run's median, used instead


def test_steady_runs_keep_the_differenced_figure(tmp_path):
    """No drift, no override: 1.3% between runs is under the limit, which
    is where every steady configuration measured so far sits."""
    n, batch = 4, 1
    rows = []
    for system in ("latentserve_kernel_graphed", "vllm"):
        for length, per_step in ((8, 0.0239), (16, 0.0242)):
            wall = 1.0 + length * n / batch * per_step
            r = _rows(system, "h", "2026-10-02T10:00", wall, wall, 40)[0]
            r["output_length"] = length
            r["extra"]["wall_s"] = wall
            r["tpot_p50_ms"] = per_step * 1000
            rows.append(r)
    out = _run(tmp_path, rows)
    assert "runs drifted" not in out