"""The GPU sampler records machine state alongside every benchmark run.

It must never break a benchmark: no nvidia-smi means no samples, said
plainly, not an exception.
"""

from __future__ import annotations

import time

import benchmarks.gpu_sampler as gs
from benchmarks.gpu_sampler import GpuSampler, decode_throttle


def test_missing_nvidia_smi_records_nothing_and_does_not_raise(monkeypatch):
    monkeypatch.setattr(gs, "_query", lambda device, field: None)
    with GpuSampler() as s:
        pass
    assert s.summary() == {"gpu_samples": 0}


def test_throttle_bits_decode():
    assert decode_throttle(0x4 | 0x20) == ["sw_power_cap", "sw_thermal"]
    assert decode_throttle(0) == []


def test_summary_from_samples(monkeypatch):
    """Two idle samples and one power-capped one: idle is not throttling,
    the power cap is."""
    rows = iter([["1590", "5000", "61", "35.0", "0x0000000000000001"],
                 ["1590", "5000", "62", "36.0", "0x0000000000000001"],
                 ["1200", "5000", "74", "70.0", "0x0000000000000004"]])
    last = ["1200", "5000", "74", "70.0", "0x0000000000000004"]

    def fake(device, field):
        return next(rows, last)

    monkeypatch.setattr(gs, "_query", fake)
    with GpuSampler(interval_s=0.01) as s:
        time.sleep(0.2)
    out = s.summary()
    assert out["gpu_samples"] >= 3
    assert out["gpu_sm_clock_min_mhz"] == 1200
    assert out["gpu_temp_max_c"] == 74
    assert out["gpu_throttle_reasons"] == ["sw_power_cap"]
    assert 0 < out["gpu_throttled_frac"] <= 1