"""
What state the GPU was in while a benchmark ran.

Phase 14 found two harnesses measuring the same engine on the same host
and stack 2.5 ms per token apart, at batch 1. Short runs with pauses
between them are measured on a GPU that has time to cool; a long run of
back-to-back prefills and decodes is measured on one that may have hit
its power or thermal limit and lowered its clocks. A T4 is passively
cooled with a 70 W cap, which makes that plausible rather than exotic.

So each run records the SM clock, temperature, power and active throttle
reasons, sampled once a second in a background thread. It costs one
`nvidia-smi` call per second and reads nothing from the process being
measured. If `nvidia-smi` is missing, the sampler records nothing and
says so, rather than failing the benchmark.
"""

from __future__ import annotations

import statistics
import subprocess
import threading
import time

FIELDS = ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw")
# Renamed in newer drivers; whichever answers is used.
THROTTLE_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")

THROTTLE_BITS = {
    0x1: "idle",
    0x2: "app_clocks",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
}


def decode_throttle(mask: int) -> list[str]:
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


def _query(device: int, throttle_field: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(FIELDS + (throttle_field,))}",
             "--format=csv,noheader,nounits", "-i", str(device)],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [x.strip() for x in out.stdout.strip().split(",")]


class GpuSampler:
    """Context manager: samples GPU state for the duration of a `with`."""

    def __init__(self, device: int = 0, interval_s: float = 1.0):
        self.device, self.interval = device, interval_s
        self.samples: list[tuple] = []
        self._stop = threading.Event()
        self._thread = None
        self._field = None

    def _probe(self) -> bool:
        for field in THROTTLE_FIELDS:
            if _query(self.device, field) is not None:
                self._field = field
                return True
        return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            row = _query(self.device, self._field)
            if row is not None and len(row) == 5:
                try:
                    sm, mem, temp, power = (float(v) for v in row[:4])
                    mask = int(row[4], 16) if row[4].startswith("0x") else int(row[4])
                    self.samples.append((sm, mem, temp, power, mask))
                except ValueError:
                    pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if self._probe():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return False

    def summary(self) -> dict:
        if not self.samples:
            return {"gpu_samples": 0}
        sm = [s[0] for s in self.samples]
        mask = 0
        for s in self.samples:
            mask |= s[4]
        # Throttled = clocks lowered for a reason other than being idle.
        throttled = [s for s in self.samples if s[4] & ~0x1]
        return {
            "gpu_samples": len(self.samples),
            "gpu_sm_clock_mean_mhz": statistics.fmean(sm),
            "gpu_sm_clock_min_mhz": min(sm),
            "gpu_mem_clock_mean_mhz": statistics.fmean(s[1] for s in self.samples),
            "gpu_temp_max_c": max(s[2] for s in self.samples),
            "gpu_power_mean_w": statistics.fmean(s[3] for s in self.samples),
            "gpu_throttle_reasons": decode_throttle(mask & ~0x1),
            "gpu_throttled_frac": len(throttled) / len(self.samples),
        }
