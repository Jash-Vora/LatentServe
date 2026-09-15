"""
Phase 5 — the benchmark harness.

docs/methodology.md Phase 5: "Before introducing MLA, make measurement
trustworthy."

That instruction earned itself in Phase 4. Three of five predictions came
back falsified, and two of those were the harness answering a different
question than the one asked:

  * inter-token latency percentiles taken over per-request *means*, so a
    2 s stall spread across 229 steps vanished;
  * then over the decode *call* duration rather than the wall gap between
    tokens, so prefill blocking stayed invisible a second time.

Neither was visible in the code. Both were visible the moment a specific
numeric prediction disagreed with a measurement. This module exists so
those mistakes are made once, in one place, rather than re-invented per
runner.

Three jobs:

  1. **Statistics** — percentiles, bootstrap confidence intervals, and a
     bimodal-aware latency summary, so "p99" always means the same thing.
  2. **Repeatability** — quantify run-to-run variation, so a 3% delta is
     reported as noise when noise is 3%.
  3. **Fairness guards** — refuse to compute a ratio between two runs
     that were not measured under matched conditions. Phase 6 compares
     against vLLM, where an unmatched control (sampling, output length,
     prefix caching) produces a number that looks fine and means nothing.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

# Controls that must match before two rows may be compared. From
# methodology Section 33 ("Fairness Rules for vLLM"): model weights,
# tokenizer, precision, GPU, input/output tokens, concurrency, sampling,
# context length.
FAIRNESS_KEYS = (
    "model",
    "dtype",
    "batch_size",
    "context_length",
    "output_length",
    "num_gpus",
    "gpu_name",
    "sampling",
    "workload",
    "arrival_rate",
)


def percentile(xs: Sequence[float], p: float) -> Optional[float]:
    """Nearest-rank percentile: the smallest value at or above which p of
    the sample lies. One definition, used everywhere from Phase 5 on.

    Three definitions were available and the choice is worth stating in
    the report rather than inheriting:

      * numpy's default interpolates between neighbours, inventing values
        that never occurred — wrong for a latency distribution where the
        interesting mode is a real 2.3 s stall, not an average of one;
      * the Phase 1-4 runners used ``round(p * (n - 1))``, which inherits
        Python's banker's rounding: ``round(4.5) == 4`` but
        ``round(5.5) == 6``, so p50 of an even-length sample tilts in a
        direction that depends on the index's parity;
      * nearest-rank, below, is the textbook definition, returns an
        observed value, and has no rounding-mode surprises.

    Switching shifts an index by at most one position (well under 1% of a
    sample of hundreds), so Phase 1-4 headline numbers are unaffected in
    any way that matters — but the change is recorded here because a
    percentile definition that drifts mid-project is exactly the sort of
    thing that costs an afternoon in month four.
    """
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    import math

    rank = max(1, math.ceil(p * len(xs)))
    return xs[min(len(xs), rank) - 1]


def bootstrap_ci(
    xs: Sequence[float], confidence: float = 0.95, resamples: int = 1000, seed: int = 0
) -> tuple[Optional[float], Optional[float]]:
    """Percentile-bootstrap CI for the median.

    Methodology asks for confidence intervals "where practical". With 3
    measured trials a parametric CI would assume a normality that three
    points cannot support; the bootstrap at least makes no distributional
    claim. With n < 3 it returns (None, None) rather than a number that
    would look more authoritative than it is.
    """
    xs = [x for x in xs if x is not None]
    if len(xs) < 3:
        return None, None
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(xs, k=len(xs))) for _ in range(resamples)
    ]
    tail = (1 - confidence) / 2
    return percentile(medians, tail), percentile(medians, 1 - tail)


def coefficient_of_variation(xs: Sequence[float]) -> Optional[float]:
    """Std / mean. The number that says whether a measured difference is
    bigger than the harness's own noise floor."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    mean = statistics.mean(xs)
    return statistics.stdev(xs) / mean if mean else None


@dataclass
class LatencySummary:
    """Distribution summary for one latency population.

    `stall_rate` and `stalled_ms_total` exist because inter-token latency
    under a serving runtime is bimodal: a normal population around the
    decode step, and a stall population at the length of whatever prefill
    was admitted. In Phase 4 roughly 1% of gaps were stalls, which put
    p99 exactly on the boundary between the two populations, so it moved
    with sampling noise rather than with the scheduler being tested.
    Where a distribution is bimodal, report how often the second mode
    happens — not a percentile that lands inside it by luck.
    """

    count: int
    p50: Optional[float]
    p90: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    p999: Optional[float]
    mean: Optional[float]
    max: Optional[float]
    stall_rate: Optional[float]
    stalled_ms_total: Optional[float]
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    def as_dict(self, prefix: str) -> dict:
        return {f"{prefix}_{k}": v for k, v in self.__dict__.items()}


def summarise_latency(
    xs: Sequence[float], stall_factor: float = 10.0, with_ci: bool = False
) -> LatencySummary:
    xs = [x for x in xs if x is not None]
    if not xs:
        return LatencySummary(0, *([None] * 9))
    p50 = percentile(xs, 0.50)
    threshold = stall_factor * (p50 or 0)
    stalls = [x for x in xs if x > threshold] if p50 else []
    ci_low, ci_high = bootstrap_ci(xs) if with_ci else (None, None)
    return LatencySummary(
        count=len(xs),
        p50=p50,
        p90=percentile(xs, 0.90),
        p95=percentile(xs, 0.95),
        p99=percentile(xs, 0.99),
        p999=percentile(xs, 0.999),
        mean=statistics.mean(xs),
        max=max(xs),
        stall_rate=len(stalls) / len(xs),
        stalled_ms_total=sum(stalls),
        ci_low=ci_low,
        ci_high=ci_high,
    )


# ----------------------------------------------------------------------
# Fairness guards
# ----------------------------------------------------------------------


class UnfairComparison(AssertionError):
    """Raised when two runs differ on a controlled variable.

    A hard failure rather than a warning. The failure mode this prevents
    — comparing LatentServe at 128 output tokens against vLLM at 256, or
    against a vLLM that silently enabled prefix caching — produces a
    plausible number, and a plausible wrong number is worse than a crash.
    """


def _get(row: dict, key: str):
    """Rows carry phase-specific fields inside `extra`; look in both."""
    if key in row:
        return row[key]
    return (row.get("extra") or {}).get(key)


def comparison_mismatches(
    a: dict, b: dict, keys: Iterable[str] = FAIRNESS_KEYS
) -> dict[str, tuple]:
    """Controlled variables on which two rows disagree.

    A key missing from *both* rows is not a mismatch — not every phase
    records every control. A key present in one and absent from the other
    is, because that usually means one side quietly defaulted.
    """
    out = {}
    for key in keys:
        va, vb = _get(a, key), _get(b, key)
        if va is None and vb is None:
            continue
        if va != vb:
            out[key] = (va, vb)
    return out


def assert_comparable(a: dict, b: dict, keys: Iterable[str] = FAIRNESS_KEYS) -> None:
    mismatches = comparison_mismatches(a, b, keys)
    if mismatches:
        detail = ", ".join(f"{k}: {va!r} vs {vb!r}" for k, (va, vb) in mismatches.items())
        raise UnfairComparison(
            f"refusing to compare {a.get('system')} against {b.get('system')}: "
            f"controlled variables differ ({detail})"
        )


def speedup(a: dict, b: dict, metric: str, higher_is_better: bool = True) -> float:
    """Ratio of `a` to `b` on `metric`, after checking they are comparable."""
    assert_comparable(a, b)
    va, vb = _get(a, metric), _get(b, metric)
    if va is None or vb is None:
        raise ValueError(f"metric {metric!r} missing from one of the rows")
    if not vb:
        raise ValueError(f"metric {metric!r} is zero in the baseline row")
    return va / vb if higher_is_better else vb / va


def check_environment(rows: Sequence[dict]) -> dict:
    """Flag reproducibility drift across a set of rows.

    Rows produced by different commits, library versions or GPUs are
    still *reportable* — sometimes that is the only data you have — but
    the drift has to travel with the numbers rather than be discovered
    later. Returns what varied; empty means a clean set.
    """
    drift: dict[str, set] = {}
    for key in ("git_commit", "cuda_version", "hostname"):
        values = {row.get(key) for row in rows if row.get(key) is not None}
        if len(values) > 1:
            drift[key] = values
    libs = {tuple(sorted((row.get("lib_versions") or {}).items())) for row in rows}
    if len(libs) > 1:
        drift["lib_versions"] = libs
    gpus = {
        tuple(g.get("name") for g in (row.get("gpu_info") or [])) for row in rows
    }
    if len(gpus) > 1:
        drift["gpu_info"] = gpus
    return drift


def repeatability_report(trials: Sequence[dict], metric: str) -> dict:
    """Run-to-run variation for one metric across repeated identical runs.

    The noise floor. Phase 4's scheduler sweep reproduced TTFT to within
    1.2% across two runs, which is what licenses calling a 3.3% throughput
    difference real and a 0.4% one noise. Without this number, every
    small delta in the report is an assertion.
    """
    values = [_get(t, metric) for t in trials]
    values = [v for v in values if v is not None]
    low, high = bootstrap_ci(values)
    return {
        "metric": metric,
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "cv": coefficient_of_variation(values),
        "ci_low": low,
        "ci_high": high,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
