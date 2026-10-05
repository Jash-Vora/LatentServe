"""
Phase 17 — the adaptive runtime's sparsity policy.

The plan (methodology §24): turn the techniques into a runtime policy, with
thresholds *derived from benchmark data, not assumed*, and compare fixed
against adaptive strategies. Phase 15 measured what sparse decode costs —
net answers lost, ~0.4% at 50% of pages and ~1.2% at 37.5% — and Phase 14
what it buys, which ranges from nothing at batch 1 to ~1.5x at batch 8 /
32K. A fixed sparse setting pays its cost on every step, including the ones
where it buys nothing. This policy pays it only where the measured step
time says it is bought:

    choose(batch, context) -> the fastest budget the quality tier allows,
                              if it beats dense by at least `min_gain`;
                              otherwise dense (None)

Tiers: "strict" (dense only), "balanced" (50%), "relaxed" (50% or 37.5%).
Step times come from a calibration table measured on the target machine
(benchmarks/runners/phase17_calibrate.py), predicted between its points.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

TIERS = {"strict": (), "balanced": (0.5,), "relaxed": (0.5, 0.375)}

# Phase 15, pooled net answers lost per budget (descriptive: several seeds
# were exploratory). For expected-loss accounting, not for decisions.
ANSWER_LOSS = {None: 0.0, 0.5: 0.004, 0.375: 0.012, 0.25: 0.055}


@dataclass
class StepTable:
    """Measured decode-step times, predicted between measured points.

    rows: {"batch", "ctx", "ratio" (None for dense), "ms"}. Prediction is
    bilinear in (log2 batch, log2 context) where all four corners were
    measured, nearest measured point otherwise — a cell can be missing
    because that shape did not fit in memory.
    """

    rows: list = field(default_factory=list)

    def _points(self, ratio):
        return {(r["batch"], r["ctx"]): r["ms"] for r in self.rows if r["ratio"] == ratio}

    def predict(self, batch: float, ctx: float, ratio) -> Optional[float]:
        pts = self._points(ratio)
        if not pts:
            return None
        if (batch, ctx) in pts:
            return pts[(batch, ctx)]
        bs = sorted({b for b, _ in pts})
        cs = sorted({c for _, c in pts})
        b = min(max(batch, bs[0]), bs[-1])
        c = min(max(ctx, cs[0]), cs[-1])
        b0 = max(x for x in bs if x <= b)
        b1 = min(x for x in bs if x >= b)
        c0 = max(x for x in cs if x <= c)
        c1 = min(x for x in cs if x >= c)
        corners = [(b0, c0), (b0, c1), (b1, c0), (b1, c1)]
        if all(k in pts for k in corners):
            tb = 0.0 if b1 == b0 else (math.log2(b) - math.log2(b0)) / (math.log2(b1) - math.log2(b0))
            tc = 0.0 if c1 == c0 else (math.log2(c) - math.log2(c0)) / (math.log2(c1) - math.log2(c0))
            lo = pts[(b0, c0)] * (1 - tc) + pts[(b0, c1)] * tc
            hi = pts[(b1, c0)] * (1 - tc) + pts[(b1, c1)] * tc
            return lo * (1 - tb) + hi * tb
        lb, lc = math.log2(max(batch, 1)), math.log2(max(ctx, 1))
        near = min(pts, key=lambda k: (math.log2(k[0]) - lb) ** 2 + (math.log2(k[1]) - lc) ** 2)
        return pts[near]


@dataclass
class SparsePolicy:
    table: StepTable
    tier: str = "relaxed"
    min_gain: float = 0.05

    def __post_init__(self):
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}; known: {', '.join(TIERS)}")

    @property
    def allowed(self) -> tuple:
        return TIERS[self.tier]

    @property
    def may_sparsify(self) -> bool:
        return bool(self.allowed)

    def choose(self, batch: int, ctx: float) -> Optional[float]:
        if not self.allowed or batch <= 0:
            return None
        dense = self.table.predict(batch, ctx, None)
        if dense is None:
            return None
        best, best_ms = None, dense
        for ratio in self.allowed:
            ms = self.table.predict(batch, ctx, ratio)
            if ms is not None and ms < best_ms:
                best, best_ms = ratio, ms
        if best is None or dense / best_ms - 1.0 < self.min_gain:
            return None
        return best

    @classmethod
    def from_json(cls, path, tier: str = "relaxed", min_gain: float = 0.05) -> "SparsePolicy":
        rows = json.loads(Path(path).read_text())["rows"]
        return cls(StepTable(rows), tier=tier, min_gain=min_gain)


def expected_answer_loss(tokens_by_ratio: dict) -> float:
    """Phase 15's per-budget answer-loss rates, weighted by the tokens each
    budget generated: an estimate of a strategy's quality cost."""
    total = sum(tokens_by_ratio.values())
    if not total:
        return 0.0
    return sum(n * ANSWER_LOSS.get(r, float("nan")) for r, n in tokens_by_ratio.items()) / total
