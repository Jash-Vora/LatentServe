"""
Phase 18 — a router in front of independent replicas (one engine per GPU).

Policies:

  round_robin    replicas in turn: the naive baseline
  least_loaded   the replica with the fewest outstanding requests
  prefix_aware   the replica that last served the deepest matching prefix —
                 so a conversation's turns land where its history is cached
                 (Phase 13) — unless that replica is more than `slack`
                 requests busier than the least loaded

Pure prefix affinity would send every conversation to whichever replica
first saw the shared system prompt; the slack rule (as in cache-aware routers
like SGLang's) keeps deep matches — a conversation's own history — at home
and balances shallow ones. Prefixes are identified with the prefix cache's
own chained block hashes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from cache.prefix_cache import block_keys

POLICIES = ("round_robin", "least_loaded", "prefix_aware")


@dataclass
class Router:
    replicas: int
    policy: str = "least_loaded"
    block_size: int = 16
    slack: int = 2
    max_entries: int = 200_000
    outstanding: list = field(default_factory=list)
    routed: list = field(default_factory=list)
    affinity_hits: int = 0
    _seen: OrderedDict = field(default_factory=OrderedDict)   # block hash -> replica
    _next: int = 0

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; known: {', '.join(POLICIES)}")
        self.outstanding = [0] * self.replicas
        self.routed = [0] * self.replicas

    def _least(self) -> int:
        return min(range(self.replicas), key=lambda i: (self.outstanding[i], i))

    def route(self, prompt) -> int:
        if self.policy == "round_robin":
            target = self._next % self.replicas
            self._next += 1
        elif self.policy == "least_loaded":
            target = self._least()
        else:
            keys = [k[0] for k in block_keys(prompt, self.block_size)]
            home = next((self._seen[h] for h in reversed(keys) if h in self._seen), None)
            least = self._least()
            if home is not None and self.outstanding[home] - self.outstanding[least] <= self.slack:
                target = home
                self.affinity_hits += 1
            else:
                target = least
            for h in keys:                          # this prefix now lives on `target`
                self._seen[h] = target
                self._seen.move_to_end(h)
            while len(self._seen) > self.max_entries:
                self._seen.popitem(last=False)
        self.outstanding[target] += 1
        self.routed[target] += 1
        return target

    def done(self, replica: int) -> None:
        self.outstanding[replica] -= 1


def percentiles(values, ps=(50, 95, 99)) -> dict:
    """Nearest-rank percentiles; NaN for an empty list."""
    xs = sorted(values)
    if not xs:
        return {f"p{p}": float("nan") for p in ps}
    out = {}
    for p in ps:
        k = max(0, min(len(xs) - 1, -(-p * len(xs) // 100) - 1))
        out[f"p{p}"] = xs[k]
    return out
