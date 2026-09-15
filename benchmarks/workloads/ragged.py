"""
Request-stream workloads.

docs/methodology.md Section 32 lists six workload families. This module
provides the ones Phase 3 needs — anything where sequence lengths differ
— and Phase 4's scheduler reuses it unchanged, since a scheduler and a
block allocator are stressed by the same thing: requests that arrive,
grow at different rates, and leave at different times.

A contiguous cache is indifferent to workload shape because it reserves
for the worst case regardless. Paging is not. So the workload *is* the
experiment here: benchmarking paging on fixed-length prompts would
measure only its overhead, never its benefit, and would make it look
strictly worse than contiguous.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: int
    prompt_tokens: int
    output_tokens: int
    arrival_step: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


# Workload families from docs/methodology.md Section 32. Each is
# (name, [(prompt, output, weight), ...]).
WORKLOADS: dict[str, list[tuple[int, int, float]]] = {
    # A — short interactive. Paging's weakest case: everything is small
    # and similar, so there is little for it to reclaim.
    "short_interactive": [(1024, 128, 1.0)],
    # B — long prompt, short answer. Prefill- and memory-dominated.
    "long_prompt": [(32768, 128, 0.5), (65536, 128, 0.5)],
    # C — long generation. The cache grows for a long time after prefill,
    # which is where incremental block allocation earns its keep.
    "long_generation": [(8192, 1024, 1.0)],
    # D — mixed serving. The realistic case and the one Phase 3 is really
    # about: a 64K request sharing a GPU with 1K requests forces a
    # contiguous cache to reserve as if every sequence were 64K.
    "mixed": [
        (1024, 128, 0.40),
        (4096, 256, 0.30),
        (16384, 256, 0.20),
        (32768, 256, 0.07),
        (65536, 256, 0.03),
    ],
    # E — shared prefixes, for Phase 13. Same lengths, and the sharing is
    # expressed through the allocator's refcounts rather than here.
    "shared_prefix": [(8192, 256, 1.0)],
}


def generate_requests(
    workload: str,
    num_requests: int,
    seed: int = 0,
    arrival_rate: float = 0.0,
) -> list[Request]:
    """Sample a request stream.

    `arrival_rate` is requests per decode step: 0.0 means everything is
    queued at step 0 (a burst, which maximises memory pressure and is
    the right stress test for an allocator), while a positive rate
    spreads arrivals out (closer to steady-state serving, and what
    Phase 4 wants).
    """
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}; have {sorted(WORKLOADS)}")
    rng = random.Random(seed)
    mix = WORKLOADS[workload]
    weights = [w for _, _, w in mix]

    requests = []
    step = 0
    for i in range(num_requests):
        prompt, output, _ = rng.choices(mix, weights=weights, k=1)[0]
        # +-10% jitter so lengths don't all land exactly on block
        # boundaries, which would flatter paging's fragmentation numbers.
        prompt = max(1, int(prompt * rng.uniform(0.9, 1.1)))
        output = max(1, int(output * rng.uniform(0.9, 1.1)))
        if arrival_rate > 0:
            step += int(rng.expovariate(arrival_rate))
        requests.append(Request(i, prompt, output, arrival_step=step))
    return requests


def describe(requests: list[Request]) -> dict:
    prompts = [r.prompt_tokens for r in requests]
    totals = [r.total_tokens for r in requests]
    return {
        "num_requests": len(requests),
        "prompt_min": min(prompts),
        "prompt_max": max(prompts),
        "prompt_mean": sum(prompts) / len(prompts),
        "total_tokens": sum(totals),
        "longest_sequence": max(totals),
        # The ratio that predicts how badly a contiguous cache does: it
        # sizes every slot for the longest sequence it must support.
        "max_to_mean_ratio": max(totals) / (sum(totals) / len(totals)),
    }
