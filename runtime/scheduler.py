"""
Phase 4 — scheduling policies.

docs/methodology.md Phase 4 asks for four, implemented progressively:
FIFO, length-aware, fair, SLO-aware — and for scheduler overhead to be
measured explicitly rather than assumed negligible.

A scheduler here decides one thing: **given the requests waiting and the
KV blocks free, which do we admit next?** It does not decide how to
batch (every resident sequence decodes every step — that is what
continuous batching means) and it does not decide eviction policy (the
engine owns that). Keeping the policy this small is what makes swapping
one out a controlled experiment instead of a rewrite.

Each policy is a sort key over the waiting queue. That shape makes the
overhead measurement honest too: the cost is a sort, and a sort over a
queue of tens of requests should be invisible next to a ~30 ms decode
step. Phase 4's job is to confirm that rather than assert it.
"""

from __future__ import annotations

import time
from typing import Callable, Literal

from runtime.request import ServedRequest

SchedulerName = Literal["fifo", "length_aware", "fair", "slo_aware"]


class Scheduler:
    """Base policy: orders the waiting queue and applies admission."""

    name: SchedulerName = "fifo"

    def __init__(self, max_running: int = 64):
        # Cap on resident sequences. KV capacity is the real limit, but a
        # cap keeps the decode batch from growing until per-step latency
        # violates every SLO at once — throughput and tail latency pull
        # in opposite directions here.
        self.max_running = max_running
        self.overhead_s = 0.0
        self.calls = 0

    def priority(self, request: ServedRequest, now: float) -> float:
        """Lower sorts first."""
        return request.arrival_time

    def select(
        self,
        waiting: list[ServedRequest],
        num_running: int,
        can_admit: Callable[[int], bool],
        now: float | None = None,
    ) -> list[ServedRequest]:
        """Pick requests to admit this iteration.

        Admission stops at the first request that does not fit, rather
        than skipping it to seat a smaller one behind it. Skipping raises
        throughput and starves long requests indefinitely — the same
        head-of-line trade the policies below make explicit. `fair` is
        the one that is allowed to reorder around it, by aging.
        """
        t0 = time.perf_counter()
        self.calls += 1
        now = time.perf_counter() if now is None else now

        ordered = sorted(waiting, key=lambda r: self.priority(r, now))
        admitted: list[ServedRequest] = []
        slots_left = self.max_running - num_running
        for request in ordered:
            if slots_left <= 0:
                break
            if not can_admit(request.prompt_len):
                break
            admitted.append(request)
            slots_left -= 1

        self.overhead_s += time.perf_counter() - t0
        return admitted

    def stats(self) -> dict:
        return {
            "scheduler": self.name,
            "scheduler_calls": self.calls,
            "scheduler_overhead_ms": self.overhead_s * 1000,
            "scheduler_overhead_us_per_call": (
                self.overhead_s * 1e6 / self.calls if self.calls else 0.0
            ),
        }


class FIFOScheduler(Scheduler):
    """Arrival order. The baseline every other policy is measured against.

    Fair in the weakest sense — nobody overtakes — and vulnerable to
    head-of-line blocking: one 64K prompt at the front stalls a hundred
    1K prompts behind it.
    """

    name = "fifo"


class LengthAwareScheduler(Scheduler):
    """Shortest prompt first.

    Classic SJF: minimises mean waiting time, and on the `mixed` workload
    (max/mean length ratio 7.4, measured in Phase 3) the short requests
    are the overwhelming majority, so mean and p50 TTFT should improve a
    lot. The cost is the tail — long prompts get overtaken repeatedly, so
    watch p99 TTFT and `preemptions`, not just the mean. A policy that
    improves p50 while wrecking p99 has not improved the server.
    """

    name = "length_aware"

    def priority(self, request: ServedRequest, now: float) -> float:
        return request.prompt_len


class FairScheduler(Scheduler):
    """Shortest-job-first with aging, so nothing starves.

    Priority falls as a request waits; after `aging_seconds` of waiting a
    request gains the equivalent of `aging_tokens` of shortness. The
    knobs make the trade explicit instead of hiding it: aging_seconds
    small means nearly FIFO, large means nearly pure SJF.
    """

    name = "fair"

    def __init__(self, max_running: int = 64, aging_seconds: float = 1.0,
                 aging_tokens: float = 4096.0):
        super().__init__(max_running)
        self.aging_seconds = aging_seconds
        self.aging_tokens = aging_tokens

    def priority(self, request: ServedRequest, now: float) -> float:
        waited = now - request.arrival_time
        return request.prompt_len - (waited / self.aging_seconds) * self.aging_tokens


class SLOAwareScheduler(Scheduler):
    """Earliest deadline first over TTFT targets.

    Requests carrying an `slo_ttft_ms` are ordered by how much slack they
    have left; requests without one fall back to arrival order behind
    them. This is the policy most likely to look bad on mean throughput
    and good on the metric anyone actually promises a user, which is
    precisely why both need reporting.
    """

    name = "slo_aware"

    def __init__(self, max_running: int = 64, default_slo_ms: float = 2000.0):
        super().__init__(max_running)
        self.default_slo_ms = default_slo_ms

    def priority(self, request: ServedRequest, now: float) -> float:
        slo = request.slo_ttft_ms or self.default_slo_ms
        elapsed_ms = (now - request.arrival_time) * 1000
        return slo - elapsed_ms  # smallest slack first


SCHEDULERS: dict[str, type[Scheduler]] = {
    "fifo": FIFOScheduler,
    "length_aware": LengthAwareScheduler,
    "fair": FairScheduler,
    "slo_aware": SLOAwareScheduler,
}


def build_scheduler(name: str, max_running: int = 64, **kwargs) -> Scheduler:
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {name!r}; have {sorted(SCHEDULERS)}")
    return SCHEDULERS[name](max_running=max_running, **kwargs)
