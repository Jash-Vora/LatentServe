"""
Phase 4 — request objects and lifecycle.

docs/methodology.md Phase 4:

    ARRIVED -> QUEUED -> PREFILL -> DECODING -> FINISHED

Everything measured before this phase was a property of one generation
call. From here the unit of measurement is a *request*, and the metrics
that matter are per-request and distributional: a server whose mean TTFT
looks fine while its p99 is 40x worse is a bad server, and mean latency
alone cannot tell you that (methodology Section 12, "Percentiles").

Queue time is the new quantity. Phase 1-3's TTFT was pure prefill
latency because nothing ever waited. Under load, TTFT = queue time +
prefill, and the two respond to completely different fixes — more
capacity versus a faster prefill. Keeping them separate here is what
makes the scheduler comparison in `runtime/scheduler.py` legible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RequestState(str, Enum):
    ARRIVED = "arrived"
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"
    PREEMPTED = "preempted"


@dataclass
class ServedRequest:
    """One request's lifecycle, timings and generated tokens."""

    request_id: int
    prompt_ids: list  # token ids
    max_new_tokens: int
    arrival_time: float = field(default_factory=time.perf_counter)
    slo_ttft_ms: Optional[float] = None  # target used by the SLO-aware policy

    state: RequestState = RequestState.ARRIVED
    slot: Optional[int] = None  # cache slot while resident
    output_ids: list = field(default_factory=list)

    # --- timing, all perf_counter seconds ---
    scheduled_time: Optional[float] = None  # admitted, prefill about to run
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    decode_step_ms: list = field(default_factory=list)

    # How many times this request was evicted and had to redo its prompt.
    preemptions: int = 0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def generated(self) -> int:
        return len(self.output_ids)

    @property
    def is_finished(self) -> bool:
        return self.state is RequestState.FINISHED

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.max_new_tokens

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def queue_ms(self) -> Optional[float]:
        """Arrival to admission. Zero in Phases 1-3 because nothing ever
        queued; under load it is usually the dominant term in TTFT."""
        if self.scheduled_time is None:
            return None
        return (self.scheduled_time - self.arrival_time) * 1000

    @property
    def prefill_ms(self) -> Optional[float]:
        if self.first_token_time is None or self.scheduled_time is None:
            return None
        return (self.first_token_time - self.scheduled_time) * 1000

    @property
    def ttft_ms(self) -> Optional[float]:
        """What the client experiences: queue + prefill. This is the
        number that must be compared against vLLM in Phase 6, and it is
        not the same quantity Phase 1-3 called TTFT."""
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.arrival_time) * 1000

    @property
    def tpot_ms(self) -> Optional[float]:
        if not self.decode_step_ms:
            return None
        return sum(self.decode_step_ms) / len(self.decode_step_ms)

    @property
    def e2e_ms(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return (self.finish_time - self.arrival_time) * 1000

    def metrics(self) -> dict:
        return {
            "request_id": self.request_id,
            "prompt_len": self.prompt_len,
            "generated": self.generated,
            "queue_ms": self.queue_ms,
            "prefill_ms": self.prefill_ms,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "e2e_ms": self.e2e_ms,
            "preemptions": self.preemptions,
            "state": self.state.value,
        }
