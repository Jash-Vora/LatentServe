"""
Phase 4 — decode-batch assembly.

Continuous batching's per-step work is small enough that the engine
inlines it, but the two quantities it depends on are subtle enough to be
worth naming and testing on their own:

  * **slots** — which cache slot each batch row refers to. Rows are not
    slot indices: slot 7 can be row 0 once slots 0-6 retire. Getting
    this wrong reads another sequence's KV, and the output stays fluent.
  * **positions** — each sequence's absolute position, for RoPE. In a
    ragged batch these differ per row, so a single scalar offset (all
    that Phases 1-3 ever needed) silently gives every sequence the first
    one's positional phase.

Both are pure functions of request state, so they are unit-testable
without a GPU — see tests/test_phase4_serving.py.
"""

from __future__ import annotations

from typing import Sequence

from runtime.request import ServedRequest


def batch_slots(requests: Sequence[ServedRequest]) -> list[int]:
    """Cache slot for each batch row, in row order."""
    slots = [r.slot for r in requests]
    if any(s is None for s in slots):
        raise ValueError("every resident request must hold a cache slot")
    if len(set(slots)) != len(slots):
        raise ValueError(f"duplicate cache slots in one batch: {slots}")
    return slots


def batch_positions(requests: Sequence[ServedRequest]) -> list[int]:
    """Absolute position of the token each row is about to process:
    prompt length plus tokens generated, minus the one being fed in."""
    return [r.prompt_len + r.generated - 1 for r in requests]


def batch_input_tokens(requests: Sequence[ServedRequest]) -> list[int]:
    """The last generated token of each resident sequence."""
    return [r.output_ids[-1] for r in requests]
