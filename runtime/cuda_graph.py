"""
Phase 13 — CUDA graph decode.

## Why

Phase 12's profiler, batch 1 / ctx 8192, per decode step:

    wall         40.0 ms
    GPU busy     24.6 ms
    CPU busy     42.6 ms     <- more than wall
    GPU idle     15.4 ms     (39% of the step)

CPU time exceeds wall time: the step is limited by Python dispatching
~900 kernel launches, not by the GPU executing them. Attention is 5.7 ms
of the 24.6; even a free attention kernel would leave wall near 40.
That is why three rounds of kernel tuning barely moved batch 1.

A CUDA graph records the launch sequence once and replays it with a
single call. The per-launch Python cost disappears; wall time should
fall toward the 24.6 ms the GPU actually needs.

## What a graph freezes, and how each is handled

A graph records memory **addresses** and **shapes**. On replay it reads
the same addresses with the same sizes, whatever the Python objects now
say. Anything that moves or resizes between steps replays stale data —
silently, with fluent output.

  * Cache bookkeeping is host-side Python (`advance`). It runs *outside*
    the graph, before each replay, and writes into persistent buffers
    the graph already points at (cache/paged_cache.py, Phase 13 note).
  * Token ids and positions are copied into static input buffers owned
    here.
  * The block table is held at the cache's full capacity width, so one
    graph serves a sequence that grows; the kernel masks by `seq_lens`.
  * The kernel's split count is pinned per graph, since a captured grid
    cannot change size. The kernel derives pages-per-split on the device
    from `seq_lens`, so a fixed count stays correct as the length grows.
  * RoPE is given an explicit bound, so it needs no `.item()` sync.
  * Scratch buffers are keyed by shape, so capturing a second graph with
    a different split count cannot free the first graph's memory.

## Buckets

Graphs are keyed by (batch size, context bucket). The bucket is a
*performance* refinement, not a correctness one: any graph whose bucket
covers the current length gives the right answer. What the bucket buys
is a split count suited to that length — a long-context split count on a
short sequence launches mostly-empty programs.

## Capture writes real KV, safely

Warm-up and capture both run the step, and the step writes this token's
K and V into its cache slot. That is harmless because it is idempotent:
the same token at the same position writes the same values to the same
slot every time. So capture happens *during* a real decode step, using
that step's real inputs — no throwaway slot, no rollback.

Note that capture records kernels without executing them; the graph is
replayed once immediately afterwards to actually produce this step's
logits.

## Scope

fp16 paged cache with the Triton kernel path only.

  * The contiguous cache reads a slice whose length changes every step,
    which is a shape change.
  * The INT8 cache finalises a block (quantises the residual) on a
    host-side condition — when a block fills — and its residual writes
    are not idempotent. Both make it uncapturable as written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from cache.paged_cache import PagedKVCache

DEFAULT_BUCKETS = (1024, 2048, 4096, 8192, 16384, 32768)


class GraphUnsupported(RuntimeError):
    """Raised for a configuration a graph cannot capture correctly.

    A distinct type so the serving engine can fall back to eager instead
    of crashing — a missing graph is a slowdown, not a failure.
    """


def check_capturable(model) -> None:
    cache = model.cache
    if cache is None:
        raise GraphUnsupported("allocate the cache before capturing")
    try:
        from cache.int8_paged_cache import Int8PagedKVCache

        if isinstance(cache, Int8PagedKVCache):
            raise GraphUnsupported(
                "the INT8 cache finalises blocks on a host-side condition and its "
                "residual writes are not idempotent; capture the fp16 paged cache"
            )
    except ImportError:
        pass
    if not isinstance(cache, PagedKVCache):
        raise GraphUnsupported(
            "the contiguous cache reads a slice whose length changes every step"
        )
    if any(layer.attn.attn_impl != "triton_paged" for layer in model.layers):
        raise GraphUnsupported('graph capture needs attn_impl="triton_paged" on every layer')
    if model.rope is None:
        raise GraphUnsupported('graph capture needs rope_source="latentserve"')


@dataclass
class CapturedDecode:
    """One CUDA graph for a fixed (batch size, context bucket)."""

    model: object
    batch_size: int
    bucket_tokens: int
    num_splits: int
    warmup: int = 3

    static_ids: torch.Tensor = field(init=False)
    static_pos: torch.Tensor = field(init=False)
    static_logits: Optional[torch.Tensor] = field(init=False, default=None)
    graph: Optional[torch.cuda.CUDAGraph] = field(init=False, default=None)
    replays: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        dev = self.model.device
        self.static_ids = torch.zeros(self.batch_size, 1, dtype=torch.long, device=dev)
        self.static_pos = torch.zeros(self.batch_size, 1, dtype=torch.long, device=dev)

    def _set_splits(self, value: Optional[int]) -> None:
        for layer in self.model.layers:
            layer.attn.num_splits = value

    def _run(self) -> torch.Tensor:
        return self.model.decode_forward_static(
            self.static_ids, self.static_pos, max_position=self.bucket_tokens - 1
        )

    def load(self, token_ids: torch.Tensor, positions: torch.Tensor) -> None:
        """Copy this step's inputs into the buffers the graph reads.

        The step that most often goes wrong. Skip it and the graph
        replays the previous step's tokens — no error, plausible output.
        """
        self.static_ids.copy_(token_ids)
        self.static_pos.copy_(positions)

    def capture(self, pool=None) -> None:
        """Record the step. Caller has already advanced the cache and
        called `load()` with this step's real inputs.

        `pool` is a memory pool shared with other graphs. Each graph
        otherwise gets a private pool sized to its own peak, and warming
        every batch size at two context points is 2B graphs — 32 at
        batch 16. Sharing is safe here for two reasons: graphs never run
        concurrently, and each graph's output stays referenced by its
        `static_logits`, so no later capture can reuse it. Intermediates
        are reused across graphs, which is fine because nothing reads
        them between replays.
        """
        self._set_splits(self.num_splits)
        try:
            # Warm-up on a side stream, as torch.cuda.graph requires: this
            # is also where Triton compiles and where every scratch buffer
            # gets allocated, so nothing allocates inside the capture.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self.warmup):
                    self._run()
            torch.cuda.current_stream().wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self.static_logits = self._run()
            self.graph = graph
        finally:
            # The split count is baked into the graph now; leave eager
            # calls to choose their own.
            self._set_splits(None)

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        self.replays += 1
        return self.static_logits


class GraphedDecoder:
    """Decode steps through captured graphs, eager where none applies.

    Usage per step, in place of `model.decode_step_ragged`:

        logits = decoder.step(token_ids, positions, slots)

    The returned tensor is the graph's static output buffer: it is
    overwritten by the next replay of the same graph. Read what you need
    (argmax, sampling) before the next step, or clone it.
    """

    def __init__(
        self,
        model,
        buckets: Sequence[int] = DEFAULT_BUCKETS,
        num_splits: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.model = model
        self.num_splits_override = num_splits or {}
        self.enabled = enabled and torch.cuda.is_available()
        self.graphs: dict[tuple[int, int], CapturedDecode] = {}
        self.eager_steps = 0
        self.graph_steps = 0
        self.captures = 0
        self._pool = None
        if self.enabled:
            check_capturable(model)

        capacity = model.cache.spec.max_seq_len
        # Buckets past the cache's capacity are unreachable; the capacity
        # itself becomes the last bucket so no reachable length falls off
        # the end into eager.
        self.buckets = tuple(sorted({b for b in buckets if b < capacity} | {capacity}))

        # RoPE rebuilds its tables when a position exceeds them, and a
        # rebuild *reallocates* — leaving every graph captured before it
        # pointing at freed cos/sin memory, which replays without error.
        # Build to capacity now: advance() refuses positions past
        # capacity, so no rebuild can ever happen after a capture.
        if model.rope is not None and model.rope.max_seq_len < capacity:
            model.rope._build_tables(capacity)

    def bucket_for(self, length: int) -> Optional[int]:
        for b in self.buckets:
            if length <= b:
                return b
        return None

    def _splits_for(self, batch: int, bucket: int) -> int:
        if (batch, bucket) in self.num_splits_override:
            return self.num_splits_override[(batch, bucket)]
        from kernels.gqa.paged_decode import choose_num_splits

        cache = self.model.cache
        return choose_num_splits(
            batch, cache.spec.num_kv_heads, bucket, cache.block_size, self.model.device
        )

    @torch.no_grad()
    def step(
        self, token_ids: torch.Tensor, positions: torch.Tensor, slots: Sequence[int]
    ) -> torch.Tensor:
        cache = self.model.cache
        # Host bookkeeping first, outside any graph: decides where this
        # step's KV goes and updates the persistent buffers in place.
        cache.advance(1, slots=list(slots))
        batch = len(slots)
        bucket = self.bucket_for(cache.max_len)
        rope_limit = self.model.rope.max_seq_len

        if not self.enabled or bucket is None or bucket > rope_limit:
            self.eager_steps += 1
            return self.model.decode_forward_static(
                token_ids, positions, max_position=cache.max_len
            )

        key = (batch, bucket)
        graph = self.graphs.get(key)
        if graph is None:
            graph = CapturedDecode(self.model, batch, bucket, self._splits_for(batch, bucket))
            graph.load(token_ids, positions)
            graph.capture(pool=self._pool)
            if self._pool is None:
                self._pool = graph.graph.pool()
            self.graphs[key] = graph
            self.captures += 1
        else:
            graph.load(token_ids, positions)
        self.graph_steps += 1
        return graph.replay()

    def stats(self) -> dict:
        return {
            "graphs": len(self.graphs),
            "captures": self.captures,
            "graph_steps": self.graph_steps,
            "eager_steps": self.eager_steps,
            "keys": sorted(self.graphs),
        }
