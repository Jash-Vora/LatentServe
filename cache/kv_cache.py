"""
Phase 2 — LatentServe's own contiguous KV cache.

docs/methodology.md Phase 2 specifies the representation:

    K[layer][sequence][kv_heads][head_dim]
    V[layer][sequence][kv_heads][head_dim]

and the measurements: KV bytes/token, VRAM growth, memory bandwidth,
decode latency.

Two deliberate differences from the Hugging Face cache Phase 1 used:

  * **Preallocated, not grown.** HF's DynamicCache concatenates a new
    tensor every decode step, so its KV allocation is a moving target
    and its cost is tangled up with allocator behaviour. LatentServe
    allocates the full [B, kv_heads, max_seq_len, head_dim] block once
    and writes into slices. That makes "KV memory" an exact,
    reportable number instead of an estimate, and it is what Phase 3's
    paged allocator gets compared against (contiguous vs. paged).
  * **Layout is ours to choose.** Stored head-major
    ([B, kv_heads, seq, head_dim]) so the decode read path hands SDPA a
    tensor in the layout it already wants, with no per-step transpose.
    The alternative (seq-major, [B, seq, kv_heads, head_dim]) writes
    one contiguous run per token instead of one per head, which is the
    friendlier layout for a paged allocator and for gather-based sparse
    attention. Phase 3 and Phase 14 should revisit this; Phase 11
    should measure it rather than argue about it.

The `KVHeadsMode` knob is the controlled experiment behind Phase 2's
"vary GQA configuration". Qwen2.5-1.5B-Instruct's KV head count is
fixed by its weights (12 query heads / 2 KV heads, group size 6), and
you cannot retrain it — but you can change how many KV heads the
*cache* stores:

  * ``native``   — 2 KV heads. The real thing.
  * ``mha_sim``  — expand the 2 KV heads to 12 *before* caching, i.e.
    store one KV head per query head. The attention math is unchanged,
    so logits are bit-identical to ``native``; only memory and memory
    traffic go up 6x. Any latency difference is therefore a pure
    memory-system effect, with model quality held exactly constant.
    This is the cleanest possible answer to "how does GQA affect decode
    performance?" — cleaner than comparing two different checkpoints.
  * ``mqa_sim``  — mean-pool the 2 KV heads into 1, halving KV memory.
    Unlike mha_sim this *does* change the numerics (it is not what the
    weights were trained for), so it is a memory/bandwidth probe only
    and its outputs must not be used for quality claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

KVHeadsMode = Literal["native", "mha_sim", "mqa_sim"]


def effective_kv_heads(num_kv_heads: int, num_attention_heads: int, mode: KVHeadsMode) -> int:
    """How many KV heads the cache physically stores under `mode`."""
    if mode == "native":
        return num_kv_heads
    if mode == "mha_sim":
        return num_attention_heads
    if mode == "mqa_sim":
        return 1
    raise ValueError(f"unknown kv_heads_mode {mode!r}")


@dataclass(frozen=True)
class KVCacheSpec:
    """Everything needed to size a cache, and to state its cost in bytes
    before allocating it (so a sweep can skip a configuration that
    cannot fit rather than OOM halfway through)."""

    num_layers: int
    num_kv_heads: int  # *effective* — already resolved through KVHeadsMode
    head_dim: int
    max_batch_size: int
    max_seq_len: int
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

    @property
    def dtype_bytes(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()

    @property
    def bytes_per_token(self) -> int:
        """KV bytes for one token of one sequence, summed over layers.
        The 2 is K and V. For Qwen2.5-1.5B native GQA in fp16:
        2 * 28 * 2 * 128 * 2 = 28,672 B/token (28 KiB)."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def total_bytes(self) -> int:
        return self.bytes_per_token * self.max_seq_len * self.max_batch_size

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1024 / 1024

    def describe(self) -> str:
        return (
            f"KVCacheSpec(layers={self.num_layers}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, B={self.max_batch_size}, S={self.max_seq_len}, "
            f"{self.bytes_per_token}B/token, {self.total_mb:.1f}MB total)"
        )


def max_context_for_budget(spec: KVCacheSpec, budget_bytes: int, batch_size: int) -> int:
    """Longest context whose KV cache fits in `budget_bytes` at
    `batch_size`. Used to report usable cache capacity (a Phase 3
    metric, available cheaply from Phase 2 onward) and to skip
    infeasible sweep points."""
    per_token = spec.bytes_per_token * batch_size
    return 0 if per_token == 0 else int(budget_bytes // per_token)


class ContiguousKVCache:
    """Preallocated per-layer K/V tensors, written in place.

    Phase 2 holds one shared fill length for the whole batch: the
    benchmark drives fixed-length synthetic prompts, so every sequence
    in a batch has the same length. Ragged batches arrive with the
    scheduler in Phase 4, and are the reason Phase 3's paged allocator
    exists — a contiguous cache has to pad to the longest sequence,
    which is precisely the waste we want to measure it losing.
    """

    def __init__(self, spec: KVCacheSpec):
        self.spec = spec
        self.device = torch.device(spec.device)
        self._length = 0
        self.k: list[torch.Tensor] = []
        self.v: list[torch.Tensor] = []
        for _ in range(spec.num_layers):
            shape = (spec.max_batch_size, spec.num_kv_heads, spec.max_seq_len, spec.head_dim)
            self.k.append(torch.zeros(shape, dtype=spec.dtype, device=self.device))
            self.v.append(torch.zeros(shape, dtype=spec.dtype, device=self.device))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Tokens currently cached per sequence."""
        return self._length

    def reset(self) -> None:
        """Forget contents without freeing memory. Deliberately does not
        zero the tensors: stale values past `length` are never read
        (reads are sliced to `length`), and zeroing 7 GB between trials
        would show up in the benchmark as latency that no real server
        pays."""
        self._length = 0

    def advance(self, n: int, batch_size: Optional[int] = None) -> None:
        """`batch_size` is accepted and ignored: a contiguous cache has
        one shared fill length. PagedKVCache needs it to build per-sequence
        block tables, and the two must be interchangeable from the
        executor's point of view."""
        if self._length + n > self.spec.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: {self._length} + {n} > max_seq_len="
                f"{self.spec.max_seq_len}. Size the cache for input+output tokens."
            )
        self._length += n

    # ------------------------------------------------------------------
    # Read/write path
    # ------------------------------------------------------------------

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> None:
        """Write [B, kv_heads, n, head_dim] at absolute position
        `start_pos`. In place — no allocation on the decode path, which
        is the whole point of preallocating."""
        n = k.shape[2]
        end = start_pos + n
        if end > self.spec.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow on layer {layer_idx}: writing [{start_pos}, {end}) "
                f"into max_seq_len={self.spec.max_seq_len}"
            )
        batch = k.shape[0]
        self.k[layer_idx][:batch, :, start_pos:end].copy_(k)
        self.v[layer_idx][:batch, :, start_pos:end].copy_(v)

    def read(self, layer_idx: int, batch_size: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Views of the first `length` cached positions. Views, not
        copies — a copy here would double KV traffic per decode step and
        quietly invalidate every bandwidth number Phase 2 reports."""
        return (
            self.k[layer_idx][:batch_size, :, :length],
            self.v[layer_idx][:batch_size, :, :length],
        )

    # ------------------------------------------------------------------
    # Accounting (the numbers Phase 2 exists to produce)
    # ------------------------------------------------------------------

    @property
    def allocated_bytes(self) -> int:
        return self.spec.total_bytes

    def used_bytes(self, batch_size: Optional[int] = None) -> int:
        b = self.spec.max_batch_size if batch_size is None else batch_size
        return self.spec.bytes_per_token * self._length * b

    def utilization(self, batch_size: Optional[int] = None) -> float:
        """Fraction of the allocated block actually holding live KV.
        Under a contiguous cache with uniform lengths this is just
        length/max_seq_len; it becomes the interesting number in Phase 3,
        where paging is supposed to push it toward 1.0 under ragged
        workloads."""
        alloc = self.allocated_bytes
        return 0.0 if alloc == 0 else self.used_bytes(batch_size) / alloc

    def padding_mask(self) -> None:
        """Always None: uniform lengths, nothing to mask. Present so the
        attention path can ask either cache the same question."""
        return None

    def gather_bytes_per_decode_step(self, batch_size: int) -> int:
        """Zero — a contiguous cache is read in place. The paged cache's
        non-zero answer here is exactly the cost Phase 3 measures."""
        return 0

    def bytes_read_per_decode_step(self, batch_size: int) -> int:
        """KV bytes attention must read to decode one token per sequence:
        the entire cache up to `length`, for every layer. This is the
        numerator of Phase 2's achieved-bandwidth estimate, and the
        quantity MLA (Phase 7) is trying to shrink."""
        return self.spec.bytes_per_token * self._length * batch_size

    def stats(self, batch_size: Optional[int] = None) -> dict:
        b = self.spec.max_batch_size if batch_size is None else batch_size
        return {
            "kv_bytes_per_token": self.spec.bytes_per_token,
            "kv_allocated_mb": self.allocated_bytes / 1024 / 1024,
            "kv_used_mb": self.used_bytes(b) / 1024 / 1024,
            "kv_utilization": self.utilization(b),
            "kv_length": self._length,
            "kv_heads": self.spec.num_kv_heads,
            "max_seq_len": self.spec.max_seq_len,
            "batch_size": b,
        }
