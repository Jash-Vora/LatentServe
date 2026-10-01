"""
Phase 3 — paged KV cache.

A PagedAttention-style memory manager: KV lives in a pool of fixed-size
blocks, and each sequence holds a block table mapping its logical
positions to scattered physical blocks
(`cache/block_allocator.py`). Compare against `cache/kv_cache.py`'s
contiguous cache under variable sequence lengths, concurrent requests,
request termination and high utilization.

## Set expectations honestly

Paging is a **capacity** optimization. It should make decode latency
slightly *worse*, because attention can no longer read a contiguous
tensor. Phase 2 already showed what that class of cost looks like: the
`repeat_kv` materialisation was 13x the necessary traffic and dominated
TPOT past 4K. The gather here is a 1x copy, not 13x, but it is not free
and the honest framing of Phase 3 is "what capacity does paging buy,
and what does it cost in latency" — not "paging makes things fast".

## Layout

Per layer: `[num_blocks, block_size, kv_heads, head_dim]`, viewed flat
as `[num_blocks * block_size, kv_heads, head_dim]` so a slot index
addresses one token's KV directly.

This is the seq-major layout `cache/kv_cache.py` deliberately did not
use. Phase 2's contiguous cache is head-major because SDPA wants
`[B, H, S, D]` on the read path with no transpose. Here, a token's KV
must be one contiguous run so that scattered blocks can be gathered by
a single index operation, and the transpose moves to the read. Two
caches, two layouts, for reasons that come from how each is accessed —
worth stating in the report, since "which layout is faster" has no
context-free answer.

## The gather, and why it is the Phase 11 hook

Attention needs `[B, kv_heads, L, head_dim]`. Blocks are scattered, so
either (a) gather them into a contiguous buffer each step, or (b) have
the kernel walk the block table itself with an online softmax. (b) is a
real paged-attention kernel and belongs in Phase 11. Phase 3 does (a),
measures it, and that measurement is the motivation for (b). Recording
"the gather cost X ms/token in PyTorch, so a kernel is warranted" is a
better Phase 11 setup than writing the kernel on a hunch.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from cache.block_allocator import BlockAllocator, BlockTable
from cache.kv_cache import KVCacheSpec


class PagedKVCache:
    """Block-paged KV storage with the same read/write contract as
    `ContiguousKVCache`, so `GQAAttention` cannot tell them apart."""

    def __init__(self, spec: KVCacheSpec, block_size: int = 16, num_blocks: Optional[int] = None):
        self.spec = spec
        self.block_size = block_size
        self.device = torch.device(spec.device)

        if num_blocks is None:
            # Default to the same total capacity the contiguous cache
            # would have reserved, so a like-for-like comparison starts
            # from an equal memory budget rather than an equal number of
            # sequences.
            per_seq = (spec.max_seq_len + block_size - 1) // block_size
            num_blocks = per_seq * spec.max_batch_size
        self.num_blocks = num_blocks

        self.allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
        self.tables: list[BlockTable] = [
            BlockTable(self.allocator) for _ in range(spec.max_batch_size)
        ]

        shape = (num_blocks, block_size, spec.num_kv_heads, spec.head_dim)
        self.k_pool: list[torch.Tensor] = []
        self.v_pool: list[torch.Tensor] = []
        for _ in range(spec.num_layers):
            self.k_pool.append(torch.zeros(shape, dtype=spec.dtype, device=self.device))
            self.v_pool.append(torch.zeros(shape, dtype=spec.dtype, device=self.device))

        self._flat_k = [t.view(-1, spec.num_kv_heads, spec.head_dim) for t in self.k_pool]
        self._flat_v = [t.view(-1, spec.num_kv_heads, spec.head_dim) for t in self.v_pool]

        # Slot indices are identical for all 28 layers, so they are built
        # once per step and reused. Recomputing per layer would make the
        # block-table lookup 28x more expensive than it needs to be and
        # would show up as "paging overhead" that is really a bug.
        self._read_slots: Optional[torch.Tensor] = None
        self._write_slots: Optional[torch.Tensor] = None
        self._block_tables: Optional[torch.Tensor] = None
        self._seq_lens_tensor: Optional[torch.Tensor] = None

        # ---- Phase 13: persistent decode buffers ----------------------
        #
        # A CUDA graph records memory *addresses*. Anything the decode
        # step reads must therefore live at the same address on every
        # step, with new values copied in — never a fresh tensor bound to
        # the same attribute name, which leaves the graph reading the old
        # one. These are allocated once at full capacity and written
        # into from here on; `reset()` deliberately does not reallocate
        # them, or a graph captured before a reset would break after it.
        #
        # The block table is held at *capacity* width, not at the current
        # page count. The kernel masks by `seq_lens`, so trailing entries
        # are never read, and a fixed width is what lets one captured
        # graph serve a growing sequence.
        self._capacity_pages = (spec.max_seq_len + block_size - 1) // block_size
        self._block_tables_buf = torch.zeros(
            (spec.max_batch_size, max(1, self._capacity_pages)),
            dtype=torch.int32, device=self.device,
        )
        self._seq_lens_buf = torch.zeros(spec.max_batch_size, dtype=torch.int32,
                                         device=self.device)
        self._write_slots_buf = torch.zeros((spec.max_batch_size, 1), dtype=torch.long,
                                            device=self.device)
        # (slot, pages) last written into each block-table row, so a row
        # is only rewritten when it actually changes — once per page of
        # growth rather than once per token.
        self._table_state: list = [None] * spec.max_batch_size
        # The gather path's per-token slot index is only needed by
        # read(). The kernel path never calls read(), so building it in
        # advance() — a Python loop plus a host-to-device copy per
        # sequence, every step — was pure overhead there.
        self._read_slots_dirty = True
        # Which sequence slots participate in the next forward pass, in
        # batch order. Phase 3 always used range(batch_size); continuous
        # batching (Phase 4) needs holes — slot 3 can finish and be reused
        # by a new request while slots 0, 1, 2 keep decoding.
        self._active: list[int] = list(range(spec.max_batch_size))
        self.gather_calls = 0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Uniform length, for interface parity with ContiguousKVCache.
        Raises on ragged state rather than silently returning the max —
        a caller that assumes uniformity on a ragged cache produces
        wrong attention, not an error."""
        lengths = {self.tables[i].length for i in self._active}
        if len(lengths) > 1:
            raise RuntimeError(f"cache is ragged ({sorted(lengths)}); use seq_lens")
        return lengths.pop() if lengths else 0

    @property
    def seq_lens(self) -> list[int]:
        return [self.tables[i].length for i in self._active]

    @property
    def active_slots(self) -> list[int]:
        return list(self._active)

    def set_active(self, slots: Sequence[int]) -> None:
        """Choose which slots the next forward pass covers, in batch order.

        Row j of every tensor the model passes in or gets back
        corresponds to `slots[j]`. Keeping that mapping explicit — rather
        than implying it from a batch size — is what lets the scheduler
        add and retire sequences mid-flight.
        """
        self._active = list(slots)

    @property
    def max_len(self) -> int:
        return max((self.tables[i].length for i in self._active), default=0)

    def reset(self) -> None:
        for t in self.tables:
            if t.blocks:
                t.free()
        self.allocator.reset()
        self.tables = [BlockTable(self.allocator) for _ in range(self.spec.max_batch_size)]
        self._read_slots = self._write_slots = None
        self._block_tables = self._seq_lens_tensor = None
        # Contents cleared, storage kept: see the persistent-buffer note in
        # __init__. Zeroing is for determinism only — the kernel masks by
        # length and never reads past it.
        self._block_tables_buf.zero_()
        self._seq_lens_buf.zero_()
        self._table_state = [None] * self.spec.max_batch_size
        self._read_slots_dirty = True
        self._active = list(range(self.spec.max_batch_size))
        self.gather_calls = 0

    def can_admit(self, num_tokens: int) -> bool:
        """Whether the pool can seat a prompt of this length right now.
        The scheduler's admission test — a paged cache that dies on
        exhaustion has thrown away the reason to page."""
        return self.allocator.blocks_for_tokens(num_tokens) <= self.allocator.num_free

    def free_sequence(self, index: int) -> None:
        """Return one sequence's blocks to the pool. The operation a
        contiguous cache cannot express — it can only free the whole
        batch — and the reason paging survives churn."""
        self.tables[index].free()
        self._read_slots_dirty = True
        self._table_state[index] = None

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def advance(
        self,
        n: int,
        batch_size: Optional[int] = None,
        slots: Optional[Sequence[int]] = None,
    ) -> None:
        """Reserve n more tokens for each active sequence and rebuild the
        slot index.

        `slots` names the active set directly (Phase 4). `batch_size`
        keeps the Phase 2/3 shorthand of "the first B slots".
        """
        if slots is not None:
            self.set_active(slots)
        elif batch_size is not None:
            self.set_active(range(batch_size))
        active = self._active
        starts = [self.tables[i].length for i in active]
        for i in active:
            self.tables[i].append(n)

        dev = self.device
        b = len(active)
        slots_now = [
            [self.tables[i].slot(p) for p in range(start, start + n)]
            for i, start in zip(active, starts)
        ]
        if n == 1:
            # Decode: write into the persistent buffer so the address is
            # stable across steps. One small host-to-device copy.
            self._write_slots_buf[:b].copy_(torch.tensor(slots_now, dtype=torch.long))
            self._write_slots = self._write_slots_buf[:b]
        else:
            # Prefill is never captured (it stays eager, and its block
            # count varies), so a fresh tensor is fine here.
            self._write_slots = torch.tensor(slots_now, dtype=torch.long, device=dev)

        self._read_slots_dirty = True
        self._rebuild_kernel_inputs()

    def _build_read_slots(self) -> None:
        """Per-token slot index for the gather path, built on demand.

        Pad short sequences with slot 0. Padded positions must be masked
        out by the caller (`padding_mask`); block 0 is never left
        unwritten in practice, so an unmasked pad would silently attend
        to another sequence's tokens.
        """
        dev = self.device
        active = self._active
        b = len(active)
        self._read_slots = torch.zeros((b, self.max_len), dtype=torch.long, device=dev)
        for row, i in enumerate(active):
            seq_slots = self.tables[i].slots()
            if seq_slots:
                self._read_slots[row, : len(seq_slots)] = torch.tensor(
                    seq_slots, dtype=torch.long, device=dev
                )
        self._read_slots_dirty = False

    def _rebuild_kernel_inputs(self) -> None:
        """Block table and lengths as device tensors, for the Phase 11
        kernel.

        Built **once per step, in `advance()`**, for exactly the reason
        `_read_slots` is: these are identical across all 28 layers, and
        each rebuild is a Python loop plus a host-to-device copy. Doing
        it inside attention instead cost ~20 ms per decode step — 28
        rebuilds and 28 H2D transfers — which is more than the gather
        this kernel exists to remove. The docstring above `_read_slots`
        warned about precisely this; the kernel path did it anyway.
        """
        active = self._active
        b = len(active)
        for row, i in enumerate(active):
            blocks = self.tables[i].blocks
            state = (i, len(blocks))
            # A row changes only when its sequence gains a page (every
            # block_size tokens) or the slot is reassigned to a different
            # sequence — not on every token.
            if self._table_state[row] != state:
                if len(blocks) > self._capacity_pages:
                    raise RuntimeError(
                        f"sequence in slot {i} needs {len(blocks)} pages, past the "
                        f"cache's capacity of {self._capacity_pages}"
                    )
                if blocks:
                    self._block_tables_buf[row, : len(blocks)].copy_(
                        torch.tensor(blocks, dtype=torch.int32)
                    )
                self._table_state[row] = state
        self._seq_lens_buf[:b].copy_(
            torch.tensor([self.tables[i].length for i in active], dtype=torch.int32)
        )
        self._block_tables = self._block_tables_buf[:b]
        self._seq_lens_tensor = self._seq_lens_buf[:b]

    def block_tables_tensor(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """[B, max_pages] of physical block ids, for the Phase 11 kernel.

        The gather path never needed this: it flattened the block table
        into per-token slot indices. A kernel that walks the table itself
        needs the pages, not the slots — that difference is the whole
        point, since the slot form is what forces the copy.
        """
        if self._block_tables is None:
            raise RuntimeError("call advance() before block_tables_tensor()")
        if batch_size is None or batch_size >= self._block_tables.shape[0]:
            return self._block_tables
        return self._block_tables[:batch_size]

    def seq_lens_tensor(self, batch_size: Optional[int] = None) -> torch.Tensor:
        if self._seq_lens_tensor is None:
            raise RuntimeError("call advance() before seq_lens_tensor()")
        if batch_size is None or batch_size >= self._seq_lens_tensor.shape[0]:
            return self._seq_lens_tensor
        return self._seq_lens_tensor[:batch_size]

    def padding_mask(self) -> Optional[torch.Tensor]:
        """Boolean keep-mask [B, 1, 1, max_len], or None when the batch is
        uniform and no masking is needed."""
        lens = self.seq_lens
        if len(set(lens)) <= 1:
            return None
        max_len = max(lens)
        pos = torch.arange(max_len, device=self.device)[None, :]
        keep = pos < torch.tensor(lens, device=self.device)[:, None]
        return keep[:, None, None, :]

    # ------------------------------------------------------------------
    # Read/write path
    # ------------------------------------------------------------------

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int = 0) -> None:
        """Scatter [B, kv_heads, n, head_dim] into the block pool.

        `start_pos` is ignored: the destination comes from the block
        table built by `advance()`, which is what lets sequences at
        different lengths share one call. The parameter stays for
        signature parity with ContiguousKVCache.
        """
        b, h, n, d = k.shape
        slots = self._write_slots
        if slots is None or slots.shape[0] < b or slots.shape[1] != n:
            raise RuntimeError("call advance(n, batch_size) before write()")
        flat = slots[:b].reshape(-1)
        self._flat_k[layer_idx].index_copy_(0, flat, k.permute(0, 2, 1, 3).reshape(-1, h, d))
        self._flat_v[layer_idx].index_copy_(0, flat, v.permute(0, 2, 1, 3).reshape(-1, h, d))

    def read(
        self, layer_idx: int, batch_size: int, length: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather scattered blocks into [B, kv_heads, L, head_dim].

        This copy is the price of paging. It is one pass over the live
        KV, per layer, per step — the same bytes a contiguous cache reads
        directly, moved once more. `gather_bytes_per_decode_step()`
        quantifies it and `benchmarks/runners/phase3_paged.py` measures
        whether the prediction holds.
        """
        if self._write_slots is None:
            raise RuntimeError("call advance() before read()")
        if self._read_slots_dirty:
            self._build_read_slots()
        slots = self._read_slots
        idx = slots[:batch_size] if length is None else slots[:batch_size, :length]
        self.gather_calls += 1
        k = self._flat_k[layer_idx][idx]  # [B, L, H, D]
        v = self._flat_v[layer_idx][idx]
        return k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

    # ------------------------------------------------------------------
    # Accounting
    # ------------------------------------------------------------------

    @property
    def allocated_bytes(self) -> int:
        """Whole pool, whether or not blocks are handed out — this is
        what the GPU actually holds.

        There is one pool *per layer*, so the layer count belongs in this
        product. Omitting it under-reported the allocation by 28x for
        Qwen2.5-1.5B, which showed up as a 5 MB cache where 134 MB was
        allocated. `KVCacheSpec.bytes_per_token` already folds the layers
        in, so express it in terms of that rather than re-deriving the
        product and risking the same omission twice.
        """
        return self.spec.bytes_per_token * self.num_blocks * self.block_size

    def used_bytes(self, batch_size: Optional[int] = None) -> int:
        rows = self._active if batch_size is None else list(self._active)[:batch_size]
        return self.spec.bytes_per_token * sum(self.tables[i].length for i in rows)

    def reserved_bytes(self, batch_size: Optional[int] = None) -> int:
        """Bytes in allocated blocks, including the partly-empty tail of
        each sequence. reserved - used is internal fragmentation."""
        rows = self._active if batch_size is None else list(self._active)[:batch_size]
        return self.spec.bytes_per_token * sum(self.tables[i].capacity for i in rows)

    def fragmentation(self, batch_size: Optional[int] = None) -> float:
        reserved = self.reserved_bytes(batch_size)
        return 0.0 if reserved == 0 else 1 - self.used_bytes(batch_size) / reserved

    def utilization(self, batch_size: Optional[int] = None) -> float:
        alloc = self.allocated_bytes
        return 0.0 if alloc == 0 else self.used_bytes(batch_size) / alloc

    def bytes_read_per_decode_step(self, batch_size: int) -> int:
        rows = list(self._active)[:batch_size]
        return self.spec.bytes_per_token * sum(self.tables[i].length for i in rows)

    def gather_bytes_per_decode_step(self, batch_size: int) -> int:
        """Extra traffic paging adds: the gather reads the live KV and
        writes a copy of it, per layer, per step."""
        return 2 * self.bytes_read_per_decode_step(batch_size)

    def stats(self, batch_size: Optional[int] = None) -> dict:
        b = len(self._active) if batch_size is None else batch_size
        return {
            "kv_bytes_per_token": self.spec.bytes_per_token,
            "kv_allocated_mb": self.allocated_bytes / 1024 / 1024,
            "kv_used_mb": self.used_bytes(b) / 1024 / 1024,
            "kv_reserved_mb": self.reserved_bytes(b) / 1024 / 1024,
            "kv_utilization": self.utilization(b),
            "internal_fragmentation": self.fragmentation(b),
            "block_size": self.block_size,
            "gather_calls": self.gather_calls,
            **self.allocator.stats(),
        }


def contiguous_reserved_bytes(spec: KVCacheSpec, seq_lens: Sequence[int]) -> int:
    """What a contiguous cache must reserve for these sequences: every
    slot up to max_seq_len, for every sequence, regardless of how long
    they actually get. The baseline paging is measured against."""
    return spec.bytes_per_token * spec.max_seq_len * len(seq_lens)


def paged_reserved_bytes(spec: KVCacheSpec, seq_lens: Sequence[int], block_size: int) -> int:
    """What paging reserves: each sequence rounded up to a whole block."""
    return spec.bytes_per_token * block_size * sum(
        (n + block_size - 1) // block_size for n in seq_lens
    )