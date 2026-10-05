"""
Phase 3 — block allocator and block tables.

docs/methodology.md Phase 3: a logical token sequence maps onto
physical blocks that need not be adjacent.

    Token 0 ............ Token N        (logical)
       |                    |
    Block 17  Block 3  Block 41  Block 8 (physical)

This module is deliberately pure Python/CPU and holds no tensors. That
split is what lets most of Phase 3's results — fragmentation, usable
capacity, allocation overhead — be measured by simulating thousands of
requests in seconds on a laptop, instead of burning T4 hours to learn
things that are properties of the allocation policy rather than of the
GPU. `cache/paged_cache.py` is the part that owns memory.

Reference counting is here from the start even though nothing shares
blocks until Phase 13 (prefix caching). Retrofitting refcounts into a
live allocator later means auditing every free() call site under time
pressure; carrying an always-1 counter now costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


class OutOfBlocks(RuntimeError):
    """Raised when the pool cannot satisfy an allocation.

    A distinct type because Phase 4's scheduler must treat this as
    backpressure — queue the request, preempt another — rather than as a
    crash. A paged cache that dies on exhaustion has thrown away the
    main reason to page.
    """


@dataclass
class BlockAllocator:
    """Fixed pool of equal-sized blocks with a free list.

    LIFO reuse (`pop()` from the tail) is intentional: the most recently
    freed block is the one most likely still resident in L2, and it
    keeps block ids clustered, which matters for the gather in
    `paged_cache.py`. FIFO would spread a sequence's blocks across the
    whole pool over time.
    """

    num_blocks: int
    block_size: int

    _free: list[int] = field(default_factory=list, init=False)
    _ref_count: dict[int, int] = field(default_factory=dict, init=False)
    alloc_calls: int = field(default=0, init=False)
    free_calls: int = field(default=0, init=False)
    peak_used: int = field(default=0, init=False)
    # Phase 13: a prefix cache keeps blocks no sequence references, until
    # memory is needed. `reclaimable()` counts them; `reclaimer(n)` evicts up
    # to n back into the pool. Both None without prefix caching.
    reclaimer: Optional[Callable[[int], None]] = field(default=None, init=False, repr=False)
    reclaimable: Optional[Callable[[], int]] = field(default=None, init=False, repr=False)
    # Called as on_ref_change(block, old, new) — lets the prefix cache keep
    # its reclaimable count in O(1) instead of scanning every cached block.
    on_ref_change: Optional[Callable[[int, int, int], None]] = field(default=None, init=False,
                                                                       repr=False)

    def __post_init__(self) -> None:
        if self.num_blocks <= 0 or self.block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        # Descending so that pop() hands out 0, 1, 2, ... first.
        self._free = list(range(self.num_blocks - 1, -1, -1))

    # ------------------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def utilization(self) -> float:
        return self.num_used / self.num_blocks

    def blocks_for_tokens(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    # ------------------------------------------------------------------

    @property
    def num_available(self) -> int:
        """Free blocks plus cached blocks nobody references: what an
        allocation can actually get."""
        return self.num_free + (self.reclaimable() if self.reclaimable else 0)

    def allocate(self, num_blocks: int) -> list[int]:
        if num_blocks > self.num_free and self.reclaimer is not None:
            self.reclaimer(num_blocks - self.num_free)
        if num_blocks > self.num_free:
            raise OutOfBlocks(
                f"requested {num_blocks} blocks, {self.num_free} free "
                f"({self.num_used}/{self.num_blocks} in use)"
            )
        self.alloc_calls += 1
        out = [self._free.pop() for _ in range(num_blocks)]
        for b in out:
            self._ref_count[b] = 1
        self.peak_used = max(self.peak_used, self.num_used)
        return out

    def free(self, blocks: Iterable[int]) -> None:
        """Decrement refcounts; return to the pool at zero.

        Shared blocks (Phase 13) are freed by whichever sequence finishes
        last, which is why this cannot simply push every block back.
        """
        self.free_calls += 1
        for b in blocks:
            count = self._ref_count.get(b)
            if count is None:
                raise ValueError(f"freeing block {b} that is not allocated")
            if count > 1:
                self._ref_count[b] = count - 1
            else:
                del self._ref_count[b]
                self._free.append(b)
            if self.on_ref_change is not None:
                self.on_ref_change(b, count, count - 1)

    def incref(self, blocks: Iterable[int]) -> None:
        """Mark blocks as shared by one more sequence. Unused until
        Phase 13; the fork-a-prefix operation is exactly this."""
        for b in blocks:
            if b not in self._ref_count:
                raise ValueError(f"increfing block {b} that is not allocated")
            self._ref_count[b] += 1
            if self.on_ref_change is not None:
                self.on_ref_change(b, self._ref_count[b] - 1, self._ref_count[b])

    def ref_count(self, block: int) -> int:
        return self._ref_count.get(block, 0)

    def reset(self) -> None:
        self._free = list(range(self.num_blocks - 1, -1, -1))
        self._ref_count.clear()
        self.alloc_calls = self.free_calls = self.peak_used = 0

    def stats(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "blocks_used": self.num_used,
            "blocks_free": self.num_free,
            "block_utilization": self.utilization,
            "peak_blocks_used": self.peak_used,
            "alloc_calls": self.alloc_calls,
            "free_calls": self.free_calls,
        }


@dataclass
class BlockTable:
    """One sequence's logical-to-physical map.

    Holds `blocks` (physical ids, in logical order) and `length` (tokens
    actually written). The gap between `length` and
    `len(blocks) * block_size` is **internal fragmentation** — the tail
    block is usually partly empty. That waste is bounded by block_size
    per sequence, which is the central trade in Phase 3: smaller blocks
    waste less and cost more bookkeeping.
    """

    allocator: BlockAllocator
    blocks: list[int] = field(default_factory=list)
    length: int = 0
    # Bumped on every change to `blocks`. Consumers that mirror the table
    # elsewhere — the kernel's persistent block-table rows — key on it.
    # Keying on (slot, number of blocks) instead was a real bug: a finished
    # request's slot, reused by a new request with the same block count in
    # the same batch row, kept the old request's blocks, and the kernel
    # attended over the previous request's cache. The allocator's LIFO free
    # list even handed back the same ids in reverse order, so it looked
    # plausible. A uniform burst workload — every benchmark — triggers it.
    version: int = 0

    @property
    def capacity(self) -> int:
        return len(self.blocks) * self.allocator.block_size

    @property
    def wasted_slots(self) -> int:
        return self.capacity - self.length

    def reserve(self, num_tokens: int) -> None:
        """Ensure room for `length + num_tokens`, allocating as needed.

        Growth is incremental and on demand — the entire point. A
        contiguous cache must reserve max_seq_len up front for every
        sequence whether or not it ever gets there.
        """
        needed = self.allocator.blocks_for_tokens(self.length + num_tokens)
        if needed > len(self.blocks):
            self.blocks.extend(self.allocator.allocate(needed - len(self.blocks)))
            self.version += 1

    def append(self, num_tokens: int) -> None:
        self.reserve(num_tokens)
        self.length += num_tokens

    def slot(self, position: int) -> int:
        """Flat physical slot for a logical position: the index into a
        pool viewed as [num_blocks * block_size, ...]."""
        if position >= self.length:
            raise IndexError(f"position {position} beyond length {self.length}")
        bs = self.allocator.block_size
        return self.blocks[position // bs] * bs + (position % bs)

    def slots(self, start: int = 0, end: Optional[int] = None) -> list[int]:
        return [self.slot(p) for p in range(start, self.length if end is None else end)]

    def free(self) -> None:
        self.allocator.free(self.blocks)
        self.blocks = []
        self.length = 0
        self.version += 1
