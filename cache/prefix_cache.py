"""
Phase 13 — prefix caching: shared prompt prefixes are prefilled once.

Full blocks of written tokens are identified by a *chained* hash: each
block's key covers its own tokens and everything before it, so two prompts
share a block only if they are identical up to and including it — the scheme
vLLM calls automatic prefix caching. A match is verified against the stored
tokens and parent, so a hash collision can never reuse the wrong block.

The cache holds its own reference to every block it records. A block is in
use while any sequence also references it, and evictable once only the cache
does; evictable blocks stay until memory is needed, least recently used
first. The allocator asks for them through its reclaim hook, and admission
counts them as available.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from cache.block_allocator import BlockAllocator


def block_keys(tokens: Sequence[int], block_size: int, num_blocks: Optional[int] = None) -> list:
    """(hash, parent hash, tokens) for each full block, chained."""
    n = len(tokens) // block_size if num_blocks is None else num_blocks
    keys, parent = [], None
    for i in range(n):
        chunk = tuple(tokens[i * block_size:(i + 1) * block_size])
        h = hash((parent, chunk))
        keys.append((h, parent, chunk))
        parent = h
    return keys


@dataclass
class PrefixCache:
    allocator: BlockAllocator
    block_size: int
    by_hash: dict = field(default_factory=dict)          # hash -> block
    entry: dict = field(default_factory=dict)            # block -> (hash, parent, tokens)
    lru: OrderedDict = field(default_factory=OrderedDict)  # block -> None, oldest first
    hits: int = 0
    lookups: int = 0
    evictions: int = 0
    _reclaimable: int = 0

    def __post_init__(self):
        self.allocator.reclaimer = self.reclaim
        self.allocator.reclaimable = self.num_reclaimable
        self.allocator.on_ref_change = self._on_ref_change

    def _on_ref_change(self, block: int, old: int, new: int) -> None:
        """Keep the count of cached blocks only the cache holds (count 1).
        Admission asks for it several times per step; scanning ~16K cached
        blocks each time would have shown up as prefix-caching overhead that
        was really just a lazy count."""
        if block not in self.entry:
            return
        if old == 2 and new == 1:
            self._reclaimable += 1
        elif old == 1 and new == 2:
            self._reclaimable -= 1

    # -- lookup --------------------------------------------------------------

    def match(self, tokens: Sequence[int]) -> list:
        """The longest run of cached blocks matching the start of `tokens`.
        At least the last token is left out, so prefill always has one token
        whose logits give the first output token."""
        limit = (len(tokens) - 1) // self.block_size
        out = []
        for h, parent, chunk in block_keys(tokens, self.block_size, limit):
            blk = self.by_hash.get(h)
            if blk is None or self.entry[blk][1:] != (parent, chunk):
                break
            out.append(blk)
        self.lookups += 1
        self.hits += bool(out)
        for blk in out:
            self.lru.move_to_end(blk)
        return out

    # -- recording -----------------------------------------------------------

    def register(self, tokens: Sequence[int], blocks: Sequence[int]) -> int:
        """Record the full blocks of a sequence (`tokens` written into
        `blocks`, in order). Blocks already cached under the same key are
        left as they are. Returns how many were newly recorded."""
        new = 0
        for (h, parent, chunk), blk in zip(block_keys(tokens, self.block_size), blocks):
            if h in self.by_hash:
                if self.by_hash[h] == blk:
                    self.lru.move_to_end(blk)
                continue
            if blk in self.entry:                 # recorded under another key: leave it
                continue
            self.allocator.incref([blk])          # the cache's own reference (not yet an entry:
            self.by_hash[h] = blk                 # the 1 -> 2 change is not counted)
            self.entry[blk] = (h, parent, chunk)
            self.lru[blk] = None
            new += 1
        return new

    # -- eviction ------------------------------------------------------------

    def num_reclaimable(self) -> int:
        return self._reclaimable

    def _count_reclaimable(self) -> int:
        """The O(n) definition, for tests: must always equal the O(1) count."""
        return sum(1 for b in self.entry if self.allocator.ref_count(b) == 1)

    def reclaim(self, n: int) -> int:
        """Evict up to n blocks only the cache references, oldest first."""
        freed = 0
        for blk in list(self.lru):
            if freed >= n:
                break
            if self.allocator.ref_count(blk) != 1:
                continue
            h = self.entry.pop(blk)[0]            # no longer an entry: the 1 -> 0 change
            del self.by_hash[h]                   # is not seen by the hook, so count it here
            del self.lru[blk]
            self._reclaimable -= 1
            self.allocator.free([blk])
            freed += 1
            self.evictions += 1
        return freed

    def stats(self) -> dict:
        return {"cached_blocks": len(self.entry), "reclaimable": self.num_reclaimable(),
                "lookups": self.lookups, "hits": self.hits, "evictions": self.evictions}
