"""
Phase 7.3 — INT8 KV in the actual paged decode path.

7.2 showed INT8 round-trip barely moves the model's output (KL 0.00061,
1.37% top-1 flips) using per-channel K / per-token V scales fit over an
entire offline sequence. The online-scaling check (`phase7_online_scaling`)
showed the streaming-realizable version — K's scale fit block-local,
within each 16-token page — is at least as good, and better at every
block size tried. This module is what turns that measurement into
storage: `Int8PagedKVCache` has the same read/write contract as
`PagedKVCache` (cache/paged_cache.py), so `GQAAttention` cannot tell it
apart from either paged cache, but the pool underneath is INT8.

## Why K needs a residual buffer and V does not

A `(block, head, channel)` scale for K can only be fixed once every
token in that block is known — exactly the online-scaling experiment's
"cannot look ahead" constraint. But tokens arrive one at a time on the
decode path (`write()` is called with n=1 per step), so a block is
*filling* for `block_size - 1` steps before it is fully known.

The fix is the standard streaming-quantization move: keep each
sequence's current, not-yet-full block in a small FP16 "residual"
buffer, and only quantize+scatter it into the INT8 pool the instant it
reaches `block_size` tokens. `read()` then splices two sources per
sequence: INT8 pool (dequantized) for every finalized block, and the
FP16 residual directly for the still-filling tail. The tail is
therefore always exact — never quantized — which is a free, not a
approximated, floor: it is *waiting* to be quantized, not a permanent
higher-precision allowance.

V needs none of this. Its scale is per `(token, head)` — one token's
own values, nothing else — so it is already streaming-safe the moment
it is written (`compression/truncation.py`'s note on `per_token` says
the same thing about the *simulated* version of this). V is quantized
and scattered directly in `write()`, no residual, no look-ahead.

## The traffic arithmetic this does *not* fix

Storing INT8 does not by itself halve decode memory traffic. Today's
gather still (a) reads the pool, INT8 or not, (b) dequantizes into an
FP16 buffer, and (c) hands that FP16 buffer to SDPA, which reads it
again. Call the resident KV size 1 unit (FP16). Storage drops to 0.5,
but the traffic is 0.5 (read) + 1 (write the FP16 dequant buffer) + 1
(SDPA reads it) = 2.5, against a resident-FP16 paged cache's gather
cost of 1 (read) + 1 (write the FP16 copy) + 1 (SDPA reads it) = 3 —
roughly a 17% reduction in bytes moved, not 50%. The full 2x needs
attention to consume INT8 directly without a dequantized intermediate,
which means a custom kernel (Phase 11). What *is* real here, and
measurable without a kernel, is storage: half the resident bytes, and
therefore up to ~2x the tokens or concurrent sequences at the same
memory budget — the Phase 7.5 capacity experiment this module exists
to feed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from cache.block_allocator import BlockAllocator, BlockTable
from cache.kv_cache import KVCacheSpec

_EPS = 1e-8


class Int8PagedKVCache:
    """Block-paged KV storage with K/V held as INT8, same read/write
    contract as `PagedKVCache` (and therefore `ContiguousKVCache`), so
    `GQAAttention` cannot tell any of the three apart.

    K: one scale per `(physical block, kv head, channel)`, fixed the
    moment its block fills — matching the paged cache's own page size
    by default (`block_size=16`), which is exactly what the
    online-scaling check measured as both realizable and quality-best.

    V: one scale per `(token, kv head)`, fixed the instant that token
    is written. No look-ahead, no residual.
    """

    def __init__(
        self,
        spec: KVCacheSpec,
        block_size: int = 16,
        num_blocks: Optional[int] = None,
        k_bits: int = 8,
        v_bits: int = 8,
        out_dtype: Optional[torch.dtype] = None,
    ):
        self.spec = spec
        self.block_size = block_size
        self.device = torch.device(spec.device)
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.k_qmax = 2 ** (k_bits - 1) - 1
        self.v_qmax = 2 ** (v_bits - 1) - 1
        # dtype `read()` hands back to attention. Defaults to the cache
        # spec's own dtype (fp16 in practice) so SDPA sees exactly what
        # it would from a non-quantized paged cache.
        self.out_dtype = out_dtype or spec.dtype

        if num_blocks is None:
            # Same convention as PagedKVCache: default to the contiguous
            # cache's total capacity, so a capacity comparison starts
            # from an equal memory *budget*, not an equal block count —
            # doubly important here, since the whole point of Phase 7.5
            # is to show INT8 buys more capacity from that budget.
            per_seq = (spec.max_seq_len + block_size - 1) // block_size
            num_blocks = per_seq * spec.max_batch_size
        self.num_blocks = num_blocks

        self.allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
        self.tables: list[BlockTable] = [
            BlockTable(self.allocator) for _ in range(spec.max_batch_size)
        ]

        kv_shape = (num_blocks, block_size, spec.num_kv_heads, spec.head_dim)
        self.k_pool: list[torch.Tensor] = []
        self.v_pool: list[torch.Tensor] = []
        # K scale: one value per (block, head, channel) — shared by every
        # token in the block, which is the whole memory saving over a
        # per-token scale and the thing that makes it non-streaming-safe
        # without the residual buffer below.
        self.k_scale_pool: list[torch.Tensor] = []
        # V scale: one value per (block*block_size flattened == token,
        # head) — no sharing across tokens, no residual needed.
        self.v_scale_pool: list[torch.Tensor] = []
        for _ in range(spec.num_layers):
            self.k_pool.append(torch.zeros(kv_shape, dtype=torch.int8, device=self.device))
            self.v_pool.append(torch.zeros(kv_shape, dtype=torch.int8, device=self.device))
            self.k_scale_pool.append(
                torch.ones(num_blocks, spec.num_kv_heads, spec.head_dim,
                           dtype=torch.float32, device=self.device)
            )
            self.v_scale_pool.append(
                torch.ones(num_blocks, block_size, spec.num_kv_heads,
                           dtype=torch.float32, device=self.device)
            )

        self._flat_k = [t.view(-1, spec.num_kv_heads, spec.head_dim) for t in self.k_pool]
        self._flat_v = [t.view(-1, spec.num_kv_heads, spec.head_dim) for t in self.v_pool]
        self._flat_v_scale = [t.view(-1, spec.num_kv_heads) for t in self.v_scale_pool]

        # K's not-yet-full tail block per sequence slot, held in the
        # cache's native dtype (fp16) until it fills. Sized
        # [max_batch_size, block_size, kv_heads, head_dim] per layer —
        # a fixed, small cost (one page per sequence per layer), not one
        # that grows with context length the way the pool does.
        self._k_residual: list[torch.Tensor] = [
            torch.zeros(spec.max_batch_size, block_size, spec.num_kv_heads, spec.head_dim,
                        dtype=spec.dtype, device=self.device)
            for _ in range(spec.num_layers)
        ]

        self._read_slots: Optional[torch.Tensor] = None
        self._write_slots: Optional[torch.Tensor] = None
        self._active: list[int] = list(range(spec.max_batch_size))
        self.gather_calls = 0

    # ------------------------------------------------------------------
    # State — identical to PagedKVCache; duplicated rather than shared
    # by inheritance, since the two classes' storage is different enough
    # (int8 pools + scale tables + residual vs. one fp16 pool) that
    # sharing would mean overriding most of __init__ anyway.
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
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
        self._active = list(range(self.spec.max_batch_size))
        self.gather_calls = 0
        # Not strictly required for correctness (every finalized block
        # belongs to a freed sequence and will be overwritten before it
        # is ever read again, same as the int8 pool itself), but zeroing
        # avoids a stale residual masquerading as real data if a bug
        # elsewhere ever reads before writing.
        for r in self._k_residual:
            r.zero_()

    def can_admit(self, num_tokens: int) -> bool:
        return self.allocator.blocks_for_tokens(num_tokens) <= self.allocator.num_free

    def free_sequence(self, index: int) -> None:
        self.tables[index].free()
        self._read_slots = None

    # ------------------------------------------------------------------
    # Allocation — identical to PagedKVCache.
    # ------------------------------------------------------------------

    def advance(
        self,
        n: int,
        batch_size: Optional[int] = None,
        slots: Optional[Sequence[int]] = None,
    ) -> None:
        if slots is not None:
            self.set_active(slots)
        elif batch_size is not None:
            self.set_active(range(batch_size))
        active = self._active
        starts = [self.tables[i].length for i in active]
        for i in active:
            self.tables[i].append(n)

        dev = self.device
        self._write_slots = torch.tensor(
            [
                [self.tables[i].slot(p) for p in range(start, start + n)]
                for i, start in zip(active, starts)
            ],
            dtype=torch.long,
            device=dev,
        )
        b = len(active)
        max_len = self.max_len
        self._read_slots = torch.zeros((b, max_len), dtype=torch.long, device=dev)
        for row, i in enumerate(active):
            seq_slots = self.tables[i].slots()
            if seq_slots:
                self._read_slots[row, : len(seq_slots)] = torch.tensor(
                    seq_slots, dtype=torch.long, device=dev
                )

    def padding_mask(self) -> Optional[torch.Tensor]:
        lens = self.seq_lens
        if len(set(lens)) <= 1:
            return None
        max_len = max(lens)
        pos = torch.arange(max_len, device=self.device)[None, :]
        keep = pos < torch.tensor(lens, device=self.device)[:, None]
        return keep[:, None, None, :]

    # ------------------------------------------------------------------
    # Read/write path — this is the part that differs from PagedKVCache.
    # ------------------------------------------------------------------

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int = 0) -> None:
        """Quantize and scatter [B, kv_heads, n, head_dim] into the pool.

        V is quantized and scattered unconditionally — its scale never
        depends on anything but the token being written. K is buffered
        into the FP16 residual for whatever block(s) this write touches,
        and only quantized into the INT8 pool for a block the instant
        this write completes it. A single call can complete zero, one,
        or several blocks: n=1 on the decode path completes at most one
        (the block it happens to fill), while a prefill chunk can span
        many.
        """
        b, h, n, d = k.shape
        slots = self._write_slots
        if slots is None or slots.shape[0] < b or slots.shape[1] != n:
            raise RuntimeError("call advance(n, batch_size) before write()")

        k_tok = k.permute(0, 2, 1, 3)  # [B, n, h, d]
        v_tok = v.permute(0, 2, 1, 3)

        # --- V: per-token scale, no look-ahead, quantize+scatter now. ---
        v_scale = v_tok.abs().amax(dim=-1, keepdim=True).clamp_min(_EPS)  # [B, n, h, 1]
        v_q = torch.clamp(
            torch.round(v_tok / v_scale * self.v_qmax), -self.v_qmax, self.v_qmax
        ).to(torch.int8)
        flat = slots[:b].reshape(-1)
        self._flat_v[layer_idx].index_copy_(0, flat, v_q.reshape(-1, h, d))
        # Scale pools are float32 (accumulation precision for the scale
        # itself, independent of the cache's activation dtype), while
        # v_scale inherits v's dtype (fp16 on GPU). index_copy_, unlike
        # plain indexed assignment, requires matching dtypes.
        v_scale_flat = v_scale.reshape(-1, h).to(self._flat_v_scale[layer_idx].dtype)
        self._flat_v_scale[layer_idx].index_copy_(0, flat, v_scale_flat)

        # --- K: buffer into the residual, finalize any block this write
        # completes. `slots` gives the flat physical slot for every
        # (row, new-token) pair; flat slot // block_size is the physical
        # block id and flat slot % block_size is the offset within it,
        # by construction of the pool's [num_blocks, block_size, ...]
        # layout (same identity PagedKVCache.read() relies on). ---
        block_ids = slots[:b] // self.block_size  # [B, n]
        block_offs = slots[:b] % self.block_size  # [B, n]
        residual = self._k_residual[layer_idx]
        k_pool = self.k_pool[layer_idx]
        k_scale_pool = self.k_scale_pool[layer_idx]

        for row in range(b):
            seq_idx = self._active[row]
            t = 0
            while t < n:
                blk = int(block_ids[row, t])
                off0 = int(block_offs[row, t])
                # A write's tokens land at consecutive offsets within a
                # sequence, so the run of tokens sharing this physical
                # block is contiguous — find how far it extends.
                run = 1
                while t + run < n and int(block_ids[row, t + run]) == blk:
                    run += 1
                residual[seq_idx, off0 : off0 + run] = k_tok[row, t : t + run]
                if off0 + run == self.block_size:
                    block_data = residual[seq_idx]  # [block_size, h, d], now complete
                    scale = block_data.abs().amax(dim=0).clamp_min(_EPS)  # [h, d]
                    q = torch.clamp(
                        torch.round(block_data / scale * self.k_qmax),
                        -self.k_qmax, self.k_qmax,
                    ).to(torch.int8)
                    k_pool[blk] = q
                    k_scale_pool[blk] = scale
                t += run

    def read(
        self, layer_idx: int, batch_size: int, length: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather + dequantize into [B, kv_heads, L, head_dim], in
        `out_dtype` (fp16 by default) — the same shape and dtype
        `PagedKVCache.read()` returns, so attention needs no changes.

        Per docs/paged_cache.py's own framing: this is still "gather
        into a contiguous buffer" (option (a), Phase 3's motivation for
        a Phase 11 kernel that reads the pool directly), just gathering
        from a smaller, INT8 pool instead of an FP16 one. See this
        module's docstring for the traffic consequence of that.
        """
        slots = self._read_slots
        if slots is None:
            raise RuntimeError("call advance() before read()")
        active = self._active[:batch_size]
        idx = slots[:batch_size] if length is None else slots[:batch_size, :length]
        max_len = idx.shape[1]
        self.gather_calls += 1

        # V: every gathered slot has a real scale (V has no residual
        # state), so this is a uniform gather + dequant, same recipe
        # regardless of whether the token is old or brand new.
        v_q = self._flat_v[layer_idx][idx]  # [B, L, h, d]
        v_scale = self._flat_v_scale[layer_idx][idx]  # [B, L, h]
        v = v_q.to(torch.float32) * (v_scale.unsqueeze(-1) / self.v_qmax)

        # K: dequantize everything as if it were finalized...
        k_q = self._flat_k[layer_idx][idx]  # [B, L, h, d]
        blk_id = idx // self.block_size  # [B, L] — see write()'s note on this identity
        k_scale = self.k_scale_pool[layer_idx][blk_id]  # [B, L, h, d]
        k = k_q.to(torch.float32) * (k_scale.to(torch.float32) / self.k_qmax)

        # ...then splice in each row's true FP16 values for its
        # currently-filling tail block, which has no scale yet (the
        # dequant above used whatever scale happened to be sitting in
        # that not-yet-written pool slot — garbage that this overwrites).
        residual = self._k_residual[layer_idx]
        for row, seq_idx in enumerate(active):
            seq_len = min(self.tables[seq_idx].length, max_len)
            tail_len = seq_len % self.block_size
            if tail_len:
                tail_start = seq_len - tail_len
                k[row, tail_start:seq_len] = residual[seq_idx, :tail_len].to(torch.float32)

        k = k.to(self.out_dtype).permute(0, 2, 1, 3)  # [B, h, L, D]
        v = v.to(self.out_dtype).permute(0, 2, 1, 3)
        return k, v

    # ------------------------------------------------------------------
    # Accounting
    # ------------------------------------------------------------------

    @property
    def k_bytes_per_token(self) -> int:
        """1 byte/element plus this token's share of its block's FP32
        scale, amortized over `block_size` tokens — the number that
        makes a capacity comparison against PagedKVCache honest (see
        Phase 7.5)."""
        elems = self.spec.num_kv_heads * self.spec.head_dim
        return elems + (elems * 4 + self.block_size - 1) // self.block_size

    @property
    def v_bytes_per_token(self) -> int:
        """1 byte/element plus its own FP32 per-token, per-head scale —
        not amortized, since V's scale is not shared across tokens."""
        return self.spec.num_kv_heads * self.spec.head_dim + self.spec.num_kv_heads * 4

    @property
    def bytes_per_token(self) -> int:
        """Summed over layers — the INT8-cache analogue of
        `KVCacheSpec.bytes_per_token`, used in place of it everywhere
        below so `stats()`/`used_bytes()`/etc. report the true INT8
        footprint rather than the FP16 spec's."""
        return self.spec.num_layers * (self.k_bytes_per_token + self.v_bytes_per_token)

    @property
    def allocated_bytes(self) -> int:
        """Whole pool, computed from the actual tensors rather than
        `bytes_per_token * capacity` — that product would double-count
        the way scale bytes get amortized above. Includes the residual
        buffers: real GPU memory, fixed in size (bounded by
        `max_batch_size * block_size`, not by context length), so it is
        a one-time cost that matters less the longer a sequence runs."""
        total = 0
        for layer in range(self.spec.num_layers):
            total += self.k_pool[layer].numel() * self.k_pool[layer].element_size()
            total += self.k_scale_pool[layer].numel() * self.k_scale_pool[layer].element_size()
            total += self.v_pool[layer].numel() * self.v_pool[layer].element_size()
            total += self.v_scale_pool[layer].numel() * self.v_scale_pool[layer].element_size()
            total += self._k_residual[layer].numel() * self._k_residual[layer].element_size()
        return total

    def used_bytes(self, batch_size: Optional[int] = None) -> int:
        rows = self._active if batch_size is None else list(self._active)[:batch_size]
        return self.bytes_per_token * sum(self.tables[i].length for i in rows)

    def reserved_bytes(self, batch_size: Optional[int] = None) -> int:
        rows = self._active if batch_size is None else list(self._active)[:batch_size]
        return self.bytes_per_token * sum(self.tables[i].capacity for i in rows)

    def fragmentation(self, batch_size: Optional[int] = None) -> float:
        reserved = self.reserved_bytes(batch_size)
        return 0.0 if reserved == 0 else 1 - self.used_bytes(batch_size) / reserved

    def utilization(self, batch_size: Optional[int] = None) -> float:
        alloc = self.allocated_bytes
        return 0.0 if alloc == 0 else self.used_bytes(batch_size) / alloc

    def bytes_read_per_decode_step(self, batch_size: int) -> int:
        rows = list(self._active)[:batch_size]
        return self.bytes_per_token * sum(self.tables[i].length for i in rows)

    def gather_bytes_per_decode_step(self, batch_size: int) -> int:
        """Extra traffic the gather adds, in INT8-pool bytes — the
        numerator for the "0.5 + 1 + 1 = 2.5x" arithmetic in this
        module's docstring is this plus the FP16 dequant buffer's own
        read+write, which is sized off `spec.bytes_per_token` (FP16),
        not this. `benchmarks/runners` computes that comparison; this
        method only reports the paging-specific half of it."""
        return 2 * self.bytes_read_per_decode_step(batch_size)

    def stats(self, batch_size: Optional[int] = None) -> dict:
        b = len(self._active) if batch_size is None else batch_size
        return {
            "kv_bytes_per_token": self.bytes_per_token,
            "kv_bytes_per_token_fp16_equiv": self.spec.bytes_per_token,
            "kv_allocated_mb": self.allocated_bytes / 1024 / 1024,
            "kv_used_mb": self.used_bytes(b) / 1024 / 1024,
            "kv_reserved_mb": self.reserved_bytes(b) / 1024 / 1024,
            "kv_utilization": self.utilization(b),
            "internal_fragmentation": self.fragmentation(b),
            "block_size": self.block_size,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "gather_calls": self.gather_calls,
            **self.allocator.stats(),
        }