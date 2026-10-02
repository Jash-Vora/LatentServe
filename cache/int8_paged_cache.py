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
therefore always exact — never quantized — which is a free, not an
approximated, floor: it is *waiting* to be quantized, not a permanent
higher-precision allowance.

V needs none of this. Its scale is per `(token, head)` — one token's
own values, nothing else — so it is already streaming-safe the moment
it is written (`compression/truncation.py`'s note on `per_token` says
the same thing about the *simulated* version of this). V is quantized
and scattered directly in `write()`, no residual, no look-ahead.

## Scale convention

Both scale pools hold the **dequantization step** — `absmax / qmax`,
the multiplier that turns a stored integer back into a value — not the
absmax itself. That is the standard meaning of "scale" in symmetric
quantization (`x ~= q * scale`), and it keeps the read path to a single
broadcast multiply instead of a multiply and a divide. The pools stay
FP32: fitting a step is a reduction over up to `block_size` tokens, and
a step that underflows is a whole channel silently zeroed.

## The traffic arithmetic this does *not* fix

Storing INT8 does not by itself reduce decode memory traffic, and an
earlier version of this docstring claimed a reduction the code could
not deliver. In units of the resident FP16 KV size, per layer per step,
what eager PyTorch actually moves is:

    gather whole blocks out of the INT8 pool   0.5 read + 0.5 write
    dequantize that buffer into FP16           0.5 read + 1.0 write
    SDPA reads the FP16 buffer                 1.0 read
                                               ---------------------
                                               3.5

against a resident-FP16 paged cache's 1.0 (gather read) + 1.0 (gather
write) + 1.0 (SDPA read) = 3.0. INT8 storage therefore costs roughly
17% *more* bytes moved per decode step, not less: the pool is half the
size, but it is read, expanded, and read again, and the expansion is
the largest single term. Getting under 3.0 needs the gather and the
dequantize fused into one pass — i.e. attention consuming INT8
directly, which means a custom kernel (Phase 11).

`gather_bytes_per_decode_step()` reports that number rather than the
0.5-read-plus-1.0-write an ideal fused kernel would move, so the
benchmark's printed gather figure matches what the hardware sees.

What *is* real here, and measurable without a kernel, is storage: half
the resident bytes, and therefore up to ~2x the tokens or concurrent
sequences at the same memory budget — the Phase 7.5 capacity experiment
this module exists to feed.

## Cost model for the write path

Everything structural on the write path is per-`advance()`, not
per-layer. Which block ids a write will finalize, the head/middle/tail
split of the chunk, and the row/sequence index tensors depend only on
the block tables, so they are computed once in `advance()` and replayed
by all 28 `write()` calls. Nothing on this path reads a device tensor
back to the host: doing so once per token (as an earlier version did)
cost ~129,000 device synchronizations per 4096-token prefill chunk and
dominated TTFT completely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from cache.block_allocator import BlockAllocator, BlockTable
from cache.kv_cache import KVCacheSpec

_EPS = 1e-8

# Either a basic slice (a view, no gather) or a LongTensor of ids.
# Consecutive ids are by far the common case — a plain `range(B)` batch —
# and taking the slice keeps the write path off PyTorch's advanced
# indexing machinery entirely.
Rows = Union[slice, torch.Tensor]


@dataclass
class _KWriteGroup:
    """The K-side plan for a set of rows that share a start position.

    A write of `n` tokens starting at logical position `start` splits
    into at most three pieces:

      * `head`   — tokens landing in the block already partly filled
                   (`start % block_size != 0`). They go to the residual;
                   if they reach the block boundary, `head_ids` names the
                   physical block to finalize.
      * `middle` — `full` whole blocks contained entirely in this write.
                   They never touch the residual: their scale is known
                   from the chunk itself, so they are quantized straight
                   into the pool. `mid_ids` names them, row-major.
      * `tail`   — the leftover that leaves the next block partly filled.
                   Straight to the residual, nothing finalized.

    Rows are grouped by `start` so the uniform case — every active
    sequence at the same length, which is every prefill and every
    non-ragged decode — is one group, and therefore one set of tensor
    ops for the whole batch.
    """

    rows: Rows
    seqs: Rows
    off0: int
    head: int
    full: int
    tail: int
    head_ids: Optional[torch.Tensor]
    mid_ids: Optional[torch.Tensor]


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
        asymmetric: bool = False,
        out_dtype: Optional[torch.dtype] = None,
    ):
        self.spec = spec
        self.block_size = block_size
        self.device = torch.device(spec.device)
        if not 2 <= k_bits <= 8 or not 2 <= v_bits <= 8:
            raise ValueError("k_bits/v_bits must be in [2, 8]: the pool is INT8-backed")
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.k_qmax = 2 ** (k_bits - 1) - 1
        self.v_qmax = 2 ** (v_bits - 1) - 1
        # Asymmetric (affine) quantization fits [min, max] per group
        # instead of assuming the range is centred on zero. Measured on
        # Qwen2.5-1.5B it is 6.5x better than symmetric at 8 bits
        # (KL 0.00006 vs 0.00039) and categorically better below that:
        # symmetric int6 produces max-KL of 8.2 nats — individual
        # positions destroyed — against 0.022 asymmetric. K's channels
        # carry a DC offset, and a symmetric range spends half its levels
        # on a sign those channels barely use.
        #
        # The cost is a zero-point stored alongside every scale, doubling
        # the scale metadata: 25% of the int8 K payload at block 16, 6%
        # at block 64. `k_bytes_per_token` counts it.
        self.asymmetric = asymmetric
        self.k_levels = 2**k_bits - 1
        self.v_levels = 2**v_bits - 1
        # Stored codes stay in int8 range by shifting the unsigned code
        # down by half the range: q_s = q_u - 2^(bits-1). The shift is
        # folded into the stored zero-point at write time
        # (zero' = lo + offset * step, since
        #  q_s*step + zero' = (q_u - offset)*step + lo + offset*step = q_u*step + lo)
        # so the read path is
        # `q_s * step + zero'` — one multiply-add, identical in shape to
        # the symmetric path, and with nothing added back to an int8
        # tensor. Adding it back at read time instead silently overflows:
        # int8 + 128 stays int8, so code 127 wraps to -1. That only shows
        # up at 8 bits — at 4 the range is [-8, 7] and nothing wraps —
        # which made it look like a bit-width bug rather than a dtype one.
        self.k_offset = 2 ** (k_bits - 1)
        self.v_offset = 2 ** (v_bits - 1)
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
        # K scale: one step per (block, head, channel) — shared by every
        # token in the block, which is the whole memory saving over a
        # per-token scale and the thing that makes it non-streaming-safe
        # without the residual buffer below.
        self.k_scale_pool: list[torch.Tensor] = []
        # V scale: one step per (block, offset == token, head) — no
        # sharing across tokens, no residual needed.
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
        # Zero-point pools mirror the scale pools exactly, and are only
        # allocated when asymmetric — a symmetric cache pays nothing.
        self.k_zero_pool: list[torch.Tensor] = (
            [torch.zeros_like(t) for t in self.k_scale_pool] if asymmetric else []
        )
        self.v_zero_pool: list[torch.Tensor] = (
            [torch.zeros_like(t) for t in self.v_scale_pool] if asymmetric else []
        )

        self._flat_v = [t.view(-1, spec.num_kv_heads, spec.head_dim) for t in self.v_pool]
        self._flat_v_scale = [t.view(-1, spec.num_kv_heads) for t in self.v_scale_pool]
        self._flat_v_zero = (
            [t.view(-1, spec.num_kv_heads) for t in self.v_zero_pool] if asymmetric else []
        )

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
        # Logical position each active row sat at *before* the current
        # advance() — the one thing the K write plan needs that the slot
        # tensor cannot supply without a device readback.
        self._write_starts: list[int] = []
        self._write_plan: Optional[tuple[tuple[int, int], list[_KWriteGroup]]] = None
        self._active: list[int] = list(range(spec.max_batch_size))
        self.gather_calls = 0

        # ---- Phase 14c: the INT8 cache at graph speed ------------------
        #
        # A completed block used to be quantized inside write(), on a
        # decision made in Python, per layer — exactly what a CUDA graph
        # cannot contain. The block that just filled is still intact in
        # the residual, though, so its quantization can wait until the
        # next advance(), which already runs outside the graph. Meanwhile
        # the kernel reads that page from the residual.
        #
        # Invariant, in both modes: every page of a sequence except its
        # last is in the INT8 pool; the last is in the residual, exact.
        # Immediate mode satisfies it already (it quantizes into the pool
        # and leaves the residual intact); deferred mode satisfies it by
        # quantizing any block the previous step completed before this
        # step's write. The kernel derives "last page" from the sequence
        # length alone, on the device.
        self.deferred_finalize = False
        # Blocks quantized so far, per layer and slot. Per *layer* because
        # writes happen layer by layer within a step: when layer 0 reads,
        # layers 1+ have not yet written this step's token, so a flush that
        # quantized every layer at once froze their blocks one token short.
        # The gather path caught it; the kernel path never flushes mid-step.
        self._fin = [[0] * spec.max_batch_size for _ in range(spec.num_layers)]
        bs = block_size
        self._capacity_pages = (spec.max_seq_len + bs - 1) // bs
        mb = spec.max_batch_size
        # Persistent decode buffers, allocated once and written into, for
        # the same reason as PagedKVCache's (Phase 13): a graph records
        # addresses, so the tensors it reads must never be replaced.
        self._block_tables_buf = torch.zeros((mb, max(1, self._capacity_pages)),
                                             dtype=torch.int32, device=self.device)
        self._seq_lens_buf = torch.zeros(mb, dtype=torch.int32, device=self.device)
        self._write_slots_buf = torch.zeros((mb, 1), dtype=torch.long, device=self.device)
        self._k_res_idx_buf = torch.zeros(mb, dtype=torch.long, device=self.device)
        self._res_rows_buf = torch.zeros(mb, dtype=torch.int32, device=self.device)
        self._table_state: list = [None] * mb
        self._block_tables: Optional[torch.Tensor] = None
        self._seq_lens_tensor: Optional[torch.Tensor] = None
        self._res_rows: Optional[torch.Tensor] = None
        self._graph_write = False
        self._k_res_flat = [r.view(mb * bs, spec.num_kv_heads, spec.head_dim)
                            for r in self._k_residual]

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
        self._write_starts = []
        self._write_plan = None
        self._active = list(range(self.spec.max_batch_size))
        self.gather_calls = 0
        # Phase 14c: contents cleared, storage kept, so a graph captured
        # before a reset is still valid after it.
        self._fin = [[0] * self.spec.max_batch_size for _ in range(self.spec.num_layers)]
        self._table_state = [None] * self.spec.max_batch_size
        self._block_tables = self._seq_lens_tensor = self._res_rows = None
        self._graph_write = False
        self._block_tables_buf.zero_()
        self._seq_lens_buf.zero_()
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
        self._write_plan = None
        self._set_finalized(index, 0)
        self._table_state = [None] * self.spec.max_batch_size

    # ------------------------------------------------------------------
    # Allocation — identical to PagedKVCache, plus the K write plan.
    #
    # The slot tensors are deliberately built exactly the way
    # PagedKVCache builds them, including the O(length) Python
    # `table.slots()` walk. That walk is slow for both caches, but
    # making it faster *here only* would show up as an INT8 latency win
    # that has nothing to do with INT8, and would quietly corrupt the
    # fp16-vs-int8 comparison this module exists to support.
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
        if self.deferred_finalize:
            # Blocks the previous step completed are still whole in the
            # residual; quantize them before this step overwrites it.
            self._finalize_pending(active)
        starts = [self.tables[i].length for i in active]
        for i in active:
            self.tables[i].append(n)

        dev = self.device
        self._graph_write = self.deferred_finalize and n == 1
        if n > 1 or not self.deferred_finalize:
            # The eager write path finalises every block it completes,
            # immediately.
            for i, start in zip(active, starts):
                self._set_finalized(i, (start + n) // self.block_size)
        if n == 1:
            self._fill_decode_buffers(active, starts)
        if self._graph_write:
            # Graph path: writes go through the persistent buffers, and the
            # gather path's per-token slot index is built only if read()
            # is ever called.
            self._write_starts = starts
            self._write_plan = None
            self._read_slots = None
            return
        self._write_slots = torch.tensor(
            [
                [self.tables[i].slot(p) for p in range(start, start + n)]
                for i, start in zip(active, starts)
            ],
            dtype=torch.long,
            device=dev,
        )
        self._write_starts = starts
        self._write_plan = None
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
    # Write-plan construction — host-side, once per advance()
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase 14c: deferred finalisation and the kernel's inputs
    # ------------------------------------------------------------------

    @property
    def _finalized(self) -> list[int]:
        """Blocks quantized in *every* layer, per slot."""
        return [min(layer[s] for layer in self._fin) for s in range(self.spec.max_batch_size)]

    def _set_finalized(self, slot: int, count: int) -> None:
        for layer in self._fin:
            layer[slot] = count

    def enable_deferred_finalize(self) -> None:
        """Switch to deferring block quantization to the next advance().

        Safe at any point: immediate mode has already quantized every
        complete block, so the finalized counts start from the lengths.
        """
        for i, t in enumerate(self.tables):
            self._set_finalized(i, t.length // self.block_size)
        self.deferred_finalize = True

    def _finalize_pending(self, seqs: Sequence[int], layers: Optional[Sequence[int]] = None) -> None:
        """Quantize, for every layer, any block a sequence completed but
        has not yet had quantized.

        At most one per sequence: the residual holds a single block, so a
        second pending block would already have been overwritten. That
        cannot happen if advance() runs before every write, and raising is
        the honest response if it ever does.
        """
        bs = self.block_size
        for layer in (range(self.spec.num_layers) if layers is None else layers):
            fin = self._fin[layer]
            slots, ids = [], []
            for s in seqs:
                done = self.tables[s].length // bs
                pending = done - fin[s]
                if pending <= 0:
                    continue
                if pending > 1:
                    raise RuntimeError(
                        f"sequence in slot {s} has {pending} unquantized blocks; the residual "
                        "holds one, so the earlier ones are already lost"
                    )
                slots.append(s)
                ids.append(self.tables[s].blocks[done - 1])
                fin[s] = done
            if slots:
                self._finalize_blocks(layer, self._k_residual[layer][self._ids(slots)],
                                      self._ids(ids))

    def _fill_decode_buffers(self, active: list[int], starts: list[int]) -> None:
        """Write this decode step's indices into the persistent buffers."""
        b = len(active)
        bs = self.block_size
        slots_now = [self.tables[i].slot(st) for i, st in zip(active, starts)]
        self._write_slots_buf[:b, 0].copy_(torch.tensor(slots_now, dtype=torch.long))
        self._k_res_idx_buf[:b].copy_(torch.tensor(
            [i * bs + st % bs for i, st in zip(active, starts)], dtype=torch.long))
        self._res_rows_buf[:b].copy_(torch.tensor(active, dtype=torch.int32))
        for row, i in enumerate(active):
            blocks = self.tables[i].blocks
            state = (i, len(blocks))
            if self._table_state[row] != state:
                if len(blocks) > self._capacity_pages:
                    raise RuntimeError(
                        f"sequence in slot {i} needs {len(blocks)} pages, past the "
                        f"cache's capacity of {self._capacity_pages}"
                    )
                self._block_tables_buf[row, : len(blocks)].copy_(
                    torch.tensor(blocks, dtype=torch.int32))
                self._table_state[row] = state
        self._seq_lens_buf[:b].copy_(
            torch.tensor([self.tables[i].length for i in active], dtype=torch.int32))
        self._write_slots = self._write_slots_buf[:b]
        self._block_tables = self._block_tables_buf[:b]
        self._seq_lens_tensor = self._seq_lens_buf[:b]
        self._res_rows = self._res_rows_buf[:b]

    def block_tables_tensor(self, batch_size: Optional[int] = None) -> torch.Tensor:
        if self._block_tables is None:
            raise RuntimeError("block tables exist only after a decode advance(1)")
        return self._block_tables if batch_size is None else self._block_tables[:batch_size]

    def seq_lens_tensor(self, batch_size: Optional[int] = None) -> torch.Tensor:
        if self._seq_lens_tensor is None:
            raise RuntimeError("sequence lengths exist only after a decode advance(1)")
        return self._seq_lens_tensor if batch_size is None else self._seq_lens_tensor[:batch_size]

    def residual(self, layer_idx: int) -> torch.Tensor:
        """[max_batch, block_size, kv_heads, head_dim]: each slot's last page."""
        return self._k_residual[layer_idx]

    def residual_rows_tensor(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Residual row (= slot) of each active sequence, for the kernel."""
        if self._res_rows is None:
            raise RuntimeError("residual rows exist only after a decode advance(1)")
        return self._res_rows if batch_size is None else self._res_rows[:batch_size]

    def _build_read_slots(self) -> None:
        dev = self.device
        active = self._active
        self._read_slots = torch.zeros((len(active), self.max_len), dtype=torch.long, device=dev)
        for row, i in enumerate(active):
            seq_slots = self.tables[i].slots()
            if seq_slots:
                self._read_slots[row, : len(seq_slots)] = torch.tensor(
                    seq_slots, dtype=torch.long, device=dev)

    def _rows(self, ids: list[int]) -> Rows:
        """A basic slice when `ids` are consecutive, else a LongTensor."""
        if ids and ids == list(range(ids[0], ids[0] + len(ids))):
            return slice(ids[0], ids[0] + len(ids))
        return self._ids(ids)

    def _ids(self, ids: list[int]) -> torch.Tensor:
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _build_write_plan(self, b: int, n: int) -> list[_KWriteGroup]:
        bs = self.block_size
        groups: dict[int, list[int]] = {}
        for row in range(b):
            groups.setdefault(self._write_starts[row], []).append(row)

        plan: list[_KWriteGroup] = []
        for start, rows in groups.items():
            off0 = start % bs
            # Tokens that extend (and maybe finish) the block in flight.
            head = min(bs - off0, n) if off0 else 0
            # Whole blocks wholly contained in this write. `start + head`
            # is block-aligned whenever head > 0 reached the boundary, and
            # when head == 0 `start` already was.
            full = (n - head) // bs
            tail = n - head - full * bs
            seqs = [self._active[r] for r in rows]

            head_ids = None
            if head and off0 + head == bs:
                head_ids = self._ids([self.tables[s].blocks[start // bs] for s in seqs])
            mid_ids = None
            if full:
                base = (start + head) // bs
                mid_ids = self._ids(
                    [self.tables[s].blocks[base + j] for s in seqs for j in range(full)]
                )

            plan.append(
                _KWriteGroup(
                    rows=self._rows(rows), seqs=self._rows(seqs), off0=off0,
                    head=head, full=full, tail=tail,
                    head_ids=head_ids, mid_ids=mid_ids,
                )
            )
        return plan

    def _get_write_plan(self, b: int, n: int) -> list[_KWriteGroup]:
        cached = self._write_plan
        if cached is not None and cached[0] == (b, n):
            return cached[1]
        plan = self._build_write_plan(b, n)
        self._write_plan = ((b, n), plan)
        return plan

    # ------------------------------------------------------------------
    # Read/write path — this is the part that differs from PagedKVCache.
    # ------------------------------------------------------------------

    def _finalize_blocks(
        self, layer_idx: int, data: torch.Tensor, ids: torch.Tensor
    ) -> None:
        """Quantize `data` [N, block_size, heads, dim] into the N physical
        blocks named by `ids`, fitting one step per (block, head, channel).

        The fit runs in FP32 whatever the cache's dtype. In FP16 the
        `_EPS` floor rounds to zero — FP16's smallest subnormal is ~6e-8
        — so an all-zero channel divided by its own absmax produced NaN,
        which only survived because NaN-to-int8 happens to land on a
        value the zero step then multiplies away. That is undefined
        behaviour standing in for a guard.
        """
        f = data.to(torch.float32)
        if self.asymmetric:
            lo = f.amin(dim=1)                                    # [N, heads, dim]
            hi = f.amax(dim=1)
            step = ((hi - lo) / self.k_levels).clamp_min(_EPS)
            q_u = torch.clamp(
                torch.round((f - lo.unsqueeze(1)) / step.unsqueeze(1)), 0, self.k_levels
            )
            q = (q_u - self.k_offset).to(torch.int8)
            self.k_zero_pool[layer_idx].index_copy_(0, ids, lo + self.k_offset * step)
        else:
            step = f.abs().amax(dim=1).clamp_min(_EPS) / self.k_qmax  # [N, heads, dim]
            q = torch.clamp(
                torch.round(f / step.unsqueeze(1)), -self.k_qmax, self.k_qmax
            ).to(torch.int8)
        self.k_pool[layer_idx].index_copy_(0, ids, q)
        self.k_scale_pool[layer_idx].index_copy_(0, ids, step)

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int = 0) -> None:
        """Quantize and scatter [B, kv_heads, n, head_dim] into the pool.

        V is quantized and scattered unconditionally — its scale never
        depends on anything but the token being written. K follows the
        head/middle/tail plan built by `advance()`: only the pieces that
        straddle a block boundary touch the residual, and whole blocks
        contained in the write are quantized straight into the pool.

        No device tensor is read back to the host here. Every index this
        needs comes from the block tables, which are plain Python.
        """
        b, h, n, d = k.shape
        slots = self._write_slots
        if slots is None or slots.shape[0] < b or slots.shape[1] != n:
            raise RuntimeError("call advance(n, batch_size) before write()")

        k_tok = k.permute(0, 2, 1, 3)  # [B, n, h, d]
        v_tok = v.permute(0, 2, 1, 3)
        graph_path = self._graph_write and n == 1

        # --- V: per-token scale, no look-ahead, quantize+scatter now. ---
        # The fit is FP32 for the same reason K's is (see
        # _finalize_blocks): an FP16 `_EPS` floor is no floor at all.
        v_f = v_tok.to(torch.float32)
        flat = slots[:b].reshape(-1)
        if self.asymmetric:
            v_lo = v_f.amin(dim=-1, keepdim=True)
            v_hi = v_f.amax(dim=-1, keepdim=True)
            v_step = ((v_hi - v_lo) / self.v_levels).clamp_min(_EPS)
            v_u = torch.clamp(torch.round((v_f - v_lo) / v_step), 0, self.v_levels)
            v_q = (v_u - self.v_offset).to(torch.int8)
            self._flat_v_zero[layer_idx].index_copy_(
                0, flat, (v_lo + self.v_offset * v_step).squeeze(-1).reshape(-1, h)
            )
        else:
            v_step = v_f.abs().amax(dim=-1, keepdim=True).clamp_min(_EPS) / self.v_qmax
            v_q = torch.clamp(
                torch.round(v_f / v_step), -self.v_qmax, self.v_qmax
            ).to(torch.int8)
        self._flat_v[layer_idx].index_copy_(0, flat, v_q.reshape(-1, h, d))
        self._flat_v_scale[layer_idx].index_copy_(0, flat, v_step.squeeze(-1).reshape(-1, h))

        if graph_path:
            # --- K, graph path: into the residual at a fixed index, no plan,
            # no host decision. A block this completes is quantized by the
            # next advance(); until then the kernel reads it from here. ---
            self._k_res_flat[layer_idx].index_copy_(
                0, self._k_res_idx_buf[:b], k_tok.reshape(b, h, d))
            return

        # --- K: replay the plan. ---
        bs = self.block_size
        residual = self._k_residual[layer_idx]
        for g in self._get_write_plan(b, n):
            if g.head:
                residual[g.seqs, g.off0 : g.off0 + g.head] = k_tok[g.rows, : g.head]
                if g.head_ids is not None:
                    # The block just reached block_size. Everything before
                    # g.off0 was buffered by earlier write() calls, which
                    # is precisely what the residual is for.
                    self._finalize_blocks(layer_idx, residual[g.seqs], g.head_ids)
            if g.full:
                mid = k_tok[g.rows, g.head : g.head + g.full * bs].reshape(-1, bs, h, d)
                self._finalize_blocks(layer_idx, mid, g.mid_ids)
            if g.tail:
                residual[g.seqs, : g.tail] = k_tok[g.rows, n - g.tail :]

    def _splice_residual(
        self, layer_idx: int, k: torch.Tensor, active: list[int], view_len: int
    ) -> None:
        """Overwrite each row's still-filling tail with its exact FP16
        values.

        The tail is located from the sequence's *true* length, not from
        the length of the view being read. When `read(length=L)` returns
        a prefix shorter than the sequence, the residual holds the tail
        of the whole sequence, which is not the tail of that prefix —
        splicing at `L % block_size` wrote the newest tokens over
        positions belonging to an already-finalized block.
        """
        if not active:
            return
        bs = self.block_size
        residual = self._k_residual[layer_idx]
        lens = [self.tables[i].length for i in active]

        def span(true_len: int) -> Optional[tuple[int, int]]:
            tail = true_len % bs
            if not tail:
                return None
            start = true_len - tail
            visible = min(true_len, view_len) - start
            return (start, visible) if visible > 0 else None

        if len(set(lens)) == 1:
            found = span(lens[0])
            if found is None:
                return
            start, visible = found
            k[:, start : start + visible] = residual[
                self._rows(list(active)), :visible
            ].to(k.dtype)
            return

        for row, seq_idx in enumerate(active):
            found = span(lens[row])
            if found is None:
                continue
            start, visible = found
            k[row, start : start + visible] = residual[seq_idx, :visible].to(k.dtype)

    def read(
        self, layer_idx: int, batch_size: int, length: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather + dequantize into [B, kv_heads, L, head_dim], in
        `out_dtype` (fp16 by default) — the same shape and dtype
        `PagedKVCache.read()` returns, so attention needs no changes.

        The gather is per *block*, not per token. Logical positions
        `[j*block_size, (j+1)*block_size)` of a sequence always live in
        one physical block at consecutive offsets — that is what
        `BlockTable.slot` computes — so one block id per 16 positions
        addresses the same bytes as 16 slot ids. More to the point, K's
        step table is indexed by block, so gathering it per token
        materialized a [B, L, heads, dim] FP32 tensor `block_size` times
        larger than the information in it.

        Dequantization is a single broadcast multiply straight into
        `out_dtype`. Going through FP32 cost three full-size FP32
        temporaries per layer per step — at B=1, ctx=16K that was ~63 MB
        of K traffic against the FP16 paged cache's ~17 MB, which is the
        opposite of the point of the exercise.
        """
        if self._write_slots is None:
            raise RuntimeError("call advance() before read()")
        if self.deferred_finalize:
            # The gather path reads completed pages from the pool, so a block
            # this step completed must be quantized first — for *this layer
            # only*: its write has happened, later layers' have not.
            self._finalize_pending(self._active, layers=[layer_idx])
        if self._read_slots is None:
            self._build_read_slots()
        slots = self._read_slots
        active = self._active[:batch_size]
        idx = slots[:batch_size] if length is None else slots[:batch_size, :length]
        b, view_len = idx.shape
        bs = self.block_size
        h, d = self.spec.num_kv_heads, self.spec.head_dim
        self.gather_calls += 1

        # One physical block id per block_size logical positions. Rows
        # padded out to max_len land on block 0, exactly as PagedKVCache's
        # pad-with-slot-0 does; `padding_mask()` is what makes either safe.
        blk = idx[:, ::bs] // bs  # [B, nblk]
        nblk = blk.shape[1]

        # int8 * out_dtype promotes to out_dtype in one pass: the pool is
        # read once and the FP16 buffer written once, no intermediate.
        k_q = self.k_pool[layer_idx][blk]  # [B, nblk, bs, h, d]
        k_step = self.k_scale_pool[layer_idx][blk].to(self.out_dtype)  # [B, nblk, h, d]
        v_q = self.v_pool[layer_idx][blk]  # [B, nblk, bs, h, d]
        v_step = self.v_scale_pool[layer_idx][blk].to(self.out_dtype)  # [B, nblk, bs, h]

        if self.asymmetric:
            # x = q_s * step + zero', with the code shift already folded
            # into zero' at write time. Same multiply-add as symmetric,
            # and nothing is added to an int8 tensor.
            k_zero = self.k_zero_pool[layer_idx][blk].to(self.out_dtype)
            k = k_q * k_step.unsqueeze(2) + k_zero.unsqueeze(2)
            v_zero = self.v_zero_pool[layer_idx][blk].to(self.out_dtype)
            v = v_q * v_step.unsqueeze(-1) + v_zero.unsqueeze(-1)
        else:
            k = k_q * k_step.unsqueeze(2)
            v = v_q * v_step.unsqueeze(-1)
        k = k.reshape(b, nblk * bs, h, d)[:, :view_len]
        v = v.reshape(b, nblk * bs, h, d)[:, :view_len]

        self._splice_residual(layer_idx, k, active, view_len)

        return k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)  # [B, h, L, D]

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
        params = 2 if self.asymmetric else 1          # scale, plus zero-point
        return elems + (elems * 4 * params + self.block_size - 1) // self.block_size

    @property
    def v_bytes_per_token(self) -> int:
        """1 byte/element plus its own FP32 per-token, per-head scale —
        not amortized, since V's scale is not shared across tokens."""
        params = 2 if self.asymmetric else 1
        return (self.spec.num_kv_heads * self.spec.head_dim
                + self.spec.num_kv_heads * 4 * params)

    @property
    def bytes_per_token(self) -> int:
        """Summed over layers — the INT8-cache analogue of
        `KVCacheSpec.bytes_per_token`, used in place of it everywhere
        below so `stats()`/`used_bytes()`/etc. report the true INT8
        footprint rather than the FP16 spec's.

        This is one byte per element whatever `k_bits`/`v_bits` say.
        Sub-8-bit settings change the quantization *resolution* — they
        exist so the real cache can reproduce 7.2's bit-width sweep —
        but nothing is bit-packed, so a 4-bit run occupies exactly as
        much memory as an 8-bit one. `packed_bytes_per_token` is what
        packing would buy; `stats()` reports both, so a storage claim
        cannot be read off the wrong one.
        """
        return self.spec.num_layers * (self.k_bytes_per_token + self.v_bytes_per_token)

    @property
    def packed_bytes_per_token(self) -> int:
        """Hypothetical footprint if the pools were bit-packed at
        `k_bits`/`v_bits`. Not what this cache occupies — see
        `bytes_per_token`. Reported so a `--k-bits 4` sweep shows the
        storage it is *arguing for* next to the storage it has."""
        elems = self.spec.num_kv_heads * self.spec.head_dim
        params = 2 if self.asymmetric else 1
        k = ((elems * self.k_bits + 7) // 8
             + (elems * 4 * params + self.block_size - 1) // self.block_size)
        v = (elems * self.v_bits + 7) // 8 + self.spec.num_kv_heads * 4 * params
        return self.spec.num_layers * (k + v)

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
        """Extra traffic the gather adds, per decode step, summed over
        layers — directly comparable to `PagedKVCache`'s version of this
        method, which is `2 * (its FP16 bytes)` for a read and a write.

        INT8 needs two more terms, because the buffer it gathers is not
        the buffer attention consumes:

            read the INT8 pool            1x INT8 bytes
            write the INT8 gather buffer  1x INT8 bytes
            read it back to dequantize    1x INT8 bytes
            write the FP16 buffer         1x FP16 bytes

        Reporting `2 * INT8 bytes` — as if the gather produced something
        SDPA could read — understated the real figure by about 2.4x and
        made INT8 look like a traffic reduction when it is a ~17%
        increase. The reduction needs a fused gather+dequantize kernel
        (Phase 11); see this module's docstring.
        """
        rows = list(self._active)[:batch_size]
        tokens = sum(self.tables[i].length for i in rows)
        return 3 * self.bytes_per_token * tokens + self.spec.bytes_per_token * tokens

    def stats(self, batch_size: Optional[int] = None) -> dict:
        b = len(self._active) if batch_size is None else batch_size
        return {
            "asymmetric": self.asymmetric,
            "kv_bytes_per_token": self.bytes_per_token,
            "kv_bytes_per_token_fp16_equiv": self.spec.bytes_per_token,
            "kv_bytes_per_token_if_packed": self.packed_bytes_per_token,
            "kv_allocated_mb": self.allocated_bytes / 1024 / 1024,
            "kv_used_mb": self.used_bytes(b) / 1024 / 1024,
            "kv_reserved_mb": self.reserved_bytes(b) / 1024 / 1024,
            "kv_utilization": self.utilization(b),
            "internal_fragmentation": self.fragmentation(b),
            "block_size": self.block_size,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "storage_bits_per_element": 8,
            "gather_calls": self.gather_calls,
            **self.allocator.stats(),
        }