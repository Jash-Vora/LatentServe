"""
Phase 11 — paged decode attention, read in place.

Four independent measurements point at one change:

  * Phase 3: the gather moves 2x the resident KV per step — 17.3 ms at
    8K/batch 4, and the whole of paging's latency cost (79-135 GB/s
    implied, matching this card's measured bandwidth exactly).
  * Phase 4: it makes batching non-free. TPOT rose 33 -> 65 ms from
    batch 1 to 8, where Phase 2's contiguous cache had stayed flat.
  * Phase 7.3: storing INT8 made decode 14-36% *slower*, because the
    dequantize pass is another full pass over the data.
  * Phase 7.5: it dilutes the capacity win, 1.91x analytic to 1.74x
    measured, because the fp16 gather buffer is dtype-independent.

This kernel removes all four by walking the block table inside the
kernel and consuming the cache in place, INT8 included.

## Decode only, and Phase 6 says why

vLLM runs TRITON_ATTN for both prefill and decode on sm75, and the two
went opposite ways: prefill 744 tok/s against LatentServe's SDPA at
5,238 (7x loss), decode 42.9 ms/step against 57.5 (1.34x win). Prefill
is compute-bound and O(S^2) — a tuned CUDA GEMM wins, and Triton's
tensor-core support on Turing's m16n8k8 is its weakest area. Decode is
memory-bound and GEMV-shaped: peak FLOPS is irrelevant and data layout
is everything, which is what Triton is good at.

So prefill keeps gather + SDPA, where the gather is a linear cost inside
a quadratic operation and Phase 6 measured TTFT moving only a few
percent. Only `q_len == 1` comes here — no causal mask, no square tile,
a much easier kernel than a general one.

## Split-K is not an optimisation, it is the point

Grid = (batch x kv_heads) launches **2 thread blocks at batch 1** for
Qwen2.5-1.5B on a 40-SM T4. Phase 2 measured exactly what that costs:
achieved bandwidth tracks `batch x kv_heads` almost monotonically —
2 blocks reach 95 GB/s, 12 reach 126, 48 reach 198 (62% of peak). A
kernel that ignores this would remove the gather and then leave the GPU
95% idle.

Splitting the KV length into `num_splits` chunks gives `2 x num_splits`
blocks, each producing a partial softmax `(m, l, acc)` that combines
exactly. That is flash-decoding, and here it is aimed at a measured
occupancy problem rather than adopted by convention.

## Why a PyTorch reference lives beside the Triton kernel

`paged_decode_reference` implements the *same algorithm* — same block
walk, same online softmax, same split-and-combine — in plain PyTorch. It
is not a fallback for convenience: it is the oracle that separates "the
algorithm is wrong" from "the Triton lowering is wrong", which are
otherwise indistinguishable from a bad output tensor. It also runs on
CPU, so the logic is testable without a GPU.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:  # pragma: no cover - availability is environmental
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


# ----------------------------------------------------------------------
# Reference: the algorithm, in PyTorch
# ----------------------------------------------------------------------


def _dequant_k(
    k_q: torch.Tensor, scale: Optional[torch.Tensor], zero: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """k_q [PAGE, D]; scale/zero [D] for this (block, head)."""
    if scale is None:
        return k_q.to(dtype)
    out = k_q.to(torch.float32) * scale[None, :]
    if zero is not None:
        out = out + zero[None, :]
    return out.to(dtype)


def _dequant_v(
    v_q: torch.Tensor, scale: Optional[torch.Tensor], zero: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """v_q [PAGE, D]; scale/zero [PAGE] for this (block, head) — V's scale
    is per token, so it broadcasts over the channel axis, not the token
    axis. Getting these two the wrong way round is the single easiest
    mistake in this file and produces plausible-looking output."""
    if scale is None:
        return v_q.to(dtype)
    out = v_q.to(torch.float32) * scale[:, None]
    if zero is not None:
        out = out + zero[:, None]
    return out.to(dtype)


@torch.no_grad()
def paged_decode_reference(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_zero: Optional[torch.Tensor] = None,
    v_zero: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    softmax_scale: Optional[float] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Paged decode attention, one query position per sequence.

    q            [B, H_kv, N_REP, D]  — query heads folded onto the KV head
                                        they share, as `GQAAttention` already
                                        does on its decode path
    k_pool       [num_blocks, PAGE, H_kv, D]
    v_pool       [num_blocks, PAGE, H_kv, D]
    block_tables [B, max_pages]       — physical block id per logical page
    seq_lens     [B]
    k_scale      [num_blocks, H_kv, D]        (INT8 only; per block/head/channel)
    v_scale      [num_blocks, PAGE, H_kv]     (INT8 only; per block/token/head)

    Returns [B, H_kv, N_REP, D].
    """
    b, h_kv, n_rep, d = q.shape
    page = k_pool.shape[1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    out = torch.zeros(b, h_kv, n_rep, d, dtype=q.dtype, device=q.device)

    for i in range(b):
        length = int(seq_lens[i].item())
        num_pages = (length + page - 1) // page
        pages_per_split = max(1, (num_pages + num_splits - 1) // num_splits)

        for head in range(h_kv):
            qi = q[i, head].to(compute_dtype)          # [N_REP, D]
            partial_acc, partial_m, partial_l = [], [], []

            for s in range(num_splits):
                lo = s * pages_per_split
                hi = min(lo + pages_per_split, num_pages)
                if lo >= hi:
                    continue
                # Online softmax state for this split.
                m_i = torch.full((n_rep,), float("-inf"), dtype=compute_dtype, device=q.device)
                l_i = torch.zeros(n_rep, dtype=compute_dtype, device=q.device)
                acc = torch.zeros(n_rep, d, dtype=compute_dtype, device=q.device)

                for p in range(lo, hi):
                    blk = int(block_tables[i, p].item())
                    k = _dequant_k(
                        k_pool[blk, :, head], 
                        None if k_scale is None else k_scale[blk, head],
                        None if k_zero is None else k_zero[blk, head],
                        compute_dtype,
                    )                                   # [PAGE, D]
                    v = _dequant_v(
                        v_pool[blk, :, head],
                        None if v_scale is None else v_scale[blk, :, head],
                        None if v_zero is None else v_zero[blk, :, head],
                        compute_dtype,
                    )

                    tokens = p * page + torch.arange(page, device=q.device)
                    valid = tokens < length
                    qk = (qi @ k.T) * scale             # [N_REP, PAGE]
                    qk = qk.masked_fill(~valid[None, :], float("-inf"))

                    m_new = torch.maximum(m_i, qk.max(dim=-1).values)
                    # A page entirely past the end leaves m_new at -inf;
                    # exp(-inf - -inf) is NaN, so the correction factor
                    # has to be forced to zero there rather than computed.
                    alpha = torch.where(
                        torch.isinf(m_new), torch.zeros_like(m_new), torch.exp(m_i - m_new)
                    )
                    pw = torch.exp(qk - m_new[:, None])
                    pw = torch.where(valid[None, :], pw, torch.zeros_like(pw))
                    l_i = l_i * alpha + pw.sum(dim=-1)
                    acc = acc * alpha[:, None] + pw @ v
                    m_i = m_new

                partial_acc.append(acc)
                partial_m.append(m_i)
                partial_l.append(l_i)

            if not partial_acc:
                continue
            # Combine the splits: rescale each by its own max against the
            # global one. Exactly the same correction the inner loop
            # applies between pages, applied once more between splits.
            m_all = torch.stack(partial_m)                      # [S, N_REP]
            m_global = m_all.max(dim=0).values
            w = torch.where(
                torch.isinf(m_global)[None, :],
                torch.zeros_like(m_all),
                torch.exp(m_all - m_global[None, :]),
            )
            l_total = (torch.stack(partial_l) * w).sum(dim=0)
            acc_total = (torch.stack(partial_acc) * w[:, :, None]).sum(dim=0)
            out[i, head] = (acc_total / l_total.clamp_min(1e-20)[:, None]).to(q.dtype)

    return out


# ----------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------


if HAS_TRITON:  # pragma: no cover - requires a GPU

    @triton.jit
    def _paged_decode_kernel(
        Q, K_pool, V_pool, K_scale, V_scale, K_zero, V_zero,
        BlockTables, SeqLens, PartialOut, PartialM, PartialL,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kp, stride_kh, stride_kd,
        stride_ksb, stride_ksh, stride_ksd,
        stride_vsb, stride_vsp, stride_vsh,
        stride_btb, stride_btp,
        stride_ob, stride_oh, stride_os, stride_om, stride_od,
        stride_mb, stride_mh, stride_ms, stride_mm,
        softmax_scale,
        num_splits,
        N_REP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        PAGE: tl.constexpr,
        IS_INT8: tl.constexpr,
        ASYM: tl.constexpr,
    ):
        """One program per (sequence, kv_head, split).

        BLOCK_M is 16 rather than N_REP (6 for Qwen) because `tl.dot`
        requires all three dimensions to be at least 16 on this
        architecture; the extra rows are masked and discarded.
        """
        # grid = (batch, num_splits, kv_heads)
        b = tl.program_id(0)
        split = tl.program_id(1)
        h = tl.program_id(2)

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_p = tl.arange(0, PAGE)
        row_valid = offs_m < N_REP

        q = tl.load(
            Q + b * stride_qb + h * stride_qh
            + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=row_valid[:, None], other=0.0,
        )

        seq_len = tl.load(SeqLens + b)
        num_pages = tl.cdiv(seq_len, PAGE)
        pages_per_split = tl.cdiv(num_pages, num_splits)
        lo = split * pages_per_split
        hi = tl.minimum(lo + pages_per_split, num_pages)

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for p in range(lo, hi):
            blk = tl.load(BlockTables + b * stride_btb + p * stride_btp)

            k = tl.load(
                K_pool + blk * stride_kb + offs_p[:, None] * stride_kp
                + h * stride_kh + offs_d[None, :] * stride_kd
            )
            v = tl.load(
                V_pool + blk * stride_kb + offs_p[:, None] * stride_kp
                + h * stride_kh + offs_d[None, :] * stride_kd
            )

            if IS_INT8:
                # K's scale is per (block, head, channel): broadcast over
                # tokens. V's is per (block, token, head): broadcast over
                # channels. The two axes are not interchangeable.
                ks = tl.load(K_scale + blk * stride_ksb + h * stride_ksh
                             + offs_d * stride_ksd)
                vs = tl.load(V_scale + blk * stride_vsb + offs_p * stride_vsp
                             + h * stride_vsh)
                k = k.to(tl.float32) * ks[None, :]
                v = v.to(tl.float32) * vs[:, None]
                if ASYM:
                    kz = tl.load(K_zero + blk * stride_ksb + h * stride_ksh
                                 + offs_d * stride_ksd)
                    vz = tl.load(V_zero + blk * stride_vsb + offs_p * stride_vsp
                                 + h * stride_vsh)
                    k = k + kz[None, :]
                    v = v + vz[:, None]
                k = k.to(q.dtype)
                v = v.to(q.dtype)

            tokens = p * PAGE + offs_p
            valid = tokens < seq_len

            qk = tl.dot(q, tl.trans(k)) * softmax_scale
            qk = tl.where(valid[None, :] & row_valid[:, None], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
            pw = tl.exp(qk - m_new[:, None])
            pw = tl.where(valid[None, :], pw, 0.0)
            l_i = l_i * alpha + tl.sum(pw, axis=1)
            acc = acc * alpha[:, None] + tl.dot(pw.to(v.dtype), v)
            m_i = m_new

        tl.store(
            PartialOut + b * stride_ob + h * stride_oh + split * stride_os
            + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            acc, mask=row_valid[:, None],
        )
        base = PartialM + b * stride_mb + h * stride_mh + split * stride_ms + offs_m * stride_mm
        tl.store(base, m_i, mask=row_valid)
        tl.store(
            PartialL + b * stride_mb + h * stride_mh + split * stride_ms + offs_m * stride_mm,
            l_i, mask=row_valid,
        )


def combine_splits(
    partial_acc: torch.Tensor, partial_m: torch.Tensor, partial_l: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Merge per-split partial softmaxes.

    partial_acc [B, H, S, N_REP, D]; partial_m/l [B, H, S, N_REP].

    Done in PyTorch rather than a second kernel: S is small (8-32) and
    these tensors are tiny next to the KV the main kernel just read, so a
    launch here costs less than the complexity of fusing it.
    """
    m_global = partial_m.max(dim=2, keepdim=True).values
    w = torch.where(torch.isinf(m_global), torch.zeros_like(partial_m),
                    torch.exp(partial_m - m_global))
    l_total = (partial_l * w).sum(dim=2)
    acc_total = (partial_acc * w[..., None]).sum(dim=2)
    return (acc_total / l_total.clamp_min(1e-20)[..., None]).to(out_dtype)


def paged_decode_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_zero: Optional[torch.Tensor] = None,
    v_zero: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    force_reference: bool = False,
) -> torch.Tensor:
    """Dispatch to Triton when possible, else the reference.

    `num_splits` defaults to filling the GPU: Phase 2 measured achieved
    bandwidth tracking `batch x kv_heads`, so the target is enough
    programs to cover the SMs rather than a fixed constant.
    """
    b, h_kv, n_rep, d = q.shape
    if force_reference or not HAS_TRITON or not q.is_cuda:
        return paged_decode_reference(
            q, k_pool, v_pool, block_tables, seq_lens, k_scale, v_scale,
            k_zero, v_zero, num_splits or 1, softmax_scale,
        )

    page = k_pool.shape[1]
    max_len = int(seq_lens.max().item())
    num_pages = (max_len + page - 1) // page
    if num_splits is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_splits = max(1, min(num_pages, -(-sms // max(1, b * h_kv))))

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    is_int8 = k_pool.dtype == torch.int8
    asym = k_zero is not None

    partial_acc = torch.empty(b, h_kv, num_splits, 16, d, dtype=torch.float32, device=q.device)
    partial_m = torch.full((b, h_kv, num_splits, 16), float("-inf"),
                           dtype=torch.float32, device=q.device)
    partial_l = torch.zeros(b, h_kv, num_splits, 16, dtype=torch.float32, device=q.device)

    dummy = torch.empty(1, device=q.device)
    _paged_decode_kernel[(b, num_splits, h_kv)](
        q, k_pool, v_pool,
        k_scale if k_scale is not None else dummy,
        v_scale if v_scale is not None else dummy,
        k_zero if k_zero is not None else dummy,
        v_zero if v_zero is not None else dummy,
        block_tables, seq_lens, partial_acc, partial_m, partial_l,
        *q.stride(), *k_pool.stride(),
        *(k_scale.stride() if k_scale is not None else (0, 0, 0)),
        *(v_scale.stride() if v_scale is not None else (0, 0, 0)),
        *block_tables.stride(), *partial_acc.stride(), *partial_m.stride(),
        scale, num_splits,
        N_REP=n_rep, BLOCK_M=16, BLOCK_D=d, PAGE=page,
        IS_INT8=is_int8, ASYM=asym,
    )
    return combine_splits(partial_acc, partial_m, partial_l, q.dtype)[:, :, :n_rep]