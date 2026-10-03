"""
Phase 12 — taking the decode kernel apart.

Nsight Compute is refused in this container (ERR_NVGPUCTRPERM), so the
questions a profiler answers are answered by construction instead: copies
of the production fp16 tiled kernel, each with one thing removed. Timing
them against the full kernel splits its time into parts.

    MODE_FULL         the production fp16 tiled kernel, unchanged
    MODE_NO_LOOKUP    page addresses computed directly instead of loaded from
                      the block table — same addresses, no dependent load
    MODE_LOADS_ONLY   every K/V load kept, no matrix multiply and no softmax
    MODE_COMPUTE_ONLY one tile loaded up front and reused every iteration —
                      all the arithmetic, none of the streaming

How to read them, against MODE_FULL:

  * LOADS_ONLY close to FULL: the memory path is the cost; arithmetic hides
    under it.
  * COMPUTE_ONLY close to FULL: the arithmetic is the cost — on this kernel
    most plausibly through register pressure rather than FLOPs.
  * NO_LOOKUP much faster: the dependent load (fetch a block id, only then
    fetch its page) serialises the loop, and the fix is to get the next id
    early.

`_paged_prefetch` is not an ablation but a candidate. One page per
iteration — the variant with no spills and 50% occupancy — with the next
page's block id and K/V requested *before* the current page is computed.
Triton's own pipeliner (num_stages) cannot do this for us: each page's
address comes out of a load, and it pipelines loads whose addresses depend
on the loop index.

Partial outputs use the same layout as the production kernel, so
`combine_splits` merges them and every variant is checked for the same
answer as the reference before it is timed.
"""

from __future__ import annotations

import math

import torch

try:  # pragma: no cover - availability is environmental
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

MODE_FULL, MODE_NO_LOOKUP, MODE_LOADS_ONLY, MODE_COMPUTE_ONLY = 0, 1, 2, 3
MODES = {"full": MODE_FULL, "no_lookup": MODE_NO_LOOKUP,
         "loads_only": MODE_LOADS_ONLY, "compute_only": MODE_COMPUTE_ONLY}

LAST_COMPILED: dict = {}


if HAS_TRITON:  # pragma: no cover - requires a GPU

    @triton.jit
    def _ablate_tiled(
        Q, K_pool, V_pool, BlockTables, SeqLens, PartialOut, PartialM, PartialL,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kp, stride_kh, stride_kd,
        stride_btb, stride_btp,
        stride_ob, stride_oh, stride_os, stride_om, stride_od,
        stride_mb, stride_mh, stride_ms, stride_mm,
        softmax_scale, num_splits, pages_per_seq,
        N_REP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
        PAGE: tl.constexpr, PPI: tl.constexpr, MODE: tl.constexpr,
    ):
        b = tl.program_id(0)
        split = tl.program_id(1)
        h = tl.program_id(2)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_p = tl.arange(0, PAGE)
        offs_i = tl.arange(0, PPI)
        row_valid = offs_m < N_REP
        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm
                    + offs_d[None, :] * stride_qd, mask=row_valid[:, None], other=0.0)
        seq_len = tl.load(SeqLens + b)
        num_pages = tl.cdiv(seq_len, PAGE)
        pages_per_split = tl.cdiv(num_pages, num_splits)
        lo = split * pages_per_split
        hi = tl.minimum(lo + pages_per_split, num_pages)

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        run = tl.zeros((PPI * PAGE, BLOCK_D), dtype=tl.float16)

        if MODE == 3:
            pages0 = lo + offs_i
            blk0 = tl.load(BlockTables + b * stride_btb + pages0 * stride_btp,
                           mask=pages0 < hi, other=0)
            k_fix = tl.reshape(tl.load(
                K_pool + blk0[:, None, None] * stride_kb + offs_p[None, :, None] * stride_kp
                + h * stride_kh + offs_d[None, None, :] * stride_kd), (PPI * PAGE, BLOCK_D))
            v_fix = tl.reshape(tl.load(
                V_pool + blk0[:, None, None] * stride_kb + offs_p[None, :, None] * stride_kp
                + h * stride_kh + offs_d[None, None, :] * stride_kd), (PPI * PAGE, BLOCK_D))

        for p0 in range(lo, hi, PPI):
            pages = p0 + offs_i
            page_ok = pages < hi
            if MODE == 3:
                k = k_fix
                v = v_fix
            else:
                if MODE == 1:
                    blk = b * pages_per_seq + pages
                else:
                    blk = tl.load(BlockTables + b * stride_btb + pages * stride_btp,
                                  mask=page_ok, other=0)
                k = tl.reshape(tl.load(
                    K_pool + blk[:, None, None] * stride_kb + offs_p[None, :, None] * stride_kp
                    + h * stride_kh + offs_d[None, None, :] * stride_kd), (PPI * PAGE, BLOCK_D))
                v = tl.reshape(tl.load(
                    V_pool + blk[:, None, None] * stride_kb + offs_p[None, :, None] * stride_kp
                    + h * stride_kh + offs_d[None, None, :] * stride_kd), (PPI * PAGE, BLOCK_D))
            if MODE == 2:
                # Consume the loads with an elementwise add, so they cannot be
                # eliminated, and nothing else.
                run = run + k + v
            else:
                tokens = tl.reshape(pages[:, None] * PAGE + offs_p[None, :], (PPI * PAGE,))
                valid = tl.reshape(page_ok[:, None] & (offs_p[None, :] >= 0),
                                   (PPI * PAGE,)) & (tokens < seq_len)
                qk = tl.dot(q, tl.trans(k)) * softmax_scale
                qk = tl.where(valid[None, :] & row_valid[:, None], qk, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
                pw = tl.where(valid[None, :], tl.exp(qk - m_new[:, None]), 0.0)
                l_i = l_i * alpha + tl.sum(pw, axis=1)
                acc = acc * alpha[:, None] + tl.dot(pw.to(v.dtype), v)
                m_i = m_new

        if MODE == 2:
            acc += tl.sum(run.to(tl.float32), axis=0)[None, :]
            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
        tl.store(PartialOut + b * stride_ob + h * stride_oh + split * stride_os
                 + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 acc, mask=row_valid[:, None])
        tl.store(PartialM + b * stride_mb + h * stride_mh + split * stride_ms
                 + offs_m * stride_mm, m_i, mask=row_valid)
        tl.store(PartialL + b * stride_mb + h * stride_mh + split * stride_ms
                 + offs_m * stride_mm, l_i, mask=row_valid)

    @triton.jit
    def _paged_prefetch(
        Q, K_pool, V_pool, BlockTables, SeqLens, PartialOut, PartialM, PartialL,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kp, stride_kh, stride_kd,
        stride_btb, stride_btp,
        stride_ob, stride_oh, stride_os, stride_om, stride_od,
        stride_mb, stride_mh, stride_ms, stride_mm,
        softmax_scale, num_splits,
        N_REP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
        PAGE: tl.constexpr,
    ):
        """One page per iteration, with the next page requested early.

        The block id for page p+1 and its K/V are loaded at the top of
        iteration p, before page p's arithmetic, so their latency overlaps
        the compute instead of following it. Beyond the last page the loads
        are masked to block 0, which is always allocated and finite.
        """
        b = tl.program_id(0)
        split = tl.program_id(1)
        h = tl.program_id(2)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_p = tl.arange(0, PAGE)
        row_valid = offs_m < N_REP
        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm
                    + offs_d[None, :] * stride_qd, mask=row_valid[:, None], other=0.0)
        seq_len = tl.load(SeqLens + b)
        num_pages = tl.cdiv(seq_len, PAGE)
        pages_per_split = tl.cdiv(num_pages, num_splits)
        lo = split * pages_per_split
        hi = tl.minimum(lo + pages_per_split, num_pages)

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        kv_off = offs_p[:, None] * stride_kp + h * stride_kh + offs_d[None, :] * stride_kd
        blk = tl.load(BlockTables + b * stride_btb + lo * stride_btp, mask=lo < hi, other=0)
        k_next = tl.load(K_pool + blk * stride_kb + kv_off)
        v_next = tl.load(V_pool + blk * stride_kb + kv_off)
        for p in range(lo, hi):
            k = k_next
            v = v_next
            nxt = p + 1
            blk_n = tl.load(BlockTables + b * stride_btb + nxt * stride_btp,
                            mask=nxt < hi, other=0)
            k_next = tl.load(K_pool + blk_n * stride_kb + kv_off)
            v_next = tl.load(V_pool + blk_n * stride_kb + kv_off)

            tokens = p * PAGE + offs_p
            valid = tokens < seq_len
            qk = tl.dot(q, tl.trans(k)) * softmax_scale
            qk = tl.where(valid[None, :] & row_valid[:, None], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
            pw = tl.where(valid[None, :], tl.exp(qk - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(pw, axis=1)
            acc = acc * alpha[:, None] + tl.dot(pw.to(v.dtype), v)
            m_i = m_new

        tl.store(PartialOut + b * stride_ob + h * stride_oh + split * stride_os
                 + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 acc, mask=row_valid[:, None])
        tl.store(PartialM + b * stride_mb + h * stride_mh + split * stride_ms
                 + offs_m * stride_mm, m_i, mask=row_valid)
        tl.store(PartialL + b * stride_mb + h * stride_mh + split * stride_ms
                 + offs_m * stride_mm, l_i, mask=row_valid)


def prepare(variant: str, q, k_pool, v_pool, block_tables, seq_lens, num_splits=16,
            pages_per_iter=4, num_warps=4, num_stages=2, pages_per_seq=None):
    """A callable that runs one variant: the kernel, then the production
    merge kernel, into buffers allocated once.

    Allocating partials per call and merging them with PyTorch ops costs
    about a dozen small launches — a third of a 0.17 ms batch-1 kernel —
    and would distort the very ratios the ablations exist to measure. So
    the timed path is exactly two launches, as in production.
    """
    from kernels.gqa.paged_decode import _combine_kernel

    b, h_kv, n_rep, d = q.shape
    page = k_pool.shape[1]
    scale = 1.0 / math.sqrt(d)
    dev = q.device
    acc = torch.zeros(b, h_kv, num_splits, 16, d, dtype=torch.float32, device=dev)
    m = torch.full((b, h_kv, num_splits, 16), float("-inf"), dtype=torch.float32, device=dev)
    l = torch.zeros(b, h_kv, num_splits, 16, dtype=torch.float32, device=dev)
    out = torch.empty(b, h_kv, 16, d, dtype=q.dtype, device=dev)
    common = (q, k_pool, v_pool, block_tables, seq_lens, acc, m, l,
              *q.stride(), *k_pool.stride(), *block_tables.stride(), *acc.stride(), *m.stride(),
              scale, num_splits)
    grid = (b, num_splits, h_kv)

    def run():
        if variant == "prefetch":
            LAST_COMPILED[variant] = _paged_prefetch[grid](
                *common, N_REP=n_rep, BLOCK_M=16, BLOCK_D=d, PAGE=page,
                num_warps=num_warps, num_stages=num_stages)
        else:
            LAST_COMPILED[variant] = _ablate_tiled[grid](
                *common, pages_per_seq or block_tables.shape[1],
                N_REP=n_rep, BLOCK_M=16, BLOCK_D=d, PAGE=page, PPI=pages_per_iter,
                MODE=MODES[variant], num_warps=num_warps, num_stages=num_stages)
        _combine_kernel[(b, h_kv)](acc, m, l, out, *acc.stride(), *m.stride(), *out.stride(),
                                   num_splits, N_REP=n_rep, BLOCK_M=16, BLOCK_D=d)
        return out[:, :, :n_rep]

    return run


def run_variant(variant: str, *args, **kw):
    """One-shot form, for correctness checks."""
    return prepare(variant, *args, **kw)()
