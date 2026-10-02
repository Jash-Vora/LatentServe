"""
Phase 14a, step 2 — elementwise fusion.

## Why

The settled batch-1 / 2K measurement put vLLM 12% ahead: 18.0 ms per
token against 20.4, with both systems' two half-run estimates agreeing to
0.1 ms. The gap is not attention (level or ahead at 8K) and not the
serving loop (~0.2 ms). What remains is the fixed per-token work outside
attention, which vLLM's compiler fuses and LatentServe ran as separate
kernels:

    RMSNorm           8 kernels  (upcast, square, mean, +eps, rsqrt,
                                  multiply, downcast, scale) x 2 per layer
    residual add      1 kernel   x 2 per layer
    RoPE              5 kernels  (multiply, negate, concatenate, multiply,
                                  add) x 2 (q and k)
    SiLU-and-multiply 2 kernels

About 30 per layer, 840 per token across 28 layers. Fused: one kernel per
norm with its residual add folded in, one for RoPE on q and k together,
one for SiLU-and-multiply — four per layer. Under CUDA graphs a kernel
costs ~2.6 us of fixed time (measured in 14a), so ~730 fewer is worth
about 1.9 ms: most of the gap.

## Matching the model's rounding — and where the compiler doesn't

Each kernel is *written* to reproduce Hugging Face's rounding step by
step: the RMSNorm rounds the normalised value to fp16 before applying the
weight, as `Qwen2RMSNorm` does, and the residual add is one fp32 add
rounded once, which equals an fp16 add exactly.

Written is not compiled. On the T4, RoPE differed from Hugging Face in 25%
of elements, by up to one fp16 step at the size of its *inputs*. HF's fp16
RoPE rounds each product and then their sum; where the products nearly
cancel, the small result carries their rounding error. Triton evidently
folded the intermediate roundings into one, which is closer to the true
value and differs from HF's precisely where cancellation occurs. So the
GPU test asks the property that holds either way: never further from the
exact value than the model's own fp16 computation, beyond the one
rounding every fp16 result pays. The RMSNorm also differs by the order of
the sum inside its variance.

## References run on CPU

Every kernel has a PyTorch reference that *is* the Hugging Face
computation. On CPU — and anywhere Triton is unavailable — the references
run, so the restructured layer loop (residual carried between layers,
final add fused into the final norm) is testable without a GPU. On the
GPU the Triton kernels are checked against them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # pragma: no cover - availability is environmental
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


def _use_triton(t: torch.Tensor) -> bool:
    return HAS_TRITON and t.is_cuda


# ----------------------------------------------------------------------
# References: the Hugging Face computation, verbatim
# ----------------------------------------------------------------------


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return weight * h.to(dtype)


def fused_add_rms_norm_ref(x, residual, weight, eps):
    s = residual + x
    return rms_norm_ref(s, weight, eps), s


def silu_and_mul_ref(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_qk_ref(q, k, cos, sin, hq: int, hk: int, d: int):
    """q [B, S, HQ*D], k [B, S, HK*D], cos/sin [B or 1, 1, S, D]."""
    b, s = q.shape[0], q.shape[1]
    c = cos.reshape(cos.shape[0], s, 1, d)
    sn = sin.reshape(sin.shape[0], s, 1, d)
    q4 = q.reshape(b, s, hq, d)
    k4 = k.reshape(b, s, hk, d)
    qo = (q4 * c) + (_rotate_half(q4) * sn)
    ko = (k4 * c) + (_rotate_half(k4) * sn)
    return qo.reshape(b, s, hq * d), ko.reshape(b, s, hk * d)


# ----------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------


if HAS_TRITON:  # pragma: no cover - requires a GPU

    @triton.jit
    def _rms_norm_kernel(X, R, W, Y, RES_OUT,
                         stride_x, stride_r, stride_y, stride_ro,
                         N, eps,
                         HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x_raw = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
        x = x_raw.to(tl.float32)
        if HAS_RES:
            r = tl.load(R + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)
            # fp32 add, one rounding: equal to the fp16 add the model does.
            s = (x + r).to(x_raw.dtype)
            tl.store(RES_OUT + row * stride_ro + offs, s, mask=mask)
            x = s.to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        # Round to the model dtype *before* the weight, as Qwen2RMSNorm does.
        y = (x * tl.rsqrt(var + eps)).to(x_raw.dtype).to(tl.float32)
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + row * stride_y + offs, (w * y).to(x_raw.dtype), mask=mask)

    @triton.jit
    def _silu_mul_kernel(GU, OUT, stride_gu, stride_out, I, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < I
        g_raw = tl.load(GU + row * stride_gu + offs, mask=mask, other=0.0)
        u_raw = tl.load(GU + row * stride_gu + I + offs, mask=mask, other=0.0)
        g = g_raw.to(tl.float32)
        # F.silu on fp16 computes in fp32 and rounds once; the multiply by
        # `up` is then an fp16 multiply, i.e. one more rounding.
        act = (g * tl.sigmoid(g)).to(g_raw.dtype).to(tl.float32)
        tl.store(OUT + row * stride_out + offs,
                 (act * u_raw.to(tl.float32)).to(u_raw.dtype), mask=mask)

    @triton.jit
    def _rope_kernel(Q, K, COS, SIN, QO, KO,
                     stride_q, stride_k, stride_c, stride_s, stride_qo, stride_ko,
                     HQ, HALF: tl.constexpr):
        """One program per (row, head). Rows are flattened [B, S]; heads run
        over q's heads then k's, so q and k share one launch."""
        row = tl.program_id(0)
        head = tl.program_id(1)
        if head < HQ:
            src = Q + row * stride_q + head * (2 * HALF)
            dst = QO + row * stride_qo + head * (2 * HALF)
        else:
            src = K + row * stride_k + (head - HQ) * (2 * HALF)
            dst = KO + row * stride_ko + (head - HQ) * (2 * HALF)
        offs = tl.arange(0, HALF)
        x1_raw = tl.load(src + offs)
        x1 = x1_raw.to(tl.float32)
        x2 = tl.load(src + HALF + offs).to(tl.float32)
        c1 = tl.load(COS + row * stride_c + offs).to(tl.float32)
        c2 = tl.load(COS + row * stride_c + HALF + offs).to(tl.float32)
        s1 = tl.load(SIN + row * stride_s + offs).to(tl.float32)
        s2 = tl.load(SIN + row * stride_s + HALF + offs).to(tl.float32)
        dt = x1_raw.dtype
        # q*cos + rotate_half(q)*sin with rotate_half = [-x2, x1]: each
        # product rounded, then the sum rounded — three fp16 roundings, as
        # the unfused expression performs them.
        o1 = ((x1 * c1).to(dt).to(tl.float32) + (-(x2 * s1)).to(dt).to(tl.float32)).to(dt)
        o2 = ((x2 * c2).to(dt).to(tl.float32) + (x1 * s2).to(dt).to(tl.float32)).to(dt)
        tl.store(dst + offs, o1)
        tl.store(dst + HALF + offs, o2)


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------


def _rows(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1])


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not _use_triton(x):
        return rms_norm_ref(x, weight, eps)
    x2 = _rows(x)
    y = torch.empty_like(x2)
    n = x2.shape[-1]
    _rms_norm_kernel[(x2.shape[0],)](
        x2, x2, weight, y, y, x2.stride(0), x2.stride(0), y.stride(0), y.stride(0),
        n, eps, HAS_RES=False, BLOCK=triton.next_power_of_2(n),
    )
    return y.reshape(x.shape)


def fused_add_rms_norm(x, residual, weight, eps):
    """(norm(residual + x), residual + x) in one kernel."""
    if not _use_triton(x):
        return fused_add_rms_norm_ref(x, residual, weight, eps)
    x2, r2 = _rows(x), _rows(residual)
    y = torch.empty_like(x2)
    res_out = torch.empty_like(x2)
    n = x2.shape[-1]
    _rms_norm_kernel[(x2.shape[0],)](
        x2, r2, weight, y, res_out, x2.stride(0), r2.stride(0), y.stride(0),
        res_out.stride(0), n, eps, HAS_RES=True, BLOCK=triton.next_power_of_2(n),
    )
    return y.reshape(x.shape), res_out.reshape(x.shape)


def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    if not _use_triton(gate_up):
        return silu_and_mul_ref(gate_up)
    gu = _rows(gate_up)
    inter = gu.shape[-1] // 2
    out = torch.empty((gu.shape[0], inter), dtype=gu.dtype, device=gu.device)
    block = 1024
    _silu_mul_kernel[(gu.shape[0], triton.cdiv(inter, block))](
        gu, out, gu.stride(0), out.stride(0), inter, BLOCK=block,
    )
    return out.reshape(*gate_up.shape[:-1], inter)


def rope_qk(q, k, cos, sin, hq: int, hk: int, d: int):
    """RoPE on q and k in one launch, on the [B, S, H*D] layout the fused
    projection produces — before the transpose to [B, H, S, D]."""
    if not _use_triton(q):
        return rope_qk_ref(q, k, cos, sin, hq, hk, d)
    b, s = q.shape[0], q.shape[1]
    q2 = q.reshape(b * s, hq * d)
    k2 = k.reshape(b * s, hk * d)
    c2 = cos.expand(b, 1, s, d).reshape(b * s, d)
    s2 = sin.expand(b, 1, s, d).reshape(b * s, d)
    qo = torch.empty((b * s, hq * d), dtype=q.dtype, device=q.device)
    ko = torch.empty((b * s, hk * d), dtype=k.dtype, device=k.device)
    _rope_kernel[(b * s, hq + hk)](
        q2, k2, c2, s2, qo, ko,
        q2.stride(0), k2.stride(0), c2.stride(0), s2.stride(0), qo.stride(0), ko.stride(0),
        hq, HALF=d // 2,
    )
    return qo.reshape(b, s, hq * d), ko.reshape(b, s, hk * d)