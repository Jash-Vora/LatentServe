"""
Phase 7.2 — does reconstruction error become output damage?

7.1 measured how well a rank-r subspace reconstructs K and V. It could
not say whether that error matters: a 30% relative error on V is only
bad if it moves the model's output distribution, and attention is a
softmax over dot products, which may be far more or far less forgiving
than the Frobenius norm suggests.

This module answers that directly, and cheaply: **simulate** the
compression inside the forward pass and measure logit KL divergence
against the unmodified model. No cache, no kernel, no MLA
implementation. If 19% V error moves KL by 0.01 nats, MLA is alive and
worth building; if it moves KL by 0.5, the phase ends here.

## Why simulation rather than implementation

Every method in Phase 7 — low-rank, adaptive rank, MLA, INT8 — has the
same effect on the model's arithmetic: the KV that attention sees is a
lossy version of the KV it would otherwise see. The runtime differences
(what is stored, what is recomputed, which kernel runs) change speed and
memory, not output. So the quality question can be settled for all of
them before any of them exists, by perturbing the forward pass in place.

This is the cheap exploration layer the plan calls for. Only the
representations that survive it get a runtime.

## The wrapper trick

An MLA latent compresses K and V *jointly*, but `k_proj` and `v_proj`
are separate modules called in sequence, so a forward hook on `k_proj`
cannot see V yet. Both wrappers therefore recompute both projections
from the shared input — the same hidden state feeds each — form the
joint vector, project it, and return their own slice. That doubles the
projection compute, which is irrelevant here because nothing about this
module is being timed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn

QuantMode = Literal["none", "per_tensor", "per_channel", "per_token"]


# ----------------------------------------------------------------------
# Low-rank projection
# ----------------------------------------------------------------------


def fit_basis(gram: torch.Tensor, rank: int) -> torch.Tensor:
    """Top-`rank` eigenvectors of a Gram matrix, as columns.

    This is the optimal rank-r subspace for the *data*, which is what
    an activation-aware low-rank method fits. It is not the same as
    truncating the weight matrix's own SVD, and 7.1 measured the gap:
    the weights need 433 of 512 dimensions, the activations 165.
    """
    vals, vecs = torch.linalg.eigh(gram.detach().to(device="cpu", dtype=torch.float64))
    order = torch.argsort(vals, descending=True)
    return vecs[:, order[:rank]].contiguous()


class JointKVProjection(nn.Module):
    """Wraps `k_proj` or `v_proj`, returning its slice of a rank-r
    projection of the joint [K | V] vector."""

    def __init__(self, k_proj: nn.Module, v_proj: nn.Module, basis: torch.Tensor, which: str):
        super().__init__()
        self.k_proj, self.v_proj = k_proj, v_proj
        self.which = which
        self.register_buffer("basis", basis, persistent=False)
        self.k_dim = k_proj.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.k_proj(x)
        v = self.v_proj(x)
        joint = torch.cat([k, v], dim=-1)
        basis = self.basis.to(joint.dtype)
        projected = (joint @ basis) @ basis.T
        return projected[..., : self.k_dim] if self.which == "k" else projected[..., self.k_dim :]


class SeparateProjection(nn.Module):
    """Rank-r projection of one of K or V in its own basis.

    The control for the joint version. 7.1 suggested these should be
    close — V is the binding constraint either way — and a measurement
    beats the suggestion.
    """

    def __init__(self, proj: nn.Module, basis: torch.Tensor):
        super().__init__()
        self.proj = proj
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        basis = self.basis.to(out.dtype)
        return (out @ basis) @ basis.T


# ----------------------------------------------------------------------
# Quantization — the control that low-rank has to beat
# ----------------------------------------------------------------------


def quantize_dequantize(
    x: torch.Tensor,
    mode: QuantMode,
    num_heads: int,
    head_dim: int,
    bits: int = 8,
    asymmetric: bool = False,
) -> torch.Tensor:
    """Symmetric INT8 round-trip, at one of three granularities.

    x is [..., num_heads * head_dim].

    * `per_tensor` — one scale for everything. Cheapest, worst.
    * `per_channel` — a scale per (head, dim) feature, shared across
      tokens. Suits K: 7.1 found pre-RoPE K has effective rank 5.6, i.e.
      a few channels carrying enormous magnitude, and a shared scale
      lets those channels not dominate everyone else's resolution.
    * `per_token` — a scale per (token, head), shared across dims. Suits
      V, whose energy is spread evenly (effective rank 72).

    That asymmetry is the KIVI arrangement, and 7.1's spectra predict it
    independently — which makes it a nice check on both.
    """
    if mode == "none":
        return x
    shaped = x.reshape(*x.shape[:-1], num_heads, head_dim)

    if mode == "per_tensor":
        dims = tuple(range(shaped.dim()))
    elif mode == "per_channel":
        dims = tuple(range(shaped.dim() - 2))
    elif mode == "per_token":
        dims = (shaped.dim() - 1,)
    else:
        raise ValueError(f"unknown quant mode {mode!r}")

    if asymmetric:
        # Affine: fit [min, max] rather than assuming symmetry about zero.
        # Worth its extra zero-point because KV activations are skewed —
        # a symmetric range spends half its levels on a sign the data
        # rarely uses, which costs a full bit exactly where bits are
        # scarce.
        lo = shaped.amin(dim=dims, keepdim=True)
        hi = shaped.amax(dim=dims, keepdim=True)
        levels = 2**bits - 1
        scale = ((hi - lo) / levels).clamp_min(1e-8)
        q = torch.clamp(torch.round((shaped - lo) / scale), 0, levels)
        return (q * scale + lo).reshape_as(x)

    qmax = 2 ** (bits - 1) - 1
    scale = shaped.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8)
    q = torch.clamp(torch.round(shaped / scale * qmax), -qmax, qmax)
    return (q * scale / qmax).reshape_as(x)


def quantize_dequantize_block_local(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
    block_size: int,
    bits: int = 8,
    asymmetric: bool = False,
) -> torch.Tensor:
    """Per-channel INT8, but with the scale fit *within* each contiguous
    block of `block_size` tokens rather than over the whole sequence.

    `quantize_dequantize(mode="per_channel")` reduces over every leading
    dim at once — batch *and* the entire sequence — which means token
    500's scale is computed knowing about token 9,000. A paged, streaming
    cache cannot do that: a block is written once as it fills and its
    scale has to be fixed from only the tokens inside it. This is the
    realizable version of per-channel K quantization, matching the
    project's `block_size: 16` paged-cache page.

    x is [..., seq, num_heads * head_dim] (batch dims before `seq` are
    fine; `seq` is assumed to be the second-to-last axis before the
    flattened head dim, matching how this module calls it elsewhere).

    `asymmetric` fits [min, max] per block instead of assuming the range
    is centred on zero. A symmetric fit spends half its levels on a sign
    the data may rarely use — a whole bit, wasted exactly where bits are
    scarce — so this matters far more at 4 bits than at 8. A real cache
    stores a zero-point alongside the scale, doubling the per-block
    scale overhead: 12.5% -> 25% at block 16, which is why the effective
    ratio has to be quoted per configuration rather than as "2x".
    """
    shaped = x.reshape(*x.shape[:-1], num_heads, head_dim)  # [..., seq, H, D]
    seq_dim = shaped.dim() - 3
    seq_len = shaped.shape[seq_dim]
    qmax = 2 ** (bits - 1) - 1

    out = torch.empty_like(shaped)
    reduce_dims = tuple(d for d in range(shaped.dim() - 2) if d != seq_dim)
    dims = reduce_dims + (seq_dim,)
    levels = 2**bits - 1
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        block = shaped.narrow(seq_dim, start, end - start)
        if asymmetric:
            lo = block.amin(dim=dims, keepdim=True)
            hi = block.amax(dim=dims, keepdim=True)
            scale = ((hi - lo) / levels).clamp_min(1e-8)
            q = torch.clamp(torch.round((block - lo) / scale), 0, levels)
            out.narrow(seq_dim, start, end - start).copy_(q * scale + lo)
        else:
            scale = block.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8)
            q = torch.clamp(torch.round(block / scale * qmax), -qmax, qmax)
            out.narrow(seq_dim, start, end - start).copy_(q * scale / qmax)
    return out.reshape_as(x)


class QuantizingProjection(nn.Module):
    """Wraps a projection and round-trips its output through INT8."""

    def __init__(self, proj: nn.Module, mode: QuantMode, num_heads: int, head_dim: int,
                 bits: int = 8, asymmetric: bool = False, block_size: Optional[int] = None):
        super().__init__()
        self.proj = proj
        self.mode, self.num_heads, self.head_dim, self.bits = mode, num_heads, head_dim, bits
        self.asymmetric = asymmetric
        # When set, `mode == "per_channel"` uses the block-local scale
        # fit instead of the global one — see quantize_dequantize_block_local.
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        if self.block_size is not None and self.mode == "per_channel":
            return quantize_dequantize_block_local(
                out, self.num_heads, self.head_dim, self.block_size, self.bits,
                asymmetric=self.asymmetric,
            )
        return quantize_dequantize(
            out, self.mode, self.num_heads, self.head_dim, self.bits, self.asymmetric
        )


# ----------------------------------------------------------------------
# Install / remove
# ----------------------------------------------------------------------


@dataclass
class Installed:
    """Remembers the original modules so a config can be undone exactly.

    Restoring matters more than it looks: every configuration is compared
    against the same baseline, so a leaked wrapper would contaminate
    every subsequent measurement in the sweep with a silent extra loss.
    """

    originals: list

    def remove(self) -> None:
        for layer, k, v in self.originals:
            layer.self_attn.k_proj = k
            layer.self_attn.v_proj = v


def install_joint_lowrank(model, bases: dict[int, torch.Tensor]) -> Installed:
    originals = []
    for idx, layer in enumerate(model.model.layers):
        k, v = layer.self_attn.k_proj, layer.self_attn.v_proj
        originals.append((layer, k, v))
        basis = bases[idx].to(next(k.parameters()).device)
        layer.self_attn.k_proj = JointKVProjection(k, v, basis, "k")
        layer.self_attn.v_proj = JointKVProjection(k, v, basis, "v")
    return Installed(originals)


def install_separate_lowrank(
    model, k_bases: dict[int, torch.Tensor], v_bases: dict[int, torch.Tensor]
) -> Installed:
    originals = []
    for idx, layer in enumerate(model.model.layers):
        k, v = layer.self_attn.k_proj, layer.self_attn.v_proj
        originals.append((layer, k, v))
        device = next(k.parameters()).device
        layer.self_attn.k_proj = SeparateProjection(k, k_bases[idx].to(device))
        layer.self_attn.v_proj = SeparateProjection(v, v_bases[idx].to(device))
    return Installed(originals)


def install_quantization(
    model, k_mode: QuantMode, v_mode: QuantMode, num_heads: int, head_dim: int,
    bits: int = 8, asymmetric: bool = False, k_bits: Optional[int] = None,
    v_bits: Optional[int] = None, k_block_size: Optional[int] = None,
    v_block_size: Optional[int] = None,
) -> Installed:
    """K and V may carry different bit-widths.

    7.2 measured K as the sensitive one — its granularity changes KL by
    35x while V's changes it by 2% — so spending bits asymmetrically is
    the obvious thing to try, and the memory cost is the mean of the two.

    `k_block_size` / `v_block_size`, when set, make a `per_channel` mode
    fit its scale locally within blocks of that many tokens rather than
    globally over the whole sequence — the scaling a streaming/paged
    cache can actually compute. `per_token` needs no such flag: it is
    already local to a single token and therefore already streaming-safe.
    """
    originals = []
    for layer in model.model.layers:
        k, v = layer.self_attn.k_proj, layer.self_attn.v_proj
        originals.append((layer, k, v))
        layer.self_attn.k_proj = QuantizingProjection(
            k, k_mode, num_heads, head_dim, k_bits or bits, asymmetric, k_block_size
        )
        layer.self_attn.v_proj = QuantizingProjection(
            v, v_mode, num_heads, head_dim, v_bits or bits, asymmetric, v_block_size
        )
    return Installed(originals)


# ----------------------------------------------------------------------
# Output-space metrics
# ----------------------------------------------------------------------


class DivergenceMeter:
    """Streaming KL(baseline || modified), top-1 agreement, and the
    baseline's own cross-entropy.

    KL in that direction answers "how much probability mass does the
    baseline put where the compressed model does not", which is the
    question for a lossy approximation of a fixed model. Top-1 agreement
    is the blunter, more legible companion: a KL of 0.01 nats means
    little on its own, while "changes the argmax on 0.3% of tokens" is
    immediately interpretable.
    """

    def __init__(self) -> None:
        self.kl_sum = 0.0
        self.agree = 0
        self.tokens = 0
        self.max_kl = 0.0
        # Baseline confidence, because top-1 flip rate cannot be read
        # without it. On natural text this model sits at ~0.99 with a
        # margin of ~0.99 and flips 0.000%; on random ids it sits at
        # ~0.02 with a margin of ~0.01 and flips 2-4%. Same cache, same
        # quantization — the difference was entirely how close the top
        # two candidates were.
        self.top1_prob_sum = 0.0
        self.margin_sum = 0.0

    @torch.no_grad()
    def update(self, baseline_logits: torch.Tensor, modified_logits: torch.Tensor) -> None:
        base = baseline_logits.reshape(-1, baseline_logits.shape[-1]).float()
        mod = modified_logits.reshape(-1, modified_logits.shape[-1]).float()
        log_p = torch.log_softmax(base, dim=-1)
        log_q = torch.log_softmax(mod, dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)
        self.kl_sum += float(kl.sum().item())
        self.max_kl = max(self.max_kl, float(kl.max().item()))
        self.agree += int((base.argmax(-1) == mod.argmax(-1)).sum().item())
        top2 = log_p.exp().topk(2, dim=-1).values
        self.top1_prob_sum += float(top2[:, 0].sum().item())
        self.margin_sum += float((top2[:, 0] - top2[:, 1]).sum().item())
        self.tokens += base.shape[0]

    def result(self) -> dict:
        n = max(1, self.tokens)
        return {
            "kl_mean_nats": self.kl_sum / n,
            "kl_max_nats": self.max_kl,
            "top1_agreement": self.agree / n,
            "top1_flip_rate": 1 - self.agree / n,
            "tokens": self.tokens,
            "baseline_top1_prob": self.top1_prob_sum / n,
            "baseline_top1_margin": self.margin_sum / n,
        }