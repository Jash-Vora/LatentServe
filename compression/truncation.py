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
    vals, vecs = torch.linalg.eigh(gram.to(torch.float64))
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


def quantize_dequantize(x: torch.Tensor, mode: QuantMode, num_heads: int, head_dim: int, bits: int = 8) -> torch.Tensor:
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
    qmax = 2 ** (bits - 1) - 1
    shaped = x.reshape(*x.shape[:-1], num_heads, head_dim)

    if mode == "per_tensor":
        scale = shaped.abs().amax().clamp_min(1e-8)
    elif mode == "per_channel":
        dims = tuple(range(shaped.dim() - 2))
        scale = shaped.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8)
    elif mode == "per_token":
        scale = shaped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        raise ValueError(f"unknown quant mode {mode!r}")

    q = torch.clamp(torch.round(shaped / scale * qmax), -qmax, qmax)
    return (q * scale / qmax).reshape_as(x)


class QuantizingProjection(nn.Module):
    """Wraps a projection and round-trips its output through INT8."""

    def __init__(self, proj: nn.Module, mode: QuantMode, num_heads: int, head_dim: int, bits: int = 8):
        super().__init__()
        self.proj = proj
        self.mode, self.num_heads, self.head_dim, self.bits = mode, num_heads, head_dim, bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quantize_dequantize(
            self.proj(x), self.mode, self.num_heads, self.head_dim, self.bits
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
    model, k_mode: QuantMode, v_mode: QuantMode, num_heads: int, head_dim: int, bits: int = 8
) -> Installed:
    originals = []
    for layer in model.model.layers:
        k, v = layer.self_attn.k_proj, layer.self_attn.v_proj
        originals.append((layer, k, v))
        layer.self_attn.k_proj = QuantizingProjection(k, k_mode, num_heads, head_dim, bits)
        layer.self_attn.v_proj = QuantizingProjection(v, v_mode, num_heads, head_dim, bits)
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
        self.tokens += base.shape[0]

    def result(self) -> dict:
        n = max(1, self.tokens)
        return {
            "kl_mean_nats": self.kl_sum / n,
            "kl_max_nats": self.max_kl,
            "top1_agreement": self.agree / n,
            "top1_flip_rate": 1 - self.agree / n,
            "tokens": self.tokens,
        }