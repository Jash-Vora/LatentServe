"""
Phase 7.1 — KV compressibility analysis.

docs/phase7.md. Before implementing any latent representation, find out
whether Qwen2.5-1.5B's KV state is compressible at all, and where.

## What is actually being measured

For a collection of KV vectors stacked as A (N tokens x d dims), the
singular values of A say how much of the representation lives in how few
directions. Retained energy at rank r is

    E(r) = sum_{i<=r} sigma_i^2 / sum_i sigma_i^2

and the rank needed for 99% energy is the honest answer to "how small
could this be".

**Gram accumulation, not stored activations.** sigma_i(A)^2 are the
eigenvalues of A^T A, which is d x d regardless of how many tokens pass
through. So this streams: 32K tokens cost the same memory as 32, and the
512x512 accumulator is 2 MB. Storing activations instead would be
28 layers x 32K x 512 x 4 B = 1.8 GB and would cap the sample size for
no benefit.

Accumulated in float64. fp16 activations summed over tens of thousands
of tokens lose the small singular values to rounding, and the small
singular values are exactly what the question is about.

## Two spectra, and why the difference matters

**Weight spectrum** — singular values of W_K / W_V themselves. Data-free,
and what a naive "SVD the projection" approach optimises.

**Activation spectrum** — singular values of the actual K/V produced on
real text. This is what matters, and it is not the same thing: for
K = H W_K, the activation Gram is W_K^T (H^T H) W_K, i.e. the weight
matrix seen through the input covariance. A direction the weights treat
as important but the data never excites is free to discard; a direction
the weights barely touch but the data hammers is not.

That equivalence is worth stating plainly because it means **the
activation spectrum already is the activation-aware weight analysis** —
no separate whitening step is needed, which is the expensive part of
methods like SVD-LLM.

## The RoPE trap

K is rotated by position after projection, and the rotation mixes
dimensions differently at every position. Taking the spectrum of
post-RoPE K therefore measures a matrix whose structure has been
deliberately smeared across positions, and will make K look far less
compressible than it is.

So both are measured. Compressing pre-RoPE (cache the latent,
reconstruct K, then rotate at the reconstructed position) is the
structure MLA actually uses, and the gap between the two spectra is a
concrete measurement of why DeepSeek puts the positional path outside
the compressed one.

V has no RoPE, so expect V to compress more readily than post-RoPE K.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


class SpectrumAccumulator:
    """Streaming A^T A for one named tensor slot."""

    def __init__(self, dim: int, device: str | torch.device = "cpu"):
        self.dim = dim
        self.device = torch.device(device)
        self.gram = torch.zeros((dim, dim), dtype=torch.float64, device=self.device)
        self.count = 0

    def update(self, x: torch.Tensor) -> None:
        """x: [..., dim]. Flattened over every leading axis.

        Moved to the accumulator's own device and dtype rather than
        assuming the caller matched them: activations arrive wherever the
        model lives, while the Gram may be held on CPU to keep VRAM free
        for the model itself.
        """
        flat = x.reshape(-1, self.dim).to(device=self.device, dtype=torch.float64)
        self.gram += flat.T @ flat
        self.count += flat.shape[0]

    def eigenvalues(self) -> torch.Tensor:
        """Descending eigenvalues of the Gram = squared singular values.

        `eigvalsh` exploits symmetry and returns ascending real values;
        tiny negatives can appear from accumulation and are clamped, since
        a negative variance is numerical noise rather than information.
        """
        if self.count == 0:
            return torch.zeros(self.dim, dtype=torch.float64)
        eigs = torch.linalg.eigvalsh(self.gram)
        return eigs.clamp_min(0).flip(0)


def energy_curve(eigenvalues: torch.Tensor) -> torch.Tensor:
    total = eigenvalues.sum()
    if total <= 0:
        return torch.zeros_like(eigenvalues)
    return torch.cumsum(eigenvalues, dim=0) / total


def rank_for_energy(eigenvalues: torch.Tensor, threshold: float) -> int:
    """Smallest rank retaining `threshold` of the energy."""
    curve = energy_curve(eigenvalues)
    idx = torch.nonzero(curve >= threshold)
    return int(idx[0].item()) + 1 if len(idx) else len(eigenvalues)


def effective_rank(eigenvalues: torch.Tensor) -> float:
    """exp(entropy of the normalised spectrum).

    A threshold-free summary: flat spectra give a value near d, spectra
    dominated by a few directions give a small one. Useful because the
    99%-energy rank is sensitive to exactly where the threshold sits,
    and this is not.
    """
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    p = eigenvalues / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()).item())


@dataclass
class SpectrumReport:
    name: str
    layer: int
    dim: int
    tokens: int
    eigenvalues: torch.Tensor = field(repr=False)

    def summary(self, thresholds=(0.90, 0.95, 0.99, 0.999)) -> dict:
        return {
            "name": self.name,
            "layer": self.layer,
            "dim": self.dim,
            "tokens": self.tokens,
            "effective_rank": effective_rank(self.eigenvalues),
            **{
                f"rank_{int(t * 1000)}": rank_for_energy(self.eigenvalues, t)
                for t in thresholds
            },
            # Fraction of the full dimension needed for 99% — the number
            # that decides whether a latent representation has room to
            # exist at all.
            "compression_at_99": rank_for_energy(self.eigenvalues, 0.99) / self.dim,
        }


def weight_spectrum(weight: torch.Tensor) -> torch.Tensor:
    """Squared singular values of an [out, in] projection weight.

    Data-free: says what the projection *could* express, not what the
    data makes it express. Kept as the contrast against the activation
    spectra.
    """
    sv = torch.linalg.svdvals(weight.to(torch.float64))
    return (sv**2).flip(0).flip(0)  # already descending


# ----------------------------------------------------------------------
# Break-even, for Qwen2.5-1.5B specifically
# ----------------------------------------------------------------------

GQA_NUMBERS_PER_TOKEN_PER_LAYER = 512  # 2 kv heads x 128 dims, K and V


def gqa_numbers_per_token_per_layer(num_kv_heads: int, head_dim: int) -> int:
    """K and V, for every KV head. 512 for Qwen2.5-1.5B."""
    return 2 * num_kv_heads * head_dim


def mla_break_even(rope_dim: int = 64, gqa_numbers: int = GQA_NUMBERS_PER_TOKEN_PER_LAYER) -> int:
    """Largest MLA latent dimension that is smaller than the GQA cache.

    Qwen already caches 512 numbers per token per layer, so a latent of
    `latent_dim + rope_dim` only saves anything below this. With
    rope_dim 64 that is 448 — which is why three of the five values in
    the methodology doc's Phase 10 sweep (512, 768, 1024) would make the
    cache *larger* than the baseline they are meant to improve on.
    """
    return gqa_numbers - rope_dim


def compression_ratio(
    latent_dim: int, rope_dim: int = 64, gqa_numbers: int = GQA_NUMBERS_PER_TOKEN_PER_LAYER
) -> float:
    return gqa_numbers / (latent_dim + rope_dim)