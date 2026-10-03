"""
Phase 14 — sparse decode attention, as reference math.

Sparse attention is a decode-time technique: the prompt is prefilled
densely, and each generated token attends to a *selected subset* of the
cached pages. Selection is per KV head and shared by the query heads in its
group (6 for Qwen2.5-1.5B), so a kernel reads each selected page once for
all of them — selecting per query head would break that sharing.

Policies, each given a budget of `ceil(ratio x pages)` pages:

  dense    every page (the baseline, computed by the same math)
  oracle   the pages that truly carry the most attention mass, summed over
           the group's query heads: the ceiling any indexer can reach
  bounds   Quest-style, training-free: each page's per-channel min and max
           of K bound q.k for every key in it (sum_d max(q_d min_d,
           q_d max_d)); pages ranked by the largest bound across the group,
           with the first page (attention sink) and the most recent pages
           always kept
  window   the first page plus the most recent ones, no query awareness at
           all: the control that shows whether being query-aware matters

Every call also measures how much of the *dense* attention mass the
selection captured — the most direct measure of an indexer against the
oracle. This module is the study's instrument and, later, the ground truth
for the sparse kernel's tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

POLICIES = ("dense", "oracle", "bounds", "window")


def page_mass(probs: torch.Tensor, page: int) -> torch.Tensor:
    """probs [B, H, R, T] (rows sum to 1) -> attention mass per page,
    summed over the R query heads of each group: [B, H, P]."""
    b, h, r, t = probs.shape
    pages = -(-t // page)
    padded = torch.nn.functional.pad(probs, (0, pages * page - t))
    return padded.reshape(b, h, r, pages, page).sum(dim=(2, 4))


def page_bounds(q: torch.Tensor, k: torch.Tensor, page: int) -> torch.Tensor:
    """Upper bound on q.k for every key in each page.

    q [B, H, R, D], k [B, H, T, D] -> [B, H, R, P]. With min_d and max_d the
    page's per-channel extremes, q_d k_d <= max(q_d min_d, q_d max_d) for
    every key, so the sum over channels bounds the dot product. Padding past
    T uses +inf / -inf so it never sets an extreme.
    """
    b, h, t, d = k.shape
    pages = -(-t // page)
    pad = pages * page - t
    kf = k.float()
    kmin = torch.nn.functional.pad(kf, (0, 0, 0, pad), value=float("inf"))
    kmax = torch.nn.functional.pad(kf, (0, 0, 0, pad), value=float("-inf"))
    kmin = kmin.reshape(b, h, pages, page, d).amin(dim=3)        # [B, H, P, D]
    kmax = kmax.reshape(b, h, pages, page, d).amax(dim=3)
    qf = q.float()[:, :, :, None, :]                              # [B, H, R, 1, D]
    return torch.maximum(qf * kmin[:, :, None], qf * kmax[:, :, None]).sum(-1)


def budget(ratio: float, pages: int) -> int:
    return pages if ratio >= 1.0 else max(1, min(pages, math.ceil(ratio * pages)))


def select_pages(policy: str, ratio: float, *, mass=None, bounds=None, pages: int,
                 recent: int = 2) -> torch.Tensor:
    """-> bool [B, H, P]: the pages each KV head attends to."""
    ref = mass if mass is not None else bounds
    b, h = ref.shape[0], ref.shape[1]
    k = budget(ratio, pages)
    if policy == "dense" or k >= pages:
        return torch.ones(b, h, pages, dtype=torch.bool, device=ref.device)
    if policy == "oracle":
        score = mass
    elif policy == "bounds":
        score = bounds.amax(dim=2).clone()                       # largest bound in the group
        forced = _forced(pages, recent, ref.device)
        score[..., forced] = float("inf")
    elif policy == "window":
        score = torch.arange(pages, device=ref.device, dtype=torch.float32).expand(b, h, pages).clone()
        score[..., 0] = float("inf")                             # sink first, then most recent
    else:
        raise ValueError(f"unknown policy {policy!r}")
    keep = torch.zeros(b, h, pages, dtype=torch.bool, device=ref.device)
    keep.scatter_(-1, score.topk(k, dim=-1).indices, True)
    return keep


def _forced(pages: int, recent: int, device) -> torch.Tensor:
    idx = {0, *range(max(0, pages - recent), pages)}
    return torch.tensor(sorted(idx), device=device)


@dataclass
class SparseStudy:
    """Installed on every attention layer; replaces decode attention."""

    policy: str = "dense"
    ratio: float = 1.0
    page: int = 16
    recent: int = 2
    captured: list = field(default_factory=list)   # dense mass kept, per call
    kept: list = field(default_factory=list)       # fraction of pages kept, per call

    def __call__(self, q, k_all, v_all, layer_idx=None, key_mask=None):
        """q [B, H, R, D] post-RoPE; k_all/v_all [B, H, T, D] -> [B, H, R, D]."""
        if key_mask is not None:
            raise NotImplementedError("the oracle study runs one sequence at a time")
        b, h, r, d = q.shape
        t = k_all.shape[2]
        pages = -(-t // self.page)
        scores = (q.float() @ k_all.float().transpose(-1, -2)) / math.sqrt(d)   # [B, H, R, T]
        dense = torch.softmax(scores, dim=-1)
        mass = page_mass(dense, self.page)                                   # sums to R
        bnd = page_bounds(q, k_all, self.page) if self.policy == "bounds" else None
        keep = select_pages(self.policy, self.ratio, mass=mass, bounds=bnd, pages=pages,
                            recent=self.recent)
        token_keep = keep.repeat_interleave(self.page, dim=-1)[..., :t]       # [B, H, T]
        sparse = torch.softmax(scores.masked_fill(~token_keep[:, :, None, :], float("-inf")), -1)
        self.captured.append(float((mass * keep).sum() / (b * h * r)))
        self.kept.append(float(keep.float().mean()))
        return (sparse @ v_all.float()).to(q.dtype)

    def reset_stats(self) -> None:
        self.captured.clear()
        self.kept.clear()


def install(model, study) -> None:
    """Route every layer's decode attention through `study` (None removes it)."""
    for layer in model.layers:
        layer.attn.sparse_study = study
