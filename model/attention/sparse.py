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
# Phase 15 bake-off: alternative indexers, all training-free and page-level.
#   mass    each head's bounds turned into estimated attention weights
#           (softmax over pages of bound/sqrt(d) + log tokens), summed over
#           the group: ranks like the oracle (total mass), not by the max bound
#   mean    the same estimate from q . mean(K) per page: an estimate rather
#           than a bound, and one vector per page instead of two
#   rerank  two-stage: bounds pick 2x the budget as candidates, exact scores
#           of their keys keep the true top pages
BAKEOFF_POLICIES = ("mass", "mean", "rerank")


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


def page_counts(t: int, page: int, device) -> torch.Tensor:
    """Tokens in each page of a t-token sequence: [P]."""
    pages = -(-t // page)
    counts = torch.full((pages,), float(page), device=device)
    counts[-1] = t - (pages - 1) * page
    return counts


def page_means(k: torch.Tensor, page: int) -> torch.Tensor:
    """k [B, H, T, D] -> per-page mean key [B, H, P, D] over written tokens."""
    b, h, t, d = k.shape
    pages = -(-t // page)
    padded = torch.nn.functional.pad(k.float(), (0, 0, 0, pages * page - t))
    sums = padded.reshape(b, h, pages, page, d).sum(dim=3)
    return sums / page_counts(t, page, k.device)[None, None, :, None]


def estimated_mass(page_logit: torch.Tensor, counts: torch.Tensor, scale: float) -> torch.Tensor:
    """Per-head page logits [B, H, R, P] -> estimated attention mass per page,
    summed over the R heads of the group: [B, H, P]. Each head's estimate is
    a softmax over pages, weighting a page by its token count (exp(score) per
    token, `count` tokens), so every head contributes a total of 1 — the
    oracle's normalisation."""
    logits = page_logit * scale + torch.log(counts)[None, None, None, :]
    return torch.softmax(logits, dim=-1).sum(dim=2)


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
    """Installed on every attention layer; replaces decode attention.

    `dense_layers`: the first N layers attend densely (Quest's choice: early
    layers attend diffusely, and no small page set captures them).
    """

    policy: str = "dense"
    ratio: float = 1.0
    page: int = 16
    recent: int = 2
    dense_layers: int = 0
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
        policy = self.policy
        if self.dense_layers and layer_idx is not None and layer_idx < self.dense_layers:
            policy = "dense"
        if policy in BAKEOFF_POLICIES:
            keep = self._select_bakeoff(policy, q, k_all, mass, pages, t, d)
        else:
            bnd = page_bounds(q, k_all, self.page) if policy == "bounds" else None
            keep = select_pages(policy, self.ratio, mass=mass, bounds=bnd, pages=pages,
                                recent=self.recent)
        token_keep = keep.repeat_interleave(self.page, dim=-1)[..., :t]       # [B, H, T]
        sparse = torch.softmax(scores.masked_fill(~token_keep[:, :, None, :], float("-inf")), -1)
        self.captured.append(float((mass * keep).sum() / (b * h * r)))
        self.kept.append(float(keep.float().mean()))
        return (sparse @ v_all.float()).to(q.dtype)

    def _select_bakeoff(self, policy, q, k_all, mass, pages, t, d) -> torch.Tensor:
        b, h = q.shape[0], q.shape[1]
        k = budget(self.ratio, pages)
        if k >= pages:
            return torch.ones(b, h, pages, dtype=torch.bool, device=q.device)
        scale = 1.0 / math.sqrt(d)
        counts = page_counts(t, self.page, q.device)
        if policy == "mass":
            score = estimated_mass(page_bounds(q, k_all, self.page), counts, scale)
        elif policy == "mean":
            means = page_means(k_all, self.page)                            # [B, H, P, D]
            score = estimated_mass(q.float() @ means.transpose(-1, -2), counts, scale)
        else:                                                               # rerank
            first = page_bounds(q, k_all, self.page).amax(dim=2).clone()
            first[..., _forced(pages, self.recent, q.device)] = float("inf")
            cand = first.topk(min(pages, 2 * k), dim=-1).indices
            # Exact scores of the candidates' keys rank them by their true
            # mass; the shared softmax denominator does not change the order.
            score = torch.full_like(mass, float("-inf"))
            score.scatter_(-1, cand, mass.gather(-1, cand))
        score = score.clone()
        score[..., _forced(pages, self.recent, q.device)] = float("inf")
        keep = torch.zeros(b, h, pages, dtype=torch.bool, device=q.device)
        keep.scatter_(-1, score.topk(k, dim=-1).indices, True)
        return keep

    def reset_stats(self) -> None:
        self.captured.clear()
        self.kept.clear()


def install(model, study) -> None:
    """Route every layer's decode attention through `study` (None removes it)."""
    for layer in model.layers:
        layer.attn.sparse_study = study
