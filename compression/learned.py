"""
Phase 7, Experiment B — learned latent KV.

7.2 killed post-hoc low-rank: at 2x memory it loses to INT8 by ~480x in
KL. This module tests whether that verdict is about *low-rank* or about
*how the subspace was chosen*.

## Why plain SVD cannot be improved by local training

The obvious next move — parameterise `W_down`, `W_up` and train them to
reconstruct K and V — cannot help. Minimising ||K - K_hat||^2 over rank-r
factorisations is exactly the problem the activation-Gram
eigendecomposition already solves, and 7.2 used its exact solution. A
gradient method on that objective converges back to where it started.

So a learned bottleneck can only win by optimising something **other
than reconstruction error**. Two such objectives, in increasing cost:

## B1 — output-aware low-rank (closed form, no training)

Reconstruction error treats every direction of K as equally important.
Attention does not: an error in K matters only insofar as queries see
it, and an error in V only insofar as it survives the output
projection. So weight the error by how it reaches the output.

For K, the attention logit is q . k, so the squared error that reaches
the logits is `dk^T M_K dk` with `M_K = sum_h E[q_h q_h^T]` over the
query heads sharing that KV head. For V, the error passes through W_O,
giving `M_V = sum_h W_O_h^T W_O_h`.

Minimising `||(X - X_hat) M^{1/2}||_F` has an exact solution: whiten by
`M^{1/2}`, take the top-r eigenvectors there, map back. Concretely
`W_down = M^{1/2} U_r` and `W_up = U_r^T M^{-1/2}` — the same adapter
shape a trained version would use, obtained without a training loop.

This is the output-side analogue of activation-aware SVD, and 7.1 already
showed the input-side version mattered enormously (weights said 433 of
512 dimensions were needed; activations said 165).

## B2 — KL distillation (gradient, initialised from B1)

Optimise the thing actually cared about: the divergence between the
compressed model's output distribution and the original's. Initialised
from B1 so it starts at the best closed-form solution and can only
improve — which also makes "training did not help" a meaningful result
rather than an optimisation failure.

The V metric ignores attention weighting (errors are averaged by the
attention probabilities before reaching W_O, which shrinks them), so B1
is an approximation. B2 has no such gap, which is part of what the
comparison measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from compression.spectra import SpectrumAccumulator


# ----------------------------------------------------------------------
# Metrics: how a KV error reaches the output
# ----------------------------------------------------------------------


def _cpu64(t: torch.Tensor) -> torch.Tensor:
    """Everything in this module fits subspaces on CPU in float64.

    Grams, metrics and bases are small (at most 512x512) and are solved
    with eigendecompositions where fp32 rounding is not acceptable, while
    the model itself lives on the GPU. Rather than have each call site
    remember which side it is on — the mistake that broke the first real
    run twice — every tensor entering the fitting path goes through here.
    """
    return t.detach().to(device="cpu", dtype=torch.float64)


def output_metric_for_v(o_proj_weight: torch.Tensor, group: range, head_dim: int) -> torch.Tensor:
    """M_V = sum over the query heads in this group of W_O_h^T W_O_h.

    o_proj is [hidden, num_heads * head_dim], so head h owns the column
    block h*head_dim : (h+1)*head_dim. An error dv in this KV head
    reaches the residual stream as dv @ W_O_h for every query head
    sharing it.
    """
    m = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    w = _cpu64(o_proj_weight)
    for h in group:
        block = w[:, h * head_dim : (h + 1) * head_dim]
        m += block.T @ block
    return m


class QueryMetricCollector:
    """E[q q^T] per query head, accumulated over a calibration pass.

    The query second moment is what decides which directions of K the
    model can actually see. Collected with the same streaming Gram trick
    as 7.1 — d x d regardless of token count.
    """

    def __init__(self, model, num_attention_heads: int, head_dim: int, device="cpu"):
        self.n_heads, self.head_dim = num_attention_heads, head_dim
        self.acc: dict[tuple[int, int], SpectrumAccumulator] = {}
        self.handles = []
        for idx, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.q_proj.register_forward_hook(self._hook(idx, device))
            )

    def _hook(self, layer_idx: int, device):
        def hook(_m, _i, out):
            q = out.detach().reshape(*out.shape[:-1], self.n_heads, self.head_dim)
            for h in range(self.n_heads):
                key = (layer_idx, h)
                if key not in self.acc:
                    self.acc[key] = SpectrumAccumulator(self.head_dim, device=device)
                self.acc[key].update(q[..., h, :])

        return hook

    def metric(self, layer_idx: int, group: range) -> torch.Tensor:
        m = torch.zeros(self.head_dim, self.head_dim, dtype=torch.float64)
        for h in group:
            m += self.acc[(layer_idx, h)].gram
        return m

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def joint_metric(
    query_metrics: list[torch.Tensor], value_metrics: list[torch.Tensor]
) -> torch.Tensor:
    """Block-diagonal metric over the joint [K | V] vector.

    Layout matches the joint vector the cache holds: all KV heads of K,
    then all KV heads of V. Block diagonal because an error in one head's
    K does not reach another head's logits.
    """
    blocks = list(query_metrics) + list(value_metrics)
    dim = sum(b.shape[0] for b in blocks)
    m = torch.zeros(dim, dim, dtype=torch.float64)
    offset = 0
    for b in blocks:
        d = b.shape[0]
        m[offset : offset + d, offset : offset + d] = b
        offset += d
    return m


def _sqrt_and_inverse_sqrt(m: torch.Tensor, floor_ratio: float = 1e-6):
    """M^{1/2} and M^{-1/2} with the spectrum floored.

    The metric is often near-singular — query second moments concentrate
    in few directions, exactly as 7.1's effective rank of 5.6 would
    suggest — and an unregularised inverse would send the near-null
    directions to infinity, making the fit chase directions the model
    cannot see.
    """
    vals, vecs = torch.linalg.eigh(_cpu64(m))
    vals = vals.clamp_min(vals.max() * floor_ratio)
    root = vecs @ torch.diag(vals.sqrt()) @ vecs.T
    inv_root = vecs @ torch.diag(vals.rsqrt()) @ vecs.T
    return root, inv_root


def fit_metric_basis(
    gram: torch.Tensor, metric: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
    """Optimal rank-r compression of X under the metric M.

    Returns `(down, up)` with `c = x @ down` and `x_hat = c @ up`, so the
    cached latent is r numbers per token per layer.

    With metric = I this reduces exactly to the plain eigendecomposition
    7.2 used, which is the test that pins the generalisation down.
    """
    root, inv_root = _sqrt_and_inverse_sqrt(_cpu64(metric))
    whitened = root @ _cpu64(gram) @ root
    vals, vecs = torch.linalg.eigh(whitened)
    top = vecs[:, torch.argsort(vals, descending=True)[:rank]]
    return (root @ top).contiguous(), (top.T @ inv_root).contiguous()


# ----------------------------------------------------------------------
# The adapter
# ----------------------------------------------------------------------


class LatentKVAdapter(nn.Module):
    """One layer's latent bottleneck: hidden -> c (r dims) -> [K | V].

    `down` folds the original k_proj/v_proj into the compression, so a
    real runtime computes the latent in one GEMM and never materialises
    full K/V before caching. Here both wrappers share one adapter and
    each returns its own slice, mirroring `JointKVProjection`.
    """

    def __init__(self, down: torch.Tensor, up: torch.Tensor, k_dim: int):
        super().__init__()
        self.down = nn.Parameter(down.float())
        self.up = nn.Parameter(up.float())
        self.k_dim = k_dim

    @property
    def rank(self) -> int:
        return self.down.shape[1]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        latent = hidden @ self.down.to(hidden.dtype)
        return latent @ self.up.to(hidden.dtype)

    @classmethod
    def from_projections(
        cls,
        k_proj: nn.Module,
        v_proj: nn.Module,
        post_down: torch.Tensor,
        up: torch.Tensor,
    ) -> "LatentKVAdapter":
        """Fold [W_k | W_v] into the compression.

        `post_down` maps the joint KV vector to the latent; composing it
        with the projections gives a single hidden -> latent matrix, which
        is both what MLA does architecturally and what makes the runtime
        version cheap.
        """
        w = torch.cat([_cpu64(k_proj.weight), _cpu64(v_proj.weight)], dim=0)
        post_down, up = _cpu64(post_down), _cpu64(up)
        down = w.T @ post_down          # [hidden, rank]
        bias = []
        for proj in (k_proj, v_proj):
            if getattr(proj, "bias", None) is not None:
                bias.append(_cpu64(proj.bias))
            else:
                bias.append(torch.zeros(proj.out_features, dtype=torch.float64))
        adapter = cls(down, up, k_proj.out_features)
        # Biases are constant per token, so they compress through the same
        # map rather than being dropped — dropping them would shift every
        # key and value by a fixed offset and look like a modelling loss.
        adapter.register_buffer(
            "bias_latent", (torch.cat(bias) @ post_down).float(), persistent=False
        )
        return adapter

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        latent = hidden @ self.down.to(hidden.dtype)
        if hasattr(self, "bias_latent"):
            latent = latent + self.bias_latent.to(hidden.dtype)
        return latent @ self.up.to(hidden.dtype)


class AdapterSlice(nn.Module):
    """Stands in for k_proj or v_proj, returning its half of the adapter."""

    def __init__(self, adapter: LatentKVAdapter, which: str):
        super().__init__()
        self.adapter = adapter
        self.which = which

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        out = self.adapter.project(hidden)
        return out[..., : self.adapter.k_dim] if self.which == "k" else out[..., self.adapter.k_dim :]


@dataclass
class InstalledAdapters:
    originals: list
    adapters: dict

    def remove(self) -> None:
        for layer, k, v in self.originals:
            layer.self_attn.k_proj = k
            layer.self_attn.v_proj = v

    def parameters(self):
        for a in self.adapters.values():
            yield from a.parameters()


def install_adapters(model, adapters: dict[int, LatentKVAdapter]) -> InstalledAdapters:
    originals = []
    for idx, layer in enumerate(model.model.layers):
        k, v = layer.self_attn.k_proj, layer.self_attn.v_proj
        originals.append((layer, k, v))
        adapter = adapters[idx].to(next(k.parameters()).device)
        layer.self_attn.k_proj = AdapterSlice(adapter, "k")
        layer.self_attn.v_proj = AdapterSlice(adapter, "v")
    return InstalledAdapters(originals, adapters)


# ----------------------------------------------------------------------
# B2 — KL distillation
# ----------------------------------------------------------------------


@torch.no_grad()
def baseline_logits(model, chunk: torch.Tensor) -> torch.Tensor:
    return model(input_ids=chunk, use_cache=False).logits.detach()


def distill(
    model,
    adapters: dict[int, LatentKVAdapter],
    ids: torch.Tensor,
    steps: int = 200,
    seq_len: int = 256,
    lr: float = 0.02,
    log_every: int = 25,
    seed: int = 0,
    warmup: int = 10,
    patience: int = 40,
) -> list[dict]:
    """Train the adapters to reproduce the frozen model's distribution.

    Only the adapters carry gradients; every Qwen parameter stays frozen,
    which is what keeps this an *inference-time* representation study
    rather than a fine-tune. Initialised from B1, so a failure to improve
    is evidence about the representation rather than about optimisation.

    KL(base || latent) — the direction that asks how much probability
    mass the original places where the compressed model does not, which
    is the right question for a lossy approximation of a fixed model.

    ## `lr` is relative, not absolute

    Adam takes steps of roughly `lr` whatever the parameter's magnitude.
    These parameters come from folding W_kv into an eigenbasis, so their
    entries are small — RMS around 1e-3 — and an absolute lr of 1e-3
    rewrites them completely on the first step. The first real run of
    this did exactly that: KL went 0.395 -> 5.47, an order of magnitude
    worse than the initialisation it started from.

    So `lr` here is a *fraction of each parameter's own RMS*, set per
    parameter group. 0.02 means "move each matrix by about 2% of its own
    scale per step", which is meaningful whatever that scale happens to
    be.

    Two more guards, because a diverged run that still gets scored is
    worse than no run: the best state is kept and restored at the end,
    and training stops early if nothing improves for `patience` steps.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    installed = install_adapters(model, adapters)
    params = [p for p in installed.parameters()]
    for p in params:
        p.requires_grad_(True)
    # Per-parameter lr, proportional to that parameter's own RMS.
    groups = []
    for p_ in params:
        rms = float(p_.detach().pow(2).mean().sqrt().item())
        groups.append({"params": [p_], "lr": lr * max(rms, 1e-8)})
    opt = torch.optim.Adam(groups)
    base_lrs = [g["lr"] for g in opt.param_groups]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    best = {"kl": float("inf"), "step": -1, "state": None}
    since_best = 0

    try:
        total = ids.shape[1]
        for step in range(steps):
            # Linear warmup then cosine decay. Warmup matters most here:
            # the initialisation is already a good solution, so the first
            # few full-size steps are exactly where it can be destroyed.
            if step < warmup:
                scale = (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, steps - warmup)
                scale = 0.5 * (1 + math.cos(math.pi * progress))
            for group, base in zip(opt.param_groups, base_lrs):
                group["lr"] = base * scale

            start = int(torch.randint(0, max(1, total - seq_len), (1,), generator=gen).item())
            chunk = ids[:, start : start + seq_len]

            installed.remove()
            target = baseline_logits(model, chunk)
            install_adapters(model, adapters)

            out = model(input_ids=chunk, use_cache=False).logits
            log_p = torch.log_softmax(target.float(), dim=-1)
            log_q = torch.log_softmax(out.float(), dim=-1)
            loss = (log_p.exp() * (log_p - log_q)).sum(-1).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            value = float(loss.item())
            if value < best["kl"]:
                best = {
                    "kl": value, "step": step,
                    "state": {k: v.detach().clone() for k, v in _adapter_state(adapters).items()},
                }
                since_best = 0
            else:
                since_best += 1

            if step % log_every == 0 or step == steps - 1:
                history.append({"step": step, "kl": value, "lr_scale": scale})
                print(f"    step {step:>4}  KL {value:.5f}  (best {best['kl']:.5f})")
            del target, out, log_p, log_q, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if since_best >= patience:
                print(f"    no improvement for {patience} steps; stopping at {step}")
                break
    finally:
        installed.remove()
        for p in params:
            p.requires_grad_(False)

    # Restore the best state. Scoring whatever the last step happened to
    # produce would report an optimisation accident as a property of the
    # representation.
    if best["state"] is not None:
        _load_adapter_state(adapters, best["state"])
        print(f"    restored best step {best['step']} (KL {best['kl']:.5f})")
    history.append({"step": "best", "kl": best["kl"], "best_step": best["step"]})
    return history


def _adapter_state(adapters: dict[int, LatentKVAdapter]) -> dict:
    return {
        f"{idx}.{name}": param
        for idx, adapter in adapters.items()
        for name, param in adapter.named_parameters()
    }


def _load_adapter_state(adapters: dict[int, LatentKVAdapter], state: dict) -> None:
    with torch.no_grad():
        for idx, adapter in adapters.items():
            for name, param in adapter.named_parameters():
                param.copy_(state[f"{idx}.{name}"].to(param.device))