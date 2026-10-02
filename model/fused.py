"""
Phase 14a — projection fusion.

## Why

Phase 13 put LatentServe within 2-5% of vLLM on batch-1 decode, with the
remaining gap outside attention. vLLM runs one projection for q/k/v and
one for gate/up, where LatentServe ran three and two. At batch 1 every
projection is a matrix-vector product that streams its weights once and
does very little arithmetic per byte, so the cost of a separate launch —
its fixed GPU time, its prologue, the serialisation against the next
kernel — is a real share of the work. CUDA graphs removed the *CPU* cost
of those launches. They did not remove the launches.

Merging them is 84 fewer kernels per decode step across 28 layers: two
fewer per layer in attention (q, k, v -> one) and one fewer in the MLP
(gate, up -> one).

## No extra memory

The obvious implementation keeps the original weights and a concatenated
copy: ~1.7 GB more for this model, most of it the MLP. Instead, the
concatenated tensor is built once and the checkpoint's own parameters are
re-pointed at slices of it. One copy of every weight, as before; the
original modules still work, reading the same memory.

That also means the unfused path stays live. `use_fused` switches between
them per call, so fused and unfused can be measured in one process on one
machine — which removes host-to-host noise from the comparison. Phase 13
measured that noise at 4-8%, larger than the effect being looked for.

## Graphs

A captured graph bakes in whichever path was active at capture. Switch
`use_fused` before capturing, not after: a graph captured unfused replays
unfused regardless of the flag.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def concat_into_views(modules: list[nn.Module], attr: str) -> torch.Tensor | None:
    """Concatenate `attr` of each module along dim 0 and re-point each
    module's parameter at its slice of the result.

    Returns the fused tensor, or None if none of the modules has `attr`
    (an MLP with no biases, say). Mixed presence is refused: fusing a
    biased projection with an unbiased one would need a zero bias
    invented for the latter, which is a silent change to the model.
    """
    params = [getattr(m, attr, None) for m in modules]
    if all(p is None for p in params):
        return None
    if any(p is None for p in params):
        raise ValueError(f"cannot fuse: only some modules have a {attr!r}")
    with torch.no_grad():
        fused = torch.cat([p.data for p in params], dim=0)
        offset = 0
        for p in params:
            n = p.shape[0]
            # Re-point rather than copy: the original storage is released
            # (one layer at a time, so peak memory stays at one layer's
            # worth of extra), and every existing reader — the HF reference
            # model, Phase 7's compression hooks — sees identical values at
            # the new address.
            p.data = fused[offset : offset + n]
            offset += n
    return fused


class FusedMLP(nn.Module):
    """Qwen2's SwiGLU MLP with gate and up in one projection.

    Wraps the checkpoint's own MLP rather than replacing it, so the
    unfused path is one flag away and the parameters are still the
    checkpoint's — `down_proj` is used as-is.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()
        act = getattr(mlp, "act_fn", None)
        # The fused path hard-codes silu, so refuse a checkpoint whose MLP
        # uses anything else rather than computing a different function.
        if act is not None and "silu" not in type(act).__name__.lower():
            raise ValueError(f"FusedMLP assumes SwiGLU (silu); got {act!r}")
        self.inner = mlp
        self.intermediate = mlp.gate_proj.out_features
        self.weight = concat_into_views([mlp.gate_proj, mlp.up_proj], "weight")
        self.bias = concat_into_views([mlp.gate_proj, mlp.up_proj], "bias")
        self.use_fused = True
        # Phase 14a step 2: SiLU and the multiply by `up` in one kernel,
        # reading the fused gate/up output directly.
        self.fused_act = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_fused:
            return self.inner(x)
        gate_up = F.linear(x, self.weight, self.bias)
        if self.fused_act:
            from kernels.fused_elementwise import silu_and_mul

            return self.inner.down_proj(silu_and_mul(gate_up))
        gate, up = gate_up.split(self.intermediate, dim=-1)
        return self.inner.down_proj(F.silu(gate) * up)
