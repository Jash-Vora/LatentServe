"""
Phase 2 — LatentServe's GQA attention path.

docs/methodology.md Phase 2: "build the conventional modern inference
baseline around Qwen". This module is that baseline. Every later
attention variant is a drop-in replacement for it and is measured
against it:

    reference.py (Phase 1, HF)  ->  gqa.py (here)  ->  mla.py (Phase 7)
                                                   ->  sparse.py (Phase 14)

The projections are **borrowed, not copied**: `GQAAttention` holds
references to the `q_proj`/`k_proj`/`v_proj`/`o_proj` modules of the
loaded Hugging Face model. Copying them would double weight VRAM (3.1 GB
for Qwen2.5-1.5B in fp16 — a third of a T4) and, worse, would break the
project's core methodological claim: the model is fixed and only the
execution system changes. Identical weight objects make that literally
true rather than approximately true.

Qwen2.5 detail: q/k/v projections carry biases, o_proj does not. Using
the HF `nn.Linear` modules directly means we inherit that for free
instead of re-deriving it.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

from cache.kv_cache import ContiguousKVCache, KVHeadsMode
from model.rope import apply_rope

AttnImpl = Literal["sdpa", "math", "triton_paged"]

# How the KV heads get matched up to the query heads they serve.
#
#   materialize — repeat_kv the cache out to one KV head per query head.
#                 Correct, obvious, and the reason Phase 2's first sweep
#                 measured ~13x the necessary decode memory traffic: the
#                 reshape inside repeat_kv cannot stay a view, so every
#                 decode step copies the whole cache 6x per layer, then
#                 reads the copy. Kept as a config option because the
#                 before/after is a result worth reporting, not just a
#                 bug worth deleting.
#   fold        — on the decode path, reinterpret the group of query
#                 heads sharing a KV head as extra *queries* against the
#                 un-expanded cache: [B, 12, 1, D] -> [B, 2, 6, D] against
#                 K/V of [B, 2, S, D]. Identical arithmetic, no copy, and
#                 the cache is read exactly once.
KVExpansion = Literal["materialize", "fold"]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, kv_heads, S, D] -> [B, kv_heads * n_rep, S, D].

    expand+reshape rather than repeat_interleave: the expand is a stride
    trick (no copy), and the reshape materialises exactly once. This is
    the GQA "share one KV head across a group of query heads" step, and
    for Qwen2.5-1.5B n_rep is 6.

    Note what this does *not* save. GQA shrinks the cache (2 heads
    stored, not 12) and therefore the bytes read from DRAM, but the
    attention FLOPs are unchanged: every one of the 12 query heads still
    attends over the full sequence. GQA is a memory optimization wearing
    an attention-shaped hat, which is exactly why it composes with
    DSA-style sparsity (Phase 14) — that one attacks the FLOPs.
    """
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def build_causal_mask(
    q_len: int, kv_len: int, start_pos: int, device: torch.device
) -> Optional[torch.Tensor]:
    """Boolean keep-mask [1, 1, q_len, kv_len] for a query block whose
    first row sits at absolute position `start_pos`.

    Returns None for the q_len == 1 case: a single decode query attends
    to every cached position, so there is nothing to mask and building a
    mask would add a kernel launch plus a tensor to the hottest loop in
    the system.
    """
    if q_len == 1:
        return None
    q_pos = torch.arange(start_pos, start_pos + q_len, device=device)[:, None]
    kv_pos = torch.arange(kv_len, device=device)[None, :]
    return (kv_pos <= q_pos)[None, None, :, :]


class GQAAttention(nn.Module):
    """Single-layer grouped-query attention over a LatentServe KV cache."""

    def __init__(
        self,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        o_proj: nn.Module,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        kv_heads_mode: KVHeadsMode = "native",
        attn_impl: AttnImpl = "sdpa",
        kv_expansion: KVExpansion = "fold",
    ):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = q_proj, k_proj, v_proj, o_proj
        # Phase 14a: q/k/v in one projection. Built by `fuse_projections`;
        # None until then, and `use_fused` selects the path per call.
        self._qkv_weight: Optional[torch.Tensor] = None
        self._qkv_bias: Optional[torch.Tensor] = None
        self._qkv_split: Optional[tuple[int, int, int]] = None
        self.use_fused = False
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.kv_heads_mode = kv_heads_mode
        self.attn_impl = attn_impl
        self.kv_expansion = kv_expansion
        # None lets the kernel choose per call; the Phase 13 graph runner
        # pins it per bucket, since a captured grid cannot change size.
        self.num_splits: Optional[int] = None
        self.scaling = head_dim**-0.5

    # ------------------------------------------------------------------

    def _project_kv_for_cache(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the KVHeadsMode transform on the way into the cache.

        Done *after* RoPE, so mha_sim stores rotated keys identical to
        what native would compute and then repeat — that identity is
        what makes mha_sim a pure memory-traffic experiment. Rotating
        after expanding would give the same values here but would waste
        6x the RoPE work, and would stop being equivalent the moment
        Phase 8 decouples the positional path.
        """
        if self.kv_heads_mode == "native":
            return k, v
        if self.kv_heads_mode == "mha_sim":
            n_rep = self.num_attention_heads // self.num_key_value_heads
            return repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        if self.kv_heads_mode == "mqa_sim":
            # Mean-pool to a single KV head. Changes the numerics — probe
            # for memory/bandwidth scaling only, never a quality claim.
            return k.mean(dim=1, keepdim=True), v.mean(dim=1, keepdim=True)
        raise ValueError(f"unknown kv_heads_mode {self.kv_heads_mode!r}")

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        if self.attn_impl == "sdpa":
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=is_causal, scale=self.scaling
            )
        # Explicit math path: slower and memory-hungry (it materialises the
        # full [B, H, q_len, kv_len] score matrix), kept as the numerical
        # reference that Phase 11's Triton/CUDA kernels are checked against.
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if is_causal and mask is None:
            q_len, kv_len = q.shape[2], k.shape[2]
            mask = build_causal_mask(q_len, kv_len, kv_len - q_len, q.device)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(probs, v)

    # ------------------------------------------------------------------

    def fuse_projections(self) -> None:
        """Merge q/k/v into one projection, sharing the checkpoint's memory.

        Qwen2.5's q/k/v carry biases, and those fuse too, so the merged
        projection is still a single fused matmul-plus-bias. See
        model/fused.py for why the originals are re-pointed rather than
        copied.
        """
        from model.fused import concat_into_views

        mods = [self.q_proj, self.k_proj, self.v_proj]
        self._qkv_weight = concat_into_views(mods, "weight")
        self._qkv_bias = concat_into_views(mods, "bias")
        self._qkv_split = tuple(m.out_features for m in mods)
        self.use_fused = True

    def _project_qkv(self, hidden_states: torch.Tensor):
        if self.use_fused and self._qkv_weight is not None:
            return F.linear(hidden_states, self._qkv_weight, self._qkv_bias).split(
                self._qkv_split, dim=-1
            )
        return self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

    def _pool(self, cache, name: str):
        """This layer's slice of an optional scale pool.

        Written as a method because the inline `getattr(cache, name,
        [None] * 99)[layer]` it replaces allocated a 99-element list four
        times per layer — 112 throwaway lists per decode step. Small
        individually, and this path is already dominated by per-layer
        Python cost.
        """
        pool = getattr(cache, name, None)
        return pool[self.layer_idx] if pool else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: ContiguousKVCache,
        start_pos: int,
    ) -> torch.Tensor:
        """hidden_states: [B, S, hidden]. S == 1 on the decode path,
        S == prompt length (or chunk length) on the prefill path.

        `start_pos` is the absolute position of the first token in this
        call — the single piece of state that keeps RoPE, the causal
        mask, and the cache write offset in agreement. Phase 1's Gate 1
        (teacher-forced vs. incremental logits) is exactly the test that
        catches getting it wrong.
        """
        b, s, _ = hidden_states.shape

        q, k, v = self._project_qkv(hidden_states)
        # reshape rather than view: the fused path's outputs are slices of
        # one wider tensor, so they are not contiguous across `s`. Only the
        # last dimension is being split into (heads, head_dim), which a
        # slice still supports without a copy.
        q = q.reshape(b, s, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, s, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, s, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        k, v = self._project_kv_for_cache(k, v)
        cache.write(self.layer_idx, k, v, start_pos)

        # The kernel branch has to come *before* cache.read(). Placed
        # after it, every layer gathered the whole cache into an fp16
        # buffer and the kernel then ignored it and re-read the pool —
        # paying the gather this phase exists to remove, plus the kernel.
        # torch.profiler showed it plainly: 56 `aten::index` calls per
        # decode step, two per layer, on a path that should have none.
        #
        # It also comes before `padding_mask()`, and no longer requires the
        # batch to be uniform. The kernel masks each sequence by its own
        # length from `seq_lens`, so it handles ragged batches natively —
        # the Triton tests already check lengths [130, 48] — while
        # padding_mask() builds a tensor on the host per layer for the
        # gather path's benefit. That was a correctness-shaped caution
        # that kept continuous batching off the kernel entirely.
        if (
            s == 1
            and self.attn_impl == "triton_paged"
            and hasattr(cache, "block_tables_tensor")
        ):
            from kernels.gqa.paged_decode import paged_decode_attention

            cached_kv_heads = k.shape[1]
            n_rep = self.num_attention_heads // cached_kv_heads
            q_folded = q.reshape(b, cached_kv_heads, n_rep, self.head_dim)
            out = paged_decode_attention(
                q_folded,
                cache.k_pool[self.layer_idx],
                cache.v_pool[self.layer_idx],
                cache.block_tables_tensor(b),
                cache.seq_lens_tensor(b),
                k_scale=self._pool(cache, "k_scale_pool"),
                v_scale=self._pool(cache, "v_scale_pool"),
                k_zero=self._pool(cache, "k_zero_pool"),
                v_zero=self._pool(cache, "v_zero_pool"),
                softmax_scale=self.scaling,
                max_seq_len=cache.max_len,
                # Fixed per captured graph. The kernel derives pages per
                # split on the device from seq_lens, so a fixed count
                # serves a sequence that grows within its bucket.
                num_splits=self.num_splits,
            )
            attn_out = out.reshape(b, self.num_attention_heads, 1, self.head_dim)
            attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)
            return self.o_proj(attn_out)

        # Ask the cache how much it holds rather than deriving it from
        # start_pos. In a ragged batch (Phase 4) rows sit at different
        # positions, so `start_pos + s` describes no one; and since
        # advance() now runs before the layer loop, the cache's own length
        # is already correct for the uniform case too.
        k_all, v_all = cache.read(self.layer_idx, b)
        kv_len = k_all.shape[2]

        # A paged cache pads ragged batches to the longest sequence; those
        # pad slots hold another sequence's tokens and must be masked.
        # Uniform batches (all of Phase 2 and 3's benchmarks) get None.
        # Only the gather path needs this, which is why it sits after the
        # kernel branch rather than before it.
        key_mask = cache.padding_mask()
        if key_mask is not None and s > 1:
            raise NotImplementedError(
                "ragged prefill needs a combined causal+padding mask; Phase 3 "
                "benchmarks uniform batches and ragged execution lands in Phase 4"
            )

        cached_kv_heads = k_all.shape[1]
        n_rep = self.num_attention_heads // cached_kv_heads

        if s == 1 and n_rep > 1 and self.kv_expansion == "fold":
            # Decode: no mask is needed (one query attends to everything),
            # so the group axis can be folded into the query axis and the
            # cache used as-is. This is the difference between reading the
            # KV cache once and copying it 6x per layer per token.
            q_folded = q.reshape(b, cached_kv_heads, n_rep, self.head_dim)
            out = self._attend(q_folded, k_all, v_all, mask=key_mask, is_causal=False)
            attn_out = out.reshape(b, self.num_attention_heads, 1, self.head_dim)
            attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)
            return self.o_proj(attn_out)

        # Prefill, or the materialize ablation. Prefill is compute-bound
        # (attention is O(S^2) there), so the expansion copy is a much
        # smaller share of the cost and the folded mask would itself be
        # large — [n_rep * q_len, kv_len] booleans.
        k_all = repeat_kv(k_all, n_rep)
        v_all = repeat_kv(v_all, n_rep)

        if s == 1:
            attn_out = self._attend(q, k_all, v_all, mask=key_mask, is_causal=False)
        elif start_pos == 0:
            # Square causal block — let SDPA generate the mask internally
            # (its fused path avoids materialising one).
            attn_out = self._attend(q, k_all, v_all, mask=None, is_causal=True)
        else:
            # Chunked prefill: rectangular block, q attends to the whole
            # prefix plus itself. is_causal=True would align the mask to
            # the wrong corner, so build it explicitly.
            mask = build_causal_mask(s, kv_len, start_pos, hidden_states.device)
            attn_out = self._attend(q, k_all, v_all, mask=mask, is_causal=False)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(attn_out)
