"""
Phase 2 — LatentServe's own execution path for Qwen2.5-1.5B-Instruct.

Phase 1 instrumented Hugging Face. This module *replaces* it: it drives
the decoder layer loop itself, calling LatentServe's GQA attention
(model/attention/gqa.py) against LatentServe's KV cache
(cache/kv_cache.py), while borrowing the loaded checkpoint's weight
modules unchanged. The model is fixed; the execution system is the
experimental variable (docs/methodology.md, "Model Strategy").

## Why own the layer loop instead of monkey-patching HF attention

Patching `Qwen2Attention.forward` would be less code today and a
liability for the next 20 weeks:

  * Every later phase needs control *outside* attention. Chunked
    prefill, continuous batching (Phase 4), prefix caching (Phase 13)
    and the adaptive backend policy (Phase 17) are all decisions made in
    the layer loop or above it, not inside one attention module.
  * HF's attention signature is a moving target — `past_key_value` ->
    `past_key_values`, `position_ids` -> `position_embeddings`,
    `Cache.key_cache` -> `Cache.layers[i].keys`. `model/qwen.py` already
    carries three compatibility branches in `kv_cache_bytes()` for this.
    The loop below touches only module *attributes* (`embed_tokens`,
    `input_layernorm`, `self_attn.q_proj`, `mlp`, `norm`, `lm_head`),
    which have been stable across the 4.x/5.x releases.
  * Nsight timelines (Phase 12) are readable in proportion to how much
    of the stack is ours.

## The prefill memory result this makes available immediately

HF's causal-LM forward computes logits for *every* prefill position:
[B, S, 151936]. At S=8192 in fp16 that tensor alone is 2.4 GB, and
transformers has historically upcast it to float32 (another 4.8 GB)
before returning. That is why `results/raw/phase1_reference.jsonl`
records 10.65 GB peak VRAM at 8K against a KV cache of only 227 MB, and
it is the most likely reason the 16K point is missing from that sweep.

Only the last position's logits are needed to sample the next token, so
`prefill()` below slices the hidden state to [-1:] *before* the final
norm and `lm_head`. RMSNorm is per-token, so this is numerically exact,
not an approximation. Expect peak VRAM at long context to drop by
gigabytes, and expect that to be a memory-allocation effect with no
bearing on TPOT — worth stating as a prediction before measuring it
(docs/phase2.md).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from cache.kv_cache import ContiguousKVCache, KVCacheSpec, KVHeadsMode, effective_kv_heads
from model.attention.gqa import AttnImpl, GQAAttention, KVExpansion
from model.qwen import ModelShape, QwenReference, TimedGenerationResult
from model.rope import RotaryEmbedding


@dataclass
class _Layer:
    """The pieces of one Qwen2 decoder layer we drive directly. The
    norms and MLP are the checkpoint's own modules, untouched — only
    attention is LatentServe's."""

    input_layernorm: nn.Module
    attn: GQAAttention
    post_attention_layernorm: nn.Module
    mlp: nn.Module


class LatentServeQwen:
    """Phase 2 execution path: GQA + preallocated contiguous KV cache.

    Mirrors `QwenReference`'s API (`prefill` / `decode_step` /
    `generate_greedy` / `generate_with_timing`) on purpose, so the
    benchmark runner and the correctness tests can drive either one
    through the same code path and any difference is attributable to the
    execution system rather than to the harness.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer,
        shape: ModelShape,
        device: str = "cuda",
        kv_heads_mode: KVHeadsMode = "native",
        attn_impl: AttnImpl = "sdpa",
        kv_expansion: KVExpansion = "fold",
        rope_source: str = "latentserve",
        max_seq_len_hint: int = 4096,
    ):
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.shape = shape
        self.device = torch.device(device)
        self.kv_heads_mode = kv_heads_mode
        self.attn_impl = attn_impl
        self.kv_expansion = kv_expansion
        self.rope_source = rope_source

        config = hf_model.config
        if getattr(config, "use_sliding_window", False):
            raise NotImplementedError(
                "this checkpoint enables sliding-window attention; LatentServe's "
                "Phase 2 GQA path implements full causal attention only, and "
                "running it anyway would silently change the model's math"
            )

        base = hf_model.model  # Qwen2Model
        self.embed_tokens = base.embed_tokens
        self.final_norm = base.norm
        self.lm_head = hf_model.lm_head
        self.hf_rotary = getattr(base, "rotary_emb", None)

        self.layers: list[_Layer] = []
        for idx, hf_layer in enumerate(base.layers):
            attn = GQAAttention(
                q_proj=hf_layer.self_attn.q_proj,
                k_proj=hf_layer.self_attn.k_proj,
                v_proj=hf_layer.self_attn.v_proj,
                o_proj=hf_layer.self_attn.o_proj,
                num_attention_heads=shape.num_attention_heads,
                num_key_value_heads=shape.num_key_value_heads,
                head_dim=shape.head_dim,
                layer_idx=idx,
                kv_heads_mode=kv_heads_mode,
                attn_impl=attn_impl,
                kv_expansion=kv_expansion,
            )
            self.layers.append(
                _Layer(
                    input_layernorm=hf_layer.input_layernorm,
                    attn=attn,
                    post_attention_layernorm=hf_layer.post_attention_layernorm,
                    mlp=hf_layer.mlp,
                )
            )

        if rope_source == "latentserve":
            self.rope = RotaryEmbedding.from_hf_config(
                config, max_seq_len=max_seq_len_hint, device=self.device
            )
        elif rope_source == "hf":
            if self.hf_rotary is None:
                raise ValueError('rope_source="hf" but this model exposes no model.rotary_emb')
            self.rope = None
        else:
            raise ValueError(f"unknown rope_source {rope_source!r}")

        self.cache: Optional[ContiguousKVCache] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_reference(cls, ref: QwenReference, **kwargs) -> "LatentServeQwen":
        """Build from an already-loaded `QwenReference`. Both objects
        then point at the *same* weight tensors, so a test can compare
        LatentServe against the Phase 1 oracle without loading 3 GB
        twice or risking a different checkpoint revision."""
        if ref.model is None or ref.shape is None:
            raise RuntimeError("QwenReference.load() must be called first")
        return cls(
            hf_model=ref.model,
            tokenizer=ref.tokenizer,
            shape=ref.shape,
            device=ref.device,
            **kwargs,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def weight_bytes(self) -> int:
        """Total parameter bytes. Read in full on every decode step, so
        it is the other half of the decode bandwidth story alongside the
        KV cache — see benchmarks/runners/phase2_gqa.py."""
        return sum(p.nelement() * p.element_size() for p in self.hf_model.parameters())

    # ------------------------------------------------------------------
    # Cache lifecycle
    # ------------------------------------------------------------------

    def cache_spec(self, batch_size: int, max_seq_len: int) -> KVCacheSpec:
        return KVCacheSpec(
            num_layers=self.shape.num_layers,
            num_kv_heads=effective_kv_heads(
                self.shape.num_key_value_heads,
                self.shape.num_attention_heads,
                self.kv_heads_mode,
            ),
            head_dim=self.shape.head_dim,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            dtype=self.dtype,
            device=str(self.device),
        )

    def allocate_cache(self, batch_size: int, max_seq_len: int) -> ContiguousKVCache:
        """Allocate once, up front, for input + output tokens. Reused
        across trials via `reset()` so the benchmark measures steady
        state rather than the allocator."""
        self.cache = ContiguousKVCache(self.cache_spec(batch_size, max_seq_len))
        return self.cache

    def _require_cache(self) -> ContiguousKVCache:
        if self.cache is None:
            raise RuntimeError("call allocate_cache(batch_size, max_seq_len) first")
        return self.cache

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _cos_sin(self, start_pos: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope is not None:
            return self.rope.cos_sin(start_pos, length, self.dtype)
        # rope_source="hf": bisection aid — same loop, HF's frequencies.
        positions = torch.arange(start_pos, start_pos + length, device=self.device)[None, :]
        dummy = torch.zeros(1, length, 1, device=self.device, dtype=self.dtype)
        cos, sin = self.hf_rotary(dummy, positions)
        return cos[:, None, :, :].to(self.dtype), sin[:, None, :, :].to(self.dtype)

    @torch.no_grad()
    def _forward_block(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Run one block of tokens through every layer, updating the
        cache. Returns the final hidden states [B, S, hidden] *before*
        the final norm — the caller decides how many positions are worth
        projecting to vocabulary."""
        cache = self._require_cache()
        h = self.embed_tokens(input_ids)
        cos, sin = self._cos_sin(start_pos, input_ids.shape[1])

        for layer in self.layers:
            residual = h
            h = layer.input_layernorm(h)
            h = layer.attn(h, cos, sin, cache, start_pos)
            h = residual + h

            residual = h
            h = layer.post_attention_layernorm(h)
            h = layer.mlp(h)
            h = residual + h

        cache.advance(input_ids.shape[1])
        return h

    def _to_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(hidden))

    # ------------------------------------------------------------------
    # Public step API (mirrors QwenReference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prefill(
        self, input_ids: torch.Tensor, chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """Process the prompt and return last-position logits [B, 1, V].

        `chunk_size` splits the prompt into sequential blocks. Attention
        cost is unchanged (each chunk still attends over the whole
        prefix) but the peak activation footprint falls from O(S) to
        O(chunk), which is what makes 32K+ prompts reachable on a 16 GB
        T4. It also previews the chunked-prefill scheduling decision
        Phase 4 has to make for real.
        """
        seq_len = input_ids.shape[1]
        step = chunk_size or seq_len
        hidden = None
        for start in range(0, seq_len, step):
            block = input_ids[:, start : start + step]
            hidden = self._forward_block(block, start_pos=start)
        return self._to_logits(hidden[:, -1:, :])

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor) -> torch.Tensor:
        """One token per sequence. token_ids: [B, 1] -> logits [B, 1, V]."""
        cache = self._require_cache()
        hidden = self._forward_block(token_ids, start_pos=cache.length)
        return self._to_logits(hidden)

    @torch.no_grad()
    def forward_logits_all(
        self, input_ids: torch.Tensor, chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """Logits at *every* position [B, S, V]. Correctness-harness only
        — this is the tensor `prefill()` deliberately avoids
        materialising. Used by tests/test_phase2_gqa.py to compare
        against `QwenReference.forward_teacher_forced()`."""
        seq_len = input_ids.shape[1]
        step = chunk_size or seq_len
        outs = []
        for start in range(0, seq_len, step):
            block = input_ids[:, start : start + step]
            hidden = self._forward_block(block, start_pos=start)
            outs.append(self._to_logits(hidden))
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        cache = self._require_cache()
        cache.reset()
        logits = self.prefill(input_ids, chunk_size=chunk_size)
        next_id = logits[:, -1:, :].argmax(dim=-1)
        out = [next_id]
        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(next_id)
            next_id = logits[:, -1:, :].argmax(dim=-1)
            out.append(next_id)
        return torch.cat(out, dim=1)

    # ------------------------------------------------------------------
    # Benchmark primitive
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_with_timing(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        chunk_size: Optional[int] = None,
    ) -> TimedGenerationResult:
        """Same shape of result as `QwenReference.generate_with_timing`,
        so Phase 1 and Phase 2 rows are computed by identical code and
        differ only in the system under test.

        `kv_cache_mb` here is exact (the cache knows its own bytes)
        rather than derived by walking HF `past_key_values` tensors.
        """
        cache = self._require_cache()
        cache.reset()
        batch = input_ids.shape[0]
        is_cuda = self.device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)

        t0 = time.perf_counter()
        logits = self.prefill(input_ids, chunk_size=chunk_size)
        if is_cuda:
            torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

        next_id = logits[:, -1:, :].argmax(dim=-1)
        out_ids = [next_id]
        decode_step_ms: list[float] = []
        for _ in range(max_new_tokens - 1):
            t0 = time.perf_counter()
            logits = self.decode_step(next_id)
            if is_cuda:
                torch.cuda.synchronize()
            decode_step_ms.append((time.perf_counter() - t0) * 1000)
            next_id = logits[:, -1:, :].argmax(dim=-1)
            out_ids.append(next_id)

        peak_vram_mb = (
            torch.cuda.max_memory_allocated(self.device) / 1024 / 1024 if is_cuda else 0.0
        )

        return TimedGenerationResult(
            input_ids=input_ids,
            output_ids=torch.cat(out_ids, dim=1),
            prefill_ms=prefill_ms,
            decode_step_ms=decode_step_ms,
            peak_vram_mb=peak_vram_mb,
            kv_cache_mb=cache.used_bytes(batch) / 1024 / 1024,
        )

    # ------------------------------------------------------------------

    def describe(self) -> str:
        spec = self.cache.spec if self.cache else None
        return (
            f"LatentServeQwen(layers={self.shape.num_layers}, "
            f"q_heads={self.shape.num_attention_heads}, "
            f"kv_heads={self.shape.num_key_value_heads} "
            f"(group={self.shape.gqa_group_size}), mode={self.kv_heads_mode}, "
            f"attn={self.attn_impl}, kv_expansion={self.kv_expansion}, "
            f"rope={self.rope_source}, "
            f"cache={spec.describe() if spec else 'unallocated'})"
        )
