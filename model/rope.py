"""
Phase 2 — RoPE (rotary position embeddings) for the LatentServe
execution path.

See docs/methodology.md Phase 2. Phase 1 ran Hugging Face's own
`Qwen2RotaryEmbedding` inside HF's attention; from Phase 2 onward
LatentServe drives the layer loop itself (model/latentserve_qwen.py),
so it needs its own RoPE.

Why reimplement instead of calling `hf_model.model.rotary_emb`?

  1. Phase 8 (decoupled positional representation) splits the
     positional path away from the content path. That is a change *to*
     RoPE, so RoPE has to be ours to change.
  2. Phase 11 fuses RoPE into custom Triton/CUDA kernels. You cannot
     fuse a call into `transformers`.
  3. `Qwen2RotaryEmbedding`'s constructor signature and where theta
     lives on the config have both moved across transformers releases
     (4.x keeps `config.rope_theta`; 5.x moved it into
     `config.rope_scaling["rope_theta"]`). Owning ~40 lines of RoPE is
     cheaper than tracking that.

Correctness is not assumed: `tests/test_phase2_gqa.py` asserts this
module's cos/sin match `Qwen2RotaryEmbedding`'s to float32 tolerance,
and `LatentServeQwen(rope_source="hf")` can swap HF's module back in to
bisect a suspected RoPE bug.

Layout note: this follows HF's "rotate_half" convention, where the
cos/sin tables are `head_dim` wide (each frequency duplicated in both
halves) rather than `head_dim // 2`. GPT-NeoX-style interleaving would
give different numerics against the same weights, so the convention
must match the checkpoint, not personal taste.
"""

from __future__ import annotations

from typing import Optional

import torch

# Qwen2.5 ships plain (unscaled) RoPE. YaRN/linear/dynamic scaling change
# the frequencies, so rather than silently computing the wrong thing we
# refuse and tell the caller to use rope_source="hf".
_SUPPORTED_ROPE_TYPES = {None, "default"}


def rope_theta_from_config(config) -> float:
    """Read the RoPE base frequency off an HF config across releases.

    transformers 4.x: `config.rope_theta`.
    transformers 5.x: `config.rope_scaling["rope_theta"]`.
    """
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        scaling = getattr(config, "rope_scaling", None) or {}
        theta = scaling.get("rope_theta")
    if theta is None:
        raise ValueError(
            "could not find rope_theta on the model config "
            f"({type(config).__name__}); pass theta= explicitly"
        )
    return float(theta)


def rope_type_from_config(config) -> Optional[str]:
    scaling = getattr(config, "rope_scaling", None) or {}
    return scaling.get("rope_type") or scaling.get("type")


class RotaryEmbedding:
    """Precomputed cos/sin tables for RoPE.

    Tables are built once at `max_seq_len` and sliced per step, so a
    decode step costs an index_select rather than a transcendental
    recompute. They are held in float32 regardless of activation dtype
    (matching HF) and cast at apply time: on a T4 running fp16, building
    the table in fp16 loses enough mantissa at large positions to shift
    logits measurably, which would then be misattributed to the GQA path.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 1_000_000.0,
        device: str | torch.device = "cuda",
        rope_type: Optional[str] = None,
    ):
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if rope_type not in _SUPPORTED_ROPE_TYPES:
            raise NotImplementedError(
                f"rope_type={rope_type!r} is not implemented in LatentServe's RoPE "
                "(only unscaled/default). Construct LatentServeQwen with "
                'rope_source="hf" to defer to the checkpoint\'s own rotary module.'
            )
        self.head_dim = head_dim
        self.theta = float(theta)
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len

        # inv_freq[i] = 1 / theta^(2i/d), i in [0, d/2)
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32, device=self.device) / head_dim
        self.inv_freq = 1.0 / (self.theta**exponent)

        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self._build_tables(max_seq_len)

    def _build_tables(self, length: int) -> None:
        positions = torch.arange(length, dtype=torch.float32, device=self.device)
        freqs = torch.outer(positions, self.inv_freq)  # [S, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, d] — rotate_half convention
        self._cos = emb.cos()
        self._sin = emb.sin()
        self.max_seq_len = length

    @classmethod
    def from_hf_config(
        cls,
        config,
        max_seq_len: int,
        device: str | torch.device = "cuda",
    ) -> "RotaryEmbedding":
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        return cls(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta_from_config(config),
            device=device,
            rope_type=rope_type_from_config(config),
        )

    def cos_sin(
        self, start_pos: int, length: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for absolute positions [start_pos, start_pos + length),
        shaped [1, 1, length, head_dim] to broadcast over [B, H, S, D]."""
        end = start_pos + length
        if end > self.max_seq_len:
            # Grow rather than fail: the benchmark harness sizes tables from
            # the configured context length, but ad-hoc callers shouldn't
            # have to know that.
            self._build_tables(max(end, self.max_seq_len * 2))
        cos = self._cos[start_pos:end].to(dtype)
        sin = self._sin[start_pos:end].to(dtype)
        return cos[None, None, :, :], sin[None, None, :, :]

    def cos_sin_at(self, positions: torch.Tensor, dtype: torch.dtype):
        """cos/sin for explicit per-sequence positions.

        `positions` is [B, S] of absolute positions, returning
        [B, 1, S, head_dim] to broadcast over [B, H, S, D]. Continuous
        batching (Phase 4) needs this: sequences in one decode batch sit
        at different positions, so a single scalar offset no longer
        describes the batch. Getting this wrong gives every sequence the
        first one's positional phase — plausible output, quietly wrong.
        """
        end = int(positions.max().item()) + 1
        if end > self.max_seq_len:
            self._build_tables(max(end, self.max_seq_len * 2))
        idx = positions.to(self._cos.device)
        return self._cos[idx].to(dtype)[:, None], self._sin[idx].to(dtype)[:, None]

    def nbytes(self) -> int:
        return sum(t.nelement() * t.element_size() for t in (self._cos, self._sin) if t is not None)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors shaped [B, H, S, D].

    q and k carry different head counts under GQA (12 vs 2 for
    Qwen2.5-1.5B) — that is fine, cos/sin broadcast over the head axis.
    """
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out
