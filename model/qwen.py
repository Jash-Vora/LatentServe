"""
Phase 1 — Qwen2.5-1.5B-Instruct reference + correctness baseline.

See docs/methodology.md, Phase 1. This module establishes the Hugging
Face implementation as the correctness oracle for the whole project and
instruments it for the metrics every later phase is judged against:
model load time, prefill latency, decode latency, TTFT, TPOT, E2E
latency, throughput, peak VRAM, and KV-cache memory.

Important framing (see docs methodology "Model Strategy"): at this
phase, "LatentServe" *is* the Hugging Face model. `QwenReference` below
adds timing/memory instrumentation and an explicit prefill/decode step
API around `transformers.AutoModelForCausalLM`, but changes no
computation — no custom attention, no custom KV-cache layout yet. That
starts in Phase 2 (GQA + KV cache), which will replace pieces of the
execution path around these same fixed weights. Phase 1's job is to
nail down ground-truth outputs and a trustworthy measurement harness
so that every later divergence (Phase 2 onward) can be attributed to a
specific change instead of measurement noise.

Usage:

    from model.qwen import QwenReference

    ref = QwenReference(dtype="fp16", device="cuda").load()
    result = ref.generate_with_timing(prompt="Hello,", max_new_tokens=32)
    print(result.ttft_ms, result.tpot_ms, result.e2e_latency_ms)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass
class ModelShape:
    """Architecture facts read off the loaded HF config.

    This is deliberately *not* declared in config.py (see config.py's
    module docstring: ModelConfig has no layers/heads/hidden_dim knobs
    because Qwen2.5-1.5B-Instruct is a fixed checkpoint, not a scratch
    architecture spec). Phase 2's GQA work needs
    num_attention_heads vs num_key_value_heads, so this class is the
    single place that introspection happens.
    """

    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    max_position_embeddings: int
    torch_dtype: str

    @property
    def gqa_group_size(self) -> int:
        """Query heads sharing each KV head. 1 == plain MHA."""
        if self.num_key_value_heads == 0:
            raise ValueError("num_key_value_heads must be > 0")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) not divisible by "
                f"num_key_value_heads ({self.num_key_value_heads}) — unexpected for GQA"
            )
        return self.num_attention_heads // self.num_key_value_heads

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Theoretical full (non-latent) KV-cache bytes for one token,
        summed across all layers. This is the Phase 2 GQA baseline that
        MLA (Phase 7+) is measured against — see docs/methodology.md
        Phase 9 (Question 1 — Memory)."""
        return 2 * self.num_layers * self.num_key_value_heads * self.head_dim * dtype_bytes


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def kv_cache_bytes(past_key_values) -> int:
    """Actual measured KV-cache size from a HF `past_key_values` object,
    as opposed to `ModelShape.kv_bytes_per_token`'s theoretical estimate.
    Both numbers should agree (that agreement is itself a Phase 1
    correctness check) but this one is ground truth from real tensors.

    HF's Cache object has been reshaped more than once across
    `transformers` releases, so this deliberately doesn't assume a
    single layout:
      - newest releases: `Cache.layers` is a list of per-layer cache
        objects (e.g. `DynamicLayer`) exposing `.keys` / `.values`
        tensors (either of which may still be `None` for a layer that
        hasn't been written to, e.g. an empty cache).
      - transformers >=4.40,<~4.56: `Cache` exposes flat
        `.key_cache` / `.value_cache` lists of tensors.
      - older/back-compat: `past_key_values` iterates as a legacy
        tuple-of-tuples, `((k0, v0), (k1, v1), ...)`.
    """
    if past_key_values is None:
        return 0

    total = 0

    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        for layer in layers:
            for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                if tensor is not None:
                    total += tensor.nelement() * tensor.element_size()
        return total

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for k in past_key_values.key_cache:
            if k is not None:
                total += k.nelement() * k.element_size()
        for v in past_key_values.value_cache:
            if v is not None:
                total += v.nelement() * v.element_size()
        return total

    for layer_kv in past_key_values:
        for tensor in layer_kv:
            if tensor is not None:
                total += tensor.nelement() * tensor.element_size()
    return total


@dataclass
class TimedGenerationResult:
    """Raw timing/memory data for one generate_with_timing() call.
    Maps directly onto benchmarks/schema.py::BenchmarkResult fields —
    see benchmarks/runners/phase1_reference.py for that translation."""

    input_ids: torch.Tensor
    output_ids: torch.Tensor
    prefill_ms: float
    decode_step_ms: list = field(default_factory=list)
    peak_vram_mb: float = 0.0
    kv_cache_mb: float = 0.0
    load_ms: float = 0.0

    @property
    def input_tokens(self) -> int:
        return int(self.input_ids.shape[-1])

    @property
    def output_tokens(self) -> int:
        """Total generated tokens. Note this is len(decode_step_ms) + 1,
        not len(decode_step_ms): the first output token comes from the
        prefill step, not a decode_step call, so decode_step_ms alone
        undercounts by exactly one."""
        return int(self.output_ids.shape[-1])

    @property
    def ttft_ms(self) -> float:
        """Time to first token == prefill latency (single-request, no
        queueing yet — queueing arrives with the scheduler in Phase 4)."""
        return self.prefill_ms

    @property
    def tpot_ms(self) -> float:
        """Mean time per output token, excluding the prefill step."""
        if not self.decode_step_ms:
            return 0.0
        return sum(self.decode_step_ms) / len(self.decode_step_ms)

    @property
    def e2e_latency_ms(self) -> float:
        return self.prefill_ms + sum(self.decode_step_ms)

    @property
    def throughput_tokens_sec(self) -> float:
        total_s = self.e2e_latency_ms / 1000.0
        if total_s <= 0 or self.output_tokens == 0:
            return 0.0
        return self.output_tokens / total_s

    def decode_percentiles_ms(self) -> dict:
        """p50/p95/p99 of per-token decode latency. Needs a handful of
        decode steps to be meaningful — see Phase 5 benchmark harness."""
        if not self.decode_step_ms:
            return {"p50": None, "p95": None, "p99": None}
        xs = sorted(self.decode_step_ms)

        def pct(p: float) -> float:
            idx = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
            return xs[idx]

        return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}


class QwenReference:
    """Thin, instrumented wrapper around Hugging Face's
    Qwen2.5-1.5B-Instruct. See module docstring for what this is (and
    is not) at Phase 1."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dtype: str = "fp16",
        device: str = "cuda",
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be one of {list(_DTYPE_MAP)}, got {dtype!r}")
        self.model_name = model_name
        self.device = device
        self.dtype_str = dtype
        self.torch_dtype = _DTYPE_MAP[dtype]
        self.revision = revision
        self.trust_remote_code = trust_remote_code

        self.tokenizer = None
        self.model = None
        self.shape: Optional[ModelShape] = None
        self.load_ms: float = 0.0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "QwenReference":
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.load_ms = (time.perf_counter() - t0) * 1000

        cfg = self.model.config
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.shape = ModelShape(
            num_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            torch_dtype=str(self.torch_dtype),
        )
        return self

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("QwenReference.load() must be called before use")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def encode(self, prompt: str) -> torch.Tensor:
        self._require_loaded()
        enc = self.tokenizer(prompt, return_tensors="pt")
        return enc["input_ids"].to(self.device)

    def synthesize_input_ids(self, num_tokens: int, seed: int = 0) -> torch.Tensor:
        """Build an input_ids tensor of an *exact* token length by
        sampling real vocabulary ids (excluding special tokens where
        identifiable), for the Phase 1 context-length sweep
        (1 / 16 / 1K / 4K / 8K / 16K+ — see docs/methodology.md Phase 1).
        We don't use natural text here because natural text token
        counts are awkward to hit exactly; a fixed prompt length matters
        more than prompt semantics for latency/memory measurement.
        """
        self._require_loaded()
        gen = torch.Generator().manual_seed(seed)
        vocab_size = self.shape.vocab_size if self.shape else self.model.config.vocab_size
        special_ids = set(self.tokenizer.all_special_ids or [])
        # Oversample and filter special tokens out, then trim/pad to exact length.
        ids = []
        while len(ids) < num_tokens:
            batch = torch.randint(0, vocab_size, (num_tokens * 2,), generator=gen).tolist()
            ids.extend(i for i in batch if i not in special_ids)
        ids = ids[:num_tokens]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Correctness-harness primitives (Phase 1 Gate 1: "Can Qwen generate
    # correctly?")
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_teacher_forced(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward pass, no KV cache. This is the ground
        truth every incremental (cached) path is checked against."""
        self._require_loaded()
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits

    @torch.no_grad()
    def forward_incremental(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Feed the sequence one token at a time through the KV cache
        and return the concatenated logits. If attention masking, RoPE
        position ids, and KV-cache updates are correct, this must match
        `forward_teacher_forced` within numerical tolerance (see
        tests/test_phase1_correctness.py) — that agreement is Phase 1's
        Gate 1."""
        self._require_loaded()
        past = None
        all_logits = []
        seq_len = input_ids.shape[-1]
        for t in range(seq_len):
            step_ids = input_ids[:, t : t + 1]
            out = self.model(input_ids=step_ids, past_key_values=past, use_cache=True)
            all_logits.append(out.logits)
            past = out.past_key_values
        return torch.cat(all_logits, dim=1)

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor):
        """Single multi-token prefill step. Returns (logits, past_key_values)."""
        self._require_loaded()
        out = self.model(input_ids=input_ids, use_cache=True)
        return out.logits, out.past_key_values

    @torch.no_grad()
    def decode_step(self, next_token_ids: torch.Tensor, past_key_values):
        """Single-token decode step given existing KV cache."""
        self._require_loaded()
        out = self.model(input_ids=next_token_ids, past_key_values=past_key_values, use_cache=True)
        return out.logits, out.past_key_values

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Deterministic (argmax) generation, no timing. Used by the
        determinism check: two calls with the same input must produce
        identical output ids (docs/methodology.md Phase 1, "deterministic
        generation")."""
        self._require_loaded()
        logits, past = self.prefill(input_ids)
        next_id = logits[:, -1:, :].argmax(dim=-1)
        out_ids = [next_id]
        for _ in range(max_new_tokens - 1):
            logits, past = self.decode_step(next_id, past)
            next_id = logits[:, -1:, :].argmax(dim=-1)
            out_ids.append(next_id)
        return torch.cat(out_ids, dim=1)

    # ------------------------------------------------------------------
    # Benchmark primitive (Phase 1 metrics: load, prefill, decode, TTFT,
    # TPOT, E2E, throughput, peak VRAM, KV-cache memory)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_with_timing(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
    ) -> TimedGenerationResult:
        """Greedy generation with per-step timing and memory tracking.
        Exactly one of `prompt` / `input_ids` should be given — use
        `input_ids` (e.g. from `synthesize_input_ids`) for the fixed
        context-length sweep, `prompt` for ad-hoc/manual use."""
        self._require_loaded()
        if (prompt is None) == (input_ids is None):
            raise ValueError("pass exactly one of prompt= or input_ids=")
        if prompt is not None:
            input_ids = self.encode(prompt)

        is_cuda = self.device.startswith("cuda")
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)

        # --- prefill ---
        t0 = time.perf_counter()
        logits, past = self.prefill(input_ids)
        if is_cuda:
            torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

        next_id = logits[:, -1:, :].argmax(dim=-1)
        out_ids = [next_id]

        # --- decode ---
        decode_step_ms = []
        for _ in range(max_new_tokens - 1):
            t0 = time.perf_counter()
            logits, past = self.decode_step(next_id, past)
            if is_cuda:
                torch.cuda.synchronize()
            decode_step_ms.append((time.perf_counter() - t0) * 1000)
            next_id = logits[:, -1:, :].argmax(dim=-1)
            out_ids.append(next_id)

        peak_vram_mb = (
            torch.cuda.max_memory_allocated(self.device) / 1024 / 1024 if is_cuda else 0.0
        )
        kv_mb = kv_cache_bytes(past) / 1024 / 1024

        return TimedGenerationResult(
            input_ids=input_ids,
            output_ids=torch.cat(out_ids, dim=1),
            prefill_ms=prefill_ms,
            decode_step_ms=decode_step_ms,
            peak_vram_mb=peak_vram_mb,
            kv_cache_mb=kv_mb,
            load_ms=self.load_ms,
        )


if __name__ == "__main__":
    # Minimal smoke test — mirrors benchmarks/runners/check_env.py's
    # check_qwen_loadable(), but actually runs a tiny generation.
    ref = QwenReference(dtype="fp16", device="cuda" if torch.cuda.is_available() else "cpu").load()
    print(f"Loaded {ref.model_name} in {ref.load_ms:.1f} ms")
    print(f"Shape: {ref.shape}")
    result = ref.generate_with_timing(prompt="The capital of France is", max_new_tokens=8)
    print(f"TTFT={result.ttft_ms:.1f}ms TPOT={result.tpot_ms:.1f}ms peak_vram={result.peak_vram_mb:.1f}MB")
    print(ref.tokenizer.decode(result.output_ids[0]))