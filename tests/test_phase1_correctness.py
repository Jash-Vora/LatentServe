"""
Phase 1 correctness harness (docs/methodology.md Phase 1, Gate 1: "Can
Qwen generate correctly?").

These tests require torch + transformers + a real download of
Qwen2.5-1.5B-Instruct (~3 GB) and are slow at the larger context
lengths, so they:

  - skip cleanly (not fail) if torch/transformers/CUDA/network aren't
    available, so `pytest` stays green on machines that just don't have
    the model yet (e.g. a laptop, or this repo's own CI);
  - default to small/fast token-count levels (1, 16, 128, 1024);
  - gate the full Phase 1 sweep (1K/4K/8K/16K+, per the methodology doc)
    behind LATENTSERVE_LONG_CONTEXT_TESTS=1, since those need real GPU
    memory/time and are meant to be run on the Kaggle T4 target, not on
    every local `pytest` invocation.

Run the fast subset:
    export PYTHONPATH=$(pwd):$PYTHONPATH
    pytest tests/test_phase1_correctness.py -v

Run the full context-length sweep (on a T4):
    LATENTSERVE_LONG_CONTEXT_TESTS=1 pytest tests/test_phase1_correctness.py -v
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from model.qwen import QwenReference, kv_cache_bytes  # noqa: E402

_LONG = os.environ.get("LATENTSERVE_LONG_CONTEXT_TESTS") == "1"

SHORT_LENGTHS = [1, 16, 128, 1024]
LONG_LENGTHS = [4096, 8192, 16384]


def _cuda_and_model_available() -> tuple:
    """Returns (ok, reason). We check CUDA *and* that the model is
    actually reachable/loadable (network egress + HF cache), mirroring
    benchmarks/runners/check_env.py::check_qwen_loadable — we'd rather
    skip with a clear reason than fail on a machine that just can't
    reach Hugging Face."""
    if not torch.cuda.is_available():
        return False, "no CUDA device visible"
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except Exception as e:  # noqa: BLE001 - any load failure means "skip"
        return False, f"Qwen2.5-1.5B-Instruct not reachable/loadable: {e}"
    return True, ""


_available, _skip_reason = _cuda_and_model_available()
requires_model = pytest.mark.skipif(not _available, reason=_skip_reason)


@pytest.fixture(scope="module")
def ref() -> QwenReference:
    return QwenReference(dtype="fp16", device="cuda").load()


# ---------------------------------------------------------------------
# Gate 1a — model loads and shape introspection is sane (feeds Phase 2's
# GQA group-size math directly, so wrong values here would be silently
# wrong everywhere downstream).
# ---------------------------------------------------------------------


@requires_model
def test_shape_introspection_is_consistent(ref: QwenReference):
    s = ref.shape
    assert s.num_layers > 0
    assert s.num_attention_heads > 0
    assert s.num_key_value_heads > 0
    assert s.num_attention_heads % s.num_key_value_heads == 0
    assert s.gqa_group_size == s.num_attention_heads // s.num_key_value_heads
    assert s.hidden_size == s.head_dim * s.num_attention_heads or s.head_dim > 0


# ---------------------------------------------------------------------
# Gate 1b — multi-token prefill vs single-token decode agree.
#
# This is the core correctness check: if attention masking, RoPE
# position ids, or KV-cache updates were wrong, incremental (cached)
# decode would silently diverge from a full-sequence teacher-forced
# forward pass. Every later phase's custom attention/cache path
# (Phase 2 onward) gets checked against *this* same pattern, not
# against Hugging Face directly.
# ---------------------------------------------------------------------


@requires_model
@pytest.mark.parametrize("num_tokens", SHORT_LENGTHS)
def test_incremental_matches_teacher_forced(ref: QwenReference, num_tokens: int):
    input_ids = ref.synthesize_input_ids(num_tokens, seed=0)

    teacher_forced_logits = ref.forward_teacher_forced(input_ids)
    incremental_logits = ref.forward_incremental(input_ids)

    assert teacher_forced_logits.shape == incremental_logits.shape
    # fp16 accumulation order differs slightly between the two paths
    # (batched matmul vs. per-step matmul), so this is a numerical
    # tolerance check, not bitwise equality.
    torch.testing.assert_close(
        teacher_forced_logits.float(),
        incremental_logits.float(),
        rtol=1e-2,
        atol=1e-2,
    )


@requires_model
@pytest.mark.skipif(not _LONG, reason="set LATENTSERVE_LONG_CONTEXT_TESTS=1 to run")
@pytest.mark.parametrize("num_tokens", LONG_LENGTHS)
def test_incremental_matches_teacher_forced_long_context(ref: QwenReference, num_tokens: int):
    # Long-context version of the above. Kept separate so the fast
    # subset stays fast; incremental (token-by-token) decode at 16K
    # tokens means 16K sequential forward passes and is genuinely slow.
    test_incremental_matches_teacher_forced(ref, num_tokens)


# ---------------------------------------------------------------------
# Gate 1c — multi-token prefill followed by single-token decode steps
# (the actual serving-path shape, as opposed to the fully-incremental
# check above) also agrees with the teacher-forced ground truth.
# ---------------------------------------------------------------------


@requires_model
@pytest.mark.parametrize("num_tokens", SHORT_LENGTHS)
def test_prefill_then_decode_matches_teacher_forced(ref: QwenReference, num_tokens: int):
    if num_tokens < 2:
        pytest.skip("needs at least a 1-token prompt + 1 decode step to be meaningful")

    input_ids = ref.synthesize_input_ids(num_tokens, seed=1)
    prompt_ids, last_id = input_ids[:, :-1], input_ids[:, -1:]

    teacher_forced_logits = ref.forward_teacher_forced(input_ids)

    prefill_logits, past = ref.prefill(prompt_ids)
    decode_logits, _ = ref.decode_step(last_id, past)

    # The decode step's logits (predicting the token *after* the full
    # sequence) must match the teacher-forced logits at the last position.
    torch.testing.assert_close(
        teacher_forced_logits[:, -1:, :].float(),
        decode_logits.float(),
        rtol=1e-2,
        atol=1e-2,
    )


# ---------------------------------------------------------------------
# Gate 1d — deterministic generation. Greedy decode must be reproducible:
# same input -> same output ids, every time (docs/methodology.md Phase 1).
# ---------------------------------------------------------------------


@requires_model
def test_deterministic_generation(ref: QwenReference):
    input_ids = ref.synthesize_input_ids(32, seed=2)
    out_a = ref.generate_greedy(input_ids, max_new_tokens=16)
    out_b = ref.generate_greedy(input_ids, max_new_tokens=16)
    assert torch.equal(out_a, out_b)


# ---------------------------------------------------------------------
# Gate 1e — KV-cache memory: the actual measured bytes in
# past_key_values must match the theoretical per-token estimate from
# ModelShape (docs/methodology.md Phase 9, Question 1 — this is the GQA
# baseline every later MLA compression ratio is computed against).
# ---------------------------------------------------------------------


@requires_model
@pytest.mark.parametrize("num_tokens", [16, 128])
def test_kv_cache_memory_matches_theoretical_estimate(ref: QwenReference, num_tokens: int):
    input_ids = ref.synthesize_input_ids(num_tokens, seed=3)
    _, past = ref.prefill(input_ids)

    measured_bytes = kv_cache_bytes(past)
    dtype_bytes = torch.tensor([], dtype=ref.torch_dtype).element_size()
    theoretical_bytes = ref.shape.kv_bytes_per_token(dtype_bytes) * num_tokens

    # Small tolerance for cache-implementation padding/alignment, but
    # this should be very close — a large mismatch would mean either
    # ModelShape's head/layer counts or the actual cache layout is wrong.
    assert measured_bytes == pytest.approx(theoretical_bytes, rel=0.05)


# ---------------------------------------------------------------------
# Gate 1f — generate_with_timing() produces sane, internally-consistent
# metrics (this is the harness benchmarks/runners/phase1_reference.py
# relies on for every later phase's numbers, so it gets its own check).
# ---------------------------------------------------------------------


@requires_model
def test_generate_with_timing_metrics_are_consistent(ref: QwenReference):
    input_ids = ref.synthesize_input_ids(64, seed=4)
    result = ref.generate_with_timing(input_ids=input_ids, max_new_tokens=8)

    assert result.input_tokens == 64
    assert result.output_tokens == 8
    assert result.ttft_ms > 0
    assert result.tpot_ms > 0
    assert result.e2e_latency_ms == pytest.approx(
        result.prefill_ms + sum(result.decode_step_ms), rel=1e-6
    )
    assert result.throughput_tokens_sec > 0
    assert result.peak_vram_mb > 0
    assert result.kv_cache_mb > 0
