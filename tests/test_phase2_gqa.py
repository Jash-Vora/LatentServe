"""
Phase 2 correctness harness — GQA + LatentServe KV cache.

docs/methodology.md Gate 2: "Can LatentServe execute cached decoding?"
That question is only meaningful if LatentServe's execution path
reproduces the Hugging Face oracle, so most of this file is an
equivalence test.

Two tiers, deliberately:

  * **Tier 1 (default, fast, CPU, no download).** A *randomly
    initialised* Qwen2 model with Qwen2.5-1.5B's shape signature
    (grouped KV heads, group size > 1) but tiny dimensions. Random
    weights are not a weakness here — attention masking, RoPE phase,
    cache write offsets and head grouping are weight-independent, so a
    bug in any of them shows up just as loudly, in two seconds, on a
    laptop, with no Hugging Face network access. Run in float32 so a
    real logic error can't hide behind fp16 noise.
  * **Tier 2 (gated).** The real Qwen2.5-1.5B-Instruct checkpoint
    against `QwenReference`, i.e. the Phase 1 oracle. Needs the weights
    and a GPU; skips cleanly otherwise, matching
    tests/test_phase1_correctness.py's convention.

Run tier 1:
    export PYTHONPATH=$(pwd):$PYTHONPATH
    pytest tests/test_phase2_gqa.py -v

Run tier 2 as well (on the T4):
    LATENTSERVE_REAL_MODEL_TESTS=1 pytest tests/test_phase2_gqa.py -v
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from cache.kv_cache import ContiguousKVCache, KVCacheSpec, effective_kv_heads  # noqa: E402
from model.attention.gqa import build_causal_mask, repeat_kv  # noqa: E402
from model.latentserve_qwen import LatentServeQwen  # noqa: E402
from model.qwen import ModelShape  # noqa: E402
from model.rope import RotaryEmbedding, apply_rope  # noqa: E402

_REAL = os.environ.get("LATENTSERVE_REAL_MODEL_TESTS") == "1"

# Tiny, but shaped like the real thing: 8 query heads over 2 KV heads is
# a group size of 4, so any bug that only manifests when query heads
# share a KV head (the entire point of GQA) is in scope.
TINY = dict(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=512,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = Qwen2Config(**TINY)
    model = Qwen2ForCausalLM(config).to(torch.float32).eval()
    return model


@pytest.fixture(scope="module")
def tiny_shape(tiny_model) -> ModelShape:
    cfg = tiny_model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return ModelShape(
        num_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        hidden_size=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.max_position_embeddings,
        torch_dtype="torch.float32",
    )


def build(tiny_model, tiny_shape, **kwargs) -> LatentServeQwen:
    return LatentServeQwen(
        hf_model=tiny_model,
        tokenizer=None,
        shape=tiny_shape,
        device="cpu",
        max_seq_len_hint=TINY["max_position_embeddings"],
        **kwargs,
    )


# ----------------------------------------------------------------------
# RoPE
# ----------------------------------------------------------------------


def test_rope_matches_hf(tiny_model):
    """LatentServe's RoPE tables must equal the checkpoint's own. If
    they don't, every later attention variant inherits a wrong
    positional phase and the Phase 7 MLA comparison is measuring the
    bug, not the idea."""
    hf_rotary = tiny_model.model.rotary_emb
    seq_len = 37
    positions = torch.arange(seq_len)[None, :]
    dummy = torch.zeros(1, seq_len, 1)
    hf_cos, hf_sin = hf_rotary(dummy, positions)

    rope = RotaryEmbedding.from_hf_config(
        tiny_model.config, max_seq_len=seq_len, device="cpu"
    )
    cos, sin = rope.cos_sin(0, seq_len, torch.float32)

    torch.testing.assert_close(cos[0, 0], hf_cos[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sin[0, 0], hf_sin[0], rtol=1e-5, atol=1e-6)


def test_rope_offset_is_absolute_position(tiny_model):
    """cos_sin(start_pos=k, 1) must equal position k of the full table.
    This is the decode path's contract: a decode step at position k gets
    the same phase the prefill would have given it."""
    rope = RotaryEmbedding.from_hf_config(tiny_model.config, max_seq_len=64, device="cpu")
    full_cos, full_sin = rope.cos_sin(0, 64, torch.float32)
    for k in (0, 1, 17, 63):
        cos_k, sin_k = rope.cos_sin(k, 1, torch.float32)
        torch.testing.assert_close(cos_k[0, 0, 0], full_cos[0, 0, k])
        torch.testing.assert_close(sin_k[0, 0, 0], full_sin[0, 0, k])


def test_rope_is_a_rotation():
    """RoPE must preserve the norm of each 2D frequency pair."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32, theta=1e6, device="cpu")
    q = torch.randn(1, 2, 8, 16)
    cos, sin = rope.cos_sin(0, 8, torch.float32)
    q_rot, _ = apply_rope(q, q, cos, sin)
    torch.testing.assert_close(q.norm(dim=-1), q_rot.norm(dim=-1), rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------
# KV cache
# ----------------------------------------------------------------------


def test_kv_bytes_per_token_matches_model_shape(tiny_shape):
    """The cache's byte accounting and ModelShape's theoretical formula
    (Phase 1) must agree — they are the two numbers every MLA memory
    claim in Phase 7+ is a ratio of."""
    spec = KVCacheSpec(
        num_layers=tiny_shape.num_layers,
        num_kv_heads=tiny_shape.num_key_value_heads,
        head_dim=tiny_shape.head_dim,
        max_batch_size=1,
        max_seq_len=16,
        dtype=torch.float16,
        device="cpu",
    )
    assert spec.bytes_per_token == tiny_shape.kv_bytes_per_token(dtype_bytes=2)


def test_qwen15b_kv_bytes_per_token_is_28kib():
    """Regression guard on the real model's headline number: 28 layers x
    2 KV heads x 128 head_dim x 2 (K and V) x 2 bytes = 28,672 B/token.
    Phase 1 measured 115.47 MB at 4096+128 tokens, which is this number
    x 4224 — the two agree, so this constant is the thing MLA must beat."""
    spec = KVCacheSpec(
        num_layers=28, num_kv_heads=2, head_dim=128, max_batch_size=1, max_seq_len=1,
        dtype=torch.float16, device="cpu",
    )
    assert spec.bytes_per_token == 28_672


def test_effective_kv_heads_modes():
    assert effective_kv_heads(2, 12, "native") == 2
    assert effective_kv_heads(2, 12, "mha_sim") == 12
    assert effective_kv_heads(2, 12, "mqa_sim") == 1


def test_cache_write_read_roundtrip():
    spec = KVCacheSpec(
        num_layers=2, num_kv_heads=2, head_dim=4, max_batch_size=2, max_seq_len=8,
        dtype=torch.float32, device="cpu",
    )
    cache = ContiguousKVCache(spec)
    k1 = torch.randn(2, 2, 3, 4)
    cache.write(0, k1, k1 * 2, start_pos=0)
    cache.advance(3)
    k2 = torch.randn(2, 2, 1, 4)
    cache.write(0, k2, k2 * 2, start_pos=3)
    cache.advance(1)

    k_all, v_all = cache.read(0, batch_size=2, length=4)
    torch.testing.assert_close(k_all[:, :, :3], k1)
    torch.testing.assert_close(k_all[:, :, 3:4], k2)
    torch.testing.assert_close(v_all, k_all * 2)
    assert cache.length == 4
    assert cache.used_bytes(2) == spec.bytes_per_token * 4 * 2


def test_cache_overflow_raises():
    spec = KVCacheSpec(
        num_layers=1, num_kv_heads=1, head_dim=4, max_batch_size=1, max_seq_len=4,
        dtype=torch.float32, device="cpu",
    )
    cache = ContiguousKVCache(spec)
    with pytest.raises(RuntimeError):
        cache.write(0, torch.zeros(1, 1, 5, 4), torch.zeros(1, 1, 5, 4), start_pos=0)


def test_repeat_kv_semantics():
    """repeat_kv must group-repeat (head h -> rows h*n..h*n+n-1), not
    tile. Getting this backwards pairs query heads with the wrong KV
    head and still produces plausible-looking output."""
    x = torch.arange(2 * 2 * 1 * 3, dtype=torch.float32).view(2, 2, 1, 3)
    torch.testing.assert_close(repeat_kv(x, 3), x.repeat_interleave(3, dim=1))
    assert repeat_kv(x, 1) is x


def test_causal_mask_offsets():
    mask = build_causal_mask(q_len=2, kv_len=5, start_pos=3, device=torch.device("cpu"))
    # Query at absolute position 3 sees keys 0..3; position 4 sees 0..4.
    expected = torch.tensor([[True, True, True, True, False], [True] * 5])
    assert torch.equal(mask[0, 0], expected)
    assert build_causal_mask(1, 5, 4, torch.device("cpu")) is None


# ----------------------------------------------------------------------
# Tier 1 — equivalence against Hugging Face on a tiny random model
# ----------------------------------------------------------------------


def test_prefill_logits_match_hf(tiny_model, tiny_shape):
    """The Gate 2 test: LatentServe's own layer loop, RoPE, GQA and KV
    cache must reproduce HF's full-sequence forward pass."""
    ls = build(tiny_model, tiny_shape)
    ids = torch.randint(0, TINY["vocab_size"], (2, 24))

    with torch.no_grad():
        hf_logits = tiny_model(input_ids=ids, use_cache=False).logits

    ls.allocate_cache(batch_size=2, max_seq_len=64)
    ls.cache.reset()
    ls_logits = ls.forward_logits_all(ids)

    torch.testing.assert_close(ls_logits, hf_logits, rtol=1e-4, atol=1e-4)


def test_incremental_decode_matches_teacher_forced(tiny_model, tiny_shape):
    """Cached single-token decode must agree with the no-cache
    full-sequence pass at the same positions. This is what catches a
    start_pos/RoPE/mask desync — the failure mode that silently degrades
    long-context quality rather than crashing."""
    ls = build(tiny_model, tiny_shape)
    ids = torch.randint(0, TINY["vocab_size"], (1, 12))

    with torch.no_grad():
        hf_logits = tiny_model(input_ids=ids, use_cache=False).logits

    ls.allocate_cache(batch_size=1, max_seq_len=32)
    ls.cache.reset()
    ls.prefill(ids[:, :4])
    for t in range(4, 12):
        step_logits = ls.decode_step(ids[:, t : t + 1])
        torch.testing.assert_close(
            step_logits[:, 0], hf_logits[:, t], rtol=1e-4, atol=1e-4
        )


def test_chunked_prefill_matches_unchunked(tiny_model, tiny_shape):
    """Chunked prefill is a memory optimization, not a math change, so
    it must be numerically transparent."""
    ls = build(tiny_model, tiny_shape)
    ids = torch.randint(0, TINY["vocab_size"], (1, 20))

    ls.allocate_cache(batch_size=1, max_seq_len=32)
    ls.cache.reset()
    full = ls.prefill(ids)
    ls.cache.reset()
    chunked = ls.prefill(ids, chunk_size=7)

    torch.testing.assert_close(full, chunked, rtol=1e-4, atol=1e-4)


def test_prefill_last_logits_equal_full_logits(tiny_model, tiny_shape):
    """The VRAM optimization (project only the final position through
    lm_head) must return exactly what the full [B, S, V] tensor's last
    row contains."""
    ls = build(tiny_model, tiny_shape)
    ids = torch.randint(0, TINY["vocab_size"], (1, 16))

    ls.allocate_cache(batch_size=1, max_seq_len=32)
    ls.cache.reset()
    all_logits = ls.forward_logits_all(ids)
    ls.cache.reset()
    last_logits = ls.prefill(ids)

    torch.testing.assert_close(last_logits[:, 0], all_logits[:, -1], rtol=1e-5, atol=1e-5)


def test_mha_sim_is_numerically_equivalent(tiny_model, tiny_shape):
    """mha_sim stores one KV head per query head instead of sharing.
    Same math, 4x the cache (group size 4 in TINY) — so logits must
    match native to tolerance while `kv_bytes_per_token` does not. That
    equality is what licenses attributing any measured latency
    difference purely to memory traffic."""
    native = build(tiny_model, tiny_shape, kv_heads_mode="native")
    simulated = build(tiny_model, tiny_shape, kv_heads_mode="mha_sim")
    ids = torch.randint(0, TINY["vocab_size"], (1, 16))

    native.allocate_cache(1, 32)
    native.cache.reset()
    a = native.forward_logits_all(ids)

    simulated.allocate_cache(1, 32)
    simulated.cache.reset()
    b = simulated.forward_logits_all(ids)

    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    group = tiny_shape.gqa_group_size
    assert simulated.cache.spec.bytes_per_token == native.cache.spec.bytes_per_token * group


def test_math_and_sdpa_impls_agree(tiny_model, tiny_shape):
    """The explicit-math attention path is the reference Phase 11's
    custom kernels get checked against, so it must already agree with
    SDPA today."""
    sdpa = build(tiny_model, tiny_shape, attn_impl="sdpa")
    math = build(tiny_model, tiny_shape, attn_impl="math")
    ids = torch.randint(0, TINY["vocab_size"], (1, 18))

    sdpa.allocate_cache(1, 32)
    sdpa.cache.reset()
    a = sdpa.forward_logits_all(ids)

    math.allocate_cache(1, 32)
    math.cache.reset()
    b = math.forward_logits_all(ids)

    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_rope_source_hf_matches_latentserve(tiny_model, tiny_shape):
    """The bisection escape hatch must be equivalent, or it is useless
    as a bisection escape hatch."""
    ours = build(tiny_model, tiny_shape, rope_source="latentserve")
    theirs = build(tiny_model, tiny_shape, rope_source="hf")
    ids = torch.randint(0, TINY["vocab_size"], (1, 14))

    ours.allocate_cache(1, 32)
    ours.cache.reset()
    a = ours.forward_logits_all(ids)

    theirs.allocate_cache(1, 32)
    theirs.cache.reset()
    b = theirs.forward_logits_all(ids)

    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_greedy_generation_is_deterministic(tiny_model, tiny_shape):
    ls = build(tiny_model, tiny_shape)
    ids = torch.randint(0, TINY["vocab_size"], (1, 10))
    ls.allocate_cache(1, 32)
    first = ls.generate_greedy(ids, max_new_tokens=8)
    second = ls.generate_greedy(ids, max_new_tokens=8)
    assert torch.equal(first, second)


def test_batched_matches_unbatched(tiny_model, tiny_shape):
    """Batch dimension must not leak across sequences. Cheap to get
    wrong once the cache is a shared preallocated block."""
    ls = build(tiny_model, tiny_shape)
    ids = torch.randint(0, TINY["vocab_size"], (3, 16))

    ls.allocate_cache(3, 32)
    ls.cache.reset()
    batched = ls.forward_logits_all(ids)

    singles = []
    for i in range(3):
        ls.allocate_cache(1, 32)
        ls.cache.reset()
        singles.append(ls.forward_logits_all(ids[i : i + 1]))

    torch.testing.assert_close(batched, torch.cat(singles, dim=0), rtol=1e-4, atol=1e-4)


# ----------------------------------------------------------------------
# Tier 2 — the real checkpoint against the Phase 1 oracle
# ----------------------------------------------------------------------


def _real_model_available() -> tuple:
    if not _REAL:
        return False, "set LATENTSERVE_REAL_MODEL_TESTS=1 to run against Qwen2.5-1.5B-Instruct"
    if not torch.cuda.is_available():
        return False, "no CUDA device visible"
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except Exception as e:  # noqa: BLE001
        return False, f"Qwen2.5-1.5B-Instruct not reachable/loadable: {e}"
    return True, ""


_real_ok, _real_reason = _real_model_available()
requires_real_model = pytest.mark.skipif(not _real_ok, reason=_real_reason)


@pytest.fixture(scope="module")
def real_pair():
    from model.qwen import QwenReference

    # fp32 for the numerical gates, for the same reason
    # tests/test_phase1_correctness.py uses an fp32 fixture: on
    # out-of-distribution synthetic token ids, fp16 accumulation-order
    # differences compound across 28 layers into divergence that has
    # nothing to do with the GQA path under test. The fp16 behaviour is
    # covered by the benchmark sweep, not here.
    ref = QwenReference(dtype="fp32", device="cuda").load()
    ls = LatentServeQwen.from_reference(ref, max_seq_len_hint=2048)
    return ref, ls


@requires_real_model
def test_real_model_matches_phase1_oracle(real_pair):
    """Gate 2 on the real checkpoint: LatentServe's execution path
    against `QwenReference.forward_teacher_forced()`, the Phase 1
    ground truth."""
    ref, ls = real_pair
    ids = ref.synthesize_input_ids(512, seed=0)

    oracle = ref.forward_teacher_forced(ids)
    ls.allocate_cache(batch_size=1, max_seq_len=1024)
    ls.cache.reset()
    ours = ls.forward_logits_all(ids)

    torch.testing.assert_close(ours, oracle, rtol=2e-3, atol=2e-3)


@requires_real_model
def test_real_model_greedy_output_matches(real_pair):
    """Same prompt, same deterministic output ids. The end-to-end check
    a reader of the final report actually cares about."""
    ref, ls = real_pair
    ids = ref.encode("The capital of France is")

    expected = ref.generate_greedy(ids, max_new_tokens=16)
    ls.allocate_cache(batch_size=1, max_seq_len=64)
    ours = ls.generate_greedy(ids, max_new_tokens=16)

    assert torch.equal(ours, expected), (
        f"LatentServe: {ref.tokenizer.decode(ours[0])!r} != "
        f"HF: {ref.tokenizer.decode(expected[0])!r}"
    )


@requires_real_model
def test_real_model_kv_accounting_matches_measured(real_pair):
    """LatentServe's exact cache accounting must match Phase 1's
    measured HF `past_key_values` size for the same token count."""
    from model.qwen import kv_cache_bytes

    ref, ls = real_pair
    ids = ref.synthesize_input_ids(256, seed=0)

    _, past = ref.prefill(ids)
    hf_bytes = kv_cache_bytes(past)

    ls.allocate_cache(batch_size=1, max_seq_len=512)
    ls.cache.reset()
    ls.prefill(ids)

    assert ls.cache.used_bytes(1) == hf_bytes
