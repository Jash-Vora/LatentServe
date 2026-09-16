"""
Phase 7.3 tests — INT8 paged KV cache.

Two things have to hold for this cache to be usable at all, and both are
the kind of bug that would not show up as a crash:

  1. The residual/finalize split must be *exactly* transparent to the
     reader. A still-filling block's tokens must come back bit-identical
     (no quantization has happened to them yet), and a finalized block's
     tokens must come back with error bounded by its own scale — not
     zero, not unboundedly large.
  2. Attaching this cache to the real decode path (LatentServeQwen) must
     still produce a coherent model: logits close to the FP16 paged
     cache's, not garbage. That's the Gate 4-style equivalence check
     test_phase3_paged.py runs for PagedKVCache, loosened from exact
     match to a small-KL bound — INT8 is lossy by construction, so exact
     equivalence is the wrong bar; "close enough to match 7.2's offline
     prediction" is the right one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from cache.int8_paged_cache import Int8PagedKVCache  # noqa: E402
from cache.kv_cache import KVCacheSpec  # noqa: E402
from compression.truncation import DivergenceMeter  # noqa: E402
from model.latentserve_qwen import LatentServeQwen  # noqa: E402
from model.qwen import ModelShape  # noqa: E402

TINY = dict(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(**TINY)).to(torch.float32).eval()


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


def build(tiny_model, tiny_shape) -> LatentServeQwen:
    return LatentServeQwen(
        hf_model=tiny_model, tokenizer=None, shape=tiny_shape, device="cpu",
        max_seq_len_hint=TINY["max_position_embeddings"],
    )


def tiny_spec(batch=1, max_seq=64, heads=2, dim=4, layers=2) -> KVCacheSpec:
    return KVCacheSpec(
        num_layers=layers, num_kv_heads=heads, head_dim=dim,
        max_batch_size=batch, max_seq_len=max_seq,
        dtype=torch.float32, device="cpu",
    )


# ----------------------------------------------------------------------
# Storage correctness — residual vs. finalized
# ----------------------------------------------------------------------


def test_still_filling_block_reads_back_exactly():
    """Tokens in a block that has not yet reached block_size must come
    back bit-identical: they have not been quantized, only buffered."""
    spec = tiny_spec(batch=1, max_seq=32)
    cache = Int8PagedKVCache(spec, block_size=8)
    cache.advance(5, batch_size=1)  # 5 < block_size=8: block never fills
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    cache.write(0, k, v)

    k_out, v_out = cache.read(0, batch_size=1)
    torch.testing.assert_close(k_out, k, rtol=0, atol=0)
    # V is never buffered (always quantized immediately), so it is lossy
    # even within the first partial block — bounded, not exact.
    assert (v_out - v).abs().max() <= v.abs().max() / cache.v_qmax * 0.51 + 1e-6


def test_finalized_block_is_lossy_but_bounded():
    """Once a block fills, its tokens go through the INT8 round-trip:
    not exact, but within half a quantization step of its own scale."""
    spec = tiny_spec(batch=1, max_seq=32)
    cache = Int8PagedKVCache(spec, block_size=8)
    cache.advance(8, batch_size=1)  # exactly one full block
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)
    cache.write(0, k, v)

    k_out, _ = cache.read(0, batch_size=1)
    assert not torch.equal(k_out, k)
    assert (k_out - k).abs().max() <= k.abs().max() / cache.k_qmax * 0.51 + 1e-4


def test_block_completed_across_multiple_writes_matches_single_write():
    """A block filled one decode-token at a time (n=1 per write, as the
    real decode path does) must finalize to the same INT8 values as the
    same tokens written in one shot — the scale must not depend on how
    the write calls were chunked, only on which tokens ended up in the
    block."""
    spec = tiny_spec(batch=1, max_seq=32)
    torch.manual_seed(0)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)

    one_shot = Int8PagedKVCache(spec, block_size=8)
    one_shot.advance(8, batch_size=1)
    one_shot.write(0, k, v)
    k_one_shot, _ = one_shot.read(0, batch_size=1)

    token_at_a_time = Int8PagedKVCache(spec, block_size=8)
    for t in range(8):
        token_at_a_time.advance(1, batch_size=1)
        token_at_a_time.write(0, k[:, :, t : t + 1], v[:, :, t : t + 1])
    k_streamed, _ = token_at_a_time.read(0, batch_size=1)

    torch.testing.assert_close(k_one_shot, k_streamed)


def test_write_spanning_a_block_boundary_finalizes_only_the_completed_block():
    """A single write of 5 tokens into an empty block_size=4 cache must
    finalize block 0 (positions 0-3) and leave position 4 in residual —
    not finalize early, not silently drop the boundary-crossing case."""
    spec = tiny_spec(batch=1, max_seq=32)
    cache = Int8PagedKVCache(spec, block_size=4)
    cache.advance(5, batch_size=1)
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    cache.write(0, k, v)

    k_out, _ = cache.read(0, batch_size=1)
    assert not torch.equal(k_out[:, :, :4], k[:, :, :4])  # finalized: lossy
    torch.testing.assert_close(k_out[:, :, 4:5], k[:, :, 4:5])  # residual: exact


def test_ragged_sequences_each_keep_their_own_residual():
    """Two sequences at different lengths must not cross-contaminate
    each other's still-filling tail block."""
    spec = tiny_spec(batch=2, max_seq=32)
    cache = Int8PagedKVCache(spec, block_size=8)
    cache.advance(3, batch_size=2)
    k = torch.stack([torch.full((2, 3, 4), 1.0), torch.full((2, 3, 4), -5.0)])
    v = torch.zeros(2, 2, 3, 4)
    cache.write(0, k, v)

    k_out, _ = cache.read(0, batch_size=2)
    torch.testing.assert_close(k_out[0], k[0], rtol=0, atol=0)
    torch.testing.assert_close(k_out[1], k[1], rtol=0, atol=0)


def test_reset_clears_residual_state():
    """A residual block left over from a freed sequence must not leak
    into whatever reuses that sequence slot next."""
    spec = tiny_spec(batch=1, max_seq=32)
    cache = Int8PagedKVCache(spec, block_size=8)
    cache.advance(3, batch_size=1)
    cache.write(0, torch.full((1, 2, 3, 4), 9.0), torch.zeros(1, 2, 3, 4))
    cache.reset()
    cache.advance(3, batch_size=1)
    cache.write(0, torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4))
    k_out, _ = cache.read(0, batch_size=1)
    assert (k_out == 0).all(), "stale residual from before reset() leaked through"


# ----------------------------------------------------------------------
# Capacity — the Phase 7.5 number this module exists to feed
# ----------------------------------------------------------------------


def test_int8_cache_uses_roughly_half_the_bytes_of_fp16_paged():
    """The storage claim this whole phase rests on: INT8 K/V at ~1 byte/
    element plus amortized scale overhead should land close to half of
    FP16's 2 bytes/element, not some other fraction that the scale
    bookkeeping quietly ate.

    Sized to Qwen2.5-1.5B's actual kv_heads=2, head_dim=128 rather than
    the toy shapes the other tests use: at head_dim=4 the fixed FP32
    per-token/per-block scale overhead is a large fraction of a tiny
    8-element vector, which pulls the ratio well below 0.5 for reasons
    that have nothing to do with the method — a real head_dim amortizes
    that overhead away, which is the regime this claim is actually made
    for.

    The comparison is against the FP16 byte count specifically (2
    bytes/element), not whatever dtype this test's spec happens to use
    (`tiny_spec` defaults to fp32, matching the rest of this file's
    CPU-friendly convention) — a production PagedKVCache is fp16, and
    that is the "2x capacity" claim being checked.
    """
    spec = tiny_spec(batch=1, max_seq=256, heads=2, dim=128, layers=28)
    fp16_bytes_per_token = 2 * spec.num_layers * spec.num_kv_heads * spec.head_dim * 2  # dtype_bytes=2 for fp16
    int8 = Int8PagedKVCache(spec, block_size=16)
    ratio = int8.bytes_per_token / fp16_bytes_per_token
    assert 0.45 < ratio < 0.6, f"expected ~0.5x, got {ratio:.3f}"


# ----------------------------------------------------------------------
# End-to-end — wired into the real decode path
# ----------------------------------------------------------------------


def test_int8_paged_requires_paged_true(tiny_model, tiny_shape):
    ls = build(tiny_model, tiny_shape)
    with pytest.raises(ValueError):
        ls.allocate_cache(1, 64, paged=False, kv_dtype="int8")


def test_int8_paged_decode_runs_and_stays_close_to_fp16_paged(tiny_model, tiny_shape):
    """The Gate-4-style check, loosened for a lossy cache: same prompt,
    same incremental decode, FP16 paged vs. INT8 paged. Not bit-exact —
    that would mean no quantization happened — but the KL should be
    small, in the same regime 7.2's offline simulation predicted
    (fractions of a nat, not tenths of a nat)."""
    torch.manual_seed(1)
    ids = torch.randint(0, TINY["vocab_size"], (1, 40))

    fp16_ls = build(tiny_model, tiny_shape)
    fp16_ls.allocate_cache(1, 128, paged=True, block_size=16)
    fp16_ls.cache.reset()
    fp16_ls.prefill(ids[:, :24])
    fp16_logits = [fp16_ls.decode_step(ids[:, t : t + 1]) for t in range(24, 40)]

    int8_ls = build(tiny_model, tiny_shape)
    int8_ls.allocate_cache(1, 128, paged=True, block_size=16, kv_dtype="int8")
    int8_ls.cache.reset()
    int8_ls.prefill(ids[:, :24])
    int8_logits = [int8_ls.decode_step(ids[:, t : t + 1]) for t in range(24, 40)]

    meter = DivergenceMeter()
    for base, mod in zip(fp16_logits, int8_logits):
        meter.update(base, mod)
    result = meter.result()

    assert not torch.equal(fp16_logits[-1], int8_logits[-1]), "should be lossy, not a no-op"
    assert result["kl_mean_nats"] < 0.2, result
    assert result["top1_flip_rate"] < 0.3, result


def test_int8_paged_survives_churn_and_stays_coherent(tiny_model, tiny_shape):
    """Run several generations through one pool so blocks get recycled
    and residuals get reused across sequences, then check the cache is
    still producing sane (finite, non-exploding) logits — the INT8
    analogue of test_phase3_paged's churn test, which checks exactness;
    this checks the lossy cache hasn't corrupted state instead."""
    ids = torch.randint(0, TINY["vocab_size"], (1, 20))
    ls = build(tiny_model, tiny_shape)
    ls.allocate_cache(1, 64, paged=True, block_size=4, kv_dtype="int8")
    for _ in range(5):
        ls.cache.reset()
        out = ls.forward_logits_all(ids)
    assert torch.isfinite(out).all()
    assert ls.cache.allocator.num_used == ls.cache.allocator.blocks_for_tokens(20)
