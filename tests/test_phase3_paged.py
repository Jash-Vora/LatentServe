"""
Phase 3 correctness harness — block allocator and paged KV cache.

Gate 4 (docs/methodology.md): "Can paged KV improve memory
utilization?" That only means anything if paging is first shown to be
arithmetically transparent — a paged cache that computes different
attention than a contiguous one has not improved memory utilization,
it has broken the model.

So the central test here is equivalence: the same tiny Qwen2, the same
prompts, contiguous vs. paged, identical logits. Everything else
(fragmentation, capacity, churn) is allocator bookkeeping and runs on
CPU in milliseconds — which is why most of Phase 3 needs no GPU at all.

    export PYTHONPATH=$(pwd):$PYTHONPATH
    pytest tests/test_phase3_paged.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from cache.block_allocator import BlockAllocator, BlockTable, OutOfBlocks  # noqa: E402
from cache.kv_cache import KVCacheSpec  # noqa: E402
from cache.paged_cache import (  # noqa: E402
    PagedKVCache,
    contiguous_reserved_bytes,
    paged_reserved_bytes,
)
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
# Allocator
# ----------------------------------------------------------------------


def test_allocate_and_free_roundtrip():
    a = BlockAllocator(num_blocks=8, block_size=4)
    assert a.num_free == 8
    blocks = a.allocate(3)
    assert a.num_used == 3 and len(set(blocks)) == 3
    a.free(blocks)
    assert a.num_free == 8


def test_exhaustion_raises_typed_error():
    """Phase 4's scheduler has to distinguish 'pool is full, queue this
    request' from a genuine crash, so exhaustion gets its own type."""
    a = BlockAllocator(num_blocks=2, block_size=4)
    a.allocate(2)
    with pytest.raises(OutOfBlocks):
        a.allocate(1)


def test_freed_blocks_are_reused_not_leaked():
    """Churn is the whole point: a pool that cannot recycle after a
    request finishes degrades into a contiguous cache with extra steps."""
    a = BlockAllocator(num_blocks=4, block_size=4)
    for _ in range(50):
        blocks = a.allocate(4)
        a.free(blocks)
    assert a.num_free == 4
    assert a.alloc_calls == 50


def test_refcount_defers_free_until_last_holder():
    a = BlockAllocator(num_blocks=4, block_size=4)
    blocks = a.allocate(2)
    a.incref(blocks)  # a second sequence shares them (Phase 13)
    a.free(blocks)
    assert a.num_free == 2, "still held by the second sequence"
    a.free(blocks)
    assert a.num_free == 4


def test_block_table_grows_on_demand():
    a = BlockAllocator(num_blocks=8, block_size=4)
    t = BlockTable(a)
    t.append(1)
    assert len(t.blocks) == 1 and t.length == 1
    t.append(3)
    assert len(t.blocks) == 1, "4 tokens still fit in one 4-token block"
    t.append(1)
    assert len(t.blocks) == 2


def test_internal_fragmentation_is_bounded_by_block_size():
    """The core trade: wasted slots per sequence never exceed one block,
    so smaller blocks waste less and bookkeep more."""
    a = BlockAllocator(num_blocks=64, block_size=16)
    for n in (1, 5, 16, 17, 33, 100):
        t = BlockTable(a)
        t.append(n)
        assert 0 <= t.wasted_slots < a.block_size
        t.free()


def test_slot_mapping_is_a_bijection():
    """Two logical positions must never map to one physical slot — the
    failure mode that silently corrupts one sequence with another's KV."""
    a = BlockAllocator(num_blocks=8, block_size=4)
    t1, t2 = BlockTable(a), BlockTable(a)
    t1.append(10)
    t2.append(10)
    s1, s2 = t1.slots(), t2.slots()
    assert len(set(s1)) == len(s1)
    assert not set(s1) & set(s2)


# ----------------------------------------------------------------------
# Paged cache storage
# ----------------------------------------------------------------------


def test_paged_write_read_roundtrip():
    spec = tiny_spec(batch=2, max_seq=32)
    cache = PagedKVCache(spec, block_size=8)
    cache.advance(5, batch_size=2)
    k = torch.randn(2, 2, 5, 4)
    cache.write(0, k, k * 2, start_pos=0)
    cache.advance(3, batch_size=2)
    k2 = torch.randn(2, 2, 3, 4)
    cache.write(0, k2, k2 * 2, start_pos=5)

    k_all, v_all = cache.read(0, batch_size=2)
    torch.testing.assert_close(k_all[:, :, :5], k)
    torch.testing.assert_close(k_all[:, :, 5:8], k2)
    torch.testing.assert_close(v_all, k_all * 2)


def test_paged_sequences_do_not_alias():
    """Different sequences writing at the same logical position must land
    in different physical slots."""
    spec = tiny_spec(batch=2, max_seq=32)
    cache = PagedKVCache(spec, block_size=4)
    cache.advance(6, batch_size=2)
    k = torch.stack([torch.ones(2, 6, 4), torch.full((2, 6, 4), 7.0)])
    cache.write(0, k, k, start_pos=0)
    k_all, _ = cache.read(0, batch_size=2)
    assert (k_all[0] == 1).all() and (k_all[1] == 7).all()


def test_freeing_one_sequence_returns_its_blocks():
    """The operation a contiguous cache cannot express."""
    spec = tiny_spec(batch=4, max_seq=32)
    cache = PagedKVCache(spec, block_size=8)
    cache.advance(16, batch_size=4)
    used = cache.allocator.num_used
    cache.free_sequence(1)
    assert cache.allocator.num_used == used - 2


def test_ragged_lengths_and_padding_mask():
    spec = tiny_spec(batch=2, max_seq=32)
    cache = PagedKVCache(spec, block_size=4)
    cache.advance(4, batch_size=2)
    cache.tables[1].length = 2  # sequence 1 finished shorter
    mask = cache.padding_mask()
    assert mask is not None and mask.shape == (2, 1, 1, 4)
    assert mask[0, 0, 0].all()
    assert mask[1, 0, 0].tolist() == [True, True, False, False]
    with pytest.raises(RuntimeError):
        _ = cache.length  # ragged: must not silently report a single length


# ----------------------------------------------------------------------
# Capacity accounting — the Gate 4 numbers
# ----------------------------------------------------------------------


def test_paged_reserves_less_than_contiguous_on_ragged_workloads():
    """A contiguous cache reserves max_seq_len per sequence whichever
    lengths actually arrive; paging reserves each sequence rounded up to
    a block. With a mix of short and long requests that is the whole
    argument for Phase 3."""
    spec = tiny_spec(batch=8, max_seq=2048)
    lens = [16, 32, 64, 2048, 128, 24, 48, 96]
    contig = contiguous_reserved_bytes(spec, lens)
    paged = paged_reserved_bytes(spec, lens, block_size=16)
    assert paged < contig / 4


def test_uniform_max_length_is_the_worst_case_for_paging():
    """Honesty check: when every sequence really is max_seq_len, paging
    reserves the same bytes and only adds bookkeeping. Any benchmark
    showing paging 'winning' here would be measuring a mistake."""
    spec = tiny_spec(batch=4, max_seq=256)
    lens = [256] * 4
    assert paged_reserved_bytes(spec, lens, 16) == contiguous_reserved_bytes(spec, lens)


def test_fragmentation_reported_within_one_block_per_sequence():
    spec = tiny_spec(batch=4, max_seq=256)
    cache = PagedKVCache(spec, block_size=16)
    cache.advance(20, batch_size=4)
    frag = cache.fragmentation(4)
    assert 0 < frag < 16 / 20


# ----------------------------------------------------------------------
# Equivalence — paging must not change the model's arithmetic
# ----------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_paged_matches_contiguous_logits(tiny_model, tiny_shape, block_size):
    """Gate 4's precondition, across block sizes — including block_size=1,
    the maximally scattered case, where any slot-mapping error shows up
    immediately."""
    ids = torch.randint(0, TINY["vocab_size"], (2, 20))
    outs = []
    for paged in (False, True):
        ls = build(tiny_model, tiny_shape)
        ls.allocate_cache(2, 64, paged=paged, block_size=block_size)
        ls.cache.reset()
        outs.append(ls.forward_logits_all(ids))
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-4, atol=1e-4)


def test_paged_decode_matches_contiguous(tiny_model, tiny_shape):
    """Prefill then incremental decode, which is where block boundaries
    get crossed mid-sequence."""
    ids = torch.randint(0, TINY["vocab_size"], (1, 24))
    outs = []
    for paged in (False, True):
        ls = build(tiny_model, tiny_shape)
        ls.allocate_cache(1, 64, paged=paged, block_size=4)
        ls.cache.reset()
        ls.prefill(ids[:, :10])
        outs.append(torch.cat([ls.decode_step(ids[:, t : t + 1]) for t in range(10, 24)], dim=1))
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-4, atol=1e-4)


def test_paged_survives_churn_and_still_matches(tiny_model, tiny_shape):
    """Run several generations through one pool so blocks get recycled,
    then check the last one still matches contiguous. Catches a stale
    block table or a slot index that was never rebuilt."""
    ids = torch.randint(0, TINY["vocab_size"], (1, 16))
    ls = build(tiny_model, tiny_shape)
    ls.allocate_cache(1, 64, paged=True, block_size=4)
    for _ in range(5):
        ls.cache.reset()
        paged_out = ls.forward_logits_all(ids)

    ref = build(tiny_model, tiny_shape)
    ref.allocate_cache(1, 64)
    ref.cache.reset()
    torch.testing.assert_close(paged_out, ref.forward_logits_all(ids), rtol=1e-4, atol=1e-4)
    assert ls.cache.allocator.num_used == ls.cache.allocator.blocks_for_tokens(16)
