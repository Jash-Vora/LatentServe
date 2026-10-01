"""
Phase 13 — the cache must hold still for a CUDA graph.

A captured graph records memory *addresses*, not Python objects. If a
decode step binds a fresh tensor to the same attribute name, the graph
keeps reading the old one and replays stale data — fluent, plausible,
wrong. Nothing raises.

So the property under test is not "the values are right" (the Phase 3
and 11 suites already check that) but "the values are right **at the
same address**, step after step". Every test here runs on CPU; the
address contract is identical, and the GPU-gated replay tests in
test_phase13_cuda_graph.py depend on it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cache.kv_cache import KVCacheSpec  # noqa: E402
from cache.paged_cache import PagedKVCache  # noqa: E402


def make_cache(batch=2, max_seq=256, block=16):
    spec = KVCacheSpec(num_layers=2, num_kv_heads=2, head_dim=8, max_batch_size=batch,
                       max_seq_len=max_seq, dtype=torch.float32, device="cpu")
    return PagedKVCache(spec, block_size=block)


def test_decode_buffers_keep_their_address_across_steps():
    """The contract a captured graph depends on."""
    cache = make_cache()
    cache.advance(20, batch_size=2)                 # prefill
    cache.advance(1, batch_size=2)                  # first decode step
    addrs = (cache.block_tables_tensor(2).data_ptr(),
             cache.seq_lens_tensor(2).data_ptr(),
             cache._write_slots.data_ptr())
    for _ in range(40):                             # crosses two page boundaries
        cache.advance(1, batch_size=2)
        assert cache.block_tables_tensor(2).data_ptr() == addrs[0]
        assert cache.seq_lens_tensor(2).data_ptr() == addrs[1]
        assert cache._write_slots.data_ptr() == addrs[2]


def test_values_still_update_in_place():
    """Holding still is useless if the contents go stale. Lengths must
    advance and new block ids must appear, at the same address."""
    cache = make_cache(batch=1)
    cache.advance(15, batch_size=1)
    cache.advance(1, batch_size=1)                  # length 16, still one page
    assert cache.seq_lens_tensor(1).tolist() == [16]
    cache.advance(1, batch_size=1)                  # length 17, second page
    assert cache.seq_lens_tensor(1).tolist() == [17]
    blocks = cache.tables[0].blocks
    assert cache.block_tables_tensor(1)[0, : len(blocks)].tolist() == blocks


def test_write_slots_point_at_the_next_token():
    cache = make_cache(batch=1)
    cache.advance(10, batch_size=1)
    cache.advance(1, batch_size=1)
    assert cache._write_slots.tolist() == [[cache.tables[0].slot(10)]]
    cache.advance(1, batch_size=1)
    assert cache._write_slots.tolist() == [[cache.tables[0].slot(11)]]


def test_reset_does_not_reallocate():
    """A graph captured before a reset must still be valid after it, so
    reset clears contents and keeps storage."""
    cache = make_cache()
    cache.advance(5, batch_size=2)
    cache.advance(1, batch_size=2)
    before = (cache._block_tables_buf.data_ptr(), cache._seq_lens_buf.data_ptr(),
              cache._write_slots_buf.data_ptr())
    cache.reset()
    cache.advance(5, batch_size=2)
    cache.advance(1, batch_size=2)
    after = (cache._block_tables_buf.data_ptr(), cache._seq_lens_buf.data_ptr(),
             cache._write_slots_buf.data_ptr())
    assert before == after


def test_block_table_rows_are_rewritten_only_on_change():
    """Once per page of growth, not once per token. A row is rewritten
    when its sequence gains a block or the slot changes hands."""
    cache = make_cache(batch=1)
    cache.advance(16, batch_size=1)                 # exactly one full page
    assert cache._table_state[0] == (0, 1)
    cache.advance(1, batch_size=1)                  # token 17 opens page 2
    assert cache._table_state[0] == (0, 2)
    # Tokens 18..32 all land in page 2 (positions 16..31): fifteen steps,
    # zero rewrites. A page holds block_size tokens, so the next rewrite
    # is due at token 33 and not before.
    for _ in range(15):
        cache.advance(1, batch_size=1)
        assert cache._table_state[0] == (0, 2)
    assert cache.tables[0].length == 32
    cache.advance(1, batch_size=1)                  # token 33 opens page 3
    assert cache._table_state[0] == (0, 3)


def test_read_slots_are_not_built_on_the_kernel_path():
    """The gather path needs a per-token slot index; the kernel path never
    calls read(). Building it eagerly in advance() was a Python loop plus a
    host-to-device copy per sequence, every step, for nothing."""
    cache = make_cache()
    cache.advance(30, batch_size=2)
    cache.advance(1, batch_size=2)
    assert cache._read_slots_dirty, "advance() must not build read slots"
    cache.read(0, batch_size=2)
    assert not cache._read_slots_dirty
    assert cache._read_slots.shape == (2, 31)


def test_capacity_overflow_is_loud():
    cache = make_cache(batch=1, max_seq=32, block=16)
    with pytest.raises(RuntimeError):
        cache.advance(48, batch_size=1)


# ----------------------------------------------------------------------
# The graphable decode core
# ----------------------------------------------------------------------


def _tiny():
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape

    torch.manual_seed(0)
    cfg = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
               num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)
    hf = Qwen2ForCausalLM(Qwen2Config(**cfg)).to(torch.float32).eval()
    shape = ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32")

    def build(impl="sdpa"):
        return LatentServeQwen(hf_model=hf, tokenizer=None, shape=shape, device="cpu",
                               attn_impl=impl, max_seq_len_hint=512)
    return build


def test_static_decode_matches_the_eager_decode():
    """`decode_forward_static` is what gets captured. It must compute the
    same step as `decode_step_ragged` — the only difference being that
    the host bookkeeping (`advance`) moved outside it."""
    build = _tiny()
    prompt = torch.randint(0, 128, (1, 40))
    stream = torch.randint(0, 128, (1, 12))

    eager = build()
    eager.allocate_cache(1, 128, paged=True, block_size=16)
    eager.cache.reset()
    eager.prefill_slot(prompt, slot=0)

    static = build()
    static.allocate_cache(1, 128, paged=True, block_size=16)
    static.cache.reset()
    static.prefill_slot(prompt, slot=0)

    for t in range(12):
        pos = torch.tensor([[40 + t]])
        want = eager.decode_step_ragged(stream[:, t : t + 1], pos, [0])
        static.cache.advance(1, slots=[0])            # host work, outside
        got = static.decode_forward_static(stream[:, t : t + 1], pos, max_position=511)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_rope_bound_skips_the_sync_without_changing_the_result():
    from model.rope import RotaryEmbedding

    rope = RotaryEmbedding(head_dim=16, max_seq_len=256, theta=1e6, device="cpu")
    pos = torch.tensor([[3], [190]])
    a = rope.cos_sin_at(pos, torch.float32)
    b = rope.cos_sin_at(pos, torch.float32, max_position=255)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


def test_kernel_path_handles_ragged_batches_without_a_padding_mask():
    """The kernel masks each sequence by its own length, so it never
    needed the gather path's padding mask — and requiring a uniform
    batch kept continuous batching off the kernel entirely.

    On CPU the kernel path dispatches to the reference implementation of
    the same algorithm, so this checks the decision itself: kernel path
    vs gather path on a genuinely ragged batch.
    """
    build = _tiny()
    long_prompt = torch.randint(0, 128, (1, 50))
    short_prompt = torch.randint(0, 128, (1, 9))

    outs = {}
    for impl in ("sdpa", "triton_paged"):
        m = build(impl)
        m.allocate_cache(2, 128, paged=True, block_size=16)
        m.cache.reset()
        m.prefill_slot(long_prompt, slot=0)
        m.prefill_slot(short_prompt, slot=1)
        tokens = torch.randint(0, 128, (2, 1), generator=torch.Generator().manual_seed(7))
        outs[impl] = m.decode_step_ragged(tokens, torch.tensor([[50], [9]]), [0, 1])
    torch.testing.assert_close(outs["triton_paged"], outs["sdpa"], rtol=1e-4, atol=1e-4)


def test_scratch_buffers_for_different_shapes_coexist():
    """Two captured graphs with different split counts each need their
    own scratch. Keyed by name only, the second capture reallocated the
    buffer and left the first graph pointing at freed memory."""
    from kernels.gqa.paged_decode import _scratch

    a = _scratch("acc", (1, 2, 32, 16, 8), torch.float32, torch.device("cpu"))
    b = _scratch("acc", (1, 2, 64, 16, 8), torch.float32, torch.device("cpu"))
    a_again = _scratch("acc", (1, 2, 32, 16, 8), torch.float32, torch.device("cpu"))
    assert a.data_ptr() != b.data_ptr()
    assert a_again.data_ptr() == a.data_ptr(), "the first buffer must survive the second"