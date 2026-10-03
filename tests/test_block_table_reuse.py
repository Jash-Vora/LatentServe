"""
A reused slot must never show the kernel its previous request's blocks.

The kernel's persistent block-table rows were rewritten only when a row's
(slot, block count) changed. A finished request's slot, reused by a new
request with the same block count in the same batch row, kept the old
blocks — and the allocator's LIFO free list handed back the same ids in
reverse order, so the stale row looked plausible. A uniform burst served
in waves, which is every benchmark in this project, triggers it.

The rows now key on a per-table version that moves on every change to
the block list.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cache.int8_paged_cache import Int8PagedKVCache  # noqa: E402
from cache.kv_cache import KVCacheSpec  # noqa: E402
from cache.paged_cache import PagedKVCache  # noqa: E402


def _spec():
    return KVCacheSpec(num_layers=1, num_kv_heads=2, head_dim=8, max_batch_size=2,
                       max_seq_len=128, dtype=torch.float32, device="cpu")


@pytest.mark.parametrize("cls", [PagedKVCache, Int8PagedKVCache])
def test_same_slot_same_row_same_count_gets_its_own_blocks(cls):
    c = cls(_spec(), block_size=16)
    c.advance(20, slots=[0])
    c.advance(20, slots=[1])
    c.advance(1, slots=[1, 0])                    # row 0 = slot 1, row 1 = slot 0
    c.free_sequence(0)
    c.advance(5, slots=[0])                       # shuffle the free list
    c.tables[0].free()
    c.advance(20, slots=[0])                      # new request, same block count
    c.advance(1, slots=[1, 0])                    # back in row 1
    owns = c.tables[0].blocks
    assert c.block_tables_tensor(2)[1, : len(owns)].tolist() == owns


def test_uniform_waves_through_the_kernel_match_the_gather_path():
    """The realistic trigger: equal-length requests served two at a time,
    so the second wave reuses both slots, in the same rows, with the same
    block counts. The kernel path must produce what the gather path does."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    cfg = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
               num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)

    def serve(impl):
        torch.manual_seed(0)
        hf = Qwen2ForCausalLM(Qwen2Config(**cfg)).eval()
        m = LatentServeQwen(hf_model=hf, tokenizer=None,
                            shape=ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32"),
                            device="cpu", attn_impl=impl, max_seq_len_hint=256)
        e = ServingEngine(m, max_running=2, max_seq_len=256, block_size=16)
        g = torch.Generator().manual_seed(1)
        for i in range(4):
            e.add_request(ServedRequest(request_id=i,
                          prompt_ids=torch.randint(0, 128, (20,), generator=g).tolist(),
                          max_new_tokens=8))
        return {r.request_id: r.output_ids for r in e.run()}

    assert serve("triton_paged") == serve("sdpa")
