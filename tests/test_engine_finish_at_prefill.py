"""A request can finish during prefill: it asked for one token, or its first
token is end-of-sequence. It must be retired without ever joining the decode
batch (found by the final sweep's section B: the engine crashed)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _engine(**kw):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape
    from runtime.engine import ServingEngine

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)
    ls = LatentServeQwen(hf_model=Qwen2ForCausalLM(cfg).eval(), tokenizer=None,
                         shape=ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32"),
                         device="cpu", attn_impl="sdpa", max_seq_len_hint=512)
    return ServingEngine(ls, max_running=4, max_seq_len=256, block_size=16, **kw)


def _req(i, n_prompt=30, max_new=1):
    from runtime.request import ServedRequest

    return ServedRequest(request_id=i, prompt_ids=[(7 * i + j) % 128 for j in range(n_prompt)],
                         max_new_tokens=max_new)


def test_one_token_requests_finish_at_prefill_alone_and_alongside_others():
    eng = _engine()
    eng.add_request(_req(0, max_new=1))
    eng.add_request(_req(1, max_new=6))                    # decodes alongside
    eng.add_request(_req(2, max_new=1))
    done = {r.request_id: r for r in eng.run()}
    assert [len(done[i].output_ids) for i in range(3)] == [1, 6, 1]
    assert all(r.slot is None for r in done.values())
    assert eng.cache.allocator.num_free == eng.cache.allocator.num_blocks   # nothing leaked


def test_end_of_sequence_as_the_first_token_finishes_at_prefill():
    probe = _engine()
    probe.add_request(_req(5, max_new=1))
    first = probe.run()[0].output_ids[0]
    eng = _engine()
    eng.eos_token_id = first                                # the first token is EOS
    eng.add_request(_req(5, max_new=20))
    eng.add_request(_req(6, max_new=4))
    done = {r.request_id: r for r in eng.run()}
    assert done[5].output_ids == [first]
    assert len(done[6].output_ids) >= 1


def test_one_token_requests_work_with_prefix_caching():
    eng = _engine(prefix_caching=True)
    for i in range(3):
        eng.add_request(_req(0, n_prompt=40, max_new=1))   # identical prompts: reuse
        eng.run()
    assert eng.prefix_hit_tokens == 2 * 32
