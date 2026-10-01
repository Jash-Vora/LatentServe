"""
Phase 13 — CUDA graph decode.

The CPU tests check the orchestration: which configurations are refused,
how buckets are chosen, and that the step sequence (advance outside, the
static core inside) computes the same thing as eager decode.

The GPU tests check replay itself, and the most important one is the
least glamorous: **two different inputs through the same graph must give
different outputs**. The classic graph bug is forgetting to copy a new
input into the static buffer, after which the graph replays last step's
tokens and produces fluent, plausible output for the wrong input.
Nothing raises. Every other test can pass while that one is broken.

    pytest tests/test_phase13_cuda_graph.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from model.latentserve_qwen import LatentServeQwen  # noqa: E402
from model.qwen import ModelShape  # noqa: E402
from runtime.cuda_graph import GraphedDecoder, GraphUnsupported, check_capturable  # noqa: E402

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CFG = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
           num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)
SHAPE = ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32")


def build(device="cpu", dtype=torch.float32, impl="triton_paged", paged=True,
          kv_dtype="fp16", capacity=256):
    torch.manual_seed(0)
    hf = Qwen2ForCausalLM(Qwen2Config(**CFG)).to(dtype).eval().to(device)
    m = LatentServeQwen(hf_model=hf, tokenizer=None, shape=SHAPE, device=device,
                        attn_impl=impl, max_seq_len_hint=capacity)
    kw = {"kv_dtype": kv_dtype} if paged else {}
    m.allocate_cache(2, capacity, paged=paged, block_size=16, **kw)
    m.cache.reset()
    return m


# ----------------------------------------------------------------------
# What a graph refuses, and why
# ----------------------------------------------------------------------


def test_contiguous_cache_is_refused():
    with pytest.raises(GraphUnsupported, match="slice whose length changes"):
        check_capturable(build(paged=False, impl="sdpa"))


def test_int8_cache_is_refused():
    with pytest.raises(GraphUnsupported, match="INT8"):
        check_capturable(build(kv_dtype="int8"))


def test_gather_path_is_refused():
    with pytest.raises(GraphUnsupported, match="triton_paged"):
        check_capturable(build(impl="sdpa"))


def test_buckets_end_at_capacity():
    """Buckets past capacity are unreachable, and the capacity itself must
    be the last bucket so no reachable length falls through to eager."""
    m = build(capacity=3000)
    d = GraphedDecoder(m, buckets=(1024, 2048, 4096, 8192), enabled=False)
    assert d.buckets == (1024, 2048, 3000)
    assert d.bucket_for(10) == 1024
    assert d.bucket_for(1024) == 1024
    assert d.bucket_for(1025) == 2048
    assert d.bucket_for(3000) == 3000


def test_rope_is_prebuilt_to_capacity():
    """A rebuild after capture would leave graphs pointing at freed
    cos/sin tables. Building to capacity up front makes one impossible."""
    m = build(capacity=256)
    m.rope._build_tables(64)
    GraphedDecoder(m, enabled=False)
    assert m.rope.max_seq_len >= 256


# ----------------------------------------------------------------------
# The step sequence, on CPU (eager fallback, same orchestration)
# ----------------------------------------------------------------------


def test_decoder_step_matches_eager_decode():
    """`step()` advances outside, then runs the static core. With graphs
    disabled it must reproduce `decode_step_ragged` exactly — this checks
    the orchestration on any machine, independent of capture."""
    prompt = torch.randint(0, 128, (1, 30))
    stream = torch.randint(0, 128, (1, 20))

    eager = build(impl="sdpa")
    eager.prefill_slot(prompt, slot=0)

    graphed = build()
    graphed.prefill_slot(prompt, slot=0)
    decoder = GraphedDecoder(graphed, enabled=False)

    for t in range(20):          # crosses the page boundary at 32
        pos = torch.tensor([[30 + t]])
        want = eager.decode_step_ragged(stream[:, t : t + 1], pos, [0])
        got = decoder.step(stream[:, t : t + 1], pos, [0])
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    assert decoder.eager_steps == 20


# ----------------------------------------------------------------------
# Replay (GPU)
# ----------------------------------------------------------------------


def _pair(steps=24, prompt_len=30):
    """An eager model and a graphed model on identical weights and prompt."""
    prompt = torch.randint(0, 128, (1, prompt_len)).cuda()
    eager = build("cuda", torch.float16)
    eager.prefill_slot(prompt, slot=0)
    graphed = build("cuda", torch.float16)
    graphed.prefill_slot(prompt, slot=0)
    return eager, graphed, GraphedDecoder(graphed)


@requires_gpu
def test_replay_matches_eager_across_a_page_boundary():
    """Gate 13: same logits as eager, step after step, including the step
    where the sequence gains a page and its block-table row is rewritten
    in place behind a graph that is already captured."""
    eager, graphed, decoder = _pair()
    stream = torch.randint(0, 128, (1, 24)).cuda()
    for t in range(24):
        pos = torch.tensor([[30 + t]]).cuda()
        want = eager.decode_step_ragged(stream[:, t : t + 1], pos, [0])
        got = decoder.step(stream[:, t : t + 1], pos, [0]).clone()
        torch.testing.assert_close(got, want, rtol=3e-2, atol=3e-2)
    assert decoder.captures == 1, "one bucket, one capture"
    assert decoder.graph_steps == 24


@requires_gpu
def test_different_inputs_give_different_outputs():
    """The test that matters most. If new inputs are not copied into the
    static buffers, the graph replays the previous step and the output
    for token B is the output for token A — fluent, plausible, wrong."""
    _, graphed, decoder = _pair()
    pos = torch.tensor([[30]]).cuda()
    first = decoder.step(torch.tensor([[5]]).cuda(), pos, [0]).clone()
    second = decoder.step(torch.tensor([[99]]).cuda(), pos + 1, [0]).clone()
    assert not torch.allclose(first, second), "the graph replayed stale inputs"


@requires_gpu
def test_a_second_graph_does_not_corrupt_the_first():
    """Scratch is keyed by shape so a second capture with a different
    split count cannot reallocate the first graph's buffers."""
    eager, graphed, decoder = _pair()
    decoder.num_splits_override = {(1, 1024): 4}
    stream = torch.randint(0, 128, (1, 4)).cuda()
    for t in range(2):
        pos = torch.tensor([[30 + t]]).cuda()
        eager.decode_step_ragged(stream[:, t : t + 1], pos, [0])
        decoder.step(stream[:, t : t + 1], pos, [0])

    # A second capture with a different split count. Its warm-up runs
    # this step and writes its KV; eager runs the same step, so the two
    # caches stay in step.
    from runtime.cuda_graph import CapturedDecode

    other = CapturedDecode(graphed, 1, 2048, num_splits=16)
    graphed.cache.advance(1, slots=[0])
    other.load(stream[:, 2:3], torch.tensor([[32]]).cuda())
    other.capture()
    eager.decode_step_ragged(stream[:, 2:3], torch.tensor([[32]]).cuda(), [0])

    pos = torch.tensor([[33]]).cuda()
    want = eager.decode_step_ragged(stream[:, 3:4], pos, [0])
    got = decoder.step(stream[:, 3:4], pos, [0]).clone()
    torch.testing.assert_close(got, want, rtol=3e-2, atol=3e-2)


@requires_gpu
def test_graph_survives_a_cache_reset():
    """A new generation reuses the same graph: reset clears contents and
    keeps storage, so the captured addresses are still valid."""
    eager, graphed, decoder = _pair()
    stream = torch.randint(0, 128, (1, 6)).cuda()
    for t in range(3):
        pos = torch.tensor([[30 + t]]).cuda()
        decoder.step(stream[:, t : t + 1], pos, [0])

    prompt = torch.randint(0, 128, (1, 20)).cuda()
    for m in (eager, graphed):
        m.cache.reset()
        m.prefill_slot(prompt, slot=0)
    for t in range(3, 6):
        pos = torch.tensor([[20 + t - 3]]).cuda()
        want = eager.decode_step_ragged(stream[:, t : t + 1], pos, [0])
        got = decoder.step(stream[:, t : t + 1], pos, [0]).clone()
        torch.testing.assert_close(got, want, rtol=3e-2, atol=3e-2)
    assert decoder.captures == 1, "the reset must not force a recapture"