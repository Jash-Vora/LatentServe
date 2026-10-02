"""
Phase 14b — the serving loop.

In-graph sampling and on-device token feedback change *where* the next
token is chosen and *how* it reaches the next step, never *which* token
it is: the engine is greedy, and argmax is argmax wherever it runs. So
the first property is the Gate 3 one — no request's output changes.

The second is a specific trap. Feeding last step's tokens back on the
device is only valid when this step serves exactly the same requests in
the same order. Keyed on slots, it would be wrong in the most ordinary
situation there is: one request finishes, the next is admitted into the
slot it freed, the slot list is unchanged — and the newcomer is fed the
previous occupant's last token.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from model.latentserve_qwen import LatentServeQwen  # noqa: E402
from model.qwen import ModelShape  # noqa: E402
from runtime.engine import ServingEngine  # noqa: E402
from runtime.request import ServedRequest  # noqa: E402

CFG = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
           num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)
SHAPE = ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32")
requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def engine(device="cpu", dtype=torch.float32, max_running=3, **kw):
    torch.manual_seed(0)
    hf = Qwen2ForCausalLM(Qwen2Config(**CFG)).to(dtype).eval().to(device)
    model = LatentServeQwen(hf_model=hf, tokenizer=None, shape=SHAPE, device=device,
                            attn_impl="triton_paged", max_seq_len_hint=256)
    return ServingEngine(model, max_running=max_running, max_seq_len=256, block_size=16,
                         use_cuda_graphs=True, **kw)


def requests(specs, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [ServedRequest(request_id=i,
                          prompt_ids=torch.randint(0, 128, (p,), generator=g).tolist(),
                          max_new_tokens=n)
            for i, (p, n) in enumerate(specs)]


def serve(e, specs):
    for r in requests(specs):
        e.add_request(r)
    return {r.request_id: r.output_ids for r in e.run()}


RAGGED = [(12, 9), (40, 4), (7, 14), (25, 6), (18, 11)]


def test_in_graph_sampling_changes_no_output():
    """Gate 3: ragged prompts and lengths, so the batch changes and slots
    are recycled mid-run — both the device-fed and host-fed paths run."""
    on = engine(sample_in_graph=True)
    off = engine(sample_in_graph=False)
    assert serve(on, RAGGED) == serve(off, RAGGED)
    assert on.device_fed_steps > 0 and on.host_fed_steps > 0


def test_steady_state_feeds_tokens_on_device():
    """One request decoding alone: only its first decode step needs inputs
    built on the host (the token came from prefill). Every step after
    that reuses the previous step's output in place."""
    e = engine(max_running=1)
    serve(e, [(20, 12)])
    # 12 new tokens: one from prefill, 11 decode steps.
    assert e.host_fed_steps == 1
    assert e.device_fed_steps == 10


def test_a_new_request_in_a_freed_slot_is_not_fed_the_old_token():
    """The trap: max_running=1, so request 1 is admitted into exactly the
    slot request 0 freed. Same slot list, different request. Its output
    must match what it produces when served on its own."""
    g = torch.Generator().manual_seed(3)
    prompt_a = torch.randint(0, 128, (15,), generator=g).tolist()
    prompt_b = torch.randint(0, 128, (22,), generator=g).tolist()

    def run(prompts, sample_in_graph=True):
        e = engine(max_running=1, sample_in_graph=sample_in_graph)
        for i, (prompt, n) in enumerate(prompts):
            e.add_request(ServedRequest(request_id=i, prompt_ids=prompt, max_new_tokens=n))
        return {r.request_id: r.output_ids for r in e.run()}, e

    together, e = run([(prompt_a, 5), (prompt_b, 7)])
    alone, _ = run([(prompt_b, 7)])
    host_sampled, _ = run([(prompt_a, 5), (prompt_b, 7)], sample_in_graph=False)

    assert e.cache.tables  # both requests went through the one slot
    assert together[1] == alone[0], "request B was fed request A's last token"
    assert together == host_sampled


def test_loop_profile_reports_every_phase():
    e = engine(profile_loop=True)
    serve(e, RAGGED)
    stats = e.stats()
    for phase in ("inputs", "decoder_host", "gpu_wait", "sample", "bookkeeping", "schedule"):
        assert f"loop_{phase}_ms_per_step" in stats, phase
    assert e.profiled_steps == e.decode_steps


def test_profiling_is_off_by_default():
    e = engine()
    serve(e, [(10, 4)])
    assert not any(k.startswith("loop_") for k in e.stats())


@requires_gpu
def test_in_graph_sampling_matches_host_sampling_on_gpu():
    on = serve(engine("cuda", torch.float16, sample_in_graph=True), RAGGED)
    off = serve(engine("cuda", torch.float16, sample_in_graph=False), RAGGED)
    # Same graph, same logits, argmax in or out of it: identical tokens.
    assert on == off


@requires_gpu
def test_device_feedback_runs_under_real_graphs():
    e = engine("cuda", torch.float16, max_running=1)
    serve(e, [(20, 12)])
    assert e.device_fed_steps == 10
    assert e.decoder.graph_steps > 0
