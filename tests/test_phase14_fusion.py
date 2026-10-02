"""
Phase 14a — projection fusion.

Three properties, each a way the change could be wrong while looking
fine:

  * **exactness** — fused and unfused compute the same function, so they
    must agree to float32 round-off, not just "closely";
  * **no extra memory** — the fused tensors and the checkpoint's own
    parameters must be the *same* storage, or the model quietly grows by
    ~1.7 GB;
  * **fewer launches** — the reason for doing it. Counted directly, since
    a timing improvement can come from anywhere.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from torch import nn  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from model.fused import FusedMLP, concat_into_views  # noqa: E402
from model.latentserve_qwen import LatentServeQwen  # noqa: E402
from model.qwen import ModelShape  # noqa: E402

CFG = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
           num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)
SHAPE = ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32")
requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def hf_model(device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(**CFG)).to(dtype).eval().to(device)


def build(hf, device="cpu", fuse=False, impl="triton_paged"):
    m = LatentServeQwen(hf_model=hf, tokenizer=None, shape=SHAPE, device=device,
                        attn_impl=impl, max_seq_len_hint=256, fuse_projections=fuse)
    return m


def decode_logits(m, prompt, stream):
    m.allocate_cache(1, 128, paged=True, block_size=16)
    m.cache.reset()
    m.prefill_slot(prompt, slot=0)
    out = []
    for t in range(stream.shape[1]):
        pos = torch.tensor([[prompt.shape[1] + t]], device=prompt.device)
        out.append(m.decode_step_ragged(stream[:, t : t + 1], pos, [0]))
    return torch.cat(out, dim=1)


# ----------------------------------------------------------------------
# Exactness
# ----------------------------------------------------------------------


def test_fused_matches_unfused_exactly():
    """Same weights, same arithmetic, one launch instead of several. In
    float32 the only difference is summation order inside the matmul."""
    prompt = torch.randint(0, 128, (1, 30))
    stream = torch.randint(0, 128, (1, 10))
    want = decode_logits(build(hf_model()), prompt, stream)
    got = decode_logits(build(hf_model(), fuse=True), prompt, stream)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_switching_is_exact_both_ways():
    """`set_fused` toggles one model between the two paths. They share
    every weight, so the toggle must be free and exact — that is what
    lets fused and unfused be measured in one process."""
    hf = hf_model()
    m = build(hf, fuse=True)
    prompt = torch.randint(0, 128, (1, 20))
    stream = torch.randint(0, 128, (1, 6))
    fused = decode_logits(m, prompt, stream)
    m.set_fused(False)
    unfused = decode_logits(m, prompt, stream)
    m.set_fused(True)
    again = decode_logits(m, prompt, stream)
    torch.testing.assert_close(fused, unfused, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(again, fused, rtol=0, atol=0)


def test_the_hf_reference_still_works_after_fusion():
    """LatentServe borrows the checkpoint's modules by reference, and
    fusion re-points their weights. The HF model — the Phase 1 oracle —
    must compute exactly what it did before."""
    hf = hf_model()
    ids = torch.randint(0, 128, (1, 24))
    with torch.no_grad():
        before = hf(input_ids=ids).logits
    build(hf, fuse=True)
    with torch.no_grad():
        after = hf(input_ids=ids).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)


# ----------------------------------------------------------------------
# No extra memory
# ----------------------------------------------------------------------


def _storage_bytes(module: nn.Module) -> int:
    seen, total = set(), 0
    for p in module.parameters():
        key = p.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            total += p.untyped_storage().nbytes()
    return total


def test_fusion_shares_the_checkpoints_storage():
    """The fused tensors must *be* the checkpoint's parameters, not a
    second copy of them. A copy would cost ~1.7 GB on Qwen2.5-1.5B."""
    hf = hf_model()
    m = build(hf, fuse=True)
    attn, mlp = m.layers[0].attn, m.layers[0].mlp
    assert attn.q_proj.weight.untyped_storage().data_ptr() == \
        attn._qkv_weight.untyped_storage().data_ptr()
    assert attn.v_proj.bias.untyped_storage().data_ptr() == \
        attn._qkv_bias.untyped_storage().data_ptr()
    assert mlp.inner.up_proj.weight.untyped_storage().data_ptr() == \
        mlp.weight.untyped_storage().data_ptr()


def test_fusion_does_not_grow_the_model():
    hf = hf_model()
    before = _storage_bytes(hf)
    build(hf, fuse=True)
    assert _storage_bytes(hf) == before


# ----------------------------------------------------------------------
# Fewer launches — the reason for doing it
# ----------------------------------------------------------------------


def _linear_calls_per_decode_step(m) -> int:
    m.allocate_cache(1, 128, paged=True, block_size=16)
    m.cache.reset()
    m.prefill_slot(torch.randint(0, 128, (1, 20)), slot=0)
    tok = torch.zeros(1, 1, dtype=torch.long)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        m.decode_step_ragged(tok, torch.tensor([[20]]), [0])
    return sum(e.count for e in prof.key_averages() if e.key == "aten::linear")


def test_fusion_removes_three_projections_per_layer():
    """q/k/v -> one saves two; gate/up -> one saves one. 2 layers here,
    so 6 fewer; on Qwen2.5-1.5B's 28 layers that is the 84 per step the
    phase is about."""
    unfused = _linear_calls_per_decode_step(build(hf_model()))
    fused = _linear_calls_per_decode_step(build(hf_model(), fuse=True))
    assert unfused - fused == 3 * CFG["num_hidden_layers"]


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------


def test_mixed_bias_presence_is_refused():
    """Fusing a biased projection with an unbiased one would mean
    inventing a zero bias — a silent change to the model."""
    a, b = nn.Linear(4, 3, bias=True), nn.Linear(4, 3, bias=False)
    with pytest.raises(ValueError, match="only some"):
        concat_into_views([a, b], "bias")


def test_non_silu_mlp_is_refused():
    class Gelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj, self.up_proj = nn.Linear(4, 8, bias=False), nn.Linear(4, 8, bias=False)
            self.down_proj, self.act_fn = nn.Linear(8, 4, bias=False), nn.GELU()

    with pytest.raises(ValueError, match="silu"):
        FusedMLP(Gelu())


# ----------------------------------------------------------------------
# Serving and graphs
# ----------------------------------------------------------------------


def test_engine_output_is_unchanged_by_fusion():
    """Gate 3 again: fusion must not change what any request receives,
    including through the graph path's orchestration."""
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    def serve(fuse):
        m = build(hf_model(), fuse=fuse)
        engine = ServingEngine(m, max_running=3, max_seq_len=256, block_size=16,
                               use_cuda_graphs=True)
        g = torch.Generator().manual_seed(0)
        for i, (p, n) in enumerate([(12, 9), (40, 4), (7, 14), (25, 6)]):
            engine.add_request(ServedRequest(
                request_id=i, prompt_ids=torch.randint(0, 128, (p,), generator=g).tolist(),
                max_new_tokens=n))
        return {r.request_id: r.output_ids for r in engine.run()}

    assert serve(True) == serve(False)


@requires_gpu
def test_fused_graph_matches_unfused_eager_on_gpu():
    from runtime.cuda_graph import GraphedDecoder

    prompt = torch.randint(0, 128, (1, 30)).cuda()
    stream = torch.randint(0, 128, (1, 20)).cuda()

    eager = build(hf_model("cuda", torch.float16), device="cuda")
    want = decode_logits(eager, prompt, stream)

    graphed = build(hf_model("cuda", torch.float16), device="cuda", fuse=True)
    graphed.allocate_cache(1, 128, paged=True, block_size=16)
    graphed.cache.reset()
    graphed.prefill_slot(prompt, slot=0)
    decoder = GraphedDecoder(graphed)
    got = []
    for t in range(20):
        pos = torch.tensor([[30 + t]]).cuda()
        got.append(decoder.step(stream[:, t : t + 1], pos, [0]).clone())
    torch.testing.assert_close(torch.cat(got, dim=1), want, rtol=3e-2, atol=3e-2)
