"""
Phase 14a step 2 — elementwise fusion.

Two kinds of test, for two kinds of wrong.

The **references** must be the Hugging Face computation, and the
restructured layer loop — residual carried into the next norm, the last
add folded into the final norm — must compute what the original loop
does. Both run on CPU, where the references execute, and both are exact.

The **Triton kernels** must match their references. That needs a GPU, and
is to fp16 round-off: the only remaining difference is the order of the
sum inside the RMSNorm variance.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch.nn.functional as F  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm  # noqa: E402

from kernels import fused_elementwise as fe  # noqa: E402
from model.latentserve_qwen import LatentServeQwen  # noqa: E402
from model.qwen import ModelShape  # noqa: E402
from model.rope import apply_rope  # noqa: E402

CFG = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
           num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024)
SHAPE = ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32")
requires_gpu = pytest.mark.skipif(not (torch.cuda.is_available() and fe.HAS_TRITON),
                                  reason="needs CUDA with Triton")


# ----------------------------------------------------------------------
# References are the model's own computation
# ----------------------------------------------------------------------


def test_rms_norm_ref_is_qwen2_rmsnorm():
    norm = Qwen2RMSNorm(64, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(64))
    x = torch.randn(3, 5, 64)
    torch.testing.assert_close(fe.rms_norm_ref(x, norm.weight, norm.variance_epsilon),
                               norm(x), rtol=0, atol=0)


def test_silu_and_mul_ref_is_the_mlp_activation():
    gu = torch.randn(2, 3, 32)
    gate, up = gu.chunk(2, dim=-1)
    torch.testing.assert_close(fe.silu_and_mul_ref(gu), F.silu(gate) * up, rtol=0, atol=0)


def test_rope_ref_matches_apply_rope_on_the_other_layout():
    """The fused kernel works on [B, S, H*D], before the transpose; the
    original on [B, H, S, D], after it. Same numbers either way."""
    b, s, hq, hk, d = 2, 3, 4, 2, 16
    q, k = torch.randn(b, s, hq * d), torch.randn(b, s, hk * d)
    cos, sin = torch.randn(b, 1, s, d), torch.randn(b, 1, s, d)
    want_q, want_k = apply_rope(q.reshape(b, s, hq, d).transpose(1, 2),
                                k.reshape(b, s, hk, d).transpose(1, 2), cos, sin)
    got_q, got_k = fe.rope_qk_ref(q, k, cos, sin, hq, hk, d)
    torch.testing.assert_close(got_q.reshape(b, s, hq, d).transpose(1, 2), want_q,
                               rtol=0, atol=0)
    torch.testing.assert_close(got_k.reshape(b, s, hk, d).transpose(1, 2), want_k,
                               rtol=0, atol=0)


# ----------------------------------------------------------------------
# The restructured layer loop
# ----------------------------------------------------------------------


def _hf(device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(**CFG)).to(dtype).eval().to(device)


def _decode(m, prompt, stream):
    m.allocate_cache(1, 128, paged=True, block_size=16)
    m.cache.reset()
    m.prefill_slot(prompt, slot=0)
    out = []
    for t in range(stream.shape[1]):
        pos = torch.tensor([[prompt.shape[1] + t]], device=prompt.device)
        out.append(m.decode_step_ragged(stream[:, t : t + 1], pos, [0]))
    return torch.cat(out, dim=1)


def build(hf, device="cpu", **kw):
    return LatentServeQwen(hf_model=hf, tokenizer=None, shape=SHAPE, device=device,
                           attn_impl="triton_paged", max_seq_len_hint=256, **kw)


def test_fused_loop_computes_the_original_loop_exactly():
    """Prefill and decode both run through the restructured loop, so
    both are compared. On CPU the references execute, so any difference
    is the loop's restructuring — and there must be none."""
    prompt = torch.randint(0, 128, (1, 30))
    stream = torch.randint(0, 128, (1, 10))
    want = _decode(build(_hf()), prompt, stream)
    got = _decode(build(_hf(), fuse_elementwise=True), prompt, stream)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_elementwise_implies_fused_projections():
    """The fused activation reads the merged gate/up output."""
    m = build(_hf(), fuse_elementwise=True)
    assert m.fused and m.elementwise
    assert all(layer.mlp.fused_act for layer in m.layers)


def test_switching_elementwise_is_exact_both_ways():
    m = build(_hf(), fuse_projections=True)
    prompt = torch.randint(0, 128, (1, 20))
    stream = torch.randint(0, 128, (1, 6))
    off = _decode(m, prompt, stream)
    m.set_elementwise(True)
    on = _decode(m, prompt, stream)
    m.set_elementwise(False)
    again = _decode(m, prompt, stream)
    torch.testing.assert_close(on, off, rtol=0, atol=0)
    torch.testing.assert_close(again, off, rtol=0, atol=0)


def test_engine_output_is_unchanged_by_elementwise_fusion():
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    def serve(on):
        e = ServingEngine(build(_hf(), fuse_elementwise=on, fuse_projections=True),
                          max_running=3, max_seq_len=256, block_size=16, use_cuda_graphs=True)
        g = torch.Generator().manual_seed(0)
        for i, (p, n) in enumerate([(12, 9), (40, 4), (7, 14), (25, 6)]):
            e.add_request(ServedRequest(request_id=i,
                          prompt_ids=torch.randint(0, 128, (p,), generator=g).tolist(),
                          max_new_tokens=n))
        return {r.request_id: r.output_ids for r in e.run()}

    assert serve(True) == serve(False)


# ----------------------------------------------------------------------
# Triton kernels against their references (GPU)
# ----------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("n", [1536, 128, 100])
def test_triton_rms_norm(n):
    x = torch.randn(7, n, device="cuda", dtype=torch.float16)
    w = torch.randn(n, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(fe.rms_norm(x, w, 1e-6), fe.rms_norm_ref(x, w, 1e-6),
                               rtol=4e-3, atol=4e-3)


@requires_gpu
def test_triton_fused_add_rms_norm():
    x = torch.randn(5, 1536, device="cuda", dtype=torch.float16)
    r = torch.randn(5, 1536, device="cuda", dtype=torch.float16)
    w = torch.randn(1536, device="cuda", dtype=torch.float16)
    y, s = fe.fused_add_rms_norm(x, r, w, 1e-6)
    y_ref, s_ref = fe.fused_add_rms_norm_ref(x, r, w, 1e-6)
    # The residual sum is one exact fp16 add either way.
    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
    torch.testing.assert_close(y, y_ref, rtol=4e-3, atol=4e-3)


@requires_gpu
def test_triton_silu_and_mul():
    gu = torch.randn(3, 2 * 8960, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(fe.silu_and_mul(gu), fe.silu_and_mul_ref(gu),
                               rtol=2e-3, atol=2e-3)


@requires_gpu
def test_triton_rope_on_a_strided_slice():
    """q and k arrive as slices of the fused q/k/v output, so their rows
    are strided by the full projection width, not their own."""
    b, s, hq, hk, d = 2, 1, 12, 2, 128
    qkv = torch.randn(b, s, (hq + 2 * hk) * d, device="cuda", dtype=torch.float16)
    q, k, _ = qkv.split([hq * d, hk * d, hk * d], dim=-1)
    cos = torch.randn(b, 1, s, d, device="cuda", dtype=torch.float16)
    sin = torch.randn(b, 1, s, d, device="cuda", dtype=torch.float16)
    got = fe.rope_qk(q, k, cos, sin, hq, hk, d)
    want = fe.rope_qk_ref(q, k, cos, sin, hq, hk, d)
    truth = fe.rope_qk_ref(q.float(), k.float(), cos.float(), sin.float(), hq, hk, d)
    # Neither "identical to Hugging Face" (failed on 25% of elements) nor
    # "never further from exact than HF, element by element" (failed on 28).
    # Between them those rule out both exact replication of HF's three
    # roundings and a single rounding at the end; the compiler fused part of
    # the expression, most likely one multiply into the add as an FMA.
    #
    # What holds for every such choice is the model's own error budget: each
    # output is a sum of two products, and HF's fp16 computation is only
    # guaranteed to half a step at each product's size plus half a step at
    # the result's. Any subset of those roundings stays inside it; a wrong
    # index, sign or half does not. Checked on CPU against simulated exact,
    # single-rounding and FMA kernels (accepted) and against added noise and
    # swapped halves (rejected).
    for g, t, x, h in zip(got, truth, (q, k), (hq, hk)):
        x4 = x.float().reshape(b, s, h, d)
        c = cos.float().reshape(b, s, 1, d)
        sn = sin.float().reshape(b, s, 1, d)
        products = (x4.abs() * c.abs() + fe._rotate_half(x4).abs() * sn.abs()).reshape(b, s, h * d)
        bound = 2**-11 * (products + t.abs()) + 2**-24
        err = (g.float() - t).abs()
        assert (err <= bound).all(), (
            f"outside the model's own rounding budget at {(err > bound).sum().item()} elements")


@requires_gpu
def test_graphed_elementwise_decode_matches_unfused_eager():
    from runtime.cuda_graph import GraphedDecoder

    prompt = torch.randint(0, 128, (1, 30)).cuda()
    stream = torch.randint(0, 128, (1, 16)).cuda()
    want = _decode(build(_hf("cuda", torch.float16), device="cuda"), prompt, stream)

    m = build(_hf("cuda", torch.float16), device="cuda", fuse_elementwise=True)
    m.allocate_cache(1, 128, paged=True, block_size=16)
    m.cache.reset()
    m.prefill_slot(prompt, slot=0)
    decoder = GraphedDecoder(m)
    got = [decoder.step(stream[:, t : t + 1], torch.tensor([[30 + t]]).cuda(), [0]).clone()
           for t in range(16)]
    torch.testing.assert_close(torch.cat(got, dim=1), want, rtol=3e-2, atol=3e-2)