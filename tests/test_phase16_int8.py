"""Phase 16: INT8's fused decode-step write is byte-identical to the torch path."""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")

from kernels.cuda import int8_write as iw  # noqa: E402

try:
    _, _LOG = iw.compile_cubin("sm_75")
    CAN_COMPILE = True
except Exception:  # noqa: BLE001
    CAN_COMPILE, _LOG = False, ""
requires_gpu = pytest.mark.skipif(not (CAN_COMPILE and torch.cuda.is_available()),
                                  reason="needs a GPU and CuPy")


@pytest.mark.skipif(not CAN_COMPILE, reason="needs CuPy + NVRTC")
def test_compiles_without_spills():
    assert "0 bytes spill stores" in _LOG or "registers" not in _LOG
    m = re.search(r"Used (\d+) registers", _LOG)
    assert m is None or int(m.group(1)) <= 64


def _decode_state(asym: bool, fused: bool, steps: int = 37):
    """Prefill, then `steps` decode steps through the INT8 cache's graph-path
    write; return every buffer the write touches."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from cache.int8_paged_cache import Int8PagedKVCache
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128 * 12, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                      max_position_embeddings=2048)
    hf = Qwen2ForCausalLM(cfg).half().cuda().eval()
    ls = LatentServeQwen(hf_model=hf, tokenizer=None,
                         shape=ModelShape(2, 12, 2, 128, 128 * 12, 128, 2048, "torch.float16"),
                         device="cuda", attn_impl="triton_paged", max_seq_len_hint=1024)
    # Asymmetric is a construction-time choice: its zero-point pools only
    # exist when the cache is built that way.
    ls.allocate_cache(2, 512, paged=True, block_size=16, kv_dtype="int8", asymmetric=asym)
    cache = ls.cache
    assert isinstance(cache, Int8PagedKVCache) and cache.asymmetric == asym
    cache.enable_deferred_finalize()
    cache.reset()
    before = iw.ENABLED
    iw.ENABLED = fused
    try:
        with torch.no_grad():
            ls.prefill(torch.randint(0, 128, (2, 70)).cuda())
            for t in range(steps):                              # crosses page boundaries
                ls.decode_step(torch.full((2, 1), t % 128, device="cuda"))
        torch.cuda.synchronize()
    finally:
        iw.ENABLED = before
    state = {"v": [t.clone() for t in cache._flat_v], "v_scale": [t.clone() for t in cache._flat_v_scale],
             "k_res": [t.clone() for t in cache._k_res_flat]}
    if asym:
        state["v_zero"] = [t.clone() for t in cache._flat_v_zero]
    return state


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
def test_fused_write_is_byte_identical_to_the_torch_path(asym):
    ref = _decode_state(asym, fused=False)
    got = _decode_state(asym, fused=True)
    for name in ref:
        for layer, (a, b) in enumerate(zip(ref[name], got[name])):
            assert torch.equal(a, b), f"{name}, layer {layer} differs"


@requires_gpu
def test_the_fused_path_is_actually_taken():
    """A byte-identical test passes vacuously if the fused branch never runs."""
    calls = []
    original = iw.decode_write

    def spy(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    iw.decode_write = spy
    try:
        _decode_state(False, fused=True, steps=3)
    finally:
        iw.decode_write = original
    assert len(calls) == 3 * 2                                    # steps x layers



def _torch_reference(k, v, asym, eps=1e-8, v_bits=8):
    """The torch path's arithmetic, verbatim from Int8PagedKVCache.write()."""
    qmax, levels, offset = 2 ** (v_bits - 1) - 1, 2 ** v_bits - 1, 2 ** (v_bits - 1)
    v_f = v.permute(0, 2, 1, 3).to(torch.float32)              # [B, 1, H, D]
    if asym:
        v_lo = v_f.amin(dim=-1, keepdim=True)
        v_hi = v_f.amax(dim=-1, keepdim=True)
        v_step = ((v_hi - v_lo) / levels).clamp_min(eps)
        v_u = torch.clamp(torch.round((v_f - v_lo) / v_step), 0, levels)
        v_q = (v_u - offset).to(torch.int8)
        zero = (v_lo + offset * v_step).squeeze(-1)
    else:
        v_step = v_f.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
        v_q = torch.clamp(torch.round(v_f / v_step), -qmax, qmax).to(torch.int8)
        zero = None
    return v_q[:, 0], v_step.squeeze(-1)[:, 0], None if zero is None else zero[:, 0]


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
def test_kernel_matches_the_torch_formula_on_identical_inputs(asym):
    """Isolates the kernel: same inputs, compared byte for byte, over many
    tokens and magnitudes — no model, nothing to propagate."""
    g = torch.Generator(device="cuda").manual_seed(5)
    b, h, d = 256, 2, 128
    scale = torch.logspace(-3, 2, b, device="cuda")[:, None, None, None]
    v = (torch.randn(b, h, 1, d, device="cuda", generator=g) * scale).half()
    k = torch.randn(b, h, 1, d, device="cuda", generator=g).half()
    slots = torch.randperm(4 * b, device="cuda", generator=g)[:b]
    res_idx = torch.arange(b, device="cuda")
    flat_v = torch.zeros(4 * b, h, d, dtype=torch.int8, device="cuda")
    flat_s = torch.zeros(4 * b, h, device="cuda")
    flat_z = torch.zeros(4 * b, h, device="cuda") if asym else None
    k_res = torch.zeros(b, h, d, dtype=torch.float16, device="cuda")
    iw.decode_write(k, v, slots, res_idx, flat_v, flat_s, flat_z, k_res, asym=asym, eps=1e-8,
                    qmax=127.0, levels=255.0, offset=128.0)
    torch.cuda.synchronize()
    want_q, want_s, want_z = _torch_reference(k, v, asym)
    assert torch.equal(flat_v[slots], want_q)
    assert torch.equal(flat_s[slots], want_s), "scales differ: check the reciprocal"
    if asym:
        assert torch.equal(flat_z[slots], want_z)
    assert torch.equal(k_res, k[:, :, 0])


@requires_gpu
def test_the_torch_path_is_deterministic_end_to_end():
    """Control for the end-to-end comparison: if the torch path does not
    reproduce itself, a mismatch against the fused path proves nothing."""
    a = _decode_state(False, fused=False)
    b = _decode_state(False, fused=False)
    for name in a:
        for x, y in zip(a[name], b[name]):
            assert torch.equal(x, y), name