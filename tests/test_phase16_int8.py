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
