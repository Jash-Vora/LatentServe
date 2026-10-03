"""The CUDA-core decode kernel (kernels/cuda).

Phase 12 traced the Triton kernel's ceiling to arithmetic compiled as scalar
FMA through shared memory: 255 registers, 90 spills, 40 KB, 12% occupancy.
This backend exists to not do that, so the first tests pin its resource use
from the compiler's own report — runnable wherever CuPy can reach NVRTC, no
GPU needed. The rest check it computes the reference's answer, survives
CUDA-graph capture, and matches Triton through a whole model.
"""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")

from kernels.cuda import paged_decode_cuda as pdc  # noqa: E402
from kernels.gqa import paged_decode as pd  # noqa: E402

try:
    pdc.compile_cubin("sm_75")
    CAN_COMPILE = True
except Exception:  # noqa: BLE001 - environmental
    CAN_COMPILE = False

requires_nvrtc = pytest.mark.skipif(not CAN_COMPILE, reason="needs CuPy + NVRTC")
requires_gpu = pytest.mark.skipif(
    not (CAN_COMPILE and torch.cuda.is_available() and pd.HAS_TRITON),
    reason="needs a GPU, Triton, CuPy",
)


@requires_nvrtc
@pytest.mark.parametrize("arch", ["sm_75", "sm_80"])
def test_compiles_without_spills_or_shared_memory(arch):
    _, log = pdc.compile_cubin(arch)
    regs = int(re.search(r"Used (\d+) registers", log).group(1))
    assert "0 bytes spill stores" in log and "0 bytes spill loads" in log, log
    assert regs <= 200, f"{regs} registers: heading back toward the Triton kernel's 255"
    smem = re.search(r"(\d+) bytes smem", log)
    assert smem is None or int(smem.group(1)) == 0, log


def test_ineligible_inputs_stay_on_triton():
    q = torch.zeros(1, 2, 6, 128, dtype=torch.float16)
    k = torch.zeros(4, 16, 2, 128, dtype=torch.float16)
    assert not pdc.eligible(q, k)                                    # not on a GPU
    assert not pdc.eligible(q, k.to(torch.int8))                     # INT8: Triton for now
    assert not pdc.eligible(q, k, k_scale=torch.zeros(1))


def _inputs(lens, n_rep=6, h=2, d=128, page=16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = (max(lens) + page - 1) // page
    nb = len(lens) * pages + 3
    k = torch.randn(nb, page, h, d, dtype=torch.float16, device="cuda", generator=g)
    v = torch.randn(nb, page, h, d, dtype=torch.float16, device="cuda", generator=g)
    tables = torch.randperm(nb, device="cuda", generator=g)[: len(lens) * pages]
    tables = tables.reshape(len(lens), pages).to(torch.int32)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q = torch.randn(len(lens), h, n_rep, d, dtype=torch.float16, device="cuda", generator=g)
    return q, k, v, tables, seq


@requires_gpu
@pytest.mark.parametrize("splits", [None, 1, 3, 64])
def test_matches_the_reference_on_ragged_lengths(splits):
    """64 splits over at most 64 pages leaves some splits empty for the
    shorter sequences: they must contribute nothing, not NaN."""
    lens = [1024, 300, 61, 17, 1]
    q, k, v, tables, seq = _inputs(lens)
    want = pd.paged_decode_reference(q, k, v, tables, seq, num_splits=1)
    got = pdc.paged_decode_cuda(q, k, v, tables, seq, max(lens), num_splits=splits)
    torch.testing.assert_close(got.float(), want.float(), atol=5e-3, rtol=1e-2)


@requires_gpu
def test_dispatch_through_the_normal_entry_point():
    q, k, v, tables, seq = _inputs([700, 33])
    want = pd.paged_decode_reference(q, k, v, tables, seq, num_splits=1)
    before = pd.decode_backend()
    try:
        pd.set_decode_backend("cuda")
        got = pd.paged_decode_attention(q, k, v, tables, seq, max_seq_len=700)
    finally:
        pd.set_decode_backend(before)
    torch.testing.assert_close(got.float(), want.float(), atol=5e-3, rtol=1e-2)


@requires_gpu
def test_survives_cuda_graph_capture():
    """CuPy launches on torch's current stream; under capture that stream is
    the capturing one, so the launch must be recorded and replay must read
    the *current* contents of its inputs."""
    q, k, v, tables, seq = _inputs([500, 200])
    pdc.paged_decode_cuda(q, k, v, tables, seq, 500)               # compile + warm up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = pdc.paged_decode_cuda(q, k, v, tables, seq, 500)
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.cuda.synchronize()
    want = pd.paged_decode_reference(q, k, v, tables, seq, num_splits=1)
    torch.testing.assert_close(out.float(), want.float(), atol=5e-3, rtol=1e-2)


@requires_gpu
def test_whole_model_decode_matches_triton():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape

    # head_dim must be 128, the shape the kernel is built for: a smaller
    # tiny model would silently stay on Triton and this would test nothing.
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128 * 12, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                      max_position_embeddings=1024)
    hf = Qwen2ForCausalLM(cfg).half().cuda().eval()
    shape = ModelShape(2, 12, 2, 128, 128 * 12, 128, 1024, "torch.float16")
    prompt = torch.randint(0, 128, (2, 100)).cuda()

    def decode(backend):
        before = pd.decode_backend()
        try:
            pd.set_decode_backend(backend)
            ls = LatentServeQwen(hf_model=hf, tokenizer=None, shape=shape, device="cuda",
                                 attn_impl="triton_paged", max_seq_len_hint=256)
            ls.allocate_cache(2, 256, paged=True, block_size=16)
            ls.cache.reset()
            ls.prefill(prompt)
            return ls.decode_step(torch.zeros(2, 1, dtype=torch.long, device="cuda")).float()
        finally:
            pd.set_decode_backend(before)

    torch.testing.assert_close(decode("cuda"), decode("triton"), atol=2e-2, rtol=2e-2)
