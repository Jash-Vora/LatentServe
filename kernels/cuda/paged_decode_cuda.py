"""
The CUDA-core paged decode kernel, compiled at runtime with NVRTC.

Why not a PyTorch C++ extension: the system `nvcc` on Kaggle is CUDA 12.8
and torch is built for CUDA 13.0, and torch's extension builder refuses a
major-version mismatch outright. NVRTC sidesteps the toolchain: it ships
with torch's own CUDA 13 runtime packages, compiles the kernel source on
the host — no GPU needed, so tests can compile it anywhere — and CuPy loads
and launches the result on torch's current stream, which is what lets CUDA
graphs capture it like any other kernel.

    pip install cupy-cuda13x

Partials are written in the Triton kernel's layout, so the Triton combine
kernel merges them; CUDA graphs, the engine and the tests see one more
backend behind the same function, `paged_decode_attention`.
"""

from __future__ import annotations

import math
import os
import pathlib
from typing import Optional

import torch

SOURCES = {"fp16": pathlib.Path(__file__).with_name("paged_decode_fp16.cu"),
           "int8": pathlib.Path(__file__).with_name("paged_decode_int8.cu")}
KERNELS = {"fp16": "paged_decode_fp16", "int8": "paged_decode_int8"}
SOURCE, KERNEL = SOURCES["fp16"], KERNELS["fp16"]      # kept for older callers
TOKG = 4            # tokens per half-warp per softmax group: 168 registers, no spills
# INT8 register bound (see MIN_BLOCKS in paged_decode_int8.cu). 1 leaves the
# compiler free — 199 registers, no spills, 10 warps/SM for the production
# variant; 12 holds it to fp16's 168 registers and 12 warps at the cost of
# 16 bytes of spill memory. Measured on the T4 (phase12_diag --int8): 12 is
# 11-13% faster at batch 4 and 16, 1-2% slower at batch 1. More warps means
# more loads in flight, and this kernel is limited by memory latency, not
# bytes — the same reason INT8's halved bytes did not buy time.
INT8_MIN_BLOCKS = int(os.environ.get("LATENTSERVE_INT8_MIN_BLOCKS", "12"))
TARGET_WARPS = 960  # ~2 waves of 12 resident single-warp programs on 40 SMs
MAX_SPLITS = 128

_MODULES: dict = {}
_STREAMS: dict = {}
LAST_BUILD: dict = {}


def cupy_package() -> str:
    """The CuPy build matching torch's CUDA: cu13 on vLLM's stack (torch
    built for 13.0), cu12 on Kaggle's default (12.8)."""
    major = (torch.version.cuda or "12").split(".")[0]
    return f"cupy-cuda{major}x"


def include_dirs() -> list[str]:
    """CUDA headers matching NVRTC: torch's own CUDA 13 runtime package
    first (nvidia/cu13/include), then a system toolkit as a fallback."""
    found = []
    try:
        import nvidia

        for base in nvidia.__path__:
            for sub in ("cu13/include", "cuda_runtime/include", "cuda_nvrtc/include"):
                path = os.path.join(base, sub)
                if os.path.exists(os.path.join(path, "cuda_fp16.h")):
                    found.append(path)
    except ImportError:
        pass
    for env in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(env):
            found.append(os.path.join(os.environ[env], "include"))
    found.append("/usr/local/cuda/include")
    return [p for p in dict.fromkeys(found) if os.path.exists(os.path.join(p, "cuda_fp16.h"))]


def compile_cubin(arch: str = "sm_75", head_dim: int = 128, n_rep: int = 6, page: int = 16,
                  tokg: int = TOKG, variant: str = "fp16", asym: bool = False,
                  has_res: bool = False, min_blocks: int = 1) -> tuple[bytes, str]:
    """NVRTC-compile the kernel for `arch`. Host-only: needs no GPU.

    Returns the cubin and the assembler's report, which carries registers
    and spills — the numbers Phase 12 was about.
    """
    try:
        from cupy_backends.cuda.libs import nvrtc
    except ImportError as e:  # pragma: no cover - environmental
        raise RuntimeError(f"the CUDA decode backend needs CuPy: pip install {cupy_package()}") from e
    incs = include_dirs()
    if not incs:
        raise RuntimeError("no CUDA headers (cuda_fp16.h) found for NVRTC")
    src = SOURCES[variant]
    prog = nvrtc.createProgram(src.read_text(), src.name, [], [])
    opts = [f"--gpu-architecture={arch}", "--ptxas-options=-v", "-std=c++17",
            f"-DHEAD_DIM={head_dim}", f"-DNREP={n_rep}", f"-DPAGE={page}", f"-DTOKG={tokg}",
            f"-DASYM={int(asym)}", f"-DHAS_RES={int(has_res)}", f"-DMIN_BLOCKS={min_blocks}",
            *[f"-I{p}" for p in incs]]
    try:
        nvrtc.compileProgram(prog, opts)
    except Exception as e:
        raise RuntimeError(f"NVRTC failed:\n{nvrtc.getProgramLog(prog)}") from e
    return nvrtc.getCUBIN(prog), nvrtc.getProgramLog(prog)


def _aligned(t: torch.Tensor, nbytes: int) -> bool:
    """Unit-stride last dim, and base and every row stride `nbytes`-aligned:
    what a vector load of that width needs."""
    return (t.stride(-1) == 1 and t.data_ptr() % nbytes == 0
            and all((st * t.element_size()) % nbytes == 0 for st in t.stride()[:-1]))


def eligible(q: torch.Tensor, k_pool: torch.Tensor, k_scale=None, k_residual=None,
             v_scale=None, k_zero=None, v_zero=None, res_rows=None) -> bool:
    """What these kernels handle: fp16 or INT8 cache, head_dim 128, <= 16
    query rows per KV head, pages divisible by the token grouping, and the
    alignments their vector loads need. Anything else stays on Triton
    rather than faulting."""
    if not (q.dtype == torch.float16 and q.shape[-1] == 128 and q.shape[2] <= 16
            and k_pool.shape[1] % (2 * TOKG) == 0 and q.is_cuda and _aligned(q, 16)):
        return False
    if k_pool.dtype == torch.float16:
        return k_scale is None and k_residual is None and _aligned(k_pool, 16)
    if k_pool.dtype != torch.int8:
        return False
    # INT8 rows are 8-byte loads; K's scales and zero points are read eight
    # channels at a time as two 16-byte loads; V's are scalars per token.
    if k_scale is None or v_scale is None or not _aligned(k_pool, 8):
        return False
    if k_scale.dtype != torch.float32 or v_scale.dtype != torch.float32 or not _aligned(k_scale, 16):
        return False
    if (k_zero is None) != (v_zero is None):
        return False
    if k_zero is not None and (k_zero.dtype != torch.float32 or k_zero.stride() != k_scale.stride()
                               or v_zero.stride() != v_scale.stride() or not _aligned(k_zero, 16)):
        return False
    if (k_residual is None) != (res_rows is None):
        return False
    if k_residual is not None and (k_residual.dtype != torch.float16 or not _aligned(k_residual, 16)
                                   or res_rows.dtype != torch.int32):
        return False
    return True


def _function(device: torch.device, head_dim: int, n_rep: int, page: int,
              variant: str = "fp16", asym: bool = False, has_res: bool = False,
              min_blocks: Optional[int] = None):
    import cupy

    major, minor = torch.cuda.get_device_capability(device)
    mb = (INT8_MIN_BLOCKS if min_blocks is None else min_blocks) if variant == "int8" else 1
    key = (device.index or 0, head_dim, n_rep, page, TOKG, variant, asym, has_res, mb)
    if key not in _MODULES:
        cubin, log = compile_cubin(f"sm_{major}{minor}", head_dim, n_rep, page,
                                   variant=variant, asym=asym, has_res=has_res, min_blocks=mb)
        with cupy.cuda.Device(device.index or 0):
            mod = cupy.cuda.Module()
            mod.load(cubin)
            _MODULES[key] = (mod, mod.get_function(KERNELS[variant]))
        LAST_BUILD.update(cubin=cubin, log=log, key=key)
    return _MODULES[key][1]


def kernel_resources(device: Optional[torch.device] = None, head_dim: int = 128,
                     n_rep: int = 6, page: int = 16, variant: str = "fp16",
                     asym: bool = False, has_res: bool = False,
                     min_blocks: Optional[int] = None) -> dict:
    """Registers, local memory and static shared memory of the *loaded*
    kernel, as the GPU driver reports them.

    The compiler's own report (ptxas -v) reaches NVRTC's log on some
    installs and not others — on Kaggle's the log came back empty — so the
    driver is the authority. Local memory is where spilled registers live:
    zero local bytes means zero spills.
    """
    from cupy_backends.cuda.api import driver as drv

    dev = device or torch.device("cuda", torch.cuda.current_device())
    fn = _function(dev, head_dim, n_rep, page, variant, asym, has_res, min_blocks)

    def get(attr):
        return int(drv.funcGetAttribute(attr, fn.ptr))

    return {"regs": get(drv.CU_FUNC_ATTRIBUTE_NUM_REGS),
            "local_bytes": get(drv.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
            "shared_bytes": get(drv.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)}


def _stream(device: torch.device):
    import cupy

    ptr = torch.cuda.current_stream(device).cuda_stream
    if ptr not in _STREAMS:
        _STREAMS[ptr] = cupy.cuda.ExternalStream(ptr, device.index or 0)
    return _STREAMS[ptr]


def choose_splits(batch: int, kv_heads: int, num_pages: int) -> int:
    """Enough single-warp programs for ~2 waves, never more splits than pages."""
    return max(1, min(num_pages, MAX_SPLITS, -(-TARGET_WARPS // max(1, batch * kv_heads))))


def paged_decode_cuda(q, k_pool, v_pool, block_tables, seq_lens, max_seq_len: int,
                      num_splits: Optional[int] = None,
                      softmax_scale: Optional[float] = None,
                      k_scale=None, v_scale=None, k_zero=None, v_zero=None,
                      k_residual=None, res_rows=None,
                      min_blocks: Optional[int] = None) -> torch.Tensor:
    import numpy as np

    from kernels.gqa import paged_decode as pd

    b, h_kv, n_rep, d = q.shape
    page = k_pool.shape[1]
    if q.stride(-1) != 1 or k_pool.stride(-1) != 1 or v_pool.stride() != k_pool.stride():
        raise ValueError("the CUDA kernel needs unit-stride head dims and matching K/V pools")
    if block_tables.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError("block tables and sequence lengths must be int32")
    num_pages = -(-max_seq_len // page)
    splits = num_splits or choose_splits(b, h_kv, num_pages)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)

    acc = pd._scratch("cuda_acc", (b, h_kv, splits, 16, d), torch.float32, q.device)
    m = pd._scratch("cuda_m", (b, h_kv, splits, 16), torch.float32, q.device, fill=float("-inf"))
    l = pd._scratch("cuda_l", (b, h_kv, splits, 16), torch.float32, q.device, fill=0.0)
    out = pd._scratch("cuda_out", (b, h_kv, 16, d), q.dtype, q.device)

    ptr = lambda t: np.uint64(0 if t is None else t.data_ptr())  # noqa: E731
    i64 = np.int64
    head = (ptr(q), ptr(k_pool), ptr(v_pool), ptr(block_tables), ptr(seq_lens),
            ptr(acc), ptr(m), ptr(l))
    q_k = (np.int32(block_tables.shape[1]), np.int32(splits),
           i64(q.stride(0)), i64(q.stride(1)), i64(q.stride(2)),
           i64(k_pool.stride(0)), i64(k_pool.stride(1)), i64(k_pool.stride(2)))
    tail = (i64(acc.stride(0)), i64(acc.stride(1)), i64(acc.stride(2)), i64(acc.stride(3)),
            i64(m.stride(0)), i64(m.stride(1)), i64(m.stride(2)), np.float32(scale))
    if k_pool.dtype == torch.int8:
        asym, has_res = k_zero is not None, k_residual is not None
        fn = _function(q.device, d, n_rep, page, "int8", asym, has_res, min_blocks)
        res_st = k_residual.stride()[:3] if has_res else (0, 0, 0)
        args = (*head, ptr(k_scale), ptr(v_scale), ptr(k_zero), ptr(v_zero),
                ptr(k_residual), ptr(res_rows), *q_k,
                i64(k_scale.stride(0)), i64(k_scale.stride(1)),
                i64(v_scale.stride(0)), i64(v_scale.stride(1)), i64(v_scale.stride(2)),
                *(i64(x) for x in res_st), *tail)
    else:
        fn = _function(q.device, d, n_rep, page)
        args = (*head, *q_k, *tail)
    fn((b, splits, h_kv), (32, 1, 1), args, stream=_stream(q.device))
    pd.LAST_COMPILED["combine"] = pd._combine_kernel[(b, h_kv)](
        acc, m, l, out, *acc.stride(), *m.stride(), *out.stride(), splits,
        N_REP=n_rep, BLOCK_M=16, BLOCK_D=d)
    return out[:, :, :n_rep]