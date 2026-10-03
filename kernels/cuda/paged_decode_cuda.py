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

SOURCE = pathlib.Path(__file__).with_name("paged_decode_fp16.cu")
KERNEL = "paged_decode_fp16"
TOKG = 4            # tokens per half-warp per softmax group: 168 registers, no spills
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
                  tokg: int = TOKG) -> tuple[bytes, str]:
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
    prog = nvrtc.createProgram(SOURCE.read_text(), SOURCE.name, [], [])
    opts = [f"--gpu-architecture={arch}", "--ptxas-options=-v", "-std=c++17",
            f"-DHEAD_DIM={head_dim}", f"-DNREP={n_rep}", f"-DPAGE={page}", f"-DTOKG={tokg}",
            *[f"-I{p}" for p in incs]]
    try:
        nvrtc.compileProgram(prog, opts)
    except Exception as e:
        raise RuntimeError(f"NVRTC failed:\n{nvrtc.getProgramLog(prog)}") from e
    return nvrtc.getCUBIN(prog), nvrtc.getProgramLog(prog)


def eligible(q: torch.Tensor, k_pool: torch.Tensor, k_scale=None, k_residual=None) -> bool:
    """What this kernel handles: fp16 cache, head_dim 128, <= 16 query rows
    per KV head, pages divisible by the token grouping. Everything else —
    INT8 included, for now — stays on Triton."""
    if not (k_pool.dtype == torch.float16 and q.dtype == torch.float16 and k_scale is None
            and k_residual is None and q.shape[-1] == 128 and q.shape[2] <= 16
            and k_pool.shape[1] % (2 * TOKG) == 0 and q.is_cuda):
        return False
    # Every row is read with 16-byte vector loads: base pointers and row
    # strides must be 16-byte aligned, and head dims unit-stride. A view
    # that is not falls back to Triton rather than faulting.
    for t in (q, k_pool):
        if t.stride(-1) != 1 or t.data_ptr() % 16:
            return False
        if any((st * t.element_size()) % 16 for st in t.stride()[:-1]):
            return False
    return True


def _function(device: torch.device, head_dim: int, n_rep: int, page: int):
    import cupy

    major, minor = torch.cuda.get_device_capability(device)
    key = (device.index or 0, head_dim, n_rep, page, TOKG)
    if key not in _MODULES:
        cubin, log = compile_cubin(f"sm_{major}{minor}", head_dim, n_rep, page)
        with cupy.cuda.Device(device.index or 0):
            mod = cupy.cuda.Module()
            mod.load(cubin)
            _MODULES[key] = (mod, mod.get_function(KERNEL))
        LAST_BUILD.update(cubin=cubin, log=log, key=key)
    return _MODULES[key][1]


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
                      softmax_scale: Optional[float] = None) -> torch.Tensor:
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

    fn = _function(q.device, d, n_rep, page)
    ptr = lambda t: np.uint64(t.data_ptr())  # noqa: E731
    i64 = np.int64
    fn((b, splits, h_kv), (32, 1, 1),
       (ptr(q), ptr(k_pool), ptr(v_pool), ptr(block_tables), ptr(seq_lens),
        ptr(acc), ptr(m), ptr(l),
        np.int32(block_tables.shape[1]), np.int32(splits),
        i64(q.stride(0)), i64(q.stride(1)), i64(q.stride(2)),
        i64(k_pool.stride(0)), i64(k_pool.stride(1)), i64(k_pool.stride(2)),
        i64(acc.stride(0)), i64(acc.stride(1)), i64(acc.stride(2)), i64(acc.stride(3)),
        i64(m.stride(0)), i64(m.stride(1)), i64(m.stride(2)),
        np.float32(scale)),
       stream=_stream(q.device))
    pd.LAST_COMPILED["combine"] = pd._combine_kernel[(b, h_kv)](
        acc, m, l, out, *acc.stride(), *m.stride(), *out.stride(), splits,
        N_REP=n_rep, BLOCK_M=16, BLOCK_D=d)
    return out[:, :, :n_rep]
