"""
Phase 16: INT8's decode-step write as one fused kernel (int8_write.cu).

`decode_write` replaces the torch ops in Int8PagedKVCache.write()'s graph
path — V's per-token quantize and scatter, and K's scatter into the fp16
residual — with one launch per layer, byte-identical to them. Built with
NVRTC and launched on torch's current stream, like the decode kernels, so
CUDA graphs capture it. LATENTSERVE_INT8_FUSED_WRITE=0 restores the torch
path, for A/B and for tests.
"""

from __future__ import annotations

import os
import pathlib

import torch

from kernels.cuda import paged_decode_cuda as pdc

SOURCE = pathlib.Path(__file__).with_name("int8_write.cu")
ENABLED = os.environ.get("LATENTSERVE_INT8_FUSED_WRITE", "1") != "0"
_MODULES: dict = {}


def compile_cubin(arch: str = "sm_75", head_dim: int = 128) -> tuple:
    from cupy_backends.cuda.libs import nvrtc

    prog = nvrtc.createProgram(SOURCE.read_text(), SOURCE.name, [], [])
    opts = [f"--gpu-architecture={arch}", "--ptxas-options=-v", "-std=c++17",
            f"-DHEAD_DIM={head_dim}", *[f"-I{p}" for p in pdc.include_dirs()]]
    try:
        nvrtc.compileProgram(prog, opts)
    except Exception as e:
        raise RuntimeError(f"NVRTC failed:\n{nvrtc.getProgramLog(prog)}") from e
    return nvrtc.getCUBIN(prog), nvrtc.getProgramLog(prog)


def _function(device: torch.device, head_dim: int):
    import cupy

    major, minor = torch.cuda.get_device_capability(device)
    key = (device.index or 0, head_dim)
    if key not in _MODULES:
        cubin, _ = compile_cubin(f"sm_{major}{minor}", head_dim)
        with cupy.cuda.Device(device.index or 0):
            mod = cupy.cuda.Module()
            mod.load(cubin)
            _MODULES[key] = (mod, mod.get_function("int8_decode_write"))
    return _MODULES[key][1]


def eligible(k: torch.Tensor, v: torch.Tensor) -> bool:
    """One token per sequence, fp16, head_dim a multiple of 32 (whole warps),
    channels unit-stride."""
    return (ENABLED and k.is_cuda and v.is_cuda and k.dtype == v.dtype == torch.float16
            and k.shape[2] == 1 and v.shape[2] == 1 and k.shape[-1] % 32 == 0
            and k.shape[-1] <= 1024 and k.stride(-1) == 1 and v.stride(-1) == 1)


def decode_write(k, v, slots, res_idx, flat_v, flat_v_scale, flat_v_zero, k_res, *,
                 asym: bool, eps: float, qmax: float, levels: float, offset: float) -> None:
    """k, v [B, H, 1, D] fp16; slots, res_idx [B] int64 flat indices;
    flat_v [S, H, D] int8, flat_v_scale / flat_v_zero [S, H] fp32;
    k_res [R, H, D] fp16."""
    import numpy as np

    b, h, _, d = k.shape
    fn = _function(k.device, d)
    ptr = lambda t: np.uint64(0 if t is None else t.data_ptr())  # noqa: E731
    i64 = np.int64
    fn((b, h, 1), (d, 1, 1),
       (ptr(k), ptr(v), ptr(slots), ptr(res_idx), ptr(flat_v), ptr(flat_v_scale),
        ptr(flat_v_zero), ptr(k_res),
        i64(k.stride(0)), i64(k.stride(1)), i64(v.stride(0)), i64(v.stride(1)),
        np.int32(h), np.int32(int(asym)), np.float32(eps), np.float32(qmax),
        # The reciprocal exactly as PyTorch forms it for `t / python_number`:
        # one fp32 division on the host.
        np.float32(1.0) / np.float32(qmax), np.float32(levels),
        np.float32(1.0) / np.float32(levels), np.float32(offset)),
       stream=pdc._stream(k.device))