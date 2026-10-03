"""
Phase 14, stages 3-4: sparse decode attention on CUDA cores.

    sparse_attention(q, k_pool, v_pool, block_tables, seq_lens, kmin, kmax,
                     ratio=0.25)

per layer per decode step = three steps, all on the GPU and capturable in a
CUDA graph:

  1. page_index      score every page by its Quest bound (sink and recent
                     pages forced in, pages past the end forced out)
  2. torch.topk      keep `budget(ratio)` pages per (sequence, KV head)
  3. paged_sparse    attend over just those pages, then the shared merge

and, on the write side, `update_bounds` after each token is written — one
launch per layer. Built with NVRTC through the dense kernel's machinery
(kernels/cuda/paged_decode_cuda.py), loaded with CuPy, launched on torch's
current stream.

The budget is a fixed number of pages per call — `ceil(ratio x max_pages)`
over the batch's page capacity — because a captured graph needs fixed
shapes. For a uniform batch that is the oracle study's per-sequence budget;
in a ragged batch, shorter sequences pad their selection with pages past
their end, which score -inf and are skipped by the kernel.
"""

from __future__ import annotations

import math
import pathlib
from typing import Optional

import torch

from kernels.cuda import paged_decode_cuda as pdc

SOURCE = pathlib.Path(__file__).with_name("paged_sparse_fp16.cu")
PAGES_PER_WARP = 16
_MODULES: dict = {}


def compile_cubin(arch: str = "sm_75", head_dim: int = 128, n_rep: int = 6,
                  page: int = 16) -> tuple:
    from cupy_backends.cuda.libs import nvrtc

    prog = nvrtc.createProgram(SOURCE.read_text(), SOURCE.name, [], [])
    opts = [f"--gpu-architecture={arch}", "--ptxas-options=-v", "-std=c++17",
            f"-DHEAD_DIM={head_dim}", f"-DNREP={n_rep}", f"-DPAGE={page}",
            f"-DTOKG={pdc.TOKG}", f"-DPAGES_PER_WARP={PAGES_PER_WARP}",
            *[f"-I{p}" for p in pdc.include_dirs()]]
    try:
        nvrtc.compileProgram(prog, opts)
    except Exception as e:
        raise RuntimeError(f"NVRTC failed:\n{nvrtc.getProgramLog(prog)}") from e
    return nvrtc.getCUBIN(prog), nvrtc.getProgramLog(prog)


def _functions(device: torch.device, head_dim: int, n_rep: int, page: int) -> dict:
    import cupy

    major, minor = torch.cuda.get_device_capability(device)
    key = (device.index or 0, head_dim, n_rep, page)
    if key not in _MODULES:
        cubin, _ = compile_cubin(f"sm_{major}{minor}", head_dim, n_rep, page)
        with cupy.cuda.Device(device.index or 0):
            mod = cupy.cuda.Module()
            mod.load(cubin)
            _MODULES[key] = (mod, {n: mod.get_function(n) for n in
                                   ("page_bounds_update", "page_index", "paged_sparse_fp16")})
    return _MODULES[key][1]


def budget(ratio: float, max_pages: int, recent: int = 2) -> int:
    """Pages kept per (sequence, KV head): at least the sink and recent ones."""
    if ratio >= 1.0:
        return max_pages
    return min(max_pages, max(recent + 1, math.ceil(ratio * max_pages)))


def rebuild_bounds(k_pool: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor) -> None:
    """Every block's bounds from the pool, over all PAGE rows. For setup and
    tests; a live cache maintains them with `update_bounds` as it writes."""
    kmin.copy_(k_pool.amin(dim=1))
    kmax.copy_(k_pool.amax(dim=1))


def update_bounds(k_new: torch.Tensor, slots: torch.Tensor, kmin: torch.Tensor,
                  kmax: torch.Tensor, page: int = 16) -> None:
    """k_new [B, H, D] (or [B, H, 1, D]) just written at flat `slots` [B]."""
    import numpy as np

    k = k_new.reshape(k_new.shape[0], k_new.shape[1], k_new.shape[-1])
    b, h, d = k.shape
    fn = _functions(k.device, d, 6, page)["page_bounds_update"]
    i64 = np.int64
    fn((b, h, 1), (d, 1, 1),
       (np.uint64(k.data_ptr()), np.uint64(slots.data_ptr()), np.uint64(kmin.data_ptr()),
        np.uint64(kmax.data_ptr()), i64(k.stride(0)), i64(k.stride(1)),
        i64(kmin.stride(0)), i64(kmin.stride(1))),
       stream=pdc._stream(k.device))


def page_scores(q, kmin, kmax, block_tables, seq_lens, recent: int = 2) -> torch.Tensor:
    """-> [B, H, max_pages] fp32: bound per page, +inf forced, -inf past the end."""
    import numpy as np

    from kernels.gqa import paged_decode as pd

    b, h, n_rep, d = q.shape
    max_pages = block_tables.shape[1]
    scores = pd._scratch("sparse_scores", (b, h, max_pages), torch.float32, q.device)
    fn = _functions(q.device, d, n_rep, 16)["page_index"]
    i64 = np.int64
    grid_z = -(-max_pages // (4 * PAGES_PER_WARP))
    fn((b, h, grid_z), (128, 1, 1),
       (np.uint64(q.data_ptr()), np.uint64(kmin.data_ptr()), np.uint64(kmax.data_ptr()),
        np.uint64(block_tables.data_ptr()), np.uint64(seq_lens.data_ptr()),
        np.uint64(scores.data_ptr()), np.int32(max_pages), np.int32(recent),
        i64(q.stride(0)), i64(q.stride(1)), i64(q.stride(2)),
        i64(kmin.stride(0)), i64(kmin.stride(1)), i64(scores.stride(0)), i64(scores.stride(1))),
       stream=pdc._stream(q.device))
    return scores


def select(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k pages per (sequence, head), unsorted: online softmax does not
    care about order. -> int32 [B, H, k]."""
    return torch.topk(scores, k, dim=-1, sorted=False).indices.to(torch.int32)


def sparse_decode(q, k_pool, v_pool, block_tables, seq_lens, sel,
                  num_splits: Optional[int] = None,
                  softmax_scale: Optional[float] = None) -> torch.Tensor:
    """Attention over the pages in `sel` [B, H, K] only."""
    import numpy as np

    from kernels.gqa import paged_decode as pd

    b, h, n_rep, d = q.shape
    num_sel = sel.shape[-1]
    splits = num_splits or pdc.choose_splits(b, h, num_sel)
    splits = max(1, min(splits, num_sel))
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    acc = pd._scratch("sparse_acc", (b, h, splits, 16, d), torch.float32, q.device)
    m = pd._scratch("sparse_m", (b, h, splits, 16), torch.float32, q.device, fill=float("-inf"))
    l = pd._scratch("sparse_l", (b, h, splits, 16), torch.float32, q.device, fill=0.0)
    out = pd._scratch("sparse_out", (b, h, 16, d), q.dtype, q.device)
    fn = _functions(q.device, d, n_rep, k_pool.shape[1])["paged_sparse_fp16"]
    ptr = lambda t: np.uint64(t.data_ptr())  # noqa: E731
    i64 = np.int64
    fn((b, splits, h), (32, 1, 1),
       (ptr(q), ptr(k_pool), ptr(v_pool), ptr(block_tables), ptr(seq_lens), ptr(sel),
        ptr(acc), ptr(m), ptr(l),
        np.int32(block_tables.shape[1]), np.int32(num_sel), np.int32(splits),
        i64(q.stride(0)), i64(q.stride(1)), i64(q.stride(2)),
        i64(k_pool.stride(0)), i64(k_pool.stride(1)), i64(k_pool.stride(2)),
        i64(sel.stride(0)), i64(sel.stride(1)),
        i64(acc.stride(0)), i64(acc.stride(1)), i64(acc.stride(2)), i64(acc.stride(3)),
        i64(m.stride(0)), i64(m.stride(1)), i64(m.stride(2)), np.float32(scale)),
       stream=pdc._stream(q.device))
    pd._combine_kernel[(b, h)](acc, m, l, out, *acc.stride(), *m.stride(), *out.stride(),
                               splits, N_REP=n_rep, BLOCK_M=16, BLOCK_D=d)
    return out[:, :, :n_rep]


def sparse_attention(q, k_pool, v_pool, block_tables, seq_lens, kmin, kmax,
                     ratio: float = 0.25, recent: int = 2,
                     num_splits: Optional[int] = None, max_seq_len: Optional[int] = None,
                     softmax_scale: Optional[float] = None) -> torch.Tensor:
    """Index, select, attend. ratio >= 1 still runs all three: a check that
    the sparse path reproduces dense, not a shortcut to it.

    The budget follows `max_seq_len` — the context — when given. The block
    table's width is the cache's *capacity*: budgeting from it, 25% of a
    32K-capacity cache serving a 4K context is more pages than exist, and
    the step runs dense while paying for the indexer.
    """
    pages = block_tables.shape[1] if max_seq_len is None else min(
        block_tables.shape[1], -(-max_seq_len // k_pool.shape[1]))
    k = budget(ratio, pages, recent)
    sel = select(page_scores(q, kmin, kmax, block_tables, seq_lens, recent), k)
    return sparse_decode(q, k_pool, v_pool, block_tables, seq_lens, sel, num_splits,
                         softmax_scale)


def kernel_resources(device: Optional[torch.device] = None) -> dict:
    """Registers, local (spill) memory and shared memory of each loaded kernel."""
    from cupy_backends.cuda.api import driver as drv

    dev = device or torch.device("cuda", torch.cuda.current_device())
    out = {}
    for name, fn in _functions(dev, 128, 6, 16).items():
        get = lambda a, f=fn: int(drv.funcGetAttribute(a, f.ptr))  # noqa: E731
        out[name] = {"regs": get(drv.CU_FUNC_ATTRIBUTE_NUM_REGS),
                     "local_bytes": get(drv.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
                     "shared_bytes": get(drv.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)}
    return out
