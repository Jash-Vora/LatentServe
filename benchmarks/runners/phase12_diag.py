"""
Phase 12, resumed — why the decode kernel reads the cache at ~47 GB/s.

    python -m benchmarks.runners.phase12_diag

The T4 can stream ~320 GB/s. Phase 12's sweep found the paged decode
kernel saturating near 47 GB/s at batch 4, flat across split counts — so
not occupancy in the sense of "too few programs" — and Phase 14 found the
INT8 variant *slower*, at ~22 GB/s on half the bytes. That ceiling limits
fp16 decode at long context, makes INT8 a loss, and would cap a Phase 14
sparse kernel the same way. One run, three answers:

1. **Is Nsight Compute usable here?** Containers often refuse GPU
   performance counters (ERR_NVGPUCTRPERM), which is why Phase 12 stopped
   short. Checked directly with a one-kernel probe.

2. **What did the compiler decide?** Triton records, for every compiled
   kernel, registers per thread, values spilled to local memory, and shared
   memory per program. On a T4 each SM has 65,536 registers and room for 32
   warps, so registers per thread set how many warps can be in flight — and
   warps in flight are how a GPU hides memory latency. Spills are worse:
   they turn register reads into memory traffic. The leading suspect for
   INT8 is exactly this: dequantizing into full fp32 tiles doubles their
   register footprint.

3. **What does each variant achieve?** fp16 and INT8, one page per loop
   iteration and four, at batch 1 and batch 16 at 8K — so the resource
   numbers sit next to the bandwidth they produce.

If Nsight is usable, the run then profiles the fp16 and INT8 kernels for
achieved occupancy, DRAM throughput and warp stall reasons — the answer
to *why* that the resource numbers can only suggest.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

# Turing (sm75) per-SM limits.
REGS_PER_SM = 65536
MAX_WARPS_PER_SM = 32
MAX_BLOCKS_PER_SM = 16
SMEM_PER_SM = 64 * 1024


def find_ncu() -> str | None:
    for candidate in (shutil.which("ncu"), "/usr/local/cuda/bin/ncu",
                      *sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def ncu_status() -> tuple[str, str | None]:
    """('usable' | 'missing' | 'no-permission' | 'failed', path)."""
    ncu = find_ncu()
    if ncu is None:
        return "missing", None
    probe = [ncu, "--metrics", "sm__cycles_elapsed.avg", sys.executable, "-c",
             "import torch; (torch.ones(4, device='cuda') + 1).sum().item()"]
    try:
        out = subprocess.run(probe, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return "failed", ncu
    text = out.stdout + out.stderr
    if "ERR_NVGPUCTRPERM" in text:
        return "no-permission", ncu
    if "sm__cycles_elapsed" in text:
        return "usable", ncu
    return "failed", ncu


def occupancy(regs: int, smem: int, warps: int) -> tuple[float, str]:
    """Theoretical occupancy and which resource bounds it."""
    limits = {"warp slots": MAX_WARPS_PER_SM}
    if regs:
        limits["registers"] = (REGS_PER_SM // (regs * 32)) // warps * warps
    if smem:
        limits["shared memory"] = (SMEM_PER_SM // smem) * warps
    limits["blocks per SM"] = MAX_BLOCKS_PER_SM * warps
    bound = min(limits, key=limits.get)
    return min(limits.values()) / MAX_WARPS_PER_SM, bound


def build_inputs(torch, batch, ctx, int8, page=16, h=2, d=128, n_rep=6, seed=0,
                 identity=False, ragged=None):
    """Pools, tables, lengths, query. `identity` lays sequence b's page j at
    block b*P+j, so the no-lookup ablation reads the same addresses as the
    lookup it replaces. `ragged` gives per-sequence lengths."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    lens_list = ragged or [ctx] * batch
    pages = (max(lens_list) + page - 1) // page
    nb = batch * pages + 4
    dev = "cuda"
    if int8:
        k = torch.randint(-127, 127, (nb, page, h, d), dtype=torch.int8, device=dev, generator=g)
        v = torch.randint(-127, 127, (nb, page, h, d), dtype=torch.int8, device=dev, generator=g)
        ks = torch.rand(nb, h, d, device=dev, generator=g) * 0.01 + 1e-3
        vs = torch.rand(nb, page, h, device=dev, generator=g) * 0.01 + 1e-3
        res = torch.randn(batch, page, h, d, device=dev, dtype=torch.float16, generator=g)
        rows = torch.arange(batch, dtype=torch.int32, device=dev)
        extra = dict(k_scale=ks, v_scale=vs, k_residual=res, res_rows=rows)
        # int8 K and V, K's fp32 scale per (page, head, channel), V's fp32
        # scale per (token, head).
        kv_bytes = 2 * batch * ctx * h * d + batch * pages * h * d * 4 + batch * ctx * h * 4
    else:
        k = torch.randn(nb, page, h, d, dtype=torch.float16, device=dev, generator=g)
        v = torch.randn(nb, page, h, d, dtype=torch.float16, device=dev, generator=g)
        extra = {}
        kv_bytes = 2 * batch * ctx * h * d * 2
    if identity:
        tables = torch.arange(batch * pages, dtype=torch.int32, device=dev).reshape(batch, pages)
    else:
        tables = torch.randperm(nb, device=dev, generator=g)[: batch * pages]
        tables = tables.reshape(batch, pages).to(torch.int32)
    lens = torch.tensor(lens_list, dtype=torch.int32, device=dev)
    q = torch.randn(batch, h, n_rep, d, dtype=torch.float16, device=dev, generator=g)
    return (q, k, v, tables, lens), extra, kv_bytes


def _resources(k, default_warps=4):
    meta = getattr(k, "metadata", None)
    regs = getattr(k, "n_regs", 0) or 0
    spills = getattr(k, "n_spills", 0) or 0
    smem = getattr(meta, "shared", 0) if meta is not None else 0
    warps = getattr(meta, "num_warps", default_warps) if meta is not None else default_warps
    occ, bound = occupancy(regs, smem, warps)
    return {"regs": regs, "spills": spills, "smem": smem, "warps": warps, "occ": occ,
            "bound": bound}


def _time(torch, fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def measure(torch, batch, ctx, int8, ppi, iters=50, stages=None):
    from kernels.gqa import paged_decode as pd

    args, extra, kv_bytes = build_inputs(torch, batch, ctx, int8)
    ms = _time(torch, lambda: pd.paged_decode_attention(
        *args, max_seq_len=ctx, num_splits=16, pages_per_iter=ppi, num_stages=stages,
        **extra), iters)
    return {"ms": ms, "gbs": kv_bytes / (ms / 1000) / 1e9,
            **_resources(pd.LAST_COMPILED.get("decode"), pd.NUM_WARPS)}


def measure_ablation(torch, variant, batch, ctx, ppi, stages, iters=50):
    from kernels.gqa import paged_decode_ablations as ab

    args, _, kv_bytes = build_inputs(torch, batch, ctx, False, identity=True)
    run = ab.prepare(variant, *args, pages_per_iter=ppi, num_stages=stages)
    ms = _time(torch, run, iters)
    return {"ms": ms, "gbs": kv_bytes / (ms / 1000) / 1e9,
            **_resources(ab.LAST_COMPILED.get(variant))}


def check_correctness(torch) -> bool:
    """Every timed variant against the reference, on a small ragged batch."""
    from kernels.gqa import paged_decode as pd
    from kernels.gqa import paged_decode_ablations as ab

    ok = True
    ragged = [300, 61]
    for int8 in (False, True):
        args, extra, _ = build_inputs(torch, 2, 300, int8, ragged=ragged, seed=1)
        want = pd.paged_decode_reference(*args, num_splits=1, **extra)
        for ppi in (1, 2, 4):
            for stages in (1, 2):
                got = pd.paged_decode_attention(*args, max_seq_len=300, num_splits=4,
                                                pages_per_iter=ppi, num_stages=stages, **extra)
                err = float((got.float() - want.float()).abs().max())
                good = err < 2e-2
                ok &= good
                if not good:
                    print(f"  FAIL {'int8' if int8 else 'fp16'} ppi{ppi} stages{stages}: "
                          f"max abs error {err:.3e}")
    args, _, _ = build_inputs(torch, 2, 300, False, ragged=ragged, seed=2, identity=True)
    want = pd.paged_decode_reference(*args, num_splits=1)
    for variant in ("full", "no_lookup", "prefetch"):
        got = ab.run_variant(variant, *args, num_splits=4, pages_per_iter=4 if variant != "prefetch" else 1)
        err = float((got.float() - want.float()).abs().max())
        good = err < 2e-2
        ok &= good
        if not good:
            print(f"  FAIL ablation {variant}: max abs error {err:.3e}")
    return ok


def sass_census(compiled) -> dict | None:
    """Static instruction counts from the compiled kernel's machine code.

    Not a profile: these are instructions *in* the kernel, not executions.
    But they show what no timing can — whether spill code exists (local
    loads/stores), how wide the global loads are, how much tensor-core and
    synchronisation work the compiler emitted.
    """
    import tempfile

    asm = getattr(compiled, "asm", None) or {}
    cubin = asm.get("cubin")
    tool = shutil.which("nvdisasm") or "/usr/local/cuda/bin/nvdisasm"
    if not cubin or not os.path.exists(tool):
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        path = f.name
    try:
        out = subprocess.run([tool, "-c", path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    counts = parse_sass(out.stdout)
    if not counts.get("total"):
        # nvdisasm 12.8 cannot read a cubin from NVRTC 13.0: it prints nothing
        # and the first census of the CUDA kernel came out as a row of zeros —
        # which looks like data. Say why instead.
        reason = (out.stderr or "").strip().splitlines()
        return {"error": reason[0][:70] if reason else f"nvdisasm read nothing (exit {out.returncode})"}
    return counts


def parse_sass(text: str) -> dict:
    """Opcode counts from nvdisasm output. Predicates (@P0, @!PT, @UP1) are
    skipped; labels and headers do not match. LDG is also split by width."""
    import re

    counts: dict = {}
    for line in text.splitlines():
        m = re.search(r"\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", line)
        if not m:
            continue
        op = m.group(2)
        base = op.split(".")[0]
        counts["total"] = counts.get("total", 0) + 1
        counts[base] = counts.get(base, 0) + 1
        if base == "LDG":
            width = "128" if ".128" in op else "64" if ".64" in op else "32 or less"
            counts[f"LDG {width}"] = counts.get(f"LDG {width}", 0) + 1
    return counts


CENSUS = ("LDL", "STL", "LDG 128", "LDS", "STS", "HMMA", "FFMA", "HFMA2", "FMUL", "IMAD",
          "SHFL", "BAR", "total")


def print_census(label, compiled):
    """Static counts, plus whether Triton's PTX asked for tensor cores at all.

    HMMA is the tensor-core instruction; FFMA and HFMA2 are scalar fp32 and
    paired-fp16 multiply-adds on the CUDA cores. If the matrix multiplies
    became FFMA/HFMA2 and the PTX has no `mma`, Triton never tried.
    """
    c = sass_census(compiled)
    if c is None:
        print(f"  {label:<26} (no cubin or nvdisasm)")
        return
    if "error" in c:
        print(f"  {label:<26} (unreadable: {c['error']})")
        return
    ptx = (getattr(compiled, "asm", None) or {}).get("ptx", "") or ""
    mma = ptx.count("mma.sync") + ptx.count("wgmma") + ptx.count("mma.")
    print(f"  {label:<26}" + "".join(f"{c.get(k, 0):>8}" for k in CENSUS)
          + f"{mma:>9}")


def time_in_graph(torch, fn, reps=20, iters=10) -> float:
    """Milliseconds per call, from replaying `reps` calls captured in one
    CUDA graph. Production decodes under graphs, so this is the time that
    matters — and it charges neither backend for Python launch overhead,
    which for CuPy's argument packing would otherwise swamp a 0.16 ms
    batch-1 kernel."""
    fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (iters * reps)


class _Cubin:
    """Adapter so the census can read a cubin that did not come from Triton."""

    def __init__(self, cubin):
        self.asm = {"cubin": cubin, "ptx": ""}


def cuda_section(torch) -> None:
    from kernels.cuda import paged_decode_cuda as pdc
    from kernels.gqa import paged_decode as pd

    print("\n[5] the CUDA-core kernel (kernels/cuda) against the Triton production kernel\n",
          flush=True)
    try:
        cubin, log = pdc.compile_cubin(
            "sm_{}{}".format(*torch.cuda.get_device_capability()))
    except Exception as e:  # noqa: BLE001
        print(f"  cannot build it here: {str(e).splitlines()[0]}\n"
              f"  pip install {pdc.cupy_package()}")
        return
    import cupy

    nvrtc_version = "?"
    try:
        from cupy_backends.cuda.libs import nvrtc as _nv

        nvrtc_version = ".".join(map(str, _nv.getVersion()))
    except Exception:  # noqa: BLE001
        pass
    print(f"  CuPy {cupy.__version__}, NVRTC {nvrtc_version}")
    report = [l.strip() for l in log.splitlines() if "registers" in l or "spill" in l]
    for line in report:
        print(f"  ptxas: {line}")
    if not report:
        print("  ptxas: NVRTC's log carries no register report on this install")
    res = pdc.kernel_resources()
    print(f"  driver: {res['regs']} registers, {res['local_bytes']} bytes local memory "
          f"(spills), {res['shared_bytes']} bytes shared memory", flush=True)

    args, _, _ = build_inputs(torch, 5, 1024, False, ragged=[1024, 300, 61, 17, 1], seed=3)
    want = pd.paged_decode_reference(*args, num_splits=1)
    worst = 0.0
    for splits in (None, 1, 3, 64):
        got = pdc.paged_decode_cuda(*args, 1024, num_splits=splits)
        worst = max(worst, float((got.float() - want.float()).abs().max()))
    print(f"  correctness (ragged batch, 4 split counts): max abs error {worst:.2e} -> "
          + ("OK" if worst < 5e-3 else "WRONG - timings below mean nothing"), flush=True)

    print(f"\n  {'batch':>5}{'ctx':>6}  {'triton ms':>10}{'GB/s':>7}   {'cuda ms':>9}{'GB/s':>7}"
          f"{'splits':>8}   speedup")
    for batch, ctx in ((1, 2048), (1, 8192), (4, 8192), (16, 8192), (16, 2048)):
        args, _, kv_bytes = build_inputs(torch, batch, ctx, False)
        t_ms = time_in_graph(torch, lambda: pd.paged_decode_attention(*args, max_seq_len=ctx))
        best = None
        num_pages = -(-ctx // 16)
        for splits in sorted({pdc.choose_splits(batch, 2, num_pages), 16, 32, 64, 128}):
            if splits > num_pages:
                continue
            c_ms = time_in_graph(torch, lambda s=splits: pdc.paged_decode_cuda(
                *args, ctx, num_splits=s))
            if best is None or c_ms < best[0]:
                best = (c_ms, splits)
        c_ms, splits = best
        auto = " (auto)" if splits == pdc.choose_splits(batch, 2, num_pages) else ""
        print(f"  {batch:>5}{ctx:>6}  {t_ms:>10.3f}{kv_bytes / t_ms / 1e6:>7.0f}   "
              f"{c_ms:>9.3f}{kv_bytes / c_ms / 1e6:>7.0f}{splits:>8}{auto:<7}{t_ms / c_ms:>6.2f}x",
              flush=True)

    print("\n  census of the CUDA kernel's machine code:")
    print(f"  {'kernel':<26}" + "".join(f"{k.replace(' ', ''):>8}" for k in CENSUS)
          + f"{'PTX mma':>9}")
    print_census("cuda fp16", _Cubin(cubin))
    print("\n  Loads-only reached ~243 GB/s at batch 16 / 8K: that is the ceiling to\n"
          "  compare the CUDA column against. Splits shown are the fastest of those\n"
          "  tried; '(auto)' means the kernel's own choice was already the best.")
    int8_section(torch)


def int8_section(torch) -> None:
    """The INT8 CUDA kernel against Triton INT8 and against CUDA fp16.

    Production layout: symmetric, the last page's K in the fp16 residual.
    Both register bounds are timed — unbounded (no spills, 10 warps/SM) and
    bounded to fp16's 168 registers (12 warps, small spills) — because which
    wins depends on the GPU, not on the compiler's report.
    """
    from kernels.cuda import paged_decode_cuda as pdc
    from kernels.gqa import paged_decode as pd

    print("\n[6] INT8 cache on the CUDA kernel (symmetric, last page in the fp16 residual)\n",
          flush=True)
    for mb in (1, 12):
        try:
            r = pdc.kernel_resources(variant="int8", has_res=True, min_blocks=mb)
            print(f"  min_blocks {mb:>2}: driver reports {r['regs']} registers, "
                  f"{r['local_bytes']} bytes local (spill) memory, {r['shared_bytes']} bytes shared")
        except Exception as e:  # noqa: BLE001
            print(f"  min_blocks {mb:>2}: cannot build: {str(e).splitlines()[0][:90]}")
            return

    args, extra, _ = build_inputs(torch, 5, 1024, True, ragged=[1024, 300, 61, 17, 1], seed=4)
    want = pd.paged_decode_reference(*args, num_splits=1, **extra)
    worst = 0.0
    for mb in (1, 12):
        for splits in (None, 3, 64):
            got = pdc.paged_decode_cuda(*args, 1024, num_splits=splits, min_blocks=mb, **extra)
            worst = max(worst, float((got.float() - want.float()).abs().max()))
    print(f"  correctness (ragged, 2 bounds x 3 split counts): max abs error {worst:.2e} -> "
          + ("OK" if worst < 5e-3 else "WRONG - timings below mean nothing"), flush=True)

    print(f"\n  {'batch':>5}{'ctx':>6}  {'triton int8':>12}  {'cuda int8 b1':>13}  "
          f"{'cuda int8 b12':>14}  {'cuda fp16':>10}   int8 vs fp16 (cuda)")
    before = pd.decode_backend()
    try:
        for batch, ctx in ((1, 2048), (1, 8192), (4, 8192), (16, 8192), (16, 2048)):
            a8, e8, _ = build_inputs(torch, batch, ctx, True)
            a16, _, _ = build_inputs(torch, batch, ctx, False)
            pd.set_decode_backend("triton")
            tri = time_in_graph(torch, lambda: pd.paged_decode_attention(*a8, max_seq_len=ctx, **e8))
            c1 = time_in_graph(torch, lambda: pdc.paged_decode_cuda(*a8, ctx, min_blocks=1, **e8))
            c12 = time_in_graph(torch, lambda: pdc.paged_decode_cuda(*a8, ctx, min_blocks=12, **e8))
            f16 = time_in_graph(torch, lambda: pdc.paged_decode_cuda(*a16, ctx))
            best = min(c1, c12)
            print(f"  {batch:>5}{ctx:>6}  {tri:>10.3f}ms  {c1:>11.3f}ms  {c12:>12.3f}ms  "
                  f"{f16:>8.3f}ms   {f16 / best:>5.2f}x  (best bound: {'b1' if c1 <= c12 else 'b12'})",
                  flush=True)
    finally:
        pd.set_decode_backend(before)
    print("\n  'int8 vs fp16' above 1.00x means INT8 decodes faster than fp16 on the same\n"
          "  kernel design: the bytes it saves now buy time, not just memory.")


def child(variant: str) -> int:
    import torch

    int8, ppi = variant.split("_")
    measure(torch, 16, 8192, int8 == "int8", int(ppi[3:]), iters=3)
    return 0


def profile_with_ncu(ncu: str) -> None:
    sections = ["SpeedOfLight", "Occupancy", "WarpStateStats", "MemoryWorkloadAnalysis"]
    for variant in ("fp16_ppi4", "int8_ppi4"):
        cmd = [ncu, "--kernel-name", "regex:_paged_decode", "--launch-skip", "5",
               "--launch-count", "1", "--csv", "--page", "details"]
        for sec in sections:
            cmd += ["--section", sec]
        stalls = ["long_scoreboard", "short_scoreboard", "barrier", "wait",
                  "math_pipe_throttle", "mio_throttle", "not_selected"]
        cmd += ["--metrics", ",".join(f"smsp__warp_issue_stalled_{x}_per_warp_active.pct"
                                     for x in stalls)]
        cmd += [sys.executable, "-m", "benchmarks.runners.phase12_diag", "--child", variant]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        print(f"\n--- ncu: {variant} (batch 16, ctx 8192) ---")
        for line in out.stdout.splitlines():
            if "Occupancy" in line or "Throughput" in line or "issue_stalled" in line:
                cells = [c.strip('"') for c in line.split('","')]
                if len(cells) >= 3:
                    print(f"  {cells[-3]:<50} {cells[-1]:>12} {cells[-2]}")


ROW = (f"{'variant':<24}{'batch':>6}{'ms':>8}{'GB/s':>7}{'regs':>6}{'spills':>8}"
       f"{'smem KB':>9}{'occupancy':>11}  bound by")


def _row(name, batch, m):
    print(f"{name:<24}{batch:>6}{m['ms']:>8.3f}{m['gbs']:>7.1f}{m['regs']:>6}{m['spills']:>8}"
          f"{m['smem'] / 1024:>9.1f}{m['occ']:>10.0%}  {m['bound']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    p.add_argument("--skip-ncu", action="store_true")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--int8", action="store_true",
                   help="just section [6], the INT8 CUDA kernel: about a minute")
    p.add_argument("--cuda", action="store_true",
                   help="just section [5], the CUDA-core kernel against Triton: about a minute")
    p.add_argument("--census-only", action="store_true",
                   help="just the machine-code census: seconds, not minutes")
    args = p.parse_args()
    if args.child:
        return child(args.child)

    import torch

    if not torch.cuda.is_available():
        print("[ERROR] this is a GPU diagnostic.", file=sys.stderr)
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    ctx = args.ctx

    if args.census_only or args.cuda or args.int8:
        args.skip_ncu = True
    status, ncu = ("skipped", None) if args.skip_ncu else ncu_status()
    print(f"\n[0/4] Nsight Compute: {status}" + (f" ({ncu})" if ncu else ""), flush=True)

    from kernels.gqa import paged_decode as pd
    from kernels.gqa import paged_decode_ablations as ab

    if args.census_only:
        _census_section(torch, pd, ab)
        return 0
    if args.cuda:
        cuda_section(torch)
        return 0
    if args.int8:
        int8_section(torch)
        return 0

    print("\n[1/4] correctness of every variant against the reference (ragged batch)", flush=True)
    ok = check_correctness(torch)
    print("  all variants match the reference" if ok else
          "  SOME VARIANTS ARE WRONG — their timings below mean nothing", flush=True)

    from kernels.gqa import paged_decode as pd
    from kernels.gqa import paged_decode_ablations as ab

    print(f"\n[2/4] production kernel: dtype x pages-per-iteration x pipeline stages, ctx {ctx}\n")
    print(ROW)
    for batch in (16, 1):
        for int8 in (False, True):
            for ppi in (1, 2, 4):
                for stages in (1, 2):
                    try:
                        m = measure(torch, batch, ctx, int8, ppi, stages=stages)
                        _row(f"{'int8' if int8 else 'fp16'} ppi{ppi} stages{stages}", batch, m)
                    except Exception as e:  # noqa: BLE001
                        print(f"{'int8' if int8 else 'fp16'} ppi{ppi} stages{stages}  failed: "
                              f"{str(e).splitlines()[0][:70]}")
        print(flush=True)

    print(f"[3/4] ablations (fp16, identity page layout so lookup vs no-lookup read the same "
          f"addresses), ctx {ctx}\n")
    print(ROW)
    for batch in (16, 1):
        for stages in (1, 2):
            for variant, ppi in (("full", 4), ("no_lookup", 4), ("loads_only", 4),
                                 ("compute_only", 4), ("prefetch", 1)):
                try:
                    m = measure_ablation(torch, variant, batch, ctx, ppi, stages)
                    _row(f"{variant} ppi{ppi} stages{stages}", batch, m)
                except Exception as e:  # noqa: BLE001
                    print(f"{variant} stages{stages}  failed: {str(e).splitlines()[0][:70]}")
        print(flush=True)

    _census_section(torch, pd, ab)
    cuda_section(torch)
    _reading()
    if status == "usable":
        profile_with_ncu(ncu)
    return 0


def _census_section(torch, pd, ab):
    print("[4/4] machine-code census (static instruction counts, nvdisasm)\n")
    print(f"  {'kernel':<26}" + "".join(f"{k.replace(' ', ''):>8}" for k in CENSUS)
          + f"{'PTX mma':>9}")
    for label, int8, ppi in (("fp16 ppi4 (production)", False, 4), ("fp16 ppi1", False, 1),
                             ("int8 ppi4", True, 4), ("int8 ppi1", True, 1)):
        measure(torch, 1, 1024, int8, ppi, iters=1, stages=2)
        print_census(label, pd.LAST_COMPILED.get("decode"))
    measure_ablation(torch, "prefetch", 1, 1024, 1, 2, iters=1)
    print_census("prefetch ppi1", ab.LAST_COMPILED.get("prefetch"))


def _reading():
    print("\nReading it:\n"
          "  [2] stages1 beating stages2 at ppi4 means shared memory was capping occupancy.\n"
          "      int8 regs/spills now near fp16's means the fp32-tile fix worked.\n"
          "  [3] loads_only ~ full: the memory path is the cost. compute_only ~ full: the\n"
          "      arithmetic (via register pressure) is. no_lookup << full: the dependent\n"
          "      block-table load serialises the loop. prefetch < full: overlapping that\n"
          "      load with compute is the fix.\n"
          "  [4] LDL/STL are spills made concrete; LDG 128 are fully vectorised loads.\n"
          "      HMMA = 0 with FFMA/HFMA2 in the hundreds and no PTX mma: the matrix\n"
          "      multiplies run on CUDA cores, and Triton never asked for tensor cores.")


if __name__ == "__main__":
    sys.exit(main())
