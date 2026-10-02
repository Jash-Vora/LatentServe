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


def build_inputs(torch, batch, ctx, int8, page=16, h=2, d=128, n_rep=6, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = ctx // page
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
    tables = torch.randperm(nb, device=dev, generator=g)[: batch * pages]
    tables = tables.reshape(batch, pages).to(torch.int32)
    lens = torch.full((batch,), ctx, dtype=torch.int32, device=dev)
    q = torch.randn(batch, h, n_rep, d, dtype=torch.float16, device=dev, generator=g)
    return (q, k, v, tables, lens), extra, kv_bytes


def measure(torch, batch, ctx, int8, ppi, iters=50):
    from kernels.gqa import paged_decode as pd

    args, extra, kv_bytes = build_inputs(torch, batch, ctx, int8)
    run = lambda: pd.paged_decode_attention(*args, max_seq_len=ctx, num_splits=16,  # noqa: E731
                                            pages_per_iter=ppi, **extra)
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    k = pd.LAST_COMPILED.get("decode")
    meta = getattr(k, "metadata", None)
    regs = getattr(k, "n_regs", 0) or 0
    spills = getattr(k, "n_spills", 0) or 0
    smem = getattr(meta, "shared", 0) if meta is not None else 0
    warps = getattr(meta, "num_warps", pd.NUM_WARPS) if meta is not None else pd.NUM_WARPS
    occ, bound = occupancy(regs, smem, warps)
    return {"ms": ms, "gbs": kv_bytes / (ms / 1000) / 1e9, "regs": regs,
            "spills": spills, "smem": smem, "warps": warps, "occ": occ, "bound": bound}


def child(variant: str) -> int:
    """Launch one kernel configuration a few times, for ncu to profile."""
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
        # Why warps wait, not just that they do. Long scoreboard is waiting on
        # global memory; barrier on other warps; math throttle on ALUs.
        stalls = ["long_scoreboard", "short_scoreboard", "barrier", "wait",
                  "math_pipe_throttle", "mio_throttle", "not_selected"]
        cmd += ["--metrics", ",".join(f"smsp__warp_issue_stalled_{x}_per_warp_active.pct"
                                     for x in stalls)]
        cmd += [sys.executable, "-m", "benchmarks.runners.phase12_diag", "--child", variant]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        print(f"\n--- ncu: {variant} (batch 16, ctx 8192) ---")
        wanted = ("Achieved Occupancy", "Theoretical Occupancy", "DRAM Throughput",
                  "Memory Throughput", "Compute (SM) Throughput", "Registers Per Thread",
                  "Warp Cycles Per Issued Instruction", "Stall Long Scoreboard",
                  "Stall Barrier", "Stall Math Pipe Throttle", "Duration")
        shown = 0
        for line in out.stdout.splitlines():
            if any(w in line for w in wanted) or "issue_stalled" in line:
                cells = [c.strip('"') for c in line.split('","')]
                if len(cells) >= 3:
                    print(f"  {cells[-3]:<40} {cells[-1]:>14} {cells[-2]}")
                    shown += 1
        if not shown:
            print("  (no matching metrics; raw tail below)")
            print("\n".join((out.stdout + out.stderr).splitlines()[-15:]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    p.add_argument("--skip-ncu", action="store_true")
    args = p.parse_args()
    if args.child:
        return child(args.child)

    import torch

    if not torch.cuda.is_available():
        print("[ERROR] this is a GPU diagnostic.", file=sys.stderr)
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    status, ncu = ("skipped", None) if args.skip_ncu else ncu_status()
    print(f"\n[1/3] Nsight Compute: {status}" + (f" ({ncu})" if ncu else ""))
    if status == "no-permission":
        print("      The driver refuses performance counters in this container. The\n"
              "      resource numbers below are the fallback, and usually enough.")
    elif status == "missing":
        print("      Not installed. The resource numbers below are the fallback.")

    print("\n[2/3] + [3/3] kernel resources and achieved bandwidth (16 splits)\n")
    print(f"{'variant':<12}{'batch':>6}{'ctx':>6}{'ms':>8}{'GB/s':>7}{'regs':>6}"
          f"{'spills':>8}{'smem KB':>9}{'warps':>6}{'occupancy':>11}  bound by")
    for int8 in (False, True):
        for ppi in (1, 4):
            for batch in (1, 16):
                m = measure(torch, batch, 8192, int8, ppi)
                name = f"{'int8' if int8 else 'fp16'} ppi{ppi}"
                print(f"{name:<12}{batch:>6}{8192:>6}{m['ms']:>8.3f}{m['gbs']:>7.1f}"
                      f"{m['regs']:>6}{m['spills']:>8}{m['smem'] / 1024:>9.1f}{m['warps']:>6}"
                      f"{m['occ']:>10.0%}  {m['bound']}")

    print("\nReading it: spills above zero turn register reads into memory traffic;\n"
          "occupancy well under 50% leaves too few warps to hide memory latency.\n"
          "INT8 rows with more registers or spills than their fp16 twins point at\n"
          "dequantizing into full fp32 tiles.")

    if status == "usable":
        print("\n[ncu] profiling fp16 and INT8 at batch 16 / 8K ...")
        profile_with_ncu(ncu)
    return 0


if __name__ == "__main__":
    sys.exit(main())