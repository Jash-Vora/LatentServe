"""
Phase 12 — why the kernel moves KV at ~31 GB/s when the card does ~110.

Phase 11 left two candidate explanations and no way to choose between
them from end-to-end timings:

  * **occupancy** — 2 KV heads on 40 SMs is the shortage Phase 2 already
    measured (achieved bandwidth tracked `batch x kv_heads`: 2 blocks
    reached 95 GB/s, 48 reached 198). More programs would fix it.
  * **latency** — the block-table lookup is a pointer-dependent load, so
    the fetch of page p+1 cannot start until page p's block id arrives.
    Deeper pipelining or bigger tiles would fix it.

Those point in opposite directions, and guessing has now cost two GPU
runs. This runner distinguishes them by measurement.

## The primary experiment needs no profiler

`--experiment sweep` runs the kernel **in isolation** on synthetic pools
— no model, no 28 layers, no Python dispatch — and sweeps the design
space, reporting achieved GB/s of KV read. The shape of the response is
the diagnosis:

    bandwidth rises with num_splits    -> occupancy-limited
    bandwidth rises with PPI / stages  -> latency-limited
    bandwidth rises with neither       -> something structural in the
                                          access pattern

It also isolates the kernel from the ~9 ms/step of Python and launch
overhead that Phase 11 measured separately, so the number here is the
kernel's own ceiling rather than what the model path achieves.

## Nsight is the confirmation, not the first step

`ncu` needs GPU performance-counter access, which containerised
environments frequently refuse (`ERR_NVGPUCTRPERM`). `--experiment ncu`
checks availability, prints the exact command, and parses the CSV if it
runs. `--experiment torch` is the always-available middle ground: a
`torch.profiler` trace of one real decode step, ranked by CUDA time,
which at least says how much of the step is this kernel versus
everything else.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

KV_BYTES_PER_TOKEN_PER_LAYER = 28_672


def synthetic_cache(batch, ctx, kv_heads, head_dim, page, device, dtype=torch.float16):
    """Pools and a scattered block table, sized like the real thing.

    Blocks are permuted rather than sequential: a contiguous table would
    make the pointer-dependent load look predictable to the memory
    system and flatter the very thing being measured.
    """
    pages = (ctx + page - 1) // page
    num_blocks = batch * pages
    k = torch.randn(num_blocks, page, kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(num_blocks, page, kv_heads, head_dim, dtype=dtype, device=device)
    table = torch.randperm(num_blocks, device=device)[: batch * pages]
    table = table.reshape(batch, pages).to(torch.int32)
    lens = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    return k, v, table, lens


def time_kernel(q, k, v, table, lens, iters=50, warmup=10, **kw) -> float:
    from kernels.gqa.paged_decode import paged_decode_attention

    for _ in range(warmup):
        paged_decode_attention(q, k, v, table, lens, max_seq_len=int(lens[0]), **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        paged_decode_attention(q, k, v, table, lens, max_seq_len=int(lens[0]), **kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def run_sweep(args) -> int:
    dev = torch.device("cuda")
    kv_heads, head_dim, n_rep = 2, 128, 6
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    print(f"device: {torch.cuda.get_device_name(0)}, {sms} SMs\n")

    for batch in args.batch_sizes:
        for ctx in args.context_lengths:
            k, v, table, lens = synthetic_cache(batch, ctx, kv_heads, head_dim,
                                                args.page, dev)
            q = torch.randn(batch, kv_heads, n_rep, head_dim, dtype=torch.float16, device=dev)
            kv_bytes = batch * ctx * KV_BYTES_PER_TOKEN_PER_LAYER / 28  # one layer's worth
            pages = (ctx + args.page - 1) // args.page

            print(f"=== batch={batch} ctx={ctx} ({pages} pages, "
                  f"{kv_bytes / 1024**2:.1f} MiB of KV) ===")
            print(f"{'splits':>7}{'PPI':>5}{'warps':>7}{'stages':>8}"
                  f"{'ms':>9}{'GB/s':>8}{'programs':>10}")
            best = None
            for splits in args.splits:
                if splits > pages:
                    continue
                for ppi in args.pages_per_iter:
                    for warps in args.warps:
                        for stages in args.stages:
                            try:
                                ms = time_kernel(
                                    q, k, v, table, lens, iters=args.iters,
                                    num_splits=splits, pages_per_iter=ppi,
                                    num_warps=warps, num_stages=stages,
                                )
                            except Exception as e:  # noqa: BLE001
                                print(f"{splits:>7}{ppi:>5}{warps:>7}{stages:>8}"
                                      f"   failed: {str(e)[:40]}")
                                continue
                            gbs = kv_bytes / (ms / 1000) / 1e9
                            programs = batch * kv_heads * splits
                            print(f"{splits:>7}{ppi:>5}{warps:>7}{stages:>8}"
                                  f"{ms:>9.3f}{gbs:>8.1f}{programs:>10}")
                            if best is None or gbs > best[0]:
                                best = (gbs, splits, ppi, warps, stages)
            if best:
                print(f"  best: {best[0]:.1f} GB/s at splits={best[1]} PPI={best[2]} "
                      f"warps={best[3]} stages={best[4]}\n")
    print("Reading the sweep:\n"
          "  bandwidth climbs with splits   -> occupancy-limited; the fix is more\n"
          "                                    programs, not bigger tiles\n"
          "  climbs with PPI or stages      -> latency-limited on the pointer-dependent\n"
          "                                    block-table load\n"
          "  climbs with neither            -> structural: look at the access pattern\n"
          "                                    itself, and this is where ncu earns its keep")
    return 0


def run_torch_profiler(args) -> int:
    """Where the decode step's time actually goes, without ncu."""
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    from config import load_config

    cfg = load_config(args.config)
    ref = QwenReference(model_name=cfg.model.name, dtype=cfg.model.dtype, device="cuda:0").load()
    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=args.context_lengths[0] + 64)
    for layer in model.layers:
        layer.attn.attn_impl = args.impl

    batch, ctx = args.batch_sizes[0], args.context_lengths[0]
    model.allocate_cache(batch, ctx + 64, paged=True, block_size=args.page)
    model.cache.reset()
    model.cache.advance(ctx, batch_size=batch)
    ids = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
    pos = torch.full((batch, 1), ctx - 1, dtype=torch.long, device="cuda")
    slots = list(range(batch))

    for _ in range(5):
        model.decode_step_ragged(ids, pos, slots)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(10):
            model.decode_step_ragged(ids, pos, slots)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    out = Path(args.results_dir) / f"phase12_trace_{args.impl}_b{batch}_c{ctx}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out))
    print(f"\nchrome trace: {out}")
    print("Launch count matters as much as kernel time here: Phase 11 measured ~9 ms\n"
          "per decode step of Python and launch overhead against ~5.7 ms for the SDPA\n"
          "path, across 28 layers.")
    return 0


def run_ncu(args) -> int:
    ncu = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    script = "benchmarks/runners/phase12_profile.py"
    cmd = [
        ncu or "ncu", "--target-processes", "all",
        "--kernel-name", "regex:paged_decode",
        "--launch-count", "4",
        "--section", "SpeedOfLight",
        "--section", "Occupancy",
        "--section", "MemoryWorkloadAnalysis",
        "--csv", "--log-file", str(Path(args.results_dir) / "phase12_ncu.csv"),
        sys.executable, "-m", "benchmarks.runners.phase12_profile",
        "--experiment", "sweep", "--iters", "4",
        "--splits", "32", "--pages-per-iter", "4",
        "--batch-sizes", str(args.batch_sizes[0]),
        "--context-lengths", str(args.context_lengths[0]),
    ]
    print("command:\n  " + " ".join(cmd) + "\n")
    if ncu is None:
        print("ncu not found on PATH. It ships with the CUDA toolkit; on a container\n"
              "image without it, the sweep above is the substitute — it answers the\n"
              "occupancy-vs-latency question by response shape rather than by counter.")
        return 1
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nncu failed ({e.returncode}). If the message mentions ERR_NVGPUCTRPERM,\n"
              "the driver is refusing performance-counter access — common in containers\n"
              "and not something the code can work around. Use --experiment sweep.")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=["sweep", "torch", "ncu"], default="sweep")
    p.add_argument("--config", default="configs/phase2_gqa.yaml")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--context-lengths", type=int, nargs="+", default=[8192])
    p.add_argument("--page", type=int, default=16)
    p.add_argument("--splits", type=int, nargs="+", default=[4, 16, 32, 64, 128])
    p.add_argument("--pages-per-iter", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    p.add_argument("--stages", type=int, nargs="+", default=[2, 4])
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--impl", default="triton_paged", choices=["triton_paged", "sdpa"])
    p.add_argument("--results-dir", default="results/raw")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] Phase 12 is a GPU measurement.", file=sys.stderr)
        return 1
    return {"sweep": run_sweep, "torch": run_torch_profiler, "ncu": run_ncu}[args.experiment](args)


if __name__ == "__main__":
    sys.exit(main())