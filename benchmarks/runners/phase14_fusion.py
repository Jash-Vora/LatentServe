"""
Phase 14a — what projection fusion is worth.

    export PYTHONPATH=$(pwd):$PYTHONPATH
    python -m benchmarks.runners.phase14_fusion \
        --context-lengths 2048 8192 --batch-sizes 1 4 16

## Why this is one process, alternating

Phase 13 measured machine-to-machine noise at 4-8% on Kaggle T4s, and
the effect being looked for here is about 1-1.5 ms on a ~22 ms step —
5-7%. A comparison across sessions could not distinguish them.

Fusion shares every weight with the unfused path (model/fused.py), so
one loaded model serves both. Each configuration is measured in
alternating rounds — unfused, fused, unfused, fused — so slow drift in
clocks or temperature lands on both sides equally. The reported delta is
the median of per-round differences, with its spread, so a gain smaller
than the round-to-round wobble is visible as such.

## The prediction

Batch 1 / 2K improves by most of the ~1.5 ms gap to vLLM. The saving is
per-launch fixed cost, so it should be roughly constant in milliseconds
across context lengths and shrink as a *fraction* as batch and context
grow and the step gets longer.

If batch 1 barely moves, the batch-1 gap is not launch count, and
Phase 13's explanation of it was incomplete.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

from benchmarks.schema import BenchmarkResult, ResultWriter
from config import load_config


def kernels_per_step(model, batch: int, ctx: int, block_size: int) -> int:
    """CUDA kernels in one eager decode step.

    Counted eagerly because that is where each launch is a separate
    event; under a graph they are the same kernels, replayed as one call.
    """
    model.allocate_cache(batch, ctx + 64, paged=True, block_size=block_size)
    model.cache.reset()
    model.cache.advance(ctx, batch_size=batch)
    slots = list(range(batch))
    ids = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
    pos = torch.full((batch, 1), ctx, dtype=torch.long, device="cuda")
    model.cache.advance(1, slots=slots)
    model.decode_forward_static(ids, pos, max_position=ctx + 63)      # warm
    model.cache.advance(1, slots=slots)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        model.decode_forward_static(ids, pos + 1, max_position=ctx + 63)
        torch.cuda.synchronize()
    n = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    model.cache = None
    torch.cuda.empty_cache()
    return n


def time_graphed(model, batch: int, ctx: int, block_size: int, steps: int, warmup: int) -> float:
    from runtime.cuda_graph import GraphedDecoder

    model.allocate_cache(batch, ctx + steps + warmup + 64, paged=True, block_size=block_size)
    model.cache.reset()
    model.cache.advance(ctx, batch_size=batch)
    slots = list(range(batch))
    ids = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
    # A fresh decoder per measurement: a graph bakes in whichever path was
    # active when it was captured, so reusing one across set_fused() would
    # measure the old path under the new label.
    decoder = GraphedDecoder(model)
    timings = []
    for i in range(warmup + steps):
        pos = torch.full((batch, 1), ctx + i, dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decoder.step(ids, pos, slots)
        torch.cuda.synchronize()
        if i >= warmup:
            timings.append((time.perf_counter() - t0) * 1000)
    model.cache = None
    torch.cuda.empty_cache()
    return statistics.median(timings)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase2_gqa.yaml")
    p.add_argument("--context-lengths", type=int, nargs="+", default=[2048, 8192])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--rounds", type=int, default=4,
                   help="alternating unfused/fused rounds per configuration")
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--results-dir", default="results/raw")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] Phase 14 is a GPU measurement.", file=sys.stderr)
        return 1

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    cfg = load_config(args.config)
    ref = QwenReference(model_name=cfg.model.name, dtype=cfg.model.dtype,
                        device=f"cuda:{cfg.hardware.devices[0]}").load()
    longest = max(args.context_lengths) + args.steps + args.warmup + 64
    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=longest,
                                           attn_impl="triton_paged", fuse_projections=True)
    writer = ResultWriter(results_dir=args.results_dir)

    # The mechanism, before the timing: if the kernel count does not fall
    # by ~84, nothing downstream is measuring fusion.
    ctx0 = args.context_lengths[0]
    model.set_fused(False)
    k_unfused = kernels_per_step(model, 1, ctx0, args.block_size)
    model.set_fused(True)
    k_fused = kernels_per_step(model, 1, ctx0, args.block_size)
    print(f"CUDA kernels per decode step (batch 1, ctx {ctx0}): "
          f"unfused {k_unfused}, fused {k_fused}  ({k_unfused - k_fused} fewer)\n")

    print(f"{'batch':>5}{'ctx':>7}{'unfused ms':>12}{'fused ms':>10}{'saved ms':>10}"
          f"{'saved':>8}{'round spread ms':>17}")
    for batch in args.batch_sizes:
        for ctx in args.context_lengths:
            unfused, fused = [], []
            for _ in range(args.rounds):
                for on, bucket in ((False, unfused), (True, fused)):
                    model.set_fused(on)
                    bucket.append(time_graphed(model, batch, ctx, args.block_size,
                                               args.steps, args.warmup))
            deltas = [u - f for u, f in zip(unfused, fused)]
            saved = statistics.median(deltas)
            spread = max(deltas) - min(deltas)
            u_med, f_med = statistics.median(unfused), statistics.median(fused)
            print(f"{batch:>5}{ctx:>7}{u_med:>12.2f}{f_med:>10.2f}{saved:>10.2f}"
                  f"{saved / u_med:>8.1%}{spread:>17.2f}")
            for label, value in (("unfused", u_med), ("fused", f_med)):
                writer.write(BenchmarkResult(
                    system=f"latentserve_kernel_graphed_{label}", tag="phase14_fusion",
                    attention="gqa", model=cfg.model.name, batch_size=batch,
                    context_length=ctx, output_length=args.steps, num_gpus=1,
                    tpot_ms=value, seed=0,
                    extra={"status": "ok", "fused": label == "fused", "rounds": args.rounds,
                           "saved_ms_median": saved, "saved_ms_spread": spread,
                           "kernels_unfused": k_unfused, "kernels_fused": k_fused},
                ))
    model.set_fused(True)
    print("\nA saving smaller than its round spread is not a saving. The prediction was\n"
          "most of a ~1.5 ms gap at batch 1, roughly constant in ms across contexts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
