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


MODE = {"kv_dtype": "fp16"}


def kernels_per_step(model, batch: int, ctx: int, block_size: int) -> int:
    """CUDA kernels in one eager decode step.

    Counted eagerly because that is where each launch is a separate
    event; under a graph they are the same kernels, replayed as one call.
    """
    model.allocate_cache(batch, ctx + 64, paged=True, block_size=block_size,
                         kv_dtype=MODE["kv_dtype"])
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

    model.allocate_cache(batch, ctx + steps + warmup + 64, paged=True, block_size=block_size,
                         kv_dtype=MODE["kv_dtype"])
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
    p.add_argument("--decode-backend", default=None, choices=["triton", "cuda"],
                   help="decode attention kernel for both sides; with --toggle int8 and "
                   "cuda, fp16 and INT8 caches are compared on the CUDA kernel")
    p.add_argument("--sparse-ratio", type=float, default=0.25,
                   help="--toggle sparse: fraction of pages each decode step attends to")
    p.add_argument("--sparse-recent", type=int, default=2,
                   help="--toggle sparse: most recent pages always kept")
    p.add_argument("--toggle", choices=["projections", "elementwise", "int8", "cuda_decode",
                                        "sparse"],
                   default="projections",
                   help="which fusion to A/B. 'elementwise' keeps projections fused on both "
                   "sides, so it measures elementwise fusion alone, on top of 14a.")
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
    # One switch per experiment, so each fusion is measured against its
    # own baseline: projections against unfused, elementwise against
    # projections-fused.
    def set_int8(on: bool) -> None:
        MODE["kv_dtype"] = "int8" if on else "fp16"

    if args.decode_backend:
        from kernels.gqa.paged_decode import set_decode_backend as _sdb

        _sdb(args.decode_backend)
        print(f"decode attention kernel: {args.decode_backend}")

    def set_cuda_decode(on: bool) -> None:
        from kernels.gqa.paged_decode import set_decode_backend

        set_decode_backend("cuda" if on else "triton")

    def set_sparse(on: bool) -> None:
        model.set_sparse(args.sparse_ratio if on else None, recent=args.sparse_recent)

    if args.toggle == "sparse" and args.decode_backend != "cuda":
        # Sparse is measured against the best dense path there is, not Triton.
        from kernels.gqa.paged_decode import set_decode_backend as _sdb

        _sdb("cuda")
        args.decode_backend = "cuda"
        print("--toggle sparse: dense side on the CUDA kernel (--decode-backend cuda)")
    if args.toggle in ("int8", "cuda_decode", "sparse"):
        # Measured on the best model there is: projections and elementwise
        # both fused, on both sides.
        model.set_elementwise(True)
    switch = {"projections": model.set_fused, "elementwise": model.set_elementwise,
              "int8": set_int8, "cuda_decode": set_cuda_decode,
              "sparse": set_sparse}[args.toggle]
    # Column names that say what is compared. Every toggle used to print
    # "unfused / fused", which for --toggle int8 meant fp16 / INT8 cache and
    # read as something else entirely.
    off, on = {"projections": ("unfused", "fused"), "elementwise": ("unfused", "fused"),
               "int8": ("fp16", "int8"), "cuda_decode": ("triton", "cuda"),
               "sparse": ("dense", f"sparse{args.sparse_ratio:.4g}")}[args.toggle]
    writer = ResultWriter(results_dir=args.results_dir)

    # The mechanism, before the timing: if the kernel count does not fall
    # by ~84, nothing downstream is measuring fusion.
    ctx0 = args.context_lengths[0]
    switch(False)
    k_unfused = kernels_per_step(model, 1, ctx0, args.block_size)
    switch(True)
    k_fused = kernels_per_step(model, 1, ctx0, args.block_size)
    print(f"CUDA kernels per decode step (batch 1, ctx {ctx0}): "
          f"{off} {k_unfused}, {on} {k_fused}  ({k_unfused - k_fused} fewer)\n")

    print(f"{'batch':>5}{'ctx':>7}{off + ' ms':>12}{on + ' ms':>10}{'saved ms':>10}"
          f"{'saved':>8}{'round spread ms':>17}")
    for batch in args.batch_sizes:
        for ctx in args.context_lengths:
            unfused, fused = [], []
            try:
                for _ in range(args.rounds):
                    # `state`, not `on`: reusing `on` here overwrote the column
                    # label above, and every saved row was named "..._True".
                    for state, bucket in ((False, unfused), (True, fused)):
                        switch(state)
                        bucket.append(time_graphed(model, batch, ctx, args.block_size,
                                                   args.steps, args.warmup))
            except torch.cuda.OutOfMemoryError:
                # batch x context past what this GPU's memory holds: say so
                # and go on, rather than end the sweep at its first big shape.
                model.cache = None
                torch.cuda.empty_cache()
                print(f"{batch:>5}{ctx:>7}   does not fit in GPU memory: skipped", flush=True)
                continue
            deltas = [u - f for u, f in zip(unfused, fused)]
            saved = statistics.median(deltas)
            spread = max(deltas) - min(deltas)
            u_med, f_med = statistics.median(unfused), statistics.median(fused)
            print(f"{batch:>5}{ctx:>7}{u_med:>12.2f}{f_med:>10.2f}{saved:>10.2f}"
                  f"{saved / u_med:>8.1%}{spread:>17.2f}")
            for label, value in ((off, u_med), (on, f_med)):
                writer.write(BenchmarkResult(
                    system=(f"latentserve_kernel_graphed_{args.toggle}_{label}"
                            + (f"_{args.decode_backend}decode" if args.decode_backend else "")),
                    tag="phase14_fusion",
                    attention="gqa", model=cfg.model.name, batch_size=batch,
                    context_length=ctx, output_length=args.steps, num_gpus=1,
                    tpot_ms=value, seed=0,
                    extra={"status": "ok", "fused": label == on, "rounds": args.rounds,
                           "saved_ms_median": saved, "saved_ms_spread": spread,
                           "kernels_unfused": k_unfused, "kernels_fused": k_fused},
                ))
    if args.toggle in ("cuda_decode", "sparse"):
        switch(False)       # leave the process on the default backend / dense
    else:
        switch(True)
    print("\nA saving smaller than its round spread is not a saving.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
