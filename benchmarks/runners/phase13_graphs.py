"""
Phase 13 — what CUDA graphs buy.

    export PYTHONPATH=$(pwd):$PYTHONPATH
    python -m benchmarks.runners.phase13_graphs --experiment compile
    python -m benchmarks.runners.phase13_graphs --experiment latency \
        --context-lengths 4096 8192 16384 --batch-sizes 1 4

## The prediction, written before measuring

Phase 12 measured batch 1 / ctx 8192 at 40 ms wall, 24.6 ms of GPU work
and 42.6 ms of CPU dispatch. The graph removes dispatch, not GPU work,
so graphed TPOT should fall toward **~25-28 ms** — the GPU floor plus a
replay call and the host-side `advance()`.

At batch 4 / long context the step was already GPU-bound (the kernel
was within 2.4% of SDPA at 8K), so expect much less there. If batch 4
gains as much as batch 1, the Phase 12 reading of where the time went
was wrong.

## `--experiment compile` first

`torch.compile(mode="reduce-overhead")` does graph capture
automatically. If it compiles the decode core without graph breaks,
the hand-rolled capture in runtime/cuda_graph.py is unnecessary. The
cache classes are stateful enough that breaks are expected — but that
is a thirty-minute check against weeks of work, so it runs first.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

from benchmarks.schema import BenchmarkResult, ResultWriter
from config import load_config


def _setup(cfg, ctx, batch, block):
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    ref = QwenReference(model_name=cfg.model.name, dtype=cfg.model.dtype,
                        device=f"cuda:{cfg.hardware.devices[0]}").load()
    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=ctx + 256,
                                           attn_impl="triton_paged")
    return ref, model


def run_compile(args) -> int:
    """Count graph breaks in the decode core."""
    cfg = load_config(args.config)
    ref, model = _setup(cfg, args.context_lengths[0], 1, args.block_size)
    ctx = args.context_lengths[0]
    model.allocate_cache(1, ctx + 256, paged=True, block_size=args.block_size)
    model.cache.reset()
    model.cache.advance(ctx, batch_size=1)
    model.cache.advance(1, slots=[0])
    ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    pos = torch.full((1, 1), ctx, dtype=torch.long, device="cuda")

    import torch._dynamo as dynamo

    explanation = dynamo.explain(model.decode_forward_static)(ids, pos, ctx + 255)
    print(f"graphs: {explanation.graph_count}   graph breaks: {explanation.graph_break_count}")
    for reason in explanation.break_reasons[:10]:
        print(f"  - {reason.reason}")
    if explanation.graph_break_count == 0:
        print("\nNo breaks: torch.compile(mode='reduce-overhead') can capture the core\n"
              "directly. Compare its latency against runtime/cuda_graph.py before\n"
              "keeping the hand-rolled path.")
    else:
        print("\nBreaks found. Each one splits the step into separately launched\n"
              "regions, which is the overhead graphs exist to remove. The hand-rolled\n"
              "capture in runtime/cuda_graph.py avoids them by construction.")
    return 0


def time_steps(step_fn, steps, warmup):
    timings = []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step_fn(i)
        torch.cuda.synchronize()
        if i >= warmup:
            timings.append((time.perf_counter() - t0) * 1000)
    return statistics.median(timings), sorted(timings)[int(0.95 * (len(timings) - 1))]


def run_latency(args) -> int:
    from runtime.cuda_graph import GraphedDecoder

    cfg = load_config(args.config)
    ref, model = _setup(cfg, max(args.context_lengths), max(args.batch_sizes), args.block_size)
    writer = ResultWriter(results_dir=args.results_dir)

    for batch in args.batch_sizes:
        for ctx in args.context_lengths:
            capacity = ctx + args.steps + args.warmup + 64
            results = {}
            for mode in ("eager", "graphed"):
                model.allocate_cache(batch, capacity, paged=True, block_size=args.block_size)
                model.cache.reset()
                model.cache.advance(ctx, batch_size=batch)
                slots = list(range(batch))
                ids = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
                decoder = GraphedDecoder(model, enabled=(mode == "graphed"))

                def step(i, decoder=decoder):
                    pos = torch.full((batch, 1), ctx + i, dtype=torch.long, device="cuda")
                    decoder.step(ids, pos, slots)

                tpot, p95 = time_steps(step, args.steps, args.warmup)
                results[mode] = (tpot, p95, decoder.stats())
                model.cache = None
                torch.cuda.empty_cache()

            eager, graphed = results["eager"][0], results["graphed"][0]
            print(f"batch={batch} ctx={ctx:>6}   eager {eager:6.2f} ms   graphed "
                  f"{graphed:6.2f} ms   ({graphed / eager - 1:+.1%})   "
                  f"captures={results['graphed'][2]['captures']}")
            for mode, (tpot, p95, st) in results.items():
                writer.write(BenchmarkResult(
                    system=f"latentserve_kernel_{mode}", tag="phase13_graphs",
                    attention="gqa", model=cfg.model.name, batch_size=batch,
                    context_length=ctx, output_length=args.steps, num_gpus=1,
                    tpot_ms=tpot, tpot_p95_ms=p95, seed=0,
                    extra={"status": "ok", "mode": mode, **{k: v for k, v in st.items()
                                                             if k != "keys"}},
                ))
    print("\nPrediction: batch 1 falls toward ~25-28 ms (Phase 12 measured 24.6 ms of\n"
          "GPU work under 42.6 ms of CPU dispatch). Batch 4 at long context gains\n"
          "much less, since it was already GPU-bound.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=["compile", "latency"], default="latency")
    p.add_argument("--config", default="configs/phase2_gqa.yaml")
    p.add_argument("--context-lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--results-dir", default="results/raw")
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("[ERROR] Phase 13 is a GPU measurement.", file=sys.stderr)
        return 1
    return {"compile": run_compile, "latency": run_latency}[args.experiment](args)


if __name__ == "__main__":
    sys.exit(main())
