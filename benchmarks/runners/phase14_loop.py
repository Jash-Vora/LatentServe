"""
Phase 14b — where the serving loop spends its time.

    export PYTHONPATH=$(pwd):$PYTHONPATH
    python -m benchmarks.runners.phase14_loop --context-lengths 2048 --batch-sizes 1 4 16

## Why

Phase 14a measured, on one host in one session: the bare graphed model
step at batch 1 / 2K took 18.91 ms, while the same step through the
serving engine took 19.6-20.1 ms (median gap between tokens) to 20.7 ms
(differenced mean). The remaining gap to vLLM was 1.1-1.6 ms — the same
size as the engine's own overhead. So the question became whether the
rest of the batch-1 gap lives in the serving loop rather than the model.

## What it measures

Three things per configuration, in one process:

    floor           the graphed model step alone, plus the one unavoidable
                    read of the chosen tokens back to the host
    host sampling   the engine as it was: argmax outside the graph, and
                    every step's inputs built on the host and copied up
    in-graph        argmax inside the graph, and in steady state the next
                    step's inputs taken straight from the last step's output
                    without leaving the GPU

The two engine modes alternate across rounds so slow drift lands on both.
Each loop iteration is broken into phases (inputs, decoder host work,
waiting for the GPU, sampling, request bookkeeping, scheduling); their
sum is the full per-token cost excluding prefill.

## Reading it

`engine - floor` is what the loop costs. If in-graph sampling closes most
of it, Phase 14b is done. If what remains is dominated by `bookkeeping`
or `schedule` — Python that runs while the GPU sits idle — that is the
case for overlapping it with the next step, which is a much larger and
riskier change and should be justified by this number first.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time

import torch

from benchmarks.schema import BenchmarkResult, ResultWriter
from config import load_config

PHASES = ("inputs", "decoder_host", "gpu_wait", "sample", "bookkeeping", "schedule")


def floor_ms(model, batch: int, ctx: int, block: int, steps: int, warmup: int) -> float:
    from runtime.cuda_graph import GraphedDecoder

    model.allocate_cache(batch, ctx + steps + warmup + 64, paged=True, block_size=block)
    model.cache.reset()
    model.cache.advance(ctx, batch_size=batch)
    decoder = GraphedDecoder(model, greedy=True)
    slots = list(range(batch))
    ids = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
    timings = []
    for i in range(warmup + steps):
        pos = torch.full((batch, 1), ctx + i, dtype=torch.long, device="cuda")
        t0 = time.perf_counter()
        ids = decoder.step_greedy(ids, pos, slots)
        ids.view(-1).tolist()
        if i >= warmup:
            timings.append((time.perf_counter() - t0) * 1000)
    model.cache = None
    del decoder
    gc.collect()
    torch.cuda.empty_cache()
    return statistics.median(timings)


def engine_profile(ref, model, batch: int, ctx: int, new_tokens: int, block: int,
                   sample_in_graph: bool) -> dict:
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    engine = ServingEngine(model, max_running=batch, max_seq_len=ctx + new_tokens + 64,
                           block_size=block, use_cuda_graphs=True,
                           sample_in_graph=sample_in_graph, profile_loop=True)
    # Capture before timing, at both ends of the length range the run covers.
    for point in sorted({ctx + 1, ctx + new_tokens}):
        engine.warmup_graphs(range(1, batch + 1), context_length=point)
    # Discard what warm-up accumulated; only served steps count.
    engine.loop_profile.clear()
    engine.profiled_steps = 0
    for i in range(batch):
        prompt = ref.synthesize_input_ids(ctx, seed=i)[0].tolist()
        engine.add_request(ServedRequest(request_id=i, prompt_ids=prompt,
                                         max_new_tokens=new_tokens))
    engine.run()
    steps = max(1, engine.profiled_steps)
    out = {p: engine.loop_profile.get(p, 0.0) / steps for p in PHASES}
    # The first decode step after prefill is host-fed by construction;
    # everything after it in a burst of equal-length requests is steady.
    out["device_fed_steps"] = engine.device_fed_steps
    out["host_fed_steps"] = engine.host_fed_steps
    model.cache = None
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase2_gqa.yaml")
    p.add_argument("--context-lengths", type=int, nargs="+", default=[2048])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--new-tokens", type=int, default=96)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--no-fuse", action="store_true",
                   help="measure without Phase 14a projection fusion")
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
    longest = max(args.context_lengths) + args.new_tokens + 128
    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=longest,
                                           attn_impl="triton_paged",
                                           fuse_projections=not args.no_fuse)
    writer = ResultWriter(results_dir=args.results_dir)

    head = "".join(f"{p:>13}" for p in PHASES)
    for batch in args.batch_sizes:
        for ctx in args.context_lengths:
            floor = floor_ms(model, batch, ctx, args.block_size, steps=48, warmup=8)
            runs = {False: [], True: []}
            for _ in range(args.rounds):
                for mode in (False, True):
                    runs[mode].append(engine_profile(ref, model, batch, ctx, args.new_tokens,
                                                     args.block_size, sample_in_graph=mode))

            print(f"\n=== batch {batch}  ctx {ctx}   (floor: model step alone {floor:.2f} ms) ===")
            print(f"{'mode':<16}{'per token':>11}{'over floor':>12}{head}")
            totals = {}
            for mode, label in ((False, "host sampling"), (True, "in-graph")):
                med = {p: statistics.median(r[p] for r in runs[mode]) for p in PHASES}
                total = sum(med.values())
                totals[mode] = total
                print(f"{label:<16}{total:>11.2f}{total - floor:>12.2f}"
                      + "".join(f"{med[p]:>13.3f}" for p in PHASES))
                writer.write(BenchmarkResult(
                    system=f"latentserve_engine_{'ingraph' if mode else 'hostsample'}",
                    tag="phase14_loop", attention="gqa", model=cfg.model.name,
                    batch_size=batch, context_length=ctx, output_length=args.new_tokens,
                    num_gpus=1, tpot_ms=total, seed=0,
                    extra={"status": "ok", "floor_ms": floor, "fused": not args.no_fuse,
                           **{f"{p}_ms": med[p] for p in PHASES},
                           "device_fed_steps": runs[mode][-1]["device_fed_steps"],
                           "host_fed_steps": runs[mode][-1]["host_fed_steps"]},
                ))
            saved = totals[False] - totals[True]
            gap = totals[False] - floor
            print(f"in-graph saves {saved:.2f} ms per token "
                  f"({saved / gap:.0%} of the loop's cost over the floor)" if gap > 0 else "")

    print("\nWhat is left over the floor, and which phase holds it, decides whether\n"
          "overlapping host work with the next step is worth building.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
