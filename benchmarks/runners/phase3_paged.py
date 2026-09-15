"""
Phase 3 — paged KV cache benchmark.

docs/methodology.md Phase 3: compare contiguous vs. paged KV under
variable sequence lengths, concurrent requests, request termination and
high utilization, measuring fragmentation, usable cache capacity,
allocation overhead and latency impact.

Two experiments, because those four metrics do not all need a GPU:

  **A — capacity simulation (CPU, no model, seconds).**
  Fragmentation, usable capacity and allocation overhead are properties
  of the allocation policy, not of the T4. Simulating thousands of
  requests across a block-size sweep costs nothing and produces the
  Gate 4 numbers. Burning GPU hours to rediscover arithmetic would be a
  poor trade.

  **B — latency cost (GPU, real model).**
  What paging costs per token, measured the same way Phase 2 measured
  everything else. Expect paging to be *slower*: the gather is a real
  copy of the live KV, per layer, per step. Phase 2's `repeat_kv`
  finding calibrates the expectation — that was 13x the necessary
  traffic and dominated TPOT past 4K; this is 2x (one read, one write)
  on a term that was 7-13% of decode bytes at batch 1, so predict a few
  percent at batch 1 and more at large batch where KV dominates.

Run:

    export PYTHONPATH=$(pwd):$PYTHONPATH

    # A — no GPU needed
    python -m benchmarks.runners.phase3_paged --experiment capacity

    # B — on the T4
    python -m benchmarks.runners.phase3_paged --experiment latency \\
        --context-lengths 4096 8192 16384 --batch-sizes 1 4 8
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Optional

import torch

from benchmarks.schema import BenchmarkResult, ResultWriter
from benchmarks.workloads.ragged import WORKLOADS, describe, generate_requests
from cache.block_allocator import BlockAllocator, BlockTable, OutOfBlocks
from cache.kv_cache import KVCacheSpec
from config import load_config

# Qwen2.5-1.5B-Instruct, native GQA, fp16 — measured in Phase 2 and
# asserted in tests/test_phase2_gqa.py.
KV_BYTES_PER_TOKEN = 28_672
DEFAULT_BLOCK_SIZES = [1, 8, 16, 32, 64, 128, 256]


# ----------------------------------------------------------------------
# Experiment A — capacity simulation
# ----------------------------------------------------------------------


def simulate(
    workload: str,
    num_requests: int,
    budget_bytes: int,
    block_size: int,
    seed: int,
    paged: bool,
    max_model_len: int = 65536,
) -> dict:
    """Stream requests through a fixed KV budget and report what fits.

    The two policies differ in one respect, which is the whole of Phase 3:

      * **contiguous** reserves `max_model_len` tokens per admitted
        sequence, up front. It has to — a contiguous slot cannot grow
        into its neighbour, and the output length is not known when the
        request is admitted.
      * **paged** reserves whole blocks covering the tokens that
        actually exist, and grows them one block at a time as decoding
        proceeds.

    Everything else — arrival order, admission policy, lengths — is held
    identical, so the difference in what gets served is attributable to
    the allocator alone.
    """
    requests = generate_requests(workload, num_requests, seed=seed)
    tokens_per_block = block_size
    bytes_per_block = KV_BYTES_PER_TOKEN * tokens_per_block

    if paged:
        num_blocks = budget_bytes // bytes_per_block
        allocator = BlockAllocator(num_blocks=max(1, num_blocks), block_size=block_size)
    else:
        slots = budget_bytes // (KV_BYTES_PER_TOKEN * max_model_len)
        allocator = None

    queue = list(requests)
    running: list[tuple] = []  # (request, tokens_done, BlockTable | None)
    finished = 0
    step = 0
    alloc_seconds = 0.0
    concurrency_samples: list[int] = []
    used_samples: list[int] = []
    reserved_samples: list[int] = []

    while queue or running:
        # --- admission ---
        while queue:
            req = queue[0]
            if paged:
                need = (req.prompt_tokens + block_size - 1) // block_size
                if need > allocator.num_free:
                    break
                t0 = time.perf_counter()
                table = BlockTable(allocator)
                table.append(req.prompt_tokens)
                alloc_seconds += time.perf_counter() - t0
                running.append([queue.pop(0), req.prompt_tokens, table])
            else:
                if len(running) >= slots:
                    break
                running.append([queue.pop(0), req.prompt_tokens, None])

        if not running:
            raise RuntimeError(
                f"nothing admissible at step {step}: budget too small for this workload "
                f"(longest sequence needs "
                f"{max(r.total_tokens for r in requests) * KV_BYTES_PER_TOKEN / 1024**3:.2f} GiB)"
            )

        concurrency_samples.append(len(running))
        if paged:
            used_samples.append(sum(r[2].length for r in running))
            reserved_samples.append(sum(r[2].capacity for r in running))
        else:
            used_samples.append(sum(r[1] for r in running))
            reserved_samples.append(len(running) * max_model_len)

        # --- one decode step for everything running ---
        still: list = []
        for entry in running:
            req, done, table = entry
            produced = done - req.prompt_tokens
            if produced >= req.output_tokens:
                if paged:
                    table.free()
                finished += 1
                continue
            if paged:
                t0 = time.perf_counter()
                try:
                    table.append(1)
                except OutOfBlocks:
                    # Real systems preempt here. Phase 3 records it as a
                    # capacity failure and lets Phase 4 own the policy.
                    table.free()
                    finished += 1
                    continue
                alloc_seconds += time.perf_counter() - t0
            entry[1] = done + 1
            still.append(entry)
        running = still
        step += 1

    total_tokens = sum(r.total_tokens for r in requests)
    mean_reserved = statistics.mean(reserved_samples)
    mean_used = statistics.mean(used_samples)
    return {
        "policy": "paged" if paged else "contiguous",
        "workload": workload,
        "block_size": block_size if paged else None,
        "budget_gb": budget_bytes / 1024**3,
        "steps_to_drain": step,
        "mean_concurrency": statistics.mean(concurrency_samples),
        "peak_concurrency": max(concurrency_samples),
        "mean_tokens_resident": mean_used,
        "mean_tokens_reserved": mean_reserved,
        # What fraction of reserved KV actually holds live tokens. The
        # headline Gate 4 number.
        "capacity_efficiency": mean_used / mean_reserved if mean_reserved else 0.0,
        "tokens_per_gb": mean_used / (budget_bytes / 1024**3),
        "alloc_overhead_us_per_token": alloc_seconds * 1e6 / max(1, total_tokens),
        "requests_completed": finished,
    }


def run_capacity(
    workloads: list, block_sizes: list, budget_gb: float, num_requests: int, seed: int,
    results_dir: str, tag: str, max_model_len: int = 65536,
) -> list:
    writer = ResultWriter(results_dir=results_dir)
    budget = int(budget_gb * 1024**3)
    rows = []
    for workload in workloads:
        print(f"\n=== workload={workload} | budget={budget_gb} GiB ===")
        print(" ", describe(generate_requests(workload, num_requests, seed=seed)))
        # Two contiguous baselines, because the honest comparison depends
        # on what max_model_len the deployer picked:
        #   generic — sized for the longest request the server will accept
        #             (65536). What you get without workload knowledge.
        #   oracle  — sized to the longest sequence this workload actually
        #             produces. Unachievable in practice (it needs the
        #             future), and included so paging has to beat the best
        #             possible contiguous configuration, not a strawman.
        longest = max(r.total_tokens for r in generate_requests(workload, num_requests, seed=seed))
        baseline = simulate(workload, num_requests, budget, 16, seed, paged=False,
                            max_model_len=max_model_len)
        oracle = simulate(workload, num_requests, budget, 16, seed, paged=False,
                          max_model_len=longest)
        oracle["policy"] = "contiguous_oracle"
        for name, row in (("contiguous", baseline), ("contiguous(oracle)", oracle)):
            print(
                f"  {name:<24}: concurrency {row['mean_concurrency']:6.1f}  "
                f"efficiency {row['capacity_efficiency'] * 100:5.1f}%  "
                f"{row['tokens_per_gb']:>9,.0f} tokens/GiB"
            )
        for row in [baseline, oracle] + [
            simulate(workload, num_requests, budget, bs, seed, paged=True) for bs in block_sizes
        ]:
            if row["policy"] == "paged":
                print(
                    f"  paged block_size={row['block_size']:>4}    : "
                    f"concurrency {row['mean_concurrency']:6.1f}  "
                    f"efficiency {row['capacity_efficiency'] * 100:5.1f}%  "
                    f"{row['tokens_per_gb']:>9,.0f} tokens/GiB  "
                    f"alloc {row['alloc_overhead_us_per_token']:.3f} us/token  "
                    f"({row['mean_concurrency'] / baseline['mean_concurrency']:.1f}x generic, "
                    f"{row['mean_concurrency'] / oracle['mean_concurrency']:.1f}x oracle)"
                )
            result = BenchmarkResult(
                system=f"simulation_{row['policy']}",
                tag=tag,
                attention="gqa",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                batch_size=int(row["mean_concurrency"]),
                context_length=int(row["mean_tokens_resident"] / max(1, row["mean_concurrency"])),
                output_length=0,
                num_gpus=0,
                seed=seed,
                extra={"status": "ok", "experiment": "capacity_simulation", **row},
            )
            writer.write(result)
            rows.append(result)
    return rows


# ----------------------------------------------------------------------
# Experiment B — latency cost on the GPU
# ----------------------------------------------------------------------


def run_latency(
    config_path: str, context_lengths: list, batch_sizes: list, output_tokens: int,
    repeats: int, warmup: int, block_sizes: list, results_dir: str,
) -> list:
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    cfg = load_config(config_path)
    device = f"cuda:{cfg.hardware.devices[0]}" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] no CUDA device — latency numbers meaningless.", file=sys.stderr)

    ref = QwenReference(
        model_name=cfg.model.name, dtype=cfg.model.dtype, device=device,
        revision=cfg.model.revision, trust_remote_code=cfg.model.trust_remote_code,
    ).load()
    engine = LatentServeQwen.from_reference(
        ref, max_seq_len_hint=max(context_lengths) + output_tokens
    )
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    writer = ResultWriter(results_dir=results_dir)
    rows = []

    variants = [("contiguous", None)] + [("paged", bs) for bs in block_sizes]
    for batch_size in batch_sizes:
        for ctx_len in context_lengths:
            max_seq_len = ctx_len + output_tokens
            base_ids = ref.synthesize_input_ids(ctx_len, seed=cfg.generation.seed)
            input_ids = base_ids.expand(batch_size, -1).contiguous()
            baseline_tpot = None

            for policy, block_size in variants:
                engine.allocate_cache(
                    batch_size, max_seq_len,
                    paged=(policy == "paged"), block_size=block_size or 16,
                )
                trials = []
                try:
                    for trial in range(warmup + repeats):
                        r = engine.generate_with_timing(
                            input_ids=input_ids, max_new_tokens=output_tokens,
                            chunk_size=min(4096, ctx_len),
                        )
                        if trial >= warmup:
                            trials.append(r)
                except torch.cuda.OutOfMemoryError as e:
                    engine.cache = None
                    torch.cuda.empty_cache()
                    print(f"  [OOM] {policy} B={batch_size} ctx={ctx_len}")
                    writer.write(
                        BenchmarkResult(
                            system=f"latentserve_{policy}", tag=cfg.tag, attention="gqa",
                            model=cfg.model.name, batch_size=batch_size,
                            context_length=ctx_len, output_length=output_tokens,
                            num_gpus=1, seed=cfg.generation.seed,
                            extra={"status": "oom", "error": str(e).split("\n")[0]},
                        )
                    )
                    continue

                tpot = statistics.median([t.tpot_ms for t in trials])
                if policy == "contiguous":
                    baseline_tpot = tpot
                cache = engine.cache
                gather_mb = cache.gather_bytes_per_decode_step(batch_size) / 1024 / 1024
                label = policy if block_size is None else f"{policy}(bs={block_size})"
                print(
                    f"  B={batch_size} ctx={ctx_len:>6} {label:<16} "
                    f"ttft={statistics.median([t.ttft_ms for t in trials]):8.1f}ms "
                    f"tpot={tpot:6.2f}ms "
                    + (f"({tpot / baseline_tpot - 1:+.1%} vs contiguous) " if baseline_tpot else "")
                    + f"gather={gather_mb:.0f}MB/step"
                )
                writer.write(
                    BenchmarkResult(
                        system=f"latentserve_{policy}", tag=cfg.tag, attention="gqa",
                        model=cfg.model.name, batch_size=batch_size, context_length=ctx_len,
                        output_length=output_tokens, num_gpus=1,
                        ttft_ms=statistics.median([t.ttft_ms for t in trials]),
                        tpot_ms=tpot,
                        e2e_latency_ms=statistics.median([t.e2e_latency_ms for t in trials]),
                        throughput_tokens_sec=statistics.median(
                            [t.throughput_tokens_sec * batch_size for t in trials]
                        ),
                        peak_vram_mb=max(t.peak_vram_mb for t in trials),
                        kv_cache_mb=statistics.median([t.kv_cache_mb for t in trials]),
                        seed=cfg.generation.seed,
                        extra={
                            "status": "ok", "experiment": "paging_latency", "policy": policy,
                            "block_size": block_size,
                            "gather_mb_per_step": gather_mb,
                            "paging_overhead_pct": (
                                (tpot / baseline_tpot - 1) * 100 if baseline_tpot else None
                            ),
                            "gpu_name": gpu_name,
                            **({"block_stats": cache.stats(batch_size)} if policy == "paged" else {}),
                        },
                    )
                )
                rows.append(tpot)
            engine.cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=["capacity", "latency", "both"], default="capacity")
    p.add_argument("--config", default="configs/phase3_paged.yaml")
    p.add_argument("--workloads", nargs="+", default=["mixed", "long_generation", "short_interactive"],
                   choices=sorted(WORKLOADS))
    p.add_argument("--block-sizes", type=int, nargs="+", default=DEFAULT_BLOCK_SIZES)
    p.add_argument("--budget-gb", type=float, default=8.0)
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--max-model-len", type=int, default=65536,
                   help="tokens a contiguous cache must reserve per slot")
    p.add_argument("--context-lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--latency-block-sizes", type=int, nargs="+", default=[16, 128])
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results/raw")
    args = p.parse_args()

    if args.experiment in ("capacity", "both"):
        run_capacity(args.workloads, args.block_sizes, args.budget_gb, args.num_requests,
                     args.seed, args.results_dir, tag="phase3_capacity",
                     max_model_len=args.max_model_len)
    if args.experiment in ("latency", "both"):
        run_latency(args.config, args.context_lengths, args.batch_sizes, args.output_tokens,
                    args.repeats, args.warmup, args.latency_block_sizes, args.results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
