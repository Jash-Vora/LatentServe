"""
Phase 4 — serving runtime benchmark.

docs/methodology.md Phase 4 and Section 12: continuous batching vs.
static batching, the four scheduler policies, and scheduler overhead
measured rather than assumed.

Run:

    export PYTHONPATH=$(pwd):$PYTHONPATH

    # continuous vs static, the headline comparison
    python -m benchmarks.runners.phase4_serving --config configs/phase4_serving.yaml \\
        --workload mixed --num-requests 32 --max-prompt 8192 \\
        --batch-sizes 1 4 8

    # scheduler policies at fixed concurrency
    python -m benchmarks.runners.phase4_serving --experiment schedulers \\
        --workload mixed --num-requests 32 --max-prompt 8192 --max-running 8

## What changes about the metrics here

TTFT stops meaning prefill latency. From Phase 4 on it is
queue + prefill, and under load the queue term usually dominates. Both
are recorded separately, because they respond to different fixes: queue
time wants capacity or a different admission order, prefill time wants a
faster kernel.

Percentiles stop being optional. A scheduler that improves mean TTFT by
starving its tail is a worse server, and only p95/p99 show that. Every
row here carries p50/p95/p99 for TTFT and TPOT.

## What to expect

Phase 2 measured batching as nearly free at 4K — batch 1 to 8 left TPOT
unchanged (31.0 -> 30.6 ms) for 8x the throughput, because decode is
weight-bound and the weights are read once per step regardless of batch
size. Continuous batching's win over static is therefore *not* about the
per-step cost; it is about keeping the batch full. Static batching holds
a slot until the slowest member of the group finishes, so on a workload
with mixed output lengths its mean occupancy falls well below its
nominal batch size. `mean_batch_occupancy` is the metric that shows it.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Optional

import torch

from benchmarks.schema import BenchmarkResult, ResultWriter
from benchmarks.workloads.ragged import WORKLOADS, generate_requests
from config import load_config
from runtime.engine import ServingEngine, run_static_batching
from runtime.request import ServedRequest
from runtime.scheduler import SCHEDULERS, build_scheduler


# Phase 5 owns the percentile definition now; every runner shares it so
# "p99" means one thing across the whole project.
from benchmarks.harness import percentile as _pct  # noqa: E402


def assign_slo_ms(prompt_len: int) -> float:
    """Tiered TTFT targets by prompt size.

    Without these every request carries the same default, the SLO policy's
    sort key collapses to arrival order, and it silently becomes FIFO —
    which is exactly what the first sweep measured (slo_aware and fifo
    agreed to four decimal places). A policy that cannot be distinguished
    from the baseline has not been tested.
    """
    if prompt_len <= 2048:
        return 2_000.0     # interactive
    if prompt_len <= 8192:
        return 10_000.0    # document-sized
    return 30_000.0        # batch-ish


def build_served_requests(
    ref, workload: str, num_requests: int, max_prompt: int, max_output: int, seed: int,
    arrival_rate: float = 0.0,
) -> list[ServedRequest]:
    """Turn a workload stream into ServedRequests with real token ids.

    `max_prompt` clamps the workload's long tail. The `mixed` family
    reaches 64K prompts, and a 64K prefill alone is ~70 s on a T4 — fine
    as a capacity experiment (Phase 3 simulated it for free), ruinous as
    a serving experiment where the point is to observe many requests
    interacting. Clamping is recorded in `extra` so the trim is visible
    rather than silently baked into the numbers.
    """
    import random

    specs = generate_requests(workload, num_requests, seed=seed)
    rng = random.Random(seed)
    out = []
    offset = 0.0
    for spec in specs:
        prompt_len = min(spec.prompt_tokens, max_prompt)
        ids = ref.synthesize_input_ids(prompt_len, seed=seed + spec.request_id)
        if arrival_rate > 0:
            offset += rng.expovariate(arrival_rate)
        request = ServedRequest(
            request_id=spec.request_id,
            prompt_ids=ids[0].tolist(),
            max_new_tokens=min(spec.output_tokens, max_output),
            slo_ttft_ms=assign_slo_ms(prompt_len),
        )
        request.arrival_offset_s = offset
        out.append(request)
    return out


def summarise(finished: list[ServedRequest], wall_s: float, extra: dict) -> dict:
    ttfts = [r.ttft_ms for r in finished]
    queues = [r.queue_ms for r in finished]
    # Inter-token latency percentiles must be taken over individual decode
    # steps, not over each request's mean. Averaging first hides exactly
    # the effect worth measuring: a 2 s prefill stall spread across a
    # request's 229 steps adds 9 ms to its mean and vanishes, while the
    # user saw one 2 s gap between tokens. Phases 1-3 pooled per-step
    # latencies for this reason; this runner did not, and the first
    # Phase 4 sweep reported a p99/p50 ratio of 1.2 as a result.
    steps = [ms for r in finished for ms in r.decode_step_ms]
    tpots = [r.tpot_ms for r in finished]
    total_out = sum(r.generated for r in finished)
    judged = [r for r in finished if r.met_slo is not None]
    met = [r for r in judged if r.met_slo]
    # Per-tier attainment: a policy that saves the 2 s interactive tier by
    # sacrificing the 30 s batch tier is doing its job, and the aggregate
    # rate alone would hide that.
    tiers: dict = {}
    for r in judged:
        tier = tiers.setdefault(f"slo_{int(r.slo_ttft_ms)}ms", [0, 0])
        tier[1] += 1
        tier[0] += 1 if r.met_slo else 0
    return {
        "requests": len(finished),
        "wall_s": wall_s,
        "requests_per_s": len(finished) / wall_s if wall_s else 0.0,
        "output_tokens_per_s": total_out / wall_s if wall_s else 0.0,
        "ttft_p50": _pct(ttfts, 0.50), "ttft_p95": _pct(ttfts, 0.95), "ttft_p99": _pct(ttfts, 0.99),
        "tpot_p50": _pct(steps, 0.50), "tpot_p95": _pct(steps, 0.95),
        "tpot_p99": _pct(steps, 0.99), "tpot_p999": _pct(steps, 0.999),
        "tpot_max": max(steps) if steps else None,
        # Inter-token latency is bimodal under prefill blocking: a normal
        # population around the decode step, and a stall population at
        # the length of whatever prompt was admitted. With ~1% of gaps
        # stalled, p99 lands exactly on the boundary between the two, so
        # it moves with sampling noise rather than with scheduler policy.
        # Report how often a stall happens and how bad it gets instead.
        "tpot_stall_rate": (
            sum(1 for ms in steps if ms > 10 * (_pct(steps, 0.50) or 1)) / len(steps)
            if steps else None
        ),
        "tpot_stalled_ms_total": sum(
            ms for ms in steps if ms > 10 * (_pct(steps, 0.50) or 1)
        ) if steps else None,
        # Per-request means, kept separately so the two are never confused.
        "tpot_mean_per_request_p50": _pct(tpots, 0.50),
        "tpot_mean_per_request_p95": _pct(tpots, 0.95),
        "queue_p50": _pct(queues, 0.50), "queue_p95": _pct(queues, 0.95),
        "mean_queue_ms": statistics.mean([q for q in queues if q is not None]) if queues else 0.0,
        "slo_attainment": len(met) / len(judged) if judged else None,
        **{f"{k}_attainment": v[0] / v[1] for k, v in sorted(tiers.items())},
        **extra,
    }


def _write(writer, cfg, system: str, summary: dict, batch_size: int, ctx: int) -> BenchmarkResult:
    row = BenchmarkResult(
        system=system, tag=cfg.tag, attention="gqa", model=cfg.model.name,
        batch_size=batch_size, context_length=ctx,
        output_length=int(summary.get("mean_output_tokens", 0)), num_gpus=1,
        ttft_ms=summary["ttft_p50"] or 0.0,
        tpot_ms=summary["tpot_p50"] or 0.0,
        e2e_latency_ms=summary["wall_s"] * 1000,
        throughput_tokens_sec=summary["output_tokens_per_s"],
        requests_per_sec=summary["requests_per_s"],
        ttft_p50_ms=summary["ttft_p50"], ttft_p95_ms=summary["ttft_p95"],
        ttft_p99_ms=summary["ttft_p99"],
        tpot_p50_ms=summary["tpot_p50"], tpot_p95_ms=summary["tpot_p95"],
        tpot_p99_ms=summary["tpot_p99"],
        peak_vram_mb=(
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        ),
        seed=cfg.generation.seed,
        extra={"status": "ok", **summary},
    )
    writer.write(row)
    return row


def drive(engine, requests: list, arrival_rate: float, progress) -> float:
    """Release requests into the engine and step it until everything drains.

    With `arrival_rate == 0` every request is queued at t=0: a burst,
    which maximises queueing pressure. That is a fine stress test and a
    poor starvation test — in a finite burst the last request finishes
    when the total work does, whichever order you serve it in, so no
    policy can starve anyone indefinitely. Open-loop Poisson arrivals are
    what make shortest-job-first's downside visible, and what Phase 6
    needs anyway to compare fairly against vLLM's request-rate harness.
    """
    import time

    t0 = time.perf_counter()
    pending = sorted(requests, key=lambda r: getattr(r, "arrival_offset_s", 0.0))
    engine.on_retire = progress
    i = 0
    while i < len(pending) or engine.has_work:
        now = time.perf_counter() - t0
        while i < len(pending) and getattr(pending[i], "arrival_offset_s", 0.0) <= now:
            pending[i].arrival_time = time.perf_counter()
            engine.add_request(pending[i])
            i += 1
        if engine.has_work:
            engine.step()
        elif i < len(pending):
            # Idle: wait for the next arrival rather than spinning.
            time.sleep(min(0.01, pending[i].arrival_offset_s - now))
    return time.perf_counter() - t0


def make_progress(label: str, total: int, every: int = 1):
    """Print as requests complete, so a long configuration shows life.

    At batch 1 a single configuration is minutes of silence otherwise:
    32 requests x 229 output tokens x ~31 ms is ~4 minutes before the
    summary line appears, which looks exactly like a hang.
    """
    import time

    t0 = time.perf_counter()
    state = {"done": 0}

    def on_retire(request, engine):
        state["done"] += 1
        if state["done"] % every and state["done"] != total:
            return
        elapsed = time.perf_counter() - t0
        rate = state["done"] / elapsed if elapsed else 0.0
        eta = (total - state["done"]) / rate if rate else 0.0
        print(
            f"    [{label}] {state['done']:>3}/{total} done  "
            f"{elapsed:6.0f}s elapsed  ~{eta:5.0f}s left  "
            f"running={len(engine.running)} queued={len(engine.waiting)}",
            flush=True,
        )

    return on_retire


def run(
    config_path: str, experiment: str, workload: str, num_requests: int, max_prompt: int,
    max_output: int, batch_sizes: list, max_running: int, schedulers: list,
    block_size: int, results_dir: str, progress_every: int = 4,
    arrival_rate: float = 0.0,
) -> None:
    import time

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    cfg = load_config(config_path)
    device = f"cuda:{cfg.hardware.devices[0]}" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] no CUDA device — plumbing check only.", file=sys.stderr)

    ref = QwenReference(
        model_name=cfg.model.name, dtype=cfg.model.dtype, device=device,
        revision=cfg.model.revision, trust_remote_code=cfg.model.trust_remote_code,
    ).load()
    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=max_prompt + max_output)
    writer = ResultWriter(results_dir=results_dir)

    requests = build_served_requests(ref, workload, num_requests, max_prompt, max_output,
                                     cfg.generation.seed, arrival_rate=arrival_rate)
    max_seq_len = max(r.total_len for r in requests) + 8
    mean_out = statistics.mean([r.max_new_tokens for r in requests])
    print(
        f"workload={workload} n={len(requests)} "
        f"prompt {min(r.prompt_len for r in requests)}-{max(r.prompt_len for r in requests)} "
        f"(clamped at {max_prompt}), output mean {mean_out:.0f}"
    )
    common = {"workload": workload, "max_prompt": max_prompt,
              "mean_output_tokens": mean_out, "block_size": block_size,
              "arrival_rate": arrival_rate,
              "arrival_pattern": "burst" if arrival_rate <= 0 else "poisson"}

    if experiment in ("batching", "both"):
        for batch_size in batch_sizes:
            for policy in ("static", "continuous"):
                fresh = build_served_requests(ref, workload, num_requests, max_prompt,
                                              max_output, cfg.generation.seed,
                                              arrival_rate=arrival_rate)
                t0 = time.perf_counter()
                progress = make_progress(
                    f"batch={batch_size} {policy}", len(fresh), every=progress_every
                )
                if policy == "static":
                    finished, stats = run_static_batching(
                        model, fresh, batch_size=batch_size, max_seq_len=max_seq_len,
                        block_size=block_size, on_retire=progress,
                    )
                else:
                    engine = ServingEngine(
                        model, max_running=batch_size, max_seq_len=max_seq_len,
                        block_size=block_size, scheduler="fifo", on_retire=progress,
                    )
                    for r in fresh:
                        engine.add_request(r)
                    finished = engine.run()
                    stats = engine.stats()
                wall = time.perf_counter() - t0
                summary = summarise(finished, wall, {**common, "policy": policy, **stats})
                _write(writer, cfg, f"latentserve_{policy}", summary, batch_size,
                       int(statistics.mean([r.prompt_len for r in fresh])))
                print(
                    f"  batch={batch_size:>2} {policy:<10} "
                    f"{summary['output_tokens_per_s']:7.1f} tok/s  "
                    f"ttft p50/p95 {summary['ttft_p50']:8.0f}/{summary['ttft_p95']:8.0f} ms  "
                    f"tpot p50 {summary['tpot_p50']:6.2f} ms  "
                    f"occupancy {summary.get('mean_batch_occupancy', 0):.2f}/{batch_size}"
                )
                model.cache = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if experiment in ("schedulers", "both"):
        print(f"\n=== schedulers at max_running={max_running} ===")
        for name in schedulers:
            fresh = build_served_requests(ref, workload, num_requests, max_prompt, max_output,
                                          cfg.generation.seed, arrival_rate=arrival_rate)
            engine = ServingEngine(
                model, max_running=max_running, max_seq_len=max_seq_len,
                block_size=block_size,
                scheduler=build_scheduler(name, max_running=max_running),
            )
            wall = drive(
                engine, fresh, arrival_rate,
                make_progress(name, len(fresh), every=progress_every),
            )
            finished = engine.finished
            summary = summarise(finished, wall, {**common, "policy": "continuous", **engine.stats()})
            _write(writer, cfg, f"latentserve_sched_{name}", summary, max_running,
                   int(statistics.mean([r.prompt_len for r in fresh])))
            print(
                f"  {name:<14} {summary['output_tokens_per_s']:7.1f} tok/s  "
                f"ttft p50/p99 {summary['ttft_p50']:7.0f}/{summary['ttft_p99']:7.0f} ms  "
                f"itl p50/p99 {summary['tpot_p50']:6.1f}/{summary['tpot_p99']:8.1f} ms  "
                f"SLO {(summary['slo_attainment'] or 0) * 100:5.1f}%  "
                f"sched {summary['scheduler_overhead_us_per_call']:.1f} us/call"
            )
            model.cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase4_serving.yaml")
    p.add_argument("--experiment", choices=["batching", "schedulers", "both"], default="both")
    p.add_argument("--workload", default="mixed", choices=sorted(WORKLOADS))
    p.add_argument("--num-requests", type=int, default=32)
    p.add_argument("--max-prompt", type=int, default=8192,
                   help="clamp the workload's long tail; recorded in results")
    p.add_argument("--max-output", type=int, default=256,
                   help="static batching only loses when output lengths vary, so do not "
                   "clamp the workload's 128/256 mix down to a single value")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--max-running", type=int, default=8)
    p.add_argument("--schedulers", nargs="+", default=sorted(SCHEDULERS), choices=sorted(SCHEDULERS))
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--arrival-rate", type=float, default=0.0,
                   help="requests/second (Poisson). 0 = burst at t=0, which cannot "
                   "show starvation; use ~0.5-1.0 to test the fair/SJF trade")
    p.add_argument("--progress-every", type=int, default=4,
                   help="print a progress line every N completed requests (0 to disable)")
    p.add_argument("--results-dir", default="results/raw")
    args = p.parse_args()

    run(args.config, args.experiment, args.workload, args.num_requests, args.max_prompt,
        args.max_output, args.batch_sizes, args.max_running, args.schedulers,
        args.block_size, args.results_dir, progress_every=max(1, args.progress_every),
        arrival_rate=args.arrival_rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
