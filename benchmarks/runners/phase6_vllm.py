"""
Phase 6 — LatentServe vs vLLM under matched conditions.

    python -m benchmarks.runners.phase6_vllm --config configs/phase6_vllm.yaml \\
        --workload mixed --num-requests 32 --max-prompt 8192 --max-output 256 \\
        --batch-sizes 4 8

Both systems get the same weights, tokenizer, precision, GPU, prompt
*token ids*, output lengths, concurrency limit and sampling. The controls
are asserted by `benchmarks/harness.py::assert_comparable` before any
ratio is printed — an unfair comparison raises rather than producing a
plausible wrong number.

## Running order matters

vLLM and LatentServe cannot share a process: vLLM takes a large,
persistent share of VRAM at construction (`gpu_memory_utilization`), and
LatentServe's cache sizing would then be measuring whatever was left
over. Use `--system latentserve` and `--system vllm` in separate
invocations against the same results file, then compare. `--system both`
exists for CPU plumbing checks only and warns on a GPU.

## What a result means

Phase 3 measured LatentServe's paged gather at 2x resident KV per step;
Phase 4 showed it makes batching non-free (TPOT 33 -> 65 ms from batch 1
to 8). vLLM has the paged-attention kernel that removes that gather. So
the decode-throughput gap at large batch is a *measurement of what the
Phase 11 kernel is worth*, not a verdict. Report it that way.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Optional

from benchmarks.harness import (
    UnfairComparison,
    assert_comparable,
    check_environment,
    summarise_latency,
)
from benchmarks.schema import BenchmarkResult, ResultWriter
from benchmarks.workloads.ragged import WORKLOADS
from config import load_config


def _row(cfg, system: str, batch_size: int, ctx: int, out_len: int, summary: dict,
         extra: dict) -> BenchmarkResult:
    return BenchmarkResult(
        system=system, tag=cfg.tag, attention="gqa", model=cfg.model.name,
        batch_size=batch_size, context_length=ctx, output_length=out_len, num_gpus=1,
        ttft_ms=summary.get("ttft_p50") or 0.0,
        tpot_ms=summary.get("itl_p50") or 0.0,
        e2e_latency_ms=summary.get("wall_s", 0.0) * 1000,
        throughput_tokens_sec=summary.get("output_tokens_per_s", 0.0),
        requests_per_sec=summary.get("requests_per_s", 0.0),
        ttft_p50_ms=summary.get("ttft_p50"), ttft_p95_ms=summary.get("ttft_p95"),
        ttft_p99_ms=summary.get("ttft_p99"),
        tpot_p50_ms=summary.get("itl_p50"), tpot_p95_ms=summary.get("itl_p95"),
        tpot_p99_ms=summary.get("itl_p99"),
        peak_vram_mb=summary.get("peak_vram_mb", 0.0),
        seed=cfg.generation.seed,
        extra={"status": "ok", **summary, **extra},
    )


def run_latentserve(cfg, requests, batch_size: int, block_size: int, max_seq_len: int) -> dict:
    import torch

    from runtime.engine import ServingEngine

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    device = f"cuda:{cfg.hardware.devices[0]}" if torch.cuda.is_available() else "cpu"
    ref = QwenReference(model_name=cfg.model.name, dtype=cfg.model.dtype, device=device).load()
    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=max_seq_len)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    engine = ServingEngine(
        model, max_running=batch_size, max_seq_len=max_seq_len,
        block_size=block_size, scheduler="fifo",
    )
    for r in requests:
        engine.add_request(r)
    t0 = time.perf_counter()
    finished = engine.run()
    wall = time.perf_counter() - t0

    ttft = summarise_latency([r.ttft_ms for r in finished])
    itl = summarise_latency([ms for r in finished for ms in r.decode_step_ms])
    total_out = sum(r.generated for r in finished)
    return {
        "wall_s": wall,
        "requests": len(finished),
        "output_tokens": total_out,
        "output_tokens_per_s": total_out / wall if wall else 0.0,
        "requests_per_s": len(finished) / wall if wall else 0.0,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        ),
        **{f"ttft_{k}": v for k, v in ttft.__dict__.items()},
        **{f"itl_{k}": v for k, v in itl.__dict__.items()},
        **engine.stats(),
    }


def run_vllm(cfg, requests, batch_size: int, max_seq_len: int) -> dict:
    from comparisons.vllm.runner import VLLMRunner

    runner = VLLMRunner(
        model_name=cfg.model.name,
        dtype="float16" if cfg.model.dtype == "fp16" else cfg.model.dtype,
        max_model_len=max_seq_len,
        max_num_seqs=batch_size,
        enable_prefix_caching=False,
        seed=cfg.generation.seed,
    )
    prompts = [r.prompt_ids for r in requests]
    counts = [r.max_new_tokens for r in requests]

    # One warm-up pass, excluded — methodology Section 33. vLLM's first
    # call pays CUDA graph capture and allocator warm-up that no steady
    # state includes.
    runner.generate(prompts[: min(2, len(prompts))], 8, warmup=True)

    out = runner.generate(prompts, counts)
    ttft = summarise_latency(out.pop("ttft_ms"))
    itl = summarise_latency(out.pop("mean_itl_ms"))
    out.pop("e2e_ms", None)
    return {
        **out,
        "peak_vram_mb": runner.peak_vram_mb() or 0.0,
        **{f"ttft_{k}": v for k, v in ttft.__dict__.items()},
        **{f"itl_{k}": v for k, v in itl.__dict__.items()},
        # vLLM reports only a per-request mean inter-token latency, so
        # these percentiles are over request means and are NOT comparable
        # with LatentServe's pooled per-step distribution. p50 is a fair
        # comparison; p99 is not, and saying so is the difference between
        # a measurement and a claim.
        "itl_measurement": "per_request_mean",
        **runner.describe(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--system", choices=["latentserve", "vllm", "both"], default="latentserve")
    p.add_argument("--workload", default="mixed", choices=sorted(WORKLOADS))
    p.add_argument("--num-requests", type=int, default=32)
    p.add_argument("--max-prompt", type=int, default=8192)
    p.add_argument("--max-output", type=int, default=256)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--compare", action="store_true",
                   help="do not run anything; compare existing rows in the results file")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.compare:
        return compare(cfg, args.results_dir)

    import torch

    from benchmarks.runners.phase4_serving import build_served_requests
    from model.qwen import QwenReference

    if args.system == "both" and torch.cuda.is_available():
        print(
            "[WARN] --system both runs vLLM and LatentServe in one process. vLLM holds a "
            "large persistent share of VRAM, so LatentServe's cache would be sized against "
            "the leftovers. Run them as separate invocations for real numbers.",
            file=sys.stderr,
        )

    # Prompts are built once, from the same tokenizer, and reused by both
    # systems as raw token ids.
    device = f"cuda:{cfg.hardware.devices[0]}" if torch.cuda.is_available() else "cpu"
    tok_ref = QwenReference(model_name=cfg.model.name, dtype=cfg.model.dtype, device=device).load()
    writer = ResultWriter(results_dir=args.results_dir)

    for batch_size in args.batch_sizes:
        requests = build_served_requests(
            tok_ref, args.workload, args.num_requests, args.max_prompt,
            args.max_output, cfg.generation.seed,
        )
        max_seq_len = max(r.total_len for r in requests) + 8
        ctx = int(statistics.mean([r.prompt_len for r in requests]))
        out_len = int(statistics.mean([r.max_new_tokens for r in requests]))
        controls = {
            "workload": args.workload, "max_prompt": args.max_prompt,
            "dtype": cfg.model.dtype, "sampling": cfg.generation.sampling,
            "arrival_rate": 0.0, "arrival_pattern": "burst",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "prefix_caching": False, "num_requests": args.num_requests,
        }

        for system in (["latentserve", "vllm"] if args.system == "both" else [args.system]):
            fresh = build_served_requests(
                tok_ref, args.workload, args.num_requests, args.max_prompt,
                args.max_output, cfg.generation.seed,
            )
            summary = (
                run_latentserve(cfg, fresh, batch_size, args.block_size, max_seq_len)
                if system == "latentserve"
                else run_vllm(cfg, fresh, batch_size, max_seq_len)
            )
            row = _row(cfg, system, batch_size, ctx, out_len, summary, controls)
            writer.write(row)
            print(
                f"  {system:<12} batch={batch_size:>2}  "
                f"{summary['output_tokens_per_s']:7.1f} tok/s  "
                f"ttft p50 {summary.get('ttft_p50') or 0:8.0f} ms  "
                f"itl p50 {summary.get('itl_p50') or 0:6.1f} ms  "
                f"peak {summary.get('peak_vram_mb', 0):7.0f} MB"
            )
    return 0


def compare(cfg, results_dir: str) -> int:
    """Pair up rows and report ratios, refusing unmatched comparisons."""
    import json
    from pathlib import Path

    path = Path(results_dir) / f"{cfg.tag}.jsonl"
    rows = [json.loads(line) for line in path.open()]
    ours = {r["batch_size"]: r for r in rows if r["system"] == "latentserve"}
    theirs = {r["batch_size"]: r for r in rows if r["system"] == "vllm"}

    drift = check_environment(rows)
    if drift:
        print(f"[WARN] reproducibility drift across rows: {sorted(drift)}", file=sys.stderr)

    if not theirs:
        print("No vLLM rows yet — run `--system vllm` in a separate session.")
        return 0

    print(f"{'batch':>5}  {'LatentServe':>12}  {'vLLM':>12}  {'ratio':>7}  {'ttft p50 ratio':>15}")
    for batch_size in sorted(set(ours) & set(theirs)):
        a, b = ours[batch_size], theirs[batch_size]
        try:
            assert_comparable(a, b)
        except UnfairComparison as e:
            print(f"{batch_size:>5}  SKIPPED — {e}")
            continue
        ta = a["throughput_tokens_sec"]
        tb = b["throughput_tokens_sec"]
        ra = (a.get("ttft_p50_ms") or 0) / (b.get("ttft_p50_ms") or 1)
        print(f"{batch_size:>5}  {ta:12.1f}  {tb:12.1f}  {ta / tb:6.2f}x  {ra:14.2f}x")
    print(
        "\nA decode-throughput gap at large batch is largely LatentServe's paged gather "
        "(Phase 3: 2x resident KV per step), which vLLM's paged-attention kernel avoids. "
        "That makes this gap an estimate of what the Phase 11 kernel is worth."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
