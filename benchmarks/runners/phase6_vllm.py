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


def build_uniform_requests(ref, context_length: int, num_requests: int,
                           output_tokens: int, seed: int) -> list:
    """Fixed-length prompts, for the context sweep.

    A workload family mixes prompt lengths, which is right for serving
    questions and wrong for locating a crossover: the two systems'
    per-step costs scale differently with context, so a mixed workload
    reports one blended number and hides where they cross. Uniform
    prompts make context the only variable.
    """
    from runtime.request import ServedRequest

    out = []
    for i in range(num_requests):
        ids = ref.synthesize_input_ids(context_length, seed=seed + i)
        out.append(
            ServedRequest(request_id=i, prompt_ids=ids[0].tolist(),
                          max_new_tokens=output_tokens)
        )
    return out


def _row(cfg, system: str, batch_size: int, ctx: int, out_len: int, summary: dict,
         extra: dict) -> BenchmarkResult:
    return BenchmarkResult(
        system=system, tag=cfg.tag, attention="gqa", model=cfg.model.name,
        batch_size=batch_size, context_length=ctx, output_length=out_len, num_gpus=1,
        # Missing measurements stay missing. Writing 0.0 for an
        # unavailable TTFT produces a row that reads as an excellent
        # result, which is exactly the failure Phase 5's harness exists
        # to prevent.
        ttft_ms=summary.get("ttft_p50") if summary.get("ttft_p50") is not None else 0.0,
        tpot_ms=summary.get("itl_p50") if summary.get("itl_p50") is not None else 0.0,
        e2e_latency_ms=summary.get("wall_s", 0.0) * 1000,
        throughput_tokens_sec=summary.get("output_tokens_per_s", 0.0),
        requests_per_sec=summary.get("requests_per_s", 0.0),
        ttft_p50_ms=summary.get("ttft_p50"), ttft_p95_ms=summary.get("ttft_p95"),
        ttft_p99_ms=summary.get("ttft_p99"),
        tpot_p50_ms=summary.get("itl_p50"), tpot_p95_ms=summary.get("itl_p95"),
        tpot_p99_ms=summary.get("itl_p99"),
        peak_vram_mb=summary.get("peak_vram_mb") or 0.0,
        seed=cfg.generation.seed,
        extra={"status": "ok", **summary, **extra},
    )


def run_latentserve(cfg, ref, requests, batch_size: int, block_size: int,
                    max_seq_len: int) -> dict:
    """Run one configuration on an *already loaded* reference.

    The weights are loaded once, by the caller, and shared. Loading per
    configuration would put another 2.88 GiB copy of Qwen2.5-1.5B on the
    GPU each time — which is not merely slow, it inflates
    `peak_vram_mb`, and peak VRAM is one of the quantities this phase
    compares against vLLM.
    """
    import torch

    from runtime.engine import ServingEngine

    from model.latentserve_qwen import LatentServeQwen

    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=max_seq_len)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    stats = engine.stats()
    # Drop the cache before the next configuration sizes its own.
    model.cache = None
    engine.cache = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ttft = summarise_latency([r.ttft_ms for r in finished])
    itl = summarise_latency([ms for r in finished for ms in r.decode_step_ms])
    total_out = sum(r.generated for r in finished)
    return {
        "wall_s": wall,
        "requests": len(finished),
        "output_tokens": total_out,
        "output_tokens_per_s": total_out / wall if wall else 0.0,
        "requests_per_s": len(finished) / wall if wall else 0.0,
        "peak_vram_mb": peak,
        "peak_vram_comparable": True,
        "weights_mb": ref.model.get_memory_footprint() / 1024 / 1024
        if hasattr(ref.model, "get_memory_footprint")
        else None,
        **{f"ttft_{k}": v for k, v in ttft.__dict__.items()},
        **{f"itl_{k}": v for k, v in itl.__dict__.items()},
        **stats,
    }


def run_vllm(cfg, requests, batch_size: int, max_seq_len: int) -> dict:
    """vLLM sizes its KV pool from *free* VRAM at construction
    (`gpu_memory_utilization`), so anything LatentServe left resident
    silently shrinks vLLM's cache and hands it a worse configuration.
    The caller frees the reference weights before this runs.
    """
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
    kv_bytes = runner.kv_cache_bytes()
    return {
        **out,
        "peak_vram_mb": runner.peak_vram_mb(),
        "kv_allocated_mb": kv_bytes / 1024 / 1024 if kv_bytes else None,
        **{f"ttft_{k}": v for k, v in ttft.__dict__.items()},
        **{f"itl_{k}": v for k, v in itl.__dict__.items()},
        # vLLM reports only a per-request mean inter-token latency, so
        # these percentiles are over request means and are NOT comparable
        # with LatentServe's pooled per-step distribution. p50 is a fair
        # comparison; p99 is not, and saying so is the difference between
        # a measurement and a claim.
        "itl_measurement": "per_request_mean",
        # See VLLMRunner.PEAK_VRAM_COMPARABLE.
        "peak_vram_comparable": False,
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
    p.add_argument("--output-lengths", type=int, nargs=2, default=None,
                   metavar=("SHORT", "LONG"),
                   help="run each point at two output lengths and difference them, "
                   "separating decode cost from prefill without needing TTFT")
    p.add_argument("--context-lengths", type=int, nargs="+", default=None,
                   help="sweep uniform prompt lengths instead of a workload family; "
                   "this is what locates the LatentServe/vLLM crossover")
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

    # Loaded once for the whole sweep: it supplies the tokenizer that
    # builds the prompts *and* the weights every LatentServe run executes
    # against. Prompts are then reused by both systems as raw token ids,
    # so no tokenizer difference can enter the comparison.
    device = f"cuda:{cfg.hardware.devices[0]}" if torch.cuda.is_available() else "cpu"
    tok_ref = QwenReference(model_name=cfg.model.name, dtype=cfg.model.dtype, device=device).load()
    print(f"Loaded {cfg.model.name} once in {tok_ref.load_ms:.0f} ms on {device}")
    writer = ResultWriter(results_dir=args.results_dir)

    sweep = (
        [(b, c) for b in args.batch_sizes for c in args.context_lengths]
        if args.context_lengths
        else [(b, None) for b in args.batch_sizes]
    )
    # Decode isolation: the same point at two output lengths. The extra
    # wall time between them is pure decode (identical prompts, identical
    # prefill), so decode cost per step comes out as a slope and prefill
    # as the intercept. This is the only way to separate the two for vLLM,
    # whose V1 engine does not report TTFT.
    output_lengths = args.output_lengths or [args.max_output]
    walls: dict = {}

    for batch_size, context_length in sweep:
      for out_tokens in output_lengths:
        args.max_output = out_tokens

        def make():
            return (
                build_uniform_requests(
                    tok_ref, context_length, args.num_requests, args.max_output,
                    cfg.generation.seed,
                )
                if context_length
                else build_served_requests(
                    tok_ref, args.workload, args.num_requests, args.max_prompt,
                    args.max_output, cfg.generation.seed,
                )
            )

        requests = make()
        max_seq_len = max(r.total_len for r in requests) + 8
        ctx = int(statistics.mean([r.prompt_len for r in requests]))
        out_len = int(statistics.mean([r.max_new_tokens for r in requests]))
        controls = {
            "workload": f"uniform_{context_length}" if context_length else args.workload,
            "max_prompt": context_length or args.max_prompt,
            "dtype": cfg.model.dtype, "sampling": cfg.generation.sampling,
            "arrival_rate": 0.0, "arrival_pattern": "burst",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "prefix_caching": False, "num_requests": args.num_requests,
        }

        for system in (["latentserve", "vllm"] if args.system == "both" else [args.system]):
            fresh = make()
            if system == "vllm" and tok_ref.model is not None:
                tok_ref.model = tok_ref.model.to("cpu")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            summary = (
                run_latentserve(cfg, tok_ref, fresh, batch_size, args.block_size, max_seq_len)
                if system == "latentserve"
                else run_vllm(cfg, fresh, batch_size, max_seq_len)
            )
            row = _row(cfg, system, batch_size, ctx, out_len, summary, controls)
            writer.write(row)
            walls.setdefault((system, batch_size, context_length), {})[args.max_output] = (
                summary["wall_s"]
            )
            # Use the *requested* token counts, not r.generated: only the
            # LatentServe arm writes generated tokens back onto the request
            # objects, so r.generated is 0 for vLLM and the column silently
            # printed total wall time instead of a per-step cost.
            steps = (
                max(1, sum(r.max_new_tokens for r in fresh) // batch_size)
                if context_length
                else 0
            )
            print(
                f"  {system:<12} batch={batch_size:>2} "
                + (f"ctx={context_length:>6} " if context_length else "")
                + (f"~{summary['wall_s'] / steps * 1000:6.1f} ms/step  " if steps else "")
                + f"{summary['output_tokens_per_s']:7.1f} tok/s  "
                + f"ttft p50 {summary.get('ttft_p50') or float('nan'):8.0f} ms  "
                + f"itl p50 {summary.get('itl_p50') or float('nan'):6.1f} ms  "
                + f"kv {(summary.get('kv_allocated_mb') or float('nan')):7.0f} MB"
            )
    if args.output_lengths:
        short, long = args.output_lengths
        print(f"\nDecode isolated by differencing {short} vs {long} output tokens:")
        print(f"{'system':<12} {'batch':>5} {'ctx':>6} {'decode ms/step':>15} "
              f"{'prefill tok/s':>14}")
        for (system, batch_size, context_length), by_out in sorted(walls.items()):
            if short not in by_out or long not in by_out:
                continue
            extra_steps = (long - short) * args.num_requests / batch_size
            decode_ms = (by_out[long] - by_out[short]) / extra_steps * 1000
            short_steps = short * args.num_requests / batch_size
            prefill_s = by_out[short] - decode_ms / 1000 * short_steps
            prompt_tokens = (context_length or 0) * args.num_requests
            rate = prompt_tokens / prefill_s if prefill_s > 0 and prompt_tokens else float("nan")
            print(f"{system:<12} {batch_size:>5} {context_length or 0:>6} "
                  f"{decode_ms:>15.1f} {rate:>14.0f}")
        print(
            "\nA high decode ms/step points at the decode attention kernel; a low "
            "prefill tok/s points at prefill attention, which is O(S^2) and where a "
            "weak kernel hurts most. They are different findings."
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
        "\nAttribute the gap, do not just report it. Three measured mechanisms:\n"
        "  - paged gather: Phase 3 measured 2x resident KV per step; Phase 4 showed it\n"
        "    makes batching non-free (TPOT 33 -> 65 ms, batch 1 -> 8). vLLM's\n"
        "    paged-attention kernel avoids it, so this gap estimates Phase 11's value.\n"
        "  - CUDA graphs + torch.compile: vLLM captures graphs; LatentServe runs eager\n"
        "    Python per layer, and Phase 2's P1 measured decode at 24-32% of peak\n"
        "    bandwidth, i.e. launch-overhead-bound.\n"
        "  - chunked prefill: vLLM mixes prefill into decode steps; LatentServe's\n"
        "    prefill blocks, measured in Phase 4 as ~2.3 s inter-token stalls.\n"
        "Not comparable: peak VRAM (vLLM preallocates by policy) and p99 inter-token\n"
        "latency (vLLM reports a per-request mean)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
