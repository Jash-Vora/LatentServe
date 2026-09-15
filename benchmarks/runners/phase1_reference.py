"""
Phase 1 — Qwen reference benchmark sweep.

Produces the "reproducible reference benchmark for Qwen2.5-1.5B-Instruct"
that docs/methodology.md Phase 1 lists as the deliverable: load time,
prefill latency, decode latency, TTFT, TPOT, E2E latency, throughput,
peak VRAM, and KV-cache memory, at a sweep of context lengths.

Run:

    export PYTHONPATH=$(pwd):$PYTHONPATH
    python -m benchmarks.runners.phase1_reference --config configs/phase1_reference.yaml

    # Override the context-length sweep from the CLI (defaults to the
    # Phase 1 levels in docs/methodology.md: 1 / 16 / 1K / 4K / 8K / 16K):
    python -m benchmarks.runners.phase1_reference \\
        --config configs/phase1_reference.yaml \\
        --context-lengths 1 16 1024 4096 8192 16384 \\
        --output-tokens 256 \\
        --repeats 3

Every row is written through benchmarks/schema.py::ResultWriter — see
docs/methodology.md "Reproducibility": no hand-typed numbers.
"""

from __future__ import annotations

import argparse
import statistics
import sys

from benchmarks.schema import BenchmarkResult, ResultWriter
from config import load_config

DEFAULT_CONTEXT_LENGTHS = [1, 16, 1024, 4096, 8192, 16384]


def run_sweep(
    config_path: str,
    context_lengths: list,
    output_tokens: int,
    repeats: int,
    warmup: int,
    results_dir: str = "results/raw",
) -> list:
    # Local import so this module can be imported (e.g. by tests that only
    # check CLI wiring) on a machine without torch/transformers installed.
    import torch

    from model.qwen import QwenReference

    cfg = load_config(config_path)
    device = f"cuda:{cfg.hardware.devices[0]}" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "[WARN] no CUDA device visible — running on CPU. Fine for a quick "
            "correctness/plumbing check, meaningless for the latency/VRAM "
            "numbers this sweep exists to produce. Run on a T4 for real results.",
            file=sys.stderr,
        )

    ref = QwenReference(
        model_name=cfg.model.name,
        dtype=cfg.model.dtype,
        device=device,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
    ).load()
    print(f"Loaded {cfg.model.name} in {ref.load_ms:.1f} ms on {device}")
    print(f"Shape: {ref.shape}")

    writer = ResultWriter(results_dir=results_dir)
    written = []

    for ctx_len in context_lengths:
        if ctx_len > ref.shape.max_position_embeddings:
            print(f"[SKIP] context_length={ctx_len} exceeds max_position_embeddings, skipping")
            continue

        input_ids = ref.synthesize_input_ids(ctx_len, seed=cfg.generation.seed)

        trials = []
        for trial in range(warmup + repeats):
            result = ref.generate_with_timing(input_ids=input_ids, max_new_tokens=output_tokens)
            if trial >= warmup:
                trials.append(result)
            tag = "warmup" if trial < warmup else "measured"
            print(
                f"  ctx={ctx_len:>6} trial={trial} [{tag}] "
                f"ttft={result.ttft_ms:.1f}ms tpot={result.tpot_ms:.2f}ms "
                f"peak_vram={result.peak_vram_mb:.1f}MB kv={result.kv_cache_mb:.2f}MB"
            )

        ttfts = [t.ttft_ms for t in trials]
        tpots = [t.tpot_ms for t in trials]
        e2es = [t.e2e_latency_ms for t in trials]
        # Pool every measured trial's per-token decode latencies for
        # percentile reporting (Phase 5's p50/p90/p95/p99 requirement,
        # pulled forward here since Phase 1 is where decode timing
        # first exists).
        pooled_decode = [ms for t in trials for ms in t.decode_step_ms]

        def pct(xs: list, p: float):
            if not xs:
                return None
            xs = sorted(xs)
            idx = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
            return xs[idx]

        result = BenchmarkResult(
            system="huggingface_reference",
            tag=cfg.tag,
            attention="mha",  # Phase 1 = unmodified HF reference, no custom attention path yet
            model=cfg.model.name,
            batch_size=1,
            context_length=ctx_len,
            output_length=output_tokens,
            num_gpus=len(cfg.hardware.devices),
            ttft_ms=statistics.median(ttfts),
            tpot_ms=statistics.median(tpots),
            e2e_latency_ms=statistics.median(e2es),
            throughput_tokens_sec=statistics.median([t.throughput_tokens_sec for t in trials]),
            peak_vram_mb=max(t.peak_vram_mb for t in trials),
            kv_cache_mb=statistics.median([t.kv_cache_mb for t in trials]),
            ttft_p50_ms=pct(ttfts, 0.50),
            ttft_p95_ms=pct(ttfts, 0.95),
            ttft_p99_ms=pct(ttfts, 0.99),
            tpot_p50_ms=pct(pooled_decode, 0.50),
            tpot_p95_ms=pct(pooled_decode, 0.95),
            tpot_p99_ms=pct(pooled_decode, 0.99),
            seed=cfg.generation.seed,
        )
        path = writer.write(result)
        written.append(result)
        print(f"  -> wrote {path}")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_reference.yaml")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=DEFAULT_CONTEXT_LENGTHS)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3, help="measured trials per context length")
    parser.add_argument("--warmup", type=int, default=1, help="warm-up trials excluded from results")
    parser.add_argument("--results-dir", default="results/raw")
    args = parser.parse_args()

    run_sweep(
        config_path=args.config,
        context_lengths=args.context_lengths,
        output_tokens=args.output_tokens,
        repeats=args.repeats,
        warmup=args.warmup,
        results_dir=args.results_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
