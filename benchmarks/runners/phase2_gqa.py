"""
Phase 2 — GQA + KV-cache benchmark sweep.

docs/methodology.md Phase 2 asks for KV bytes/token, VRAM growth,
memory bandwidth and decode latency, swept over GQA configuration,
batch size and context length, to establish "the baseline for MLA".

Run:

    export PYTHONPATH=$(pwd):$PYTHONPATH
    python -m benchmarks.runners.phase2_gqa --config configs/phase2_gqa.yaml

    # The GQA-configuration experiment (see cache/kv_cache.py on why
    # mha_sim is a controlled comparison rather than a different model):
    python -m benchmarks.runners.phase2_gqa \\
        --config configs/phase2_gqa.yaml \\
        --kv-heads-modes native mha_sim \\
        --context-lengths 1024 4096 8192 \\
        --batch-sizes 1 4

    # Paired against the Phase 1 HF reference in the same process, same
    # weights, same prompts — a fairer comparison than reading two
    # separate JSONL files written on different days:
    python -m benchmarks.runners.phase2_gqa --include-hf-baseline

Every row goes through benchmarks/schema.py::ResultWriter.

## What the `extra` fields are for

A decode step reads two things from DRAM: the model weights (all of
them, every step) and the KV cache (all of it, every step). For
Qwen2.5-1.5B in fp16 that is ~3.1 GB of weights against 28,672 B/token
of KV — so at batch 1, KV traffic does not match weight traffic until
roughly 110K tokens of context.

That ratio is the single most important number Phase 2 produces,
because it bounds what Phase 7 can possibly achieve: MLA shrinks the KV
term only. If a sweep point's KV fraction is 4%, then even a perfect
KV compression can move decode latency by at most ~4%, and reporting
"MLA didn't speed up decode" from such a point would be a statement
about the workload, not about MLA. Recording it here, automatically,
means the MLA phase starts from measurement instead of surprise — and
it tells you where to run those experiments (large batch, long context).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Optional

from benchmarks.schema import BenchmarkResult, ResultWriter
from config import load_config

DEFAULT_CONTEXT_LENGTHS = [1024, 4096, 8192, 16384]
DEFAULT_BATCH_SIZES = [1, 4]

# Vendor peak DRAM bandwidth, used only to report a utilization ratio.
# Labelled theoretical everywhere it appears: the honest measured number
# is achieved GB/s, and Nsight Compute (Phase 12) is what turns that into
# an explanation.
THEORETICAL_PEAK_BW_GB_S = {
    "Tesla T4": 320.0,
    "Tesla V100-SXM2-16GB": 900.0,
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA L4": 300.0,
}


def _oom_errors():
    """torch.cuda.OutOfMemoryError and torch.OutOfMemoryError are the same
    class in recent torch but not in every release this might run on, and
    older versions raise a plain RuntimeError."""
    import torch

    errs = {RuntimeError}
    for name in ("OutOfMemoryError",):
        for mod in (torch, torch.cuda):
            err = getattr(mod, name, None)
            if err is not None:
                errs.add(err)
    return tuple(errs)


def _is_oom(e: Exception) -> bool:
    return isinstance(e, tuple(_oom_errors())) and "out of memory" in str(e).lower()


def _pct(xs: list, p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    idx = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
    return xs[idx]


def _peak_bw(gpu_name: str) -> Optional[float]:
    return THEORETICAL_PEAK_BW_GB_S.get(gpu_name)


def run_sweep(
    config_path: str,
    context_lengths: list,
    batch_sizes: list,
    output_tokens: int,
    repeats: int,
    warmup: int,
    kv_heads_modes: list,
    prefill_chunk_size: Optional[int],
    attn_impl: str,
    kv_expansion: str,
    include_hf_baseline: bool,
    kv_budget_gb: float,
    results_dir: str = "results/raw",
) -> list:
    import torch

    from cache.kv_cache import max_context_for_budget
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    cfg = load_config(config_path)
    device = f"cuda:{cfg.hardware.devices[0]}" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "[WARN] no CUDA device visible — running on CPU. Plumbing check only; "
            "the latency/VRAM/bandwidth numbers this sweep exists to produce are "
            "meaningless without a GPU.",
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
    print(f"GQA group size: {ref.shape.gqa_group_size} query heads per KV head")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    peak_bw = _peak_bw(gpu_name)
    writer = ResultWriter(results_dir=results_dir)
    written = []

    for mode in kv_heads_modes:
        engine = LatentServeQwen.from_reference(
            ref,
            kv_heads_mode=mode,
            attn_impl=attn_impl,
            kv_expansion=kv_expansion,
            max_seq_len_hint=max(context_lengths) + output_tokens,
        )
        weight_bytes = engine.weight_bytes()
        print(f"\n=== kv_heads_mode={mode} | weights={weight_bytes / 1024**3:.2f} GiB ===")

        for batch_size in batch_sizes:
            for ctx_len in context_lengths:
                max_seq_len = ctx_len + output_tokens
                spec = engine.cache_spec(batch_size, max_seq_len)

                if ctx_len > ref.shape.max_position_embeddings:
                    print(f"[SKIP] ctx={ctx_len} exceeds max_position_embeddings")
                    continue
                budget_bytes = int(kv_budget_gb * 1024**3)
                if spec.total_bytes > budget_bytes:
                    feasible = max_context_for_budget(spec, budget_bytes, batch_size)
                    print(
                        f"[SKIP] mode={mode} B={batch_size} ctx={ctx_len}: KV cache would be "
                        f"{spec.total_mb:.0f} MB > budget {kv_budget_gb} GiB "
                        f"(max feasible context at this batch: ~{feasible} tokens)"
                    )
                    continue

                engine.allocate_cache(batch_size, max_seq_len)
                base_ids = ref.synthesize_input_ids(ctx_len, seed=cfg.generation.seed)
                input_ids = base_ids.expand(batch_size, -1).contiguous()

                trials = []
                oom = None
                for trial in range(warmup + repeats):
                    try:
                        result = engine.generate_with_timing(
                            input_ids=input_ids,
                            max_new_tokens=output_tokens,
                            chunk_size=prefill_chunk_size,
                        )
                    except Exception as e:  # noqa: BLE001
                        if not _is_oom(e):
                            raise
                        oom = str(e).split("\n")[0]
                        engine.cache = None
                        torch.cuda.empty_cache()
                        print(f"  mode={mode} B={batch_size} ctx={ctx_len} [OOM] {oom}")
                        break
                    if trial >= warmup:
                        trials.append(result)
                    label = "warmup" if trial < warmup else "measured"
                    print(
                        f"  mode={mode} B={batch_size} ctx={ctx_len:>6} trial={trial} "
                        f"[{label}] ttft={result.ttft_ms:.1f}ms tpot={result.tpot_ms:.2f}ms "
                        f"peak_vram={result.peak_vram_mb:.1f}MB kv={result.kv_cache_mb:.1f}MB"
                    )

                if oom is not None or not trials:
                    row = BenchmarkResult(
                        system="latentserve_gqa",
                        tag=cfg.tag,
                        attention="gqa",
                        model=cfg.model.name,
                        batch_size=batch_size,
                        context_length=ctx_len,
                        output_length=output_tokens,
                        num_gpus=len(cfg.hardware.devices),
                        seed=cfg.generation.seed,
                        extra={
                            "status": "oom" if oom else "no_trials",
                            "error": oom,
                            "kv_heads_mode": mode,
                            "kv_allocated_mb": spec.total_mb,
                            "gpu_name": gpu_name,
                        },
                    )
                    writer.write(row)
                    written.append(row)
                    continue

                ttfts = [t.ttft_ms for t in trials]
                tpots = [t.tpot_ms for t in trials]
                pooled_decode = [ms for t in trials for ms in t.decode_step_ms]
                median_tpot = statistics.median(tpots)

                # Bytes a single decode step must pull from DRAM. KV is
                # measured at the end of generation (context + output),
                # i.e. the worst case within this run, matching the
                # steady-state tpot the same run reports.
                kv_bytes_step = engine.cache.bytes_read_per_decode_step(batch_size)
                total_bytes_step = kv_bytes_step + weight_bytes
                achieved_bw = (
                    total_bytes_step / (median_tpot / 1000) / 1e9 if median_tpot > 0 else 0.0
                )

                extra = {
                    "kv_heads_mode": mode,
                    "kv_heads_effective": spec.num_kv_heads,
                    "gqa_group_size": ref.shape.gqa_group_size,
                    "kv_bytes_per_token": spec.bytes_per_token,
                    "kv_allocated_mb": spec.total_mb,
                    "kv_utilization": engine.cache.utilization(batch_size),
                    "weights_mb": weight_bytes / 1024 / 1024,
                    "decode_kv_bytes_per_step_mb": kv_bytes_step / 1024 / 1024,
                    "decode_total_bytes_per_step_mb": total_bytes_step / 1024 / 1024,
                    "decode_kv_fraction": (
                        kv_bytes_step / total_bytes_step if total_bytes_step else 0.0
                    ),
                    "achieved_bandwidth_gb_s": achieved_bw,
                    "theoretical_peak_bw_gb_s": peak_bw,
                    "bandwidth_utilization": (achieved_bw / peak_bw) if peak_bw else None,
                    "status": "ok",
                    "prefill_chunk_size": prefill_chunk_size,
                    "attn_impl": attn_impl,
                    "kv_expansion": kv_expansion,
                    "prefill_ms": statistics.median([t.prefill_ms for t in trials]),
                    "gpu_name": gpu_name,
                    "num_trials": len(trials),
                }

                row = BenchmarkResult(
                    system="latentserve_gqa",
                    tag=cfg.tag,
                    attention="gqa",
                    model=cfg.model.name,
                    batch_size=batch_size,
                    context_length=ctx_len,
                    output_length=output_tokens,
                    num_gpus=len(cfg.hardware.devices),
                    ttft_ms=statistics.median(ttfts),
                    tpot_ms=median_tpot,
                    e2e_latency_ms=statistics.median([t.e2e_latency_ms for t in trials]),
                    throughput_tokens_sec=statistics.median(
                        [t.throughput_tokens_sec * batch_size for t in trials]
                    ),
                    peak_vram_mb=max(t.peak_vram_mb for t in trials),
                    kv_cache_mb=statistics.median([t.kv_cache_mb for t in trials]),
                    ttft_p50_ms=_pct(ttfts, 0.50),
                    ttft_p95_ms=_pct(ttfts, 0.95),
                    ttft_p99_ms=_pct(ttfts, 0.99),
                    tpot_p50_ms=_pct(pooled_decode, 0.50),
                    tpot_p95_ms=_pct(pooled_decode, 0.95),
                    tpot_p99_ms=_pct(pooled_decode, 0.99),
                    seed=cfg.generation.seed,
                    extra=extra,
                )
                writer.write(row)
                written.append(row)
                print(
                    f"  -> kv={spec.bytes_per_token}B/token, KV is "
                    f"{extra['decode_kv_fraction'] * 100:.1f}% of decode-step bytes, "
                    f"achieved {achieved_bw:.0f} GB/s"
                    + (f" ({extra['bandwidth_utilization'] * 100:.0f}% of peak)" if peak_bw else "")
                )

                if include_hf_baseline and mode == kv_heads_modes[0] and batch_size == 1:
                    # The HF reference runs out of memory long before
                    # LatentServe does (it materialises a full
                    # [B, heads, S, S] score matrix — 12 GiB in fp32 at
                    # 16K, on a 14.6 GiB card). That is a finding about
                    # the baseline and must not abort the sweep that is
                    # measuring it.
                    try:
                        written.append(
                            _run_hf_baseline(
                                ref, cfg, writer, ctx_len, output_tokens, repeats, warmup, gpu_name
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        if not _is_oom(e):
                            raise
                        torch.cuda.empty_cache()
                        row = BenchmarkResult(
                            system="huggingface_reference",
                            tag=cfg.tag,
                            attention="gqa",
                            model=cfg.model.name,
                            batch_size=1,
                            context_length=ctx_len,
                            output_length=output_tokens,
                            num_gpus=len(cfg.hardware.devices),
                            seed=cfg.generation.seed,
                            extra={
                                "status": "oom",
                                "error": str(e).split("\n")[0],
                                "paired_with": "latentserve_gqa",
                                "gpu_name": gpu_name,
                            },
                        )
                        writer.write(row)
                        written.append(row)
                        print(f"  -> [HF paired] ctx={ctx_len} OOM, recorded")

        engine.cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return written


def _run_hf_baseline(
    ref, cfg, writer, ctx_len: int, output_tokens: int, repeats: int, warmup: int, gpu_name: str
) -> BenchmarkResult:
    """Same weights, same prompt, same process — the Phase 1 path rerun
    beside the Phase 2 path so the comparison controls for machine
    state, not just for configuration."""
    import torch

    input_ids = ref.synthesize_input_ids(ctx_len, seed=cfg.generation.seed)
    trials = []
    for trial in range(warmup + repeats):
        result = ref.generate_with_timing(input_ids=input_ids, max_new_tokens=output_tokens)
        if trial >= warmup:
            trials.append(result)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    ttfts = [t.ttft_ms for t in trials]
    pooled = [ms for t in trials for ms in t.decode_step_ms]
    row = BenchmarkResult(
        system="huggingface_reference",
        tag=cfg.tag,
        attention="gqa",  # Qwen2.5 *is* GQA; Phase 1 labelled the HF path "mha" only
        model=cfg.model.name,  # because no custom attention existed then. See docs/phase2.md.
        batch_size=1,
        context_length=ctx_len,
        output_length=output_tokens,
        num_gpus=len(cfg.hardware.devices),
        ttft_ms=statistics.median(ttfts),
        tpot_ms=statistics.median([t.tpot_ms for t in trials]),
        e2e_latency_ms=statistics.median([t.e2e_latency_ms for t in trials]),
        throughput_tokens_sec=statistics.median([t.throughput_tokens_sec for t in trials]),
        peak_vram_mb=max(t.peak_vram_mb for t in trials),
        kv_cache_mb=statistics.median([t.kv_cache_mb for t in trials]),
        ttft_p50_ms=_pct(ttfts, 0.50),
        ttft_p95_ms=_pct(ttfts, 0.95),
        ttft_p99_ms=_pct(ttfts, 0.99),
        tpot_p50_ms=_pct(pooled, 0.50),
        tpot_p95_ms=_pct(pooled, 0.95),
        tpot_p99_ms=_pct(pooled, 0.99),
        seed=cfg.generation.seed,
        extra={"paired_with": "latentserve_gqa", "gpu_name": gpu_name},
    )
    writer.write(row)
    print(f"  -> [HF paired] ctx={ctx_len} ttft={row.ttft_ms:.1f}ms tpot={row.tpot_ms:.2f}ms")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_gqa.yaml")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=DEFAULT_CONTEXT_LENGTHS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--kv-heads-modes",
        nargs="+",
        default=["native"],
        choices=["native", "mha_sim", "mqa_sim"],
        help="GQA configuration experiment; see cache/kv_cache.py",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=None,
        help="split prefill into blocks of this many tokens (memory, not math)",
    )
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "math"])
    parser.add_argument(
        "--kv-expansion",
        default="fold",
        choices=["fold", "materialize"],
        help="how GQA matches KV heads to query heads on the decode path; "
        "'materialize' reproduces the first sweep's 13x decode traffic",
    )
    parser.add_argument(
        "--include-hf-baseline",
        action="store_true",
        help="rerun the Phase 1 HF path on the same points in the same process",
    )
    parser.add_argument(
        "--kv-budget-gb",
        type=float,
        default=8.0,
        help="skip sweep points whose KV cache alone would exceed this",
    )
    parser.add_argument("--results-dir", default="results/raw")
    args = parser.parse_args()

    run_sweep(
        config_path=args.config,
        context_lengths=args.context_lengths,
        batch_sizes=args.batch_sizes,
        output_tokens=args.output_tokens,
        repeats=args.repeats,
        warmup=args.warmup,
        kv_heads_modes=args.kv_heads_modes,
        prefill_chunk_size=args.prefill_chunk_size,
        attn_impl=args.attn_impl,
        kv_expansion=args.kv_expansion,
        include_hf_baseline=args.include_hf_baseline,
        kv_budget_gb=args.kv_budget_gb,
        results_dir=args.results_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
