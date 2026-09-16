"""
Phase 7.3 — INT8 KV cache in the real paged decode path.

docs/phase7.md: 7.1 (spectra) said there was room, 7.2 (offline
simulation) said INT8 K:per-channel / V:per-token barely moves the
model's output, 7.2b (online-scaling check) said that number survives
being fit block-local instead of globally. This is what turns that
measurement into a running cache: `cache/int8_paged_cache.py`'s
`Int8PagedKVCache`, wired into `model/latentserve_qwen.py` via
`allocate_cache(paged=True, kv_dtype="int8")`.

Unlike 7.1/7.2/7.2b this needs no simulation — the real cache exists and
sits behind the same read/write contract as `PagedKVCache`, so it can be
benchmarked exactly like Phase 3 benchmarked paged vs. contiguous.

Three questions, three experiments:

  **A — quality (any machine with the model; GPU recommended).**
  Same prompt, same incremental decode, FP16 paged vs. INT8 paged.
  Scored by logit KL divergence and top-1 flip rate (`DivergenceMeter`,
  the same metric 7.2/7.2b used), so the offline simulation's numbers
  and the real cache's numbers are directly comparable.

  **B — storage (CPU, no model, seconds).**
  Bytes/token, the thing this phase is *for*. No forward pass needed —
  the cache's own accounting settles it, same as `int8.bytes_per_token`
  in `tests/test_phase7_int8_cache.py`.

  **C — latency (GPU, real model).**
  What the residual-buffer bookkeeping and the dequantize-on-read costs
  in TPOT, against FP16 paged at the same block size. `int8_paged_cache.py`
  itself predicts this: storage halves, but traffic only drops ~17%
  (0.5 read + 1 dequant-write + 1 SDPA-read = 2.5, vs. FP16 paged's
  1 + 1 + 1 = 3) because attention still consumes a dequantized FP16
  buffer, not INT8 directly — the full 2x needs a Phase 11 kernel. A
  latency win here would be a bigger surprise than a small regression.

Run:

    export PYTHONPATH=$(pwd):$PYTHONPATH

    # A — quality, needs the real model (GPU recommended, CPU works but slow)
    python -m benchmarks.runners.phase7_int8_cache --experiment quality \\
        --context-length 4096 --decode-steps 64

    # B — storage, no GPU, no download
    python -m benchmarks.runners.phase7_int8_cache --experiment storage

    # C — latency, on the T4
    python -m benchmarks.runners.phase7_int8_cache --experiment latency \\
        --context-lengths 4096 8192 16384 --batch-sizes 1 4

    # everything
    python -m benchmarks.runners.phase7_int8_cache --experiment all
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
from typing import Optional

import torch

from benchmarks.schema import BenchmarkResult, ResultWriter
from cache.kv_cache import KVCacheSpec
from cache.paged_cache import PagedKVCache
from cache.int8_paged_cache import Int8PagedKVCache
from compression.truncation import DivergenceMeter
from config import load_config

DEFAULT_BLOCK_SIZE = 16
# Qwen2.5-1.5B-Instruct, native GQA — 2 kv heads x 128 dims, 28 layers.
# Same constant Phase 3's runner uses for the fp16 byte count.
KV_BYTES_PER_TOKEN_FP16 = 28_672


# ----------------------------------------------------------------------
# Experiment A — quality: FP16 paged vs. INT8 paged, same decode
# ----------------------------------------------------------------------


@torch.no_grad()
def run_quality(
    config_path: str, context_length: int, decode_steps: int, block_sizes: list,
    seed: int, results_dir: str, source: str = "default",
) -> list:
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    cfg = load_config(config_path)
    device = f"cuda:{cfg.hardware.devices[0]}" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] no CUDA device — this will be slow but is correct.", file=sys.stderr)

    ref = QwenReference(
        model_name=cfg.model.name, dtype=cfg.model.dtype, device=device,
        revision=cfg.model.revision, trust_remote_code=cfg.model.trust_remote_code,
    ).load()

    # One fixed prompt + a fixed, model-independent stream of "arriving"
    # tokens to decode against, so FP16 and INT8 walk through the exact
    # same positions and the only thing that differs is the cache.
    torch.manual_seed(seed)
    # Prompt and decode stream come from one source. Real text for the
    # prompt but random ids for the decode stream would put the model back
    # into the flat-distribution regime for exactly the positions scored.
    from benchmarks.runners.phase7_spectra import build_calibration_ids

    full = build_calibration_ids(ref, context_length + decode_steps, source, seed).to(device)
    prompt_ids = full[:, :context_length]
    decode_ids = full[:, context_length : context_length + decode_steps]

    writer = ResultWriter(results_dir=results_dir)
    rows = []
    print(f"\n=== quality | context={context_length} decode_steps={decode_steps} ===")

    for block_size in block_sizes:
        fp16 = LatentServeQwen.from_reference(ref, max_seq_len_hint=context_length + decode_steps)
        fp16.allocate_cache(1, context_length + decode_steps, paged=True, block_size=block_size)
        fp16.cache.reset()
        fp16.prefill(prompt_ids, chunk_size=min(4096, context_length))
        fp16_logits = [
            fp16.decode_step(decode_ids[:, t : t + 1]) for t in range(decode_steps)
        ]
        fp16.cache = None

        int8 = LatentServeQwen.from_reference(ref, max_seq_len_hint=context_length + decode_steps)
        int8.allocate_cache(
            1, context_length + decode_steps, paged=True, block_size=block_size, kv_dtype="int8",
        )
        int8.cache.reset()
        int8.prefill(prompt_ids, chunk_size=min(4096, context_length))
        int8_logits = [
            int8.decode_step(decode_ids[:, t : t + 1]) for t in range(decode_steps)
        ]
        int8.cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        meter = DivergenceMeter()
        for base, mod in zip(fp16_logits, int8_logits):
            meter.update(base, mod)
        result = meter.result()

        # How confident the *baseline* is at these positions. Top-1 flip
        # rate cannot be interpreted without it: a model choosing between
        # two near-tied candidates flips under any perturbation, so a high
        # flip rate against a low top-1 probability says more about the
        # prompt than about the cache.
        top2 = [torch.softmax(l[:, -1].float(), -1).topk(2, -1).values for l in fp16_logits]
        result["baseline_top1_prob"] = float(torch.stack([t[:, 0] for t in top2]).mean())
        result["baseline_top1_margin"] = float(
            torch.stack([t[:, 0] - t[:, 1] for t in top2]).mean()
        )
        result["source"] = "random" if source == "random" else "text"

        print(
            f"  ctx={context_length:>6} block={block_size:>4}  "
            f"kl_mean={result['kl_mean_nats']:.5f}  "
            f"top1_flip={result['top1_flip_rate']:.3%}  "
            f"baseline top1_p={result['baseline_top1_prob']:.3f} "
            f"margin={result['baseline_top1_margin']:.3f}"
        )
        writer.write(
            BenchmarkResult(
                system="latentserve_int8_paged", tag="phase7_int8_quality", attention="gqa",
                model=cfg.model.name, batch_size=1, context_length=context_length,
                output_length=decode_steps, num_gpus=1 if torch.cuda.is_available() else 0,
                seed=seed,
                extra={
                    "status": "ok", "experiment": "int8_quality", "block_size": block_size,
                    "decode_steps": decode_steps,
                    **result,
                },
            )
        )
        rows.append(result)
    return rows


# ----------------------------------------------------------------------
# Experiment D — capacity: what the memory saving actually buys
# ----------------------------------------------------------------------


def _decode_fits(model, batch: int, ctx: int, block_size: int, kv_dtype: str,
                 asymmetric: bool, k_bits: int, v_bits: int) -> bool:
    """Can this runtime hold `batch` sequences of `ctx` tokens and take a
    decode step?

    The cache is advanced to `ctx` without a real prefill. That is
    deliberate: prefill peak is an *activation* question, already bounded
    by chunking (Phase 2), and running it would make this sweep hours
    instead of minutes. What is being measured is the steady-state decode
    ceiling — the pool, the gather buffer and attention — which is what
    decides how many sequences a server can hold.
    """
    import torch

    cache = None
    try:
        cache = model.allocate_cache(
            batch, ctx + 8, paged=True, block_size=block_size, kv_dtype=kv_dtype,
            k_bits=k_bits, v_bits=v_bits,
            **({"asymmetric": asymmetric} if kv_dtype == "int8" else {}),
        )
        cache.reset()
        cache.advance(ctx, batch_size=batch)
        ids = torch.zeros(batch, 1, dtype=torch.long, device=model.device)
        positions = torch.full((batch, 1), ctx - 1, dtype=torch.long, device=model.device)
        model.decode_step_ragged(ids, positions, list(range(batch)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        return False
    finally:
        model.cache = None
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _largest_that_fits(probe, lo: int, hi: int) -> int:
    """Doubling search then bisection. `probe(n) -> bool`, monotone."""
    if not probe(lo):
        return 0
    best = lo
    n = lo
    while n * 2 <= hi and probe(n * 2):
        best = n = n * 2
    low, high = best, min(best * 2, hi)
    while low + 1 < high:
        mid = (low + high) // 2
        if probe(mid):
            low = mid
        else:
            high = mid
    return low


def run_capacity(
    config_path: str, contexts: list, batch_sizes: list, block_sizes: list,
    k_bits: int, v_bits: int, max_batch: int, max_context: int,
    results_dir: str,
) -> list:
    """FP16 paged vs INT8 paged: how much more actually fits.

    Two directions, because a server runs out of room in both:
      * fixed context, raise concurrency until it will not fit;
      * fixed concurrency, raise context until it will not fit.

    Reported against the analytic bytes/token ratio, which is the
    prediction. A measured ratio well below it means something other
    than the KV cache is binding — weights, activations, the gather
    buffer — and that is worth knowing before claiming the saving.
    """
    import torch

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    cfg = load_config(config_path)
    device = f"cuda:{cfg.hardware.devices[0]}" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("[WARN] capacity is a VRAM question; on CPU this measures nothing.",
              file=sys.stderr)

    ref = QwenReference(
        model_name=cfg.model.name, dtype=cfg.model.dtype, device=device,
        revision=cfg.model.revision, trust_remote_code=cfg.model.trust_remote_code,
    ).load()
    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=max_context)
    writer = ResultWriter(results_dir=results_dir)
    rows = []

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"\n=== capacity | {free / 1024**3:.2f} GiB free of "
              f"{total / 1024**3:.2f} GiB after weights ===")

    variants = [("fp16", False)] + [("int8", a) for a in (False, True)]

    for block_size in block_sizes:
        for ctx in contexts:
            print(f"\n-- max concurrency at ctx={ctx}, block={block_size} --")
            baseline = None
            for kv_dtype, asym in variants:
                label = kv_dtype + (" asym" if asym else "")
                n = _largest_that_fits(
                    lambda b: _decode_fits(model, b, ctx, block_size, kv_dtype, asym,
                                           k_bits, v_bits),
                    lo=1, hi=max_batch,
                )
                if baseline is None:
                    baseline = n
                print(f"   {label:<10} max batch {n:>4}"
                      + (f"   ({n / baseline:.2f}x fp16)" if baseline else ""))
                row = {"experiment": "capacity_concurrency", "kv_dtype": kv_dtype,
                       "asymmetric": asym, "block_size": block_size, "context_length": ctx,
                       "max_batch": n, "vs_fp16": n / baseline if baseline else None,
                       "k_bits": k_bits, "v_bits": v_bits, "status": "ok"}
                rows.append(row)
                writer.write(BenchmarkResult(
                    system=f"latentserve_{kv_dtype}_paged", tag="phase7_capacity",
                    attention="gqa", model=cfg.model.name, batch_size=n,
                    context_length=ctx, output_length=0, num_gpus=1, seed=0, extra=row,
                ))

        for batch in batch_sizes:
            print(f"\n-- max context at batch={batch}, block={block_size} --")
            baseline = None
            for kv_dtype, asym in variants:
                label = kv_dtype + (" asym" if asym else "")
                n = _largest_that_fits(
                    lambda c: _decode_fits(model, batch, c, block_size, kv_dtype, asym,
                                           k_bits, v_bits),
                    lo=512, hi=max_context,
                )
                if baseline is None:
                    baseline = n
                print(f"   {label:<10} max context {n:>7}"
                      + (f"   ({n / baseline:.2f}x fp16)" if baseline else ""))
                row = {"experiment": "capacity_context", "kv_dtype": kv_dtype,
                       "asymmetric": asym, "block_size": block_size, "batch_size": batch,
                       "max_context": n, "vs_fp16": n / baseline if baseline else None,
                       "k_bits": k_bits, "v_bits": v_bits, "status": "ok"}
                rows.append(row)
                writer.write(BenchmarkResult(
                    system=f"latentserve_{kv_dtype}_paged", tag="phase7_capacity",
                    attention="gqa", model=cfg.model.name, batch_size=batch,
                    context_length=n, output_length=0, num_gpus=1, seed=0, extra=row,
                ))
    return rows


# ----------------------------------------------------------------------
# Experiment B — storage: bytes/token, no model, no GPU
# ----------------------------------------------------------------------


def run_storage(
    block_sizes: list, k_bits: int, v_bits: int, results_dir: str, tag: str,
) -> list:
    # Qwen2.5-1.5B's real shape — the regime this phase's claim is made
    # for (see test_int8_cache_uses_roughly_half_the_bytes_of_fp16_paged's
    # docstring on why a toy head_dim distorts the ratio).
    spec = KVCacheSpec(
        num_layers=28, num_kv_heads=2, head_dim=128,
        max_batch_size=1, max_seq_len=256, dtype=torch.float16, device="cpu",
    )
    writer = ResultWriter(results_dir=results_dir)
    rows = []
    print("\n=== storage | Qwen2.5-1.5B shape (28 layers, 2 kv heads, head_dim=128) ===")
    fp16_bytes = KV_BYTES_PER_TOKEN_FP16
    for block_size in block_sizes:
        int8 = Int8PagedKVCache(spec, block_size=block_size, k_bits=k_bits, v_bits=v_bits)
        int8_bytes = int8.bytes_per_token
        ratio = int8_bytes / fp16_bytes
        print(
            f"  block_size={block_size:>4}  int8={int8_bytes:>7,d} B/token  "
            f"fp16={fp16_bytes:>7,d} B/token  ratio={ratio:.3f}  "
            f"({1 / ratio:.2f}x capacity at fixed budget)"
        )
        row = {
            "block_size": block_size, "k_bits": k_bits, "v_bits": v_bits,
            "int8_bytes_per_token": int8_bytes, "fp16_bytes_per_token": fp16_bytes,
            "ratio": ratio, "capacity_multiplier": 1 / ratio,
        }
        writer.write(
            BenchmarkResult(
                system="latentserve_int8_paged", tag=tag, attention="gqa",
                model="Qwen/Qwen2.5-1.5B-Instruct", batch_size=1, context_length=0,
                output_length=0, num_gpus=0,
                extra={"status": "ok", "experiment": "int8_storage", **row},
            )
        )
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# Experiment C — latency: TPOT, FP16 paged vs. INT8 paged, same block size
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

    print(f"\n=== latency | block_sizes={block_sizes} output_tokens={output_tokens} ===")
    for batch_size in batch_sizes:
        for ctx_len in context_lengths:
            max_seq_len = ctx_len + output_tokens
            base_ids = ref.synthesize_input_ids(ctx_len, seed=cfg.generation.seed)
            input_ids = base_ids.expand(batch_size, -1).contiguous()
            baseline_tpot = None

            for block_size in block_sizes:
                for kv_dtype in ("fp16", "int8"):
                    engine.allocate_cache(
                        batch_size, max_seq_len, paged=True, block_size=block_size,
                        kv_dtype=kv_dtype,
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
                        print(f"  [OOM] {kv_dtype} bs={block_size} B={batch_size} ctx={ctx_len}")
                        writer.write(
                            BenchmarkResult(
                                system=f"latentserve_paged_{kv_dtype}", tag="phase7_int8_latency",
                                attention="gqa", model=cfg.model.name, batch_size=batch_size,
                                context_length=ctx_len, output_length=output_tokens,
                                num_gpus=1, seed=cfg.generation.seed,
                                extra={"status": "oom", "error": str(e).split("\n")[0]},
                            )
                        )
                        continue

                    tpot = statistics.median([t.tpot_ms for t in trials])
                    if kv_dtype == "fp16":
                        baseline_tpot = tpot
                    cache = engine.cache
                    gather_mb = cache.gather_bytes_per_decode_step(batch_size) / 1024 / 1024
                    label = f"{kv_dtype}(bs={block_size})"
                    print(
                        f"  B={batch_size} ctx={ctx_len:>6} {label:<16} "
                        f"ttft={statistics.median([t.ttft_ms for t in trials]):8.1f}ms "
                        f"tpot={tpot:6.2f}ms "
                        + (f"({tpot / baseline_tpot - 1:+.1%} vs fp16 paged) "
                           if baseline_tpot and kv_dtype == "int8" else "")
                        + f"gather={gather_mb:.0f}MB/step "
                        f"kv_used={cache.used_bytes(batch_size) / 1024 / 1024:.0f}MB"
                    )
                    writer.write(
                        BenchmarkResult(
                            system=f"latentserve_paged_{kv_dtype}", tag="phase7_int8_latency",
                            attention="gqa",
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
                                "status": "ok", "experiment": "int8_latency", "kv_dtype": kv_dtype,
                                "block_size": block_size,
                                "gather_mb_per_step": gather_mb,
                                "int8_overhead_pct": (
                                    (tpot / baseline_tpot - 1) * 100
                                    if baseline_tpot and kv_dtype == "int8" else None
                                ),
                                "gpu_name": gpu_name,
                                "block_stats": cache.stats(batch_size),
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
    p.add_argument("--experiment",
                   choices=["quality", "storage", "latency", "capacity", "all"],
                   default="storage")
    p.add_argument("--asymmetric", action="store_true",
                   help="affine quantization: fits [min, max] per group instead of assuming "
                   "symmetry about zero. Measured 6.5x better in KL at 8 bits and "
                   "categorically better below (symmetric int6 reaches max-KL 8.2 nats). "
                   "Costs a zero-point per scale, which bytes_per_token counts.")
    p.add_argument("--max-batch", type=int, default=256)
    p.add_argument("--max-context", type=int, default=131072)
    p.add_argument("--config", default="configs/phase2_gqa.yaml")
    p.add_argument("--block-sizes", type=int, nargs="+", default=[DEFAULT_BLOCK_SIZE])
    p.add_argument("--k-bits", type=int, default=8)
    p.add_argument("--v-bits", type=int, default=8)
    # Experiment A
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--decode-steps", type=int, default=64)
    p.add_argument("--quality-contexts", type=int, nargs="+", default=None,
                   help="sweep the prompt length for experiment A. Quantization damage should "
                   "grow with the number of cached keys competing in the softmax, and the "
                   "offline simulation averaged that away by scoring every position including "
                   "ones with almost no cache behind them.")
    p.add_argument("--text-file", default=None,
                   help="real text for the prompt and decode stream")
    p.add_argument("--random-tokens", action="store_true",
                   help="out-of-distribution ids instead (the original behaviour). Top-1 flip "
                   "rate is a threshold metric, so a near-flat distribution inflates it: when "
                   "the top two candidates sit a hair apart, any perturbation flips them.")
    # Experiment C
    p.add_argument("--context-lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results/raw")
    args = p.parse_args()

    if args.experiment in ("quality", "all"):
        source = "random" if args.random_tokens else (
            Path(args.text_file).read_text() if args.text_file else "default"
        )
        for ctx in (args.quality_contexts or [args.context_length]):
            run_quality(args.config, ctx, args.decode_steps, args.block_sizes,
                        args.seed, args.results_dir, source=source)
    if args.experiment in ("capacity", "all"):
        run_capacity(args.config, args.context_lengths, args.batch_sizes, args.block_sizes,
                     args.k_bits, args.v_bits, args.max_batch, args.max_context,
                     args.results_dir)
    if args.experiment in ("storage", "all"):
        run_storage(args.block_sizes, args.k_bits, args.v_bits, args.results_dir,
                    tag="phase7_int8_storage")
    if args.experiment in ("latency", "all"):
        run_latency(args.config, args.context_lengths, args.batch_sizes, args.output_tokens,
                    args.repeats, args.warmup, args.block_sizes, args.results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())