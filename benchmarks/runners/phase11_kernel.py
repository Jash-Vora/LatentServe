"""
Phase 11 — what the paged-decode kernel is worth.

    export PYTHONPATH=$(pwd):$PYTHONPATH
    python -m benchmarks.runners.phase11_kernel \
        --context-lengths 4096 8192 16384 --batch-sizes 1 4 8

Compares, at each point, the decode step of:

    contiguous + SDPA     Phase 2's baseline. No gather, but cannot page.
    paged + SDPA          Phase 3's path. The gather costs 2x resident KV.
    paged + kernel        this phase. Reads the cache in place.

## The pre-registered prediction

At 8K / batch 4, Phase 3 measured the gather at 17.3 ms and Phase 4
measured paged+SDPA at 49.2 ms/step. The weight read is ~28 ms
(3.09 GB at the ~110 GB/s this card sustains) and is irreducible here.
So the kernel should land near **33 ms**, against 49.2 today.

If it lands near 50, the four measurements pointing at the gather were
incomplete and something else binds. If it beats 33, something beyond
the known costs improved — most likely that the fp16 staging buffer was
also costing cache pressure the traffic model does not capture.

## Why TPOT and not end-to-end

Phase 6 established that end-to-end at 8K is 46% prefill, and prefill
does not use this kernel at all (see kernels/gqa/paged_decode.py on why
decode-only). Measuring end-to-end would dilute the effect by roughly
half and report a smaller win than the change actually delivers.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

from benchmarks.schema import BenchmarkResult, ResultWriter
from config import load_config

WEIGHT_BYTES = 3.09e9
CARD_BW = 110e9


def time_decode(model, batch: int, ctx: int, steps: int, warmup: int,
                paged: bool, impl: str, block_size: int, kv_dtype: str) -> dict:
    """Median decode-step latency at a fixed cache occupancy.

    The cache is advanced to `ctx` without a real prefill, as in Phase
    7.5's capacity search: what is being timed is the steady-state step,
    and a real prefill would add minutes per point for a number this
    does not use.
    """
    import time

    model.attn_impl = impl
    for layer in model.layers:
        layer.attn.attn_impl = impl

    cache = model.allocate_cache(
        batch, ctx + steps + 8, paged=paged, block_size=block_size, kv_dtype=kv_dtype
    )
    cache.reset()
    cache.advance(ctx, batch_size=batch)

    ids = torch.zeros(batch, 1, dtype=torch.long, device=model.device)
    positions = torch.full((batch, 1), ctx - 1, dtype=torch.long, device=model.device)
    slots = list(range(batch))

    timings = []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.decode_step_ragged(ids, positions, slots)
        torch.cuda.synchronize()
        if i >= warmup:
            timings.append((time.perf_counter() - t0) * 1000)

    stats = cache.stats(batch) if hasattr(cache, "stats") else {}
    model.cache = None
    torch.cuda.empty_cache()
    return {"tpot_ms": statistics.median(timings), "p95_ms": sorted(timings)[int(0.95 * (len(timings) - 1))],
            "kv_used_mb": stats.get("kv_used_mb", 0.0)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase2_gqa.yaml")
    p.add_argument("--context-lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--results-dir", default="results/raw")
    args = p.parse_args()

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        print("[WARN] no CUDA device; this measures nothing.", file=sys.stderr)
        return 1

    ref = QwenReference(model_name=cfg.model.name, dtype=cfg.model.dtype,
                        device=f"cuda:{cfg.hardware.devices[0]}").load()
    model = LatentServeQwen.from_reference(
        ref, max_seq_len_hint=max(args.context_lengths) + args.steps + 8
    )
    writer = ResultWriter(results_dir=args.results_dir)

    variants = [
        ("contiguous+sdpa", dict(paged=False, impl="sdpa", kv_dtype="fp16")),
        ("paged+sdpa", dict(paged=True, impl="sdpa", kv_dtype="fp16")),
        ("paged+kernel", dict(paged=True, impl="triton_paged", kv_dtype="fp16")),
    ]

    for batch in args.batch_sizes:
        for ctx in args.context_lengths:
            print(f"\n=== batch={batch} ctx={ctx} block={args.block_size} ===")
            baseline = None
            for label, kw in variants:
                try:
                    r = time_decode(model, batch, ctx, args.steps, args.warmup,
                                    block_size=args.block_size, **kw)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {label:<18} OOM")
                    torch.cuda.empty_cache()
                    continue
                if label == "paged+sdpa":
                    baseline = r["tpot_ms"]
                kv_gb = r["kv_used_mb"] / 1024
                implied_bw = (WEIGHT_BYTES + kv_gb * 1024**3) / (r["tpot_ms"] / 1000) / 1e9
                delta = f"{r['tpot_ms'] / baseline - 1:+.1%} vs paged+sdpa" if baseline else ""
                print(f"  {label:<18} tpot={r['tpot_ms']:6.2f}ms  p95={r['p95_ms']:6.2f}  "
                      f"implied {implied_bw:5.0f} GB/s  {delta}")
                writer.write(BenchmarkResult(
                    system=f"latentserve_{label.replace('+', '_')}", tag="phase11_kernel",
                    attention="gqa", model=cfg.model.name, batch_size=batch,
                    context_length=ctx, output_length=args.steps, num_gpus=1,
                    tpot_ms=r["tpot_ms"], kv_cache_mb=r["kv_used_mb"], seed=0,
                    extra={"status": "ok", "variant": label, "block_size": args.block_size,
                           "implied_bandwidth_gb_s": implied_bw,
                           "vs_paged_sdpa": r["tpot_ms"] / baseline if baseline else None,
                           **r},
                ))
    print("\nTarget: ~33 ms at 8K/batch 4 (49.2 measured for paged+sdpa in Phase 4,\n"
          "minus the 17.3 ms gather Phase 3 isolated). The weight read is ~28 ms and\n"
          "is irreducible without touching the model itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
