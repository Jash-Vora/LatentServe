"""
Phase 7 — online-scaling check (must run before 7.3 builds anything).

    export PYTHONPATH=$(pwd):$PYTHONPATH
    python -m benchmarks.runners.phase7_online_scaling --context-length 4096

## Why this has to come first

7.2's headline INT8 number — `int8 K:channel V:token`, KL 0.00061, top-1
flips 1.37% — used `quantize_dequantize(mode="per_channel")`, which fits
one scale per (head, dim) channel over the **entire sequence at once**:
`dims = tuple(range(shaped.dim() - 2))` reduces over batch *and* every
token together. That is not a configuration a streaming cache can run.
A paged KV cache (this project's own `block_size: 16`) writes a block
once, as it fills, and can never know the max magnitude of a channel in
a block that has not been written yet. Token 500's scale cannot depend
on token 9,000's.

So before 7.3 spends any effort wiring INT8 into the actual decode path,
this checks the load-bearing assumption underneath the whole plan: does
per-channel K quantization still work when the scale is fit **locally,
per 16-token block**, rather than globally?

Two outcomes:

* block-local KL/flips are close to the global numbers -> 7.3 proceeds
  as planned, storing one scale per (block, head, channel).
* block-local is much worse -> 7.3 would be implementing a
  configuration whose quality was never actually measured, and needs a
  different granularity (e.g. a larger block, or falling back to
  per-token for K too) before any runtime work starts.

## What this does NOT test

`per_token` (V's granularity) is already local to a single token, so it
is already exactly what a streaming cache can compute — nothing to
check there. This script only varies K's granularity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from benchmarks.runners.phase7_spectra import build_calibration_ids
from compression.truncation import DivergenceMeter, install_quantization
from config import load_config

DEFAULT_BLOCK_SIZES = [8, 16, 32, 64, 128]


@torch.no_grad()
def measure(ref, ids, chunk_size: int, install_fn) -> dict:
    meter = DivergenceMeter()
    for start in range(0, ids.shape[1], chunk_size):
        chunk = ids[:, start : start + chunk_size]
        base = ref.model(input_ids=chunk, use_cache=False).logits
        installed = install_fn()
        try:
            mod = ref.model(input_ids=chunk, use_cache=False).logits
        finally:
            installed.remove()
        meter.update(base, mod)
        del base, mod
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return meter.result()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase2_gqa.yaml")
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--block-sizes", type=int, nargs="+", default=DEFAULT_BLOCK_SIZES,
                   help="candidate K scale-fitting windows, tokens (16 == this "
                        "project's paged-cache block size)")
    p.add_argument("--text-file", default=None)
    p.add_argument("--random-tokens", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--results-dir", default="results/raw")
    args = p.parse_args()

    from model.qwen import QwenReference

    cfg = load_config(args.config)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ref = QwenReference(model_name=cfg.model.name, dtype="fp32", device=device).load()
    shape = ref.shape
    heads, head_dim = shape.num_key_value_heads, shape.head_dim

    source = "random" if args.random_tokens else (
        Path(args.text_file).read_text() if args.text_file else "default"
    )
    ids = build_calibration_ids(ref, args.context_length, source, cfg.generation.seed).to(device)

    rows = []

    def record(label: str, result: dict, **extra) -> None:
        rows.append({"method": label, **result, **extra})
        print(
            f"  {label:<32}KL {result['kl_mean_nats']:.5f} nats  "
            f"(max {result['kl_max_nats']:.3f})  top-1 flips "
            f"{result['top1_flip_rate'] * 100:5.2f}%"
        )

    print(f"=== online-scaling check, K:per_channel V:per_token, {ids.shape[1]} tokens ===\n")

    print("--- global K scale (7.2's number — NOT streaming-realizable) ---")
    record(
        "global",
        measure(ref, ids, args.chunk_size,
                lambda: install_quantization(ref.model, "per_channel", "per_token",
                                              heads, head_dim)),
    )

    print("\n--- block-local K scale (what a paged cache can actually compute) ---")
    for bs in args.block_sizes:
        record(
            f"block_local (block={bs})",
            measure(ref, ids, args.chunk_size,
                    lambda bs=bs: install_quantization(
                        ref.model, "per_channel", "per_token", heads, head_dim,
                        k_block_size=bs)),
            block_size=bs,
        )

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase7_online_scaling.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps({**row, "context_length": ids.shape[1],
                                "source": "random" if args.random_tokens else "text"}) + "\n")

    verdict(rows)
    print(f"\nwrote {len(rows)} rows to {path}")
    return 0


def verdict(rows: list[dict]) -> None:
    global_row = next((r for r in rows if r["method"] == "global"), None)
    block_rows = [r for r in rows if r["method"].startswith("block_local")]
    if global_row is None or not block_rows:
        return

    at_16 = next((r for r in block_rows if r.get("block_size") == 16), None)
    print("\n=== verdict ===")
    print(f"global (non-realizable): KL {global_row['kl_mean_nats']:.5f}  "
          f"flips {global_row['top1_flip_rate'] * 100:.2f}%")
    if at_16 is not None:
        ratio = at_16["kl_mean_nats"] / max(global_row["kl_mean_nats"], 1e-12)
        print(f"block=16 (realizable):   KL {at_16['kl_mean_nats']:.5f}  "
              f"flips {at_16['top1_flip_rate'] * 100:.2f}%  ({ratio:.1f}x global KL)")
        if ratio < 3.0:
            print(
                "\nBlock-local scaling at the project's own page size holds up. 7.3 can "
                "proceed: store one INT8 scale per (block, head, channel) for K."
            )
        else:
            print(
                "\nBlock-local scaling at block=16 is meaningfully worse than the global "
                "number 7.2 reported. Check the larger block sizes above before wiring "
                "INT8 into the decode path — 7.3 should target whichever block size (or "
                "fallback granularity) actually holds quality, not the number 7.2 quoted."
            )
    else:
        print("(no block_size=16 row — pass --block-sizes 16 to compare directly)")


if __name__ == "__main__":
    sys.exit(main())
