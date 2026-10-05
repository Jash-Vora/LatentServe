"""
Phase 17: measure the step-time table the sparsity policy decides from.

    python -m benchmarks.runners.phase17_calibrate      # ~15 min on a T4

"Thresholds derived from benchmark data, not assumed" (methodology §24):
decode-step time for dense, 50% and 37.5% of pages, at batch 1-32 and
2K-32K contexts, under CUDA graphs with the production model setup (fused
projections and elementwise ops, CUDA decode kernel). Budgets alternate
within each round so slow drift lands on all of them. Shapes that do not fit
in memory are skipped and recorded. Output: results/raw/phase17/step_table.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--context-lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.375])
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--results-dir", default="results/raw/phase17")
    args = p.parse_args()

    import torch

    from benchmarks.runners.phase14_fusion import time_graphed
    from config import load_config
    from kernels.gqa.paged_decode import set_decode_backend
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    if not torch.cuda.is_available():
        print("[ERROR] needs a GPU", file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    ref = QwenReference(model_name=cfg.model.name, dtype="fp16", device="cuda:0").load()
    set_decode_backend("cuda")
    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=max(args.context_lengths) + 512,
                                           attn_impl="triton_paged", fuse_projections=True)
    model.set_elementwise(True)
    budgets = [None] + args.ratios
    rows, skipped = [], []
    print(f"{'batch':>5}{'ctx':>7}{'dense ms':>10}" + "".join(f"{f'{r:.1%} ms (x)':>17}" for r in args.ratios))
    for batch in args.batch_sizes:
        for ctx in args.context_lengths:
            times = {r: [] for r in budgets}
            try:
                for _ in range(args.rounds):
                    for r in budgets:
                        model.set_sparse(r)
                        times[r].append(time_graphed(model, batch, ctx, 16, args.steps, 8))
            except torch.cuda.OutOfMemoryError:
                model.cache = None
                torch.cuda.empty_cache()
                skipped.append({"batch": batch, "ctx": ctx})
                print(f"{batch:>5}{ctx:>7}   does not fit: skipped", flush=True)
                continue
            model.set_sparse(None)
            med = {r: statistics.median(ts) for r, ts in times.items()}
            for r, ms in med.items():
                rows.append({"batch": batch, "ctx": ctx, "ratio": r, "ms": ms,
                             "spread": max(times[r]) - min(times[r])})
            print(f"{batch:>5}{ctx:>7}{med[None]:>10.2f}"
                  + "".join(f"{med[r]:>9.2f} ({med[None] / med[r]:>4.2f}x)" for r in args.ratios),
                  flush=True)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "step_table.json").write_text(json.dumps(
        {"rows": rows, "skipped": skipped, "device": torch.cuda.get_device_name(0),
         "args": vars(args)}, indent=1))
    print(f"\nwritten: {out / 'step_table.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
