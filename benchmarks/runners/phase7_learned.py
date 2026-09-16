"""
Phase 7, Experiment B — can a better-chosen latent beat SVD?

    export PYTHONPATH=$(pwd):$PYTHONPATH
    python -m benchmarks.runners.phase7_learned --context-length 8192 --ranks 128 192 256

Three subspaces at each rank, all with identical memory:

    plain SVD      the 7.2 result. Optimal for reconstruction error.
    output-aware   closed form, weighted by how each direction reaches
                   the output (queries for K, W_O for V).
    distilled      gradient-trained on logit KL, initialised from
                   output-aware so it cannot start worse.

Scored the same way as 7.2 — logit KL and top-1 flip rate against the
unmodified fp32 model — so every number here is directly comparable with
the INT8 bar (KL 0.00061, 1.37% flips at 2x).

## The gate

If the best learned latent cannot clearly beat the SVD frontier at equal
memory, stop: no kernels, no cache, no runtime. And even beating SVD is
not enough on its own — it has to approach **INT8**, which achieves 2x
with no reconstruction compute at all, and reconstruction compute was
already the term most likely to make a latent cache net-slower on a T4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from benchmarks.runners.phase7_spectra import KVCapture, build_calibration_ids
from benchmarks.runners.phase7_truncation import measure
from compression.learned import (
    LatentKVAdapter,
    QueryMetricCollector,
    distill,
    fit_metric_basis,
    install_adapters,
    joint_metric,
    output_metric_for_v,
)
from compression.spectra import compression_ratio
from compression.truncation import fit_basis, install_quantization
from config import load_config


def build_metrics(ref, ids, chunk_size: int, device: str) -> dict[int, torch.Tensor]:
    """Per layer, the block-diagonal metric over the joint [K | V] vector."""
    shape = ref.shape
    group_size = shape.num_attention_heads // shape.num_key_value_heads
    collector = QueryMetricCollector(ref.model, shape.num_attention_heads, shape.head_dim)
    with torch.no_grad():
        for start in range(0, ids.shape[1], chunk_size):
            ref.model(input_ids=ids[:, start : start + chunk_size], use_cache=False)
    collector.close()

    metrics = {}
    for idx, layer in enumerate(ref.model.model.layers):
        o_weight = layer.self_attn.o_proj.weight
        k_blocks, v_blocks = [], []
        for g in range(shape.num_key_value_heads):
            group = range(g * group_size, (g + 1) * group_size)
            k_blocks.append(collector.metric(idx, group))
            v_blocks.append(output_metric_for_v(o_weight, group, shape.head_dim))
        metrics[idx] = joint_metric(k_blocks, v_blocks)
    return metrics


def adapters_from_basis(ref, post_down: dict, up: dict) -> dict[int, LatentKVAdapter]:
    return {
        idx: LatentKVAdapter.from_projections(
            layer.self_attn.k_proj, layer.self_attn.v_proj, post_down[idx], up[idx]
        )
        for idx, layer in enumerate(ref.model.model.layers)
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase2_gqa.yaml")
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--ranks", type=int, nargs="+", default=[128, 192, 256])
    p.add_argument("--distill-steps", type=int, default=150)
    p.add_argument("--distill-seq-len", type=int, default=256)
    p.add_argument("--distill-lr", type=float, default=1e-3)
    p.add_argument("--skip-distill", action="store_true",
                   help="run the closed-form comparison only; B1 is free, B2 is not")
    p.add_argument("--text-file", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--results-dir", default="results/raw")
    args = p.parse_args()

    from model.qwen import QwenReference
    from model.rope import RotaryEmbedding

    cfg = load_config(args.config)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ref = QwenReference(model_name=cfg.model.name, dtype="fp32", device=device).load()
    shape = ref.shape
    heads, head_dim = shape.num_key_value_heads, shape.head_dim
    kv_dim = 2 * heads * head_dim

    source = Path(args.text_file).read_text() if args.text_file else "default"
    ids = build_calibration_ids(ref, args.context_length, source, cfg.generation.seed).to(device)
    rope = RotaryEmbedding.from_hf_config(ref.model.config, max_seq_len=ids.shape[1], device=device)

    print("collecting KV Gram matrices ...")
    capture = KVCapture(ref.model, shape, rope, device="cpu")
    with torch.no_grad():
        for start in range(0, ids.shape[1], args.chunk_size):
            ref.model(input_ids=ids[:, start : start + args.chunk_size], use_cache=False)
    capture.close()
    grams = {layer: acc.gram for (layer, name), acc in capture.acc.items() if name == "kv_joint"}

    print("collecting query / output metrics ...")
    metrics = build_metrics(ref, ids, args.chunk_size, device)

    rows = []

    def record(label: str, rank: int, result: dict, **extra) -> None:
        ratio = compression_ratio(rank, 64, kv_dim)
        rows.append({"method": label, "rank": rank, "cache_ratio": ratio, **result, **extra})
        print(
            f"  {label:<24} r={rank:<4}{ratio:>6.2f}x  KL {result['kl_mean_nats']:.5f}  "
            f"top-1 flips {result['top1_flip_rate'] * 100:5.2f}%"
        )

    print("\n=== INT8 bar (2x, no reconstruction compute) ===")
    int8 = measure(ref, ids, args.chunk_size,
                   lambda: install_quantization(ref.model, "per_channel", "per_token",
                                                heads, head_dim))
    rows.append({"method": "int8 K:channel V:token", "rank": None, "cache_ratio": 2.0, **int8})
    print(f"  {'int8 K:ch V:tok':<24} {'':<6}{2.0:>6.2f}x  KL {int8['kl_mean_nats']:.5f}  "
          f"top-1 flips {int8['top1_flip_rate'] * 100:5.2f}%")

    for rank in args.ranks:
        print(f"\n=== rank {rank} ({compression_ratio(rank, 64, kv_dim):.2f}x) ===")

        plain = {i: fit_basis(g, rank) for i, g in grams.items()}
        record("plain SVD (7.2)", rank,
               measure(ref, ids, args.chunk_size,
                       lambda a=adapters_from_basis(ref, plain, {i: b.T for i, b in plain.items()}):
                       install_adapters(ref.model, a)))

        fitted = {i: fit_metric_basis(grams[i], metrics[i], rank) for i in grams}
        down = {i: d for i, (d, _) in fitted.items()}
        up = {i: u for i, (_, u) in fitted.items()}
        aware = adapters_from_basis(ref, down, up)
        record("output-aware (B1)", rank,
               measure(ref, ids, args.chunk_size, lambda a=aware: install_adapters(ref.model, a)))

        if not args.skip_distill:
            print(f"  distilling {args.distill_steps} steps from the B1 initialisation ...")
            trained = adapters_from_basis(ref, down, up)
            history = distill(
                ref.model, trained, ids, steps=args.distill_steps,
                seq_len=args.distill_seq_len, lr=args.distill_lr, seed=cfg.generation.seed,
            )
            record("distilled (B2)", rank,
                   measure(ref, ids, args.chunk_size,
                           lambda a=trained: install_adapters(ref.model, a)),
                   distill_steps=args.distill_steps, final_train_kl=history[-1]["kl"])

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase7_learned.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps({**row, "context_length": ids.shape[1]}) + "\n")

    verdict(rows, int8)
    print(f"\nwrote {len(rows)} rows to {path}")
    return 0


def verdict(rows: list[dict], int8: dict) -> None:
    latents = [r for r in rows if r["rank"] is not None]
    if not latents:
        return
    best = min(latents, key=lambda r: r["kl_mean_nats"])
    at_2x = min(latents, key=lambda r: abs(r["cache_ratio"] - 2.0))
    plain_at_2x = min(
        (r for r in latents if r["method"].startswith("plain")),
        key=lambda r: abs(r["cache_ratio"] - 2.0),
        default=None,
    )

    print("\n=== verdict ===")
    print(f"best latent overall: {best['method']} r={best['rank']} "
          f"({best['cache_ratio']:.2f}x)  KL {best['kl_mean_nats']:.5f}")
    print(f"INT8 bar (2.00x):    KL {int8['kl_mean_nats']:.5f}")
    if plain_at_2x:
        gain = plain_at_2x["kl_mean_nats"] / max(at_2x["kl_mean_nats"], 1e-12)
        print(f"at ~2x, the best latent is {gain:.2f}x better than plain SVD")

    if at_2x["kl_mean_nats"] <= int8["kl_mean_nats"]:
        print(
            "\nA latent beats INT8 at equal memory on quality. Now the runtime question:\n"
            "reconstruction costs ~T x latent_dim x 512 MACs per layer per decode step,\n"
            "which INT8 does not pay. Phase 7.4 measures whether the quality win survives."
        )
    else:
        print(
            "\nINT8 still wins at equal memory, and it has no reconstruction cost.\n"
            "Per the gate in docs/phase7.md: stop here. Do not build a cache, a kernel\n"
            "or a runtime for this representation. The negative result — with the\n"
            "closed-form optimum and a trained version both measured — is the finding."
        )


if __name__ == "__main__":
    sys.exit(main())