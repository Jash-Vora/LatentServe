"""
Phase 7.2 — quality cost of KV compression, measured in output space.

    export PYTHONPATH=$(pwd):$PYTHONPATH
    python -m benchmarks.runners.phase7_truncation --context-length 4096

Sweeps low-rank projections and INT8 quantization through the forward
pass and reports logit KL divergence and top-1 flip rate against the
unmodified model. Simulation only — no cache, no kernel, no MLA
implementation — which is what makes it an hour rather than a phase.

## What decides the phase

7.1 found that at 2.0x compression a rank-192 joint projection leaves V
with 19% relative reconstruction error. Whether that matters is not a
question reconstruction error can answer; attention is a softmax over
dot products and may be far more or far less forgiving than a Frobenius
norm suggests.

The number to compare against is **INT8**, which gives exactly 2x with
no reconstruction compute at all. If low-rank at 2.0x is worse than INT8
at 2.0x, the sophisticated method loses to the trivial one at equal
memory, and 7.3-7.4 need a different justification than memory saving.

## Reading the output

`top1_flip_rate` is the legible metric: the fraction of positions where
compression changes the model's most likely next token. `kl_mean_nats`
is the sensitive one — it moves before the argmax does, so it separates
configurations that both look lossless by flip rate.

Both are computed against the same fp32 baseline, on the same tokens, in
the same process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from benchmarks.runners.phase7_spectra import KVCapture, build_calibration_ids
from compression.spectra import compression_ratio
from compression.truncation import (
    DivergenceMeter,
    fit_basis,
    install_joint_lowrank,
    install_quantization,
    install_separate_lowrank,
)
from config import load_config

DEFAULT_RANKS = [64, 96, 128, 192, 256, 384]


def collect_grams(ref, ids, rope, chunk_size: int, device: str):
    """Pass 0: the Gram matrices the subspaces are fitted from.

    Fitted on the same tokens the quality is then measured on. That is
    deliberate and needs stating in the report: it is the *optimistic*
    case, an upper bound on what a fitted low-rank method can achieve.
    A held-out calibration set would be the honest generalisation test,
    and 7.1 already showed the spectra barely move between real text and
    random ids, so the gap is expected to be small — but "expected" is
    not "measured".
    """
    capture = KVCapture(ref.model, ref.shape, rope, device="cpu")
    with torch.no_grad():
        for start in range(0, ids.shape[1], chunk_size):
            ref.model(input_ids=ids[:, start : start + chunk_size], use_cache=False)
    capture.close()

    joint, k_only, v_only = {}, {}, {}
    for (layer, name), acc in capture.acc.items():
        if name == "kv_joint":
            joint[layer] = acc.gram
        elif name == "k_pre_rope":
            k_only[layer] = acc.gram
        elif name == "v":
            v_only[layer] = acc.gram
    return joint, k_only, v_only


@torch.no_grad()
def measure(ref, ids, chunk_size: int, install_fn) -> dict:
    """Baseline vs modified logits on identical tokens.

    The baseline is recomputed per chunk rather than cached across the
    sweep: [4096, 151936] in fp32 is 2.5 GB, and holding it while a
    modified forward runs is a straightforward way to run out of a T4.
    The extra forwards are cheap next to that.
    """
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
    p.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    p.add_argument("--text-file", default=None)
    p.add_argument("--random-tokens", action="store_true")
    p.add_argument("--skip-separate", action="store_true",
                   help="skip the per-block low-rank control")
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

    source = "random" if args.random_tokens else (
        Path(args.text_file).read_text() if args.text_file else "default"
    )
    ids = build_calibration_ids(ref, args.context_length, source, cfg.generation.seed).to(device)
    rope = RotaryEmbedding.from_hf_config(ref.model.config, max_seq_len=ids.shape[1], device=device)

    print(f"fitting subspaces on {ids.shape[1]} tokens ...")
    joint_grams, k_grams, v_grams = collect_grams(ref, ids, rope, args.chunk_size, device)

    rows = []

    def record(label: str, ratio: float, result: dict, **extra) -> None:
        rows.append({"method": label, "cache_ratio": ratio, **result, **extra})
        print(
            f"  {label:<28}{ratio:>6.2f}x  KL {result['kl_mean_nats']:.5f} nats  "
            f"(max {result['kl_max_nats']:.3f})  top-1 flips "
            f"{result['top1_flip_rate'] * 100:5.2f}%"
        )

    print("\n=== sanity: no modification ===")
    record("baseline (identity)", 1.0, measure(ref, ids, args.chunk_size,
                                               lambda: install_quantization(
                                                   ref.model, "none", "none", heads, head_dim)))

    print("\n=== INT8 quantization (the bar to clear: exactly 2x, no reconstruction) ===")
    for k_mode, v_mode in (("per_tensor", "per_tensor"), ("per_channel", "per_token"),
                           ("per_token", "per_token"), ("per_channel", "per_channel")):
        record(
            f"int8 K:{k_mode[4:]} V:{v_mode[4:]}", 2.0,
            measure(ref, ids, args.chunk_size,
                    lambda k=k_mode, v=v_mode: install_quantization(
                        ref.model, k, v, heads, head_dim)),
            k_mode=k_mode, v_mode=v_mode,
        )

    print("\n=== joint low-rank (the MLA representation) ===")
    for rank in args.ranks:
        if rank >= kv_dim:
            continue
        bases = {layer: fit_basis(g, rank) for layer, g in joint_grams.items()}
        record(
            f"joint low-rank r={rank}", compression_ratio(rank, 64, kv_dim),
            measure(ref, ids, args.chunk_size,
                    lambda b=bases: install_joint_lowrank(ref.model, b)),
            rank=rank,
        )

    if not args.skip_separate:
        print("\n=== separate K/V low-rank (control) ===")
        for rank in args.ranks:
            half = rank // 2
            if half >= kv_dim // 2:
                continue
            kb = {layer: fit_basis(g, half) for layer, g in k_grams.items()}
            vb = {layer: fit_basis(g, half) for layer, g in v_grams.items()}
            record(
                f"separate K/V r={half}+{half}", compression_ratio(rank, 64, kv_dim),
                measure(ref, ids, args.chunk_size,
                        lambda a=kb, b=vb: install_separate_lowrank(ref.model, a, b)),
                rank=rank,
            )

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase7_truncation.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps({**row, "context_length": ids.shape[1],
                                "source": "random" if args.random_tokens else "text"}) + "\n")
    verdict(rows)
    print(f"\nwrote {len(rows)} rows to {path}")
    return 0


def verdict(rows: list[dict]) -> None:
    int8 = [r for r in rows if r["method"].startswith("int8")]
    lowrank = [r for r in rows if r["method"].startswith("joint low-rank")]
    if not int8 or not lowrank:
        return
    best_int8 = min(int8, key=lambda r: r["kl_mean_nats"])
    # The like-for-like comparison: low-rank at the same 2x memory.
    at_2x = min(lowrank, key=lambda r: abs(r["cache_ratio"] - 2.0))

    print("\n=== verdict ===")
    print(f"best INT8:        {best_int8['method']:<28} "
          f"KL {best_int8['kl_mean_nats']:.5f}  flips "
          f"{best_int8['top1_flip_rate'] * 100:.2f}%")
    print(f"low-rank at ~2x:  {at_2x['method']:<28} "
          f"KL {at_2x['kl_mean_nats']:.5f}  flips {at_2x['top1_flip_rate'] * 100:.2f}%")
    if at_2x["kl_mean_nats"] > best_int8["kl_mean_nats"]:
        print(
            "\nINT8 wins at equal memory, with no reconstruction compute. A latent\n"
            "representation then needs a justification other than memory saving —\n"
            "or a rank where it clearly beats this, which the sweep above locates."
        )
    else:
        print(
            "\nLow-rank beats INT8 at equal memory, so the latent representation is\n"
            "worth building. Note the two compose: a quantized latent cache is the\n"
            "product of both ratios, and is the obvious thing to try next."
        )


if __name__ == "__main__":
    sys.exit(main())