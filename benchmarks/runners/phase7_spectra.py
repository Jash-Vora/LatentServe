"""
Phase 7.1 — KV compressibility map.

    export PYTHONPATH=$(pwd):$PYTHONPATH
    python -m benchmarks.runners.phase7_spectra --context-length 8192

Produces, per layer: the rank needed for 90/95/99/99.9% of the energy,
for K (pre- and post-RoPE), V, the joint KV vector MLA would compress,
and each KV head separately — plus the data-free weight spectra as a
contrast.

Runs on CPU in minutes and needs no benchmark. Its output decides
whether 7.2-7.4 are worth building: if 128 dimensions capture 99% of the
joint KV, a latent representation has room to exist; if 400 are needed
for 95%, there is nothing to compress and six weeks are saved.

## Why the text matters

Activation spectra depend on the input distribution, so **random token
ids are not a valid sample**: they are out of distribution for a trained
model and produce activation statistics that say nothing about real
serving. Phases 1-6 used random ids happily because they were measuring
time, and time does not care what the tokens mean. This does.

Default is a set of embedded natural-language passages, tiled to the
requested length. `--text-file` takes a real corpus and is better.
`--random-tokens` runs the invalid version deliberately, as a control:
the gap between the two is itself worth reporting, because it shows how
much a compressibility claim depends on the calibration data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from compression.spectra import (
    SpectrumAccumulator,
    SpectrumReport,
    block_reconstruction_error,
    compression_ratio,
    mla_break_even,
    rank_for_energy,
    weight_spectrum,
)
from config import load_config

# Natural-language calibration text. Deliberately varied in register —
# narrative, technical, dialogue, list-like — because a single register
# would under-sample the directions the model uses and overstate
# compressibility.
CALIBRATION_PASSAGES = [
    """The tide went out further than anyone remembered. Boats that had floated
that morning now sat tilted in the mud, their hulls streaked with weed, and the
children walked out across ground that had been seabed for as long as the town
had existed. Someone said it meant a storm was coming. Someone else said it
meant nothing at all, that tides do this, that the sea keeps its own accounts
and settles them on a schedule no one has bothered to learn.""",
    """A cache is a bet about the future: that something computed once will be
wanted again, soon enough and often enough to justify the space it occupies.
Every cache design is therefore an argument about locality. Change the access
pattern and the same structure that was an optimisation becomes overhead, which
is why measuring a cache under one workload tells you almost nothing about how
it behaves under another.""",
    """"You could just ask her," he said. "I could." "But you won't." "I might."
"You've been saying that for three weeks." She turned the cup around on the
saucer without drinking from it. "There's a version of this where I ask and it
goes badly and then I still have to see her every day." "There's a version
where it goes well." "Sure," she said. "There's always that version."关于这
一点，他没有再说什么。""",
    """Ingredients: two onions, sliced thin; four cloves of garlic; a tin of
chopped tomatoes; olive oil; salt. Heat the oil until it shimmers but does not
smoke. Add the onions and cook them slowly, stirring now and then, until they
collapse and turn the colour of weak tea. This takes longer than you think —
twenty-five minutes, sometimes forty. Rushing it is the single most common
mistake, and no amount of seasoning afterwards will recover what was lost.""",
    """The theorem is usually stated for finite-dimensional spaces, though the
proof given here extends without modification provided the operator is compact.
Suppose first that the eigenvalues are distinct. Then the corresponding
eigenvectors are linearly independent, and the argument proceeds by induction on
the dimension. The degenerate case requires more care: one must choose a basis
for each eigenspace, and the choice is not canonical, which is the source of
most of the difficulty students have with this material.""",
]


def build_calibration_ids(ref, context_length: int, source: str, seed: int) -> torch.Tensor:
    if source == "random":
        print(
            "[WARN] random token ids are out of distribution for a trained model. "
            "The resulting spectra describe nothing that happens in real serving; "
            "use this only as the control it is meant to be.",
            file=sys.stderr,
        )
        return ref.synthesize_input_ids(context_length, seed=seed)

    text = source if source != "default" else "\n\n".join(CALIBRATION_PASSAGES)
    ids = ref.tokenizer(text, return_tensors="pt")["input_ids"]
    if ids.shape[1] < context_length:
        reps = context_length // ids.shape[1] + 1
        ids = ids.repeat(1, reps)
    return ids[:, :context_length]


class KVCapture:
    """Forward hooks on every layer's k_proj / v_proj.

    Hooks rather than a modified forward: this analysis wants the
    unmodified Hugging Face path, so that nothing it reports can be an
    artefact of LatentServe's execution. The projections are the same
    module objects LatentServe borrows, so the values are identical.
    """

    def __init__(self, model, shape, rope, device):
        self.shape = shape
        self.rope = rope
        self.device = device
        self.acc: dict[tuple[int, str], SpectrumAccumulator] = {}
        self.handles = []
        self._pending: dict[int, dict] = {}
        self.positions: torch.Tensor | None = None

        for idx, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.k_proj.register_forward_hook(self._make_hook(idx, "k"))
            )
            self.handles.append(
                layer.self_attn.v_proj.register_forward_hook(self._make_hook(idx, "v"))
            )

    def _slot(self, layer: int, name: str, dim: int) -> SpectrumAccumulator:
        key = (layer, name)
        if key not in self.acc:
            self.acc[key] = SpectrumAccumulator(dim, device=self.device)
        return self.acc[key]

    def _make_hook(self, layer_idx: int, which: str):
        def hook(_module, _inputs, output):
            h = self.shape.num_key_value_heads
            d = self.shape.head_dim
            x = output.detach()                      # [B, S, h*d]
            per_head = x.view(x.shape[0], x.shape[1], h, d)

            if which == "k":
                self._slot(layer_idx, "k_pre_rope", h * d).update(x)
                for head in range(h):
                    self._slot(layer_idx, f"k_pre_rope_head{head}", d).update(
                        per_head[:, :, head]
                    )
                # Post-RoPE: rotate at the true positions. Same math the
                # model applies downstream, so this is the K that a naive
                # "compress the cache as stored" approach would see.
                cos, sin = self.rope.cos_sin(0, x.shape[1], torch.float32)
                kt = per_head.permute(0, 2, 1, 3).to(torch.float32)
                from model.rope import rotate_half

                rotated = (kt * cos) + (rotate_half(kt) * sin)
                rotated = rotated.permute(0, 2, 1, 3)
                self._slot(layer_idx, "k_post_rope", h * d).update(
                    rotated.reshape(x.shape[0], x.shape[1], h * d)
                )
                for head in range(h):
                    self._slot(layer_idx, f"k_post_rope_head{head}", d).update(
                        rotated[:, :, head]
                    )
                self._pending.setdefault(layer_idx, {})["k"] = x
            else:
                self._slot(layer_idx, "v", h * d).update(x)
                for head in range(h):
                    self._slot(layer_idx, f"v_head{head}", d).update(per_head[:, :, head])
                # The joint [K | V] vector is what an MLA latent compresses:
                # one shared vector per token per layer, across both heads
                # and both of K and V. That sharing is where MLA has more
                # headroom than per-head low-rank, so it needs its own row.
                k = self._pending.get(layer_idx, {}).pop("k", None)
                if k is not None:
                    self._slot(layer_idx, "kv_joint", 2 * h * d).update(
                        torch.cat([k, x], dim=-1)
                    )

        return hook

    def reports(self) -> list[SpectrumReport]:
        out = []
        for (layer, name), acc in sorted(self.acc.items()):
            out.append(
                SpectrumReport(
                    name=name, layer=layer, dim=acc.dim,
                    tokens=acc.count, eigenvalues=acc.eigenvalues(),
                )
            )
        return out

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase2_gqa.yaml")
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--chunk-size", type=int, default=2048,
                   help="forward in chunks; spectra stream so this only bounds memory")
    p.add_argument("--text-file", default=None, help="calibration corpus (better than the default)")
    p.add_argument("--random-tokens", action="store_true",
                   help="control run on out-of-distribution ids; not a valid measurement")
    p.add_argument("--device", default=None)
    p.add_argument("--gram-device", default="cpu", choices=["cpu", "cuda"],
                   help="where the d x d accumulators live. cpu keeps VRAM free for the "
                   "fp32 model (~6.2 GB) at the cost of a small transfer per chunk; "
                   "cuda is faster but adds ~125 MB and competes with the forward pass")
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--figures-dir", default="results/figures")
    args = p.parse_args()

    from model.qwen import QwenReference
    from model.rope import RotaryEmbedding

    cfg = load_config(args.config)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    # fp32: the small singular values are the entire question, and fp16
    # rounding destroys exactly those.
    ref = QwenReference(model_name=cfg.model.name, dtype="fp32", device=device).load()
    shape = ref.shape
    print(f"{cfg.model.name}: {shape.num_layers} layers, "
          f"{shape.num_key_value_heads} KV heads x {shape.head_dim} dims")
    print(f"GQA caches {2 * shape.num_key_value_heads * shape.head_dim} numbers/token/layer; "
          f"MLA breaks even below latent_dim={mla_break_even()}")

    source = "random" if args.random_tokens else (
        Path(args.text_file).read_text() if args.text_file else "default"
    )
    ids = build_calibration_ids(ref, args.context_length, source, cfg.generation.seed).to(device)
    print(f"calibration: {ids.shape[1]} tokens from "
          f"{'random ids' if args.random_tokens else (args.text_file or 'embedded passages')}")

    rope = RotaryEmbedding.from_hf_config(ref.model.config, max_seq_len=ids.shape[1], device=device)
    gram_device = device if args.gram_device == "cuda" else "cpu"
    capture = KVCapture(ref.model, shape, rope, device=gram_device)
    print(f"accumulating Gram matrices on {gram_device}")
    with torch.no_grad():
        for start in range(0, ids.shape[1], args.chunk_size):
            ref.model(input_ids=ids[:, start : start + args.chunk_size], use_cache=False)
            print(f"  captured {min(start + args.chunk_size, ids.shape[1])}/{ids.shape[1]} tokens")
    capture.close()

    rows = [r.summary() for r in capture.reports()]

    # Retained energy is norm-weighted and this model has massive
    # activations, so ask the question that actually matters: under a
    # rank-r projection of the joint KV vector, how well is each of K and
    # V reconstructed? Reported per rank, averaged over layers.
    kv_dim = 2 * shape.num_key_value_heads * shape.head_dim
    blocks = {"K": slice(0, kv_dim // 2), "V": slice(kv_dim // 2, kv_dim)}
    ranks = [r for r in (48, 64, 96, 128, 192, 256, 384) if r < kv_dim]
    recon: dict[int, list[dict]] = {r: [] for r in ranks}
    for (layer, name), acc in capture.acc.items():
        if name != "kv_joint":
            continue
        _, vecs = acc.eigendecomposition()
        for r in ranks:
            errs = block_reconstruction_error(acc.gram, vecs, r, blocks)
            recon[r].append(errs)
            rows.append({
                "name": "kv_joint_reconstruction", "layer": layer, "dim": kv_dim,
                "tokens": acc.count, "rank": r,
                "k_rel_error": errs["K"], "v_rel_error": errs["V"],
            })

    # Data-free weight spectra, for contrast.
    for idx, layer in enumerate(ref.model.model.layers):
        wk = layer.self_attn.k_proj.weight.detach().cpu()
        wv = layer.self_attn.v_proj.weight.detach().cpu()
        for name, w in (("w_k", wk), ("w_v", wv), ("w_kv_joint", torch.cat([wk, wv], dim=0))):
            eigs = weight_spectrum(w)
            rows.append(
                SpectrumReport(name=name, layer=idx, dim=min(w.shape),
                               tokens=0, eigenvalues=eigs).summary()
            )

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase7_spectra.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps({**row, "source": "random" if args.random_tokens else "text",
                                "context_length": ids.shape[1]}) + "\n")

    report(rows, gqa_numbers=kv_dim)
    report_reconstruction(recon, kv_dim)
    print(f"\nwrote {len(rows)} rows to {path}")
    plot(rows, capture.reports(), Path(args.figures_dir))
    return 0


def report(rows: list[dict], gqa_numbers: int, rope_dim: int = 64) -> None:
    import statistics

    print("\n=== rank needed for 99% of the energy (mean across layers) ===")
    print(f"{'spectrum':<24}{'dim':>6}{'r99':>8}{'r99/dim':>10}{'eff. rank':>11}")
    by_name: dict[str, list[dict]] = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r)
    for name in sorted(by_name):
        group = by_name[name]
        dim = group[0]["dim"]
        r99 = statistics.mean(x["rank_990"] for x in group)
        eff = statistics.mean(x["effective_rank"] for x in group)
        print(f"{name:<24}{dim:>6}{r99:>8.0f}{r99 / dim:>10.2f}{eff:>11.1f}")

    joint = by_name.get("kv_joint")
    if joint:
        break_even = mla_break_even(rope_dim, gqa_numbers)
        r99 = statistics.mean(x["rank_990"] for x in joint)
        worst = max(joint, key=lambda x: x["rank_990"])
        print(
            f"\nMLA verdict: the joint KV vector needs ~{r99:.0f} of {joint[0]['dim']} dims "
            f"for 99% energy on average (worst layer {worst['layer']}: {worst['rank_990']})."
        )
        # A latent must carry the positional part too, so the comparison is
        # latent + rope_dim against what GQA already stores.
        if r99 < break_even:
            ratio = compression_ratio(int(r99), rope_dim, gqa_numbers)
            print(
                f"  Below the {break_even} break-even, so a latent representation has "
                f"room: ~{ratio:.2f}x smaller cache at 99% energy."
            )
        else:
            print(
                f"  At or above the {break_even} break-even. A latent cache would not be "
                "smaller than the GQA baseline at this fidelity — which settles 7.2-7.4 "
                "before they are built."
            )

    pre = by_name.get("k_pre_rope")
    post = by_name.get("k_post_rope")
    if pre and post:
        a = statistics.mean(x["rank_990"] for x in pre)
        b = statistics.mean(x["rank_990"] for x in post)
        print(
            f"\nRoPE cost: K needs {a:.0f} dims pre-rotation and {b:.0f} post-rotation "
            f"({b / a:.2f}x). Compressing the cache as stored means paying the larger "
            "number; compressing pre-RoPE and rotating on reconstruction means paying "
            "the smaller one. This is the measurement behind MLA's decoupled "
            "positional path."
        )


def report_reconstruction(recon: dict, kv_dim: int) -> None:
    """The honest version of the compressibility question."""
    import statistics

    print("\n=== relative reconstruction error under a rank-r joint projection ===")
    print("(mean over layers; energy thresholds are norm-weighted and flatter here)")
    print(f"{'rank':>6}{'cache ratio':>13}{'K rel.err':>12}{'V rel.err':>12}")
    for rank in sorted(recon):
        entries = recon[rank]
        if not entries:
            continue
        k = statistics.mean(e["K"] for e in entries)
        v = statistics.mean(e["V"] for e in entries)
        ratio = compression_ratio(rank, 64, kv_dim)
        print(f"{rank:>6}{ratio:>12.2f}x{k:>12.3f}{v:>12.3f}")
    print(
        "\nK and V share one latent, so the usable rank is set by whichever "
        "reconstructs worse — not by the joint energy curve. Validate the chosen "
        "rank against logit KL divergence (7.2) before trusting any of these."
    )


def plot(rows: list[dict], reports: list, figures_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not available; JSONL written, no figure")
        return

    from compression.spectra import energy_curve

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    wanted = {"kv_joint", "k_pre_rope", "k_post_rope", "v"}
    for name in sorted(wanted):
        curves = [energy_curve(r.eigenvalues) for r in reports if r.name == name]
        if not curves:
            continue
        mean = torch.stack(curves).mean(0)
        axes[0].plot(range(1, len(mean) + 1), mean.numpy(), label=name)
    dims = {r["name"]: r["dim"] for r in rows}
    break_even = mla_break_even(64, dims.get("kv_joint", 512))
    axes[0].axvline(break_even, ls="--", c="k", lw=1, label=f"break-even ({break_even})")
    axes[0].axhline(0.99, ls=":", c="grey", lw=1)
    axes[0].set(xlabel="rank", ylabel="retained energy", title="Retained energy vs rank (mean over layers)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for name in ("kv_joint", "k_post_rope", "v"):
        pts = [(r["layer"], r["rank_990"]) for r in rows if r["name"] == name]
        if pts:
            pts.sort()
            axes[1].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, label=name)
    axes[1].set(xlabel="layer", ylabel="rank for 99% energy",
                title="Compressibility by layer (the 7.3 budget)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "phase7_kv_spectra.png"
    fig.savefig(out, dpi=140)
    print(f"figure: {out}")


if __name__ == "__main__":
    sys.exit(main())