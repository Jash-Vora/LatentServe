"""
Phase 15 bake-off: which training-free indexer gets 25% of pages to pass?

    python -m benchmarks.runners.phase15_bakeoff          # ~30 min on a T4

Phase 15 failed 25% with the bound-based indexer (4 paired failures of 83,
KL 0.0146 at 8K) while the oracle lost nothing up to 16K: the selection,
not sparsity, is what fails. Candidates, all in reference math
(model/attention/sparse.py):

  bounds                 today's indexer — the baseline
  bounds+window8         8 recent pages kept instead of 2
  bounds+dense2          the first 2 layers attend densely (Quest's choice)
  mass                   bounds as estimated attention mass, summed over the
                         group — ranks like the oracle does
  mean                   the same estimate from q . mean(K): an estimate, one
                         vector per page
  rerank                 bounds pick 2x the budget, exact scores keep the best
  mass/mean/rerank+dense2+window8   each with both cheap tweaks
  oracle                 the ceiling; not buildable, never selected

The rule, fixed before any result (docs/phase15_quality.md):

  * a development set: a new seed (default 1), new text, new questions — the
    83 Phase 15 cases are never reused
  * the winner is the *cheapest* buildable candidate whose mean development
    KL (8K and ~31K) is within 10% of the best; cost order: parameter
    tweaks < summed mass < mean keys < reranking
  * paired failures are reported, not ranked on: ~25 dense-correct cases
    are too few to rank by
  * the winner is then judged on a fresh seed with the Phase 15 criteria,
    through the real GPU path once built
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

VARIANTS = {
    "bounds": dict(policy="bounds", recent=2, dense_layers=0, cost=0),
    "bounds+window8": dict(policy="bounds", recent=8, dense_layers=0, cost=0),
    "bounds+dense2": dict(policy="bounds", recent=2, dense_layers=2, cost=0),
    "mass": dict(policy="mass", recent=2, dense_layers=0, cost=1),
    "mean": dict(policy="mean", recent=2, dense_layers=0, cost=2),
    "rerank": dict(policy="rerank", recent=2, dense_layers=0, cost=3),
    "mass+dense2+window8": dict(policy="mass", recent=8, dense_layers=2, cost=1),
    "mean+dense2+window8": dict(policy="mean", recent=8, dense_layers=2, cost=2),
    "rerank+dense2+window8": dict(policy="rerank", recent=8, dense_layers=2, cost=3),
    "oracle": dict(policy="oracle", recent=2, dense_layers=0, cost=None),
}
KL_SLACK = 1.10


def apply_variant(ls, cfg, recent):
    from model.attention import sparse as sp

    name, ratio = cfg
    if name == "dense":
        study = sp.SparseStudy("dense", 1.0)
    else:
        v = VARIANTS[name]
        study = sp.SparseStudy(v["policy"], ratio, recent=v["recent"],
                               dense_layers=v["dense_layers"])
    sp.install(ls, study)
    return study


def choose(rows: list) -> dict:
    """The pre-registered rule: cheapest buildable variant within 10% of the
    best mean development KL."""
    buildable = [r for r in rows if r["cost"] is not None and r["kl"] == r["kl"]]
    if not buildable:
        return {}
    best = min(r["kl"] for r in buildable)
    eligible = [r for r in buildable if r["kl"] <= KL_SLACK * best]
    return min(eligible, key=lambda r: (r["cost"], r["kl"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1, help="development seed: never 0, Phase 15's")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--results-dir", default="results/raw/phase15_bakeoff")
    args = p.parse_args()
    if args.seed == 0:
        p.error("seed 0 is Phase 15's own cases: the bake-off must not select on them")

    import torch
    from datasets import load_dataset

    from benchmarks.runners import phase15_quality as pq
    from benchmarks.runners.phase14_oracle import eval_text
    from config import load_config
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    if not torch.cuda.is_available():
        print("[ERROR] needs a GPU", file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    device = "cuda:0"
    rng = random.Random(args.seed)
    ref = QwenReference(model_name=cfg.model.name, dtype="fp16", device=device).load()
    tok = ref.tokenizer
    ls = LatentServeQwen.from_reference(ref, attn_impl="sdpa",
                                        max_seq_len_hint=max(args.lengths) + 1024)
    log = lambda m: print(m, flush=True)  # noqa: E731
    t0 = time.perf_counter()

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    stream = pq._ids(tok, "\n\n".join(t for t in wiki["text"] if t.strip()))
    filler = pq.Filler(stream[100_000:] + stream[:100_000], offset=args.seed * 50_021)
    cfgs = [("dense", 1.0)] + [(v, args.ratio) for v in args.variants]

    log(f"[1/2] text, seed {args.seed}: {len(cfgs)} configurations")
    text = {}
    cont, limit = 128, ref.shape.max_position_embeddings - 128 - 16
    offset = args.seed * 100_003
    for ctx, windows, n in ((8192, 2, cont), (min(32768, limit), 1, 64)):
        ws = []
        for _ in range(windows):
            ws.append(pq.circular(stream, offset, ctx + n))
            offset += ctx + n
        text[ctx] = eval_text(ls, ws, cfgs, ctx, 2, device, log=log, apply_fn=apply_variant)

    log(f"[2/2] retrieval and QA, seed {args.seed}")
    squad = load_dataset("rajpurkar/squad", split="validation")
    cases = (pq.build_needles(tok, filler, args.lengths, [0.1, 0.5, 0.9], 1, rng)
             + pq.build_multikey(tok, filler, args.lengths, [0.1, 0.9], 1, rng)
             + pq.build_qa(tok, squad, args.lengths, [0.1, 0.9], 2, rng))
    rows_cases = pq.eval_cases(ls, cases, cfgs, 2, tok, device, oracle_max_ctx=10**9, log=log,
                               apply_fn=apply_variant)

    rows = []
    for name in args.variants:
        key = (name, args.ratio)
        kls = {c: statistics.fmean(text[c][key]["kl"]) for c in text}
        mass = statistics.fmean(x for c in text for x in text[c][key]["captured"])
        pr = pq.paired([{**r, "policy": "gpu" if r["policy"] == name else r["policy"]}
                        for r in rows_cases if r["policy"] in ("dense", name)], "gpu", args.ratio)
        rows.append({"variant": name, "cost": VARIANTS[name]["cost"], "mass": mass,
                     "kl_by_ctx": kls, "kl": statistics.fmean(kls.values()), "paired": pr})

    ctxs = sorted(text)
    print(f"\nPhase 15 bake-off at {args.ratio:.1%} of pages, seed {args.seed} "
          f"({(time.perf_counter() - t0) / 60:.0f} min)\n")
    print(f"{'variant':<24}{'mass kept':>10}" + "".join(f"{f'KL {c // 1024}K':>10}" for c in ctxs)
          + f"{'mean KL':>10}{'paired fail/gain':>18}{'cost':>6}")
    for r in sorted(rows, key=lambda r: r["kl"]):
        pr = r["paired"]
        fails = f"{pr['failures']}/{pr['gains']} of {pr['dense_correct']}"
        cost = "-" if r["cost"] is None else str(r["cost"])
        print(f"{r['variant']:<24}{r['mass']:>10.3f}"
              + "".join(f"{r['kl_by_ctx'][c]:>10.4f}" for c in ctxs)
              + f"{r['kl']:>10.4f}{fails:>18}{cost:>6}")
    winner = choose(rows)
    if winner:
        print(f"\nSelected by the fixed rule (cheapest within {KL_SLACK - 1:.0%} of the best mean "
              f"KL): {winner['variant']} — mean KL {winner['kl']:.4f}, cost {winner['cost']}.\n"
              "Next: build it on the GPU path and judge it on a fresh seed with the Phase 15 "
              "criteria.")
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bakeoff.json").write_text(json.dumps(
        {"rows": rows, "winner": winner.get("variant"), "args": vars(args)}, default=str, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
