"""
Phase 14, stages 1-2: how sparse can decode attention get before quality goes?

    python -m benchmarks.runners.phase14_oracle            # ~40 min on a T4
    python -m benchmarks.runners.phase14_oracle --quick    # ~12 min

The gate before any sparse kernel is built. Prompts are prefilled densely;
every *decode* step then attends only to the pages a policy selects
(model/attention/sparse.py). Sixteen configurations: dense, and the oracle,
the bounds indexer and the sink+recent window at 50% down to 3.125% of pages.

Two kinds of evidence, because perplexity alone cannot see what sparse
attention breaks:

  long text   8K tokens of Wikitext prefilled, then the next 128 fed one at a
              time as sparse decode steps: perplexity, KL from dense, top-1
              agreement with dense, and the share of dense attention mass the
              selection kept
  needles     a passkey hidden at 10/50/90% depth in 4K/8K/16K of Wikitext;
              the *question* is also fed as decode steps — prefilled densely,
              the answer's first token would be computed with full attention
              and the test would measure nothing about sparsity

Each context is prefilled once and the cache rewound between
configurations. Dense's own needle accuracy is printed with the rest: where
the model fails even with full attention, that column says nothing about
sparsity.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RATIOS = (0.5, 0.25, 0.125, 0.0625, 0.03125)
POLICY_ORDER = ("oracle", "bounds", "window")


@dataclass
class NeedleCase:
    context: list            # token ids, needle inside
    question: list           # token ids, fed as decode steps
    key: str                 # what a correct answer contains
    key_ids: list            # the key's token ids (synthetic mode judges on these)
    length: int
    depth: float


def configs(policies, ratios) -> list:
    return [("dense", 1.0)] + [(p, r) for p in policies for r in ratios]


def _apply(ls, cfg, recent: int):
    """Put the model in configuration `cfg`; return the study (or None).

    ("gpu", r) runs the real path — CUDA indexer, top-k and sparse kernel,
    fp16 bounds, budget from the context — through ls.set_sparse(r). Every
    other policy installs the reference study (model/attention/sparse.py).
    """
    from model.attention import sparse as sp

    if cfg[0] == "gpu":
        sp.install(ls, None)
        ls.set_sparse(cfg[1], recent=recent)
        return None
    if hasattr(ls, "set_sparse") and getattr(ls, "sparse_ratio", None) is not None:
        ls.set_sparse(None)
    if getattr(ls, "_gpu_mode", False) and cfg[0] == "dense":
        sp.install(ls, None)          # dense on the CUDA kernel: the real baseline
        return None
    study = sp.SparseStudy(cfg[0], cfg[1], recent=recent)
    sp.install(ls, study)
    return study


def _record(r, study):
    nan = float("nan")
    r["captured"].append(statistics.fmean(study.captured) if study and study.captured else nan)
    r["kept"].append(statistics.fmean(study.kept) if study and study.kept else nan)


def _step(ls, token: int, device):
    import torch

    return ls.decode_step(torch.tensor([[token]], device=device))[0, -1]


def eval_text(ls, windows, cfgs, ctx: int, recent: int, device, log=print) -> dict:
    """Prefill each window's first `ctx` tokens densely, then teacher-force
    the rest through sparse decode steps."""
    import torch

    from model.attention import sparse as sp

    out = {c: {"nll": [], "kl": [], "agree": [], "captured": [], "kept": []} for c in cfgs}
    for wi, ids in enumerate(windows):
        t0 = time.perf_counter()
        sp.install(ls, None)
        ls.allocate_cache(1, len(ids) + 16, paged=True, block_size=16)
        ls.cache.reset()
        with torch.no_grad():
            ls.prefill(torch.tensor([ids[:ctx]], device=device))
            dense_logp = []
            for c in cfgs:
                ls.cache.rewind(ctx)
                study = _apply(ls, c, recent)
                r = out[c]
                for t in range(len(ids) - ctx - 1):
                    logp = torch.log_softmax(_step(ls, ids[ctx + t], device).float(), -1)
                    r["nll"].append(-float(logp[ids[ctx + t + 1]]))
                    if c[0] == "dense":
                        dense_logp.append(logp.half())
                        r["kl"].append(0.0)
                        r["agree"].append(1.0)
                    else:
                        ref = dense_logp[t].float()
                        r["kl"].append(max(0.0, float((ref.exp() * (ref - logp)).sum())))
                        r["agree"].append(float(int(logp.argmax()) == int(ref.argmax())))
                _record(r, study)
                sp.install(ls, None)
        log(f"      text window {wi + 1}/{len(windows)} done in {time.perf_counter() - t0:.0f}s")
    return out


def judge(case: NeedleCase, generated_ids: list, tokenizer) -> bool:
    if tokenizer is None:
        return generated_ids[: len(case.key_ids)] == case.key_ids
    text = tokenizer.decode(generated_ids)
    m = re.search(r"\d+", text)
    return bool(m) and m.group(0) == case.key


def eval_needles(ls, cases, cfgs, recent: int, answer_tokens: int, tokenizer, device,
                 log=print) -> dict:
    import torch

    from model.attention import sparse as sp

    out = {c: {"correct": [], "captured": []} for c in cfgs}
    for ci, case in enumerate(cases):
        t0 = time.perf_counter()
        n = len(case.context)
        sp.install(ls, None)
        ls.allocate_cache(1, n + len(case.question) + answer_tokens + 16, paged=True,
                          block_size=16)
        ls.cache.reset()
        with torch.no_grad():
            ls.prefill(torch.tensor([case.context], device=device))
            for c in cfgs:
                ls.cache.rewind(n)
                study = _apply(ls, c, recent)
                logits = None
                for tok in case.question:
                    logits = _step(ls, tok, device)
                gen = []
                for _ in range(answer_tokens):
                    nxt = int(logits.argmax())
                    gen.append(nxt)
                    logits = _step(ls, nxt, device)
                out[c]["correct"].append((case.length, judge(case, gen, tokenizer)))
                out[c]["captured"].append(statistics.fmean(study.captured)
                                          if study and study.captured else float("nan"))
                sp.install(ls, None)
        log(f"      needle {ci + 1}/{len(cases)} ({case.length // 1024}K, depth {case.depth:.0%}) "
            f"done in {time.perf_counter() - t0:.0f}s")
    return out


def build_needles(tokenizer, filler: list, lengths, depths, trials, seed=0) -> list:
    rng = random.Random(seed)
    cases, offset = [], 0

    def ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    question = ids("\n\nWhat is the pass key? The pass key is")
    for length in lengths:
        for depth in depths:
            for _ in range(trials):
                key = str(rng.randint(10000, 99999))
                needle = ids(f" The pass key is {key}. Remember it. {key} is the pass key. ")
                need = length - len(needle) - len(question)
                if offset + need > len(filler):
                    offset = 0
                chunk = filler[offset:offset + need]
                offset += need
                pos = int(depth * need)
                cases.append(NeedleCase(chunk[:pos] + needle + chunk[pos:], question, key,
                                        ids(" " + key), length, depth))
    return cases


def summarise(text, needles, cfgs, lengths) -> list:
    rows = []
    for c in cfgs:
        row = {"policy": c[0], "ratio": c[1]}
        if text:
            t = text[c]
            row.update(ppl=math.exp(statistics.fmean(t["nll"])), kl=statistics.fmean(t["kl"]),
                       agree=statistics.fmean(t["agree"]),
                       captured=statistics.fmean(t["captured"]), kept=statistics.fmean(t["kept"]))
        if needles:
            n = needles[c]
            row["needle"] = {L: [ok for (ln, ok) in n["correct"] if ln == L] for L in lengths}
            if "captured" not in row:
                row["captured"] = statistics.fmean(n["captured"])
        rows.append(row)
    return rows


def print_table(rows, lengths) -> None:
    head = f"{'policy':<8}{'pages':>7}"
    if "ppl" in rows[0]:
        head += f"{'ppl':>9}{'KL vs dense':>13}{'top-1 agree':>13}"
    head += f"{'mass kept':>11}"
    if "needle" in rows[0]:
        head += "".join(f"{f'needle {L // 1024}K':>11}" for L in lengths)
    print(head)
    for r in rows:
        line = f"{r['policy']:<8}{r['ratio']:>7.1%}"
        if "ppl" in r:
            line += f"{r['ppl']:>9.3f}{r['kl']:>13.4f}{r['agree']:>13.1%}"
        cap = r["captured"]
        line += f"{'-':>11}" if cap != cap else f"{cap:>11.3f}"
        if "needle" in r:
            for L in lengths:
                ok = r["needle"][L]
                line += f"{f'{sum(ok)}/{len(ok)}':>11}"
        print(line)
        if r["policy"] == "dense":
            print("-" * len(head))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    p.add_argument("--policies", nargs="+", default=list(POLICY_ORDER))
    p.add_argument("--recent", type=int, default=2, help="recent pages always kept")
    p.add_argument("--gpu", action="store_true",
                   help="the real path: CUDA indexer + sparse kernel at each ratio, against "
                   "dense on the CUDA kernel. 'mass kept' is not measurable there (no full "
                   "scores are computed) and prints as '-'")
    p.add_argument("--text-windows", type=int, default=4)
    p.add_argument("--text-ctx", type=int, default=8192)
    p.add_argument("--text-cont", type=int, default=128)
    p.add_argument("--needle-lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--needle-depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--needle-trials", type=int, default=2)
    p.add_argument("--answer-tokens", type=int, default=8)
    p.add_argument("--skip-text", action="store_true")
    p.add_argument("--skip-needle", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="2 text windows x 64 tokens, 1 needle per cell: ~12 min")
    p.add_argument("--results-dir", default="results/raw/phase14_oracle")
    args = p.parse_args()
    if args.quick:
        args.text_windows, args.text_cont, args.needle_trials = 2, 64, 1

    import torch

    from config import load_config
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    if not torch.cuda.is_available():
        print("[ERROR] the study needs a GPU", file=sys.stderr)
        return 1
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] needs `datasets`: python -m pip install datasets", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    device = "cuda:0"
    ref = QwenReference(model_name=cfg.model.name, dtype="fp16", device=device).load()
    longest = max([args.text_ctx + args.text_cont] + args.needle_lengths) + 128
    if args.gpu:
        from kernels.gqa.paged_decode import set_decode_backend

        set_decode_backend("cuda")
        ls = LatentServeQwen.from_reference(ref, attn_impl="triton_paged",
                                            max_seq_len_hint=longest)
        ls._gpu_mode = True
        args.policies = ["gpu"]
    else:
        ls = LatentServeQwen.from_reference(ref, attn_impl="sdpa", max_seq_len_hint=longest)
    tok = ref.tokenizer

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    stream = tok("\n\n".join(t for t in wiki["text"] if t.strip()),
                 add_special_tokens=False)["input_ids"]
    span = args.text_ctx + args.text_cont
    windows = [stream[i * span:(i + 1) * span] for i in range(args.text_windows)]
    filler = stream[args.text_windows * span:]
    cases = build_needles(tok, filler, args.needle_lengths, args.needle_depths,
                          args.needle_trials)
    cfgs = configs(args.policies, args.ratios)
    print(f"{len(cfgs)} configurations; {len(windows)} text windows x {args.text_cont} tokens; "
          f"{len(cases)} needles", flush=True)

    t0 = time.perf_counter()
    text = None
    if not args.skip_text:
        print("[1/2] long text: dense prefill, sparse decode", flush=True)
        text = eval_text(ls, windows, cfgs, args.text_ctx, args.recent, device,
                         log=lambda m: print(m, flush=True))
    needles = None
    if not args.skip_needle:
        print("[2/2] needles: dense prefill, question and answer as sparse decode", flush=True)
        needles = eval_needles(ls, cases, cfgs, args.recent, args.answer_tokens, tok, device,
                               log=lambda m: print(m, flush=True))
    rows = summarise(text, needles, cfgs, args.needle_lengths)
    print(f"\nPhase 14 oracle study ({(time.perf_counter() - t0) / 60:.0f} min). 'pages' is the "
          f"budget; 'mass kept' the share of dense attention mass the selection kept.\n")
    print_table(rows, args.needle_lengths)
    print("\nReading it: the oracle is the ceiling for any page-level indexer. Where the oracle\n"
          "itself degrades, sparse attention cannot be made safe at that budget; where the\n"
          "bounds indexer trails the oracle, the indexer is the problem; where the window\n"
          "control matches the bounds indexer, query awareness is not buying anything.")
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oracle_study.json").write_text(json.dumps(
        {"rows": rows, "args": vars(args)}, default=str, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
