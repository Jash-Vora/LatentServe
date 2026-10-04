"""
Phase 15 — how sparse can attention become before useful information is lost?

    python -m benchmarks.runners.phase15_quality --task needle     [--quick]
    python -m benchmarks.runners.phase15_quality --task multikey   [--quick]
    python -m benchmarks.runners.phase15_quality --task vartrack   [--quick]
    python -m benchmarks.runners.phase15_quality --task qa         [--quick]
    python -m benchmarks.runners.phase15_quality --task text       [--quick]
    python -m benchmarks.runners.phase15_quality --task gen        [--quick]
    python -m benchmarks.runners.phase15_quality --task latency    [--quick]
    python -m benchmarks.runners.phase15_quality --task curves     # verdicts + plots

The plan (methodology §22) asks for long-context retrieval, long-context QA,
language modelling and general generation, and *sparsity <-> quality <->
latency trade-off curves*. Every budget from 100% to 3.125% of pages is
measured on the **real GPU path** — the CUDA indexer and sparse kernel the
system runs (`ls.set_sparse`) — and, for retrieval and QA up to 16K, on the
**oracle** (perfect page selection, model/attention/sparse.py) as the
ceiling. Each context is prefilled densely once; the cache is rewound
between configurations; the question and answer run as sparse decode steps.

Tasks:

  needle    a passkey at 10/25/50/75/90% depth in 4K/8K/16K/32K of Wikitext
  multikey  four passkeys for four names; the question asks for one. The
            distractor pages look like the right one: the hard case for a
            bound-based indexer
  vartrack  VAR A = 41823 ... VAR B = VAR A ... VAR D = VAR C, spread across the
            context: one dropped link breaks the chain
  qa        a SQuAD paragraph and its question, buried at a depth among
            paragraphs from other articles: real questions, distant evidence
  text      long-text continuation at 8K and 32K: KL and top-1 vs dense
  gen       ~2K-token prompts, 128 greedy tokens: how long output stays
            identical to dense
  latency   whole decode steps at every budget on three shapes

Pass criteria — fixed before any result was seen (docs/phase15_quality.md):

  * retrieval + QA: paired failures (dense right, budget wrong), summed over
    the four tasks, at most 2% of dense's correct answers (at least 1
    allowed, so a small sample is not failed by one case)
  * text: mean KL from dense <= 0.01 at both 8K and 32K
  * gen: reported, not judged — no threshold was proposed in advance
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import string
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

RATIOS = (0.5, 0.25, 0.125, 0.0625, 0.03125)
CASE_TASKS = ("needle", "multikey", "vartrack", "qa")
MAX_PAIRED_FAILURE = 0.02
MAX_KL = 0.01
NAMES = ("apple", "river", "candle", "violet", "copper", "falcon", "maple", "harbor")


# ------------------------------------------------------------------ cases ---


@dataclass
class Case:
    task: str
    context: list
    question: list
    answers: list
    length: int
    depth: float
    answer_tokens: int = 8
    meta: dict = field(default_factory=dict)


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(output: str, answers) -> bool:
    out = normalize(output)
    return any(normalize(a) and normalize(a) in out for a in answers)


def _ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def _with_inserts(filler: list, inserts) -> list:
    """Insert token lists at relative positions (0..1) of `filler`."""
    out = list(filler)
    for pos, piece in sorted(inserts, key=lambda x: -x[0]):
        at = int(pos * len(filler))
        out[at:at] = piece
    return out


class Filler:
    """Wikitext tokens handed out in order, wrapping: each case gets fresh text."""

    def __init__(self, stream, offset: int = 0):
        # The starting point depends on the seed: a new seed alone would only
        # change the passkeys, leaving nearly the same contexts. Seed 0 starts
        # at 0, so the published Phase 15 run is reproducible.
        self.stream, self.offset = stream, offset % max(1, len(stream))

    def take(self, n: int) -> list:
        n = max(0, n)
        if self.offset + n > len(self.stream):
            self.offset = 0
        chunk = self.stream[self.offset:self.offset + n]
        self.offset += n
        return chunk


def build_needles(tok, filler: Filler, lengths, depths, trials, rng) -> list:
    question = _ids(tok, "\n\nWhat is the pass key? The pass key is")
    cases = []
    for length in lengths:
        for depth in depths:
            for _ in range(trials):
                key = str(rng.randint(10000, 99999))
                needle = _ids(tok, f" The pass key is {key}. Remember it. {key} is the pass key. ")
                body = filler.take(length - len(needle) - len(question) - 8)
                cases.append(Case("needle", _with_inserts(body, [(depth, needle)]), question,
                                  [key], length, depth))
    return cases


def build_multikey(tok, filler: Filler, lengths, depths, trials, rng, keys=4) -> list:
    cases = []
    for length in lengths:
        for depth in depths:
            for _ in range(trials):
                names = rng.sample(NAMES, keys)
                nums = [str(rng.randint(10000, 99999)) for _ in names]
                target = names[0]
                pieces = [_ids(tok, f" The pass key for {n} is {k}. Remember it. ")
                          for n, k in zip(names, nums)]
                spots = [depth] + [rng.uniform(0.05, 0.95) for _ in names[1:]]
                question = _ids(tok, f"\n\nWhat is the pass key for {target}? "
                                     f"The pass key for {target} is")
                body = filler.take(length - sum(map(len, pieces)) - len(question) - 8)
                cases.append(Case("multikey", _with_inserts(body, list(zip(spots, pieces))),
                                  question, [nums[0]], length, depth,
                                  meta={"distractors": dict(zip(names[1:], nums[1:]))}))
    return cases


def build_vartrack(tok, filler: Filler, lengths, trials, rng, hops=4, distractors=2) -> list:
    def name():
        return "".join(rng.choice(string.ascii_uppercase) for _ in range(5))

    cases = []
    for length in lengths:
        for _ in range(trials):
            chain = [name() for _ in range(hops)]
            value = str(rng.randint(10000, 99999))
            stmts = [f" VAR {chain[0]} = {value}. "]
            stmts += [f" VAR {chain[i]} = VAR {chain[i - 1]}. " for i in range(1, hops)]
            pieces = [((i + 0.5) / hops, _ids(tok, s)) for i, s in enumerate(stmts)]
            pieces += [(rng.uniform(0.05, 0.95),
                        _ids(tok, f" VAR {name()} = {rng.randint(10000, 99999)}. "))
                       for _ in range(distractors)]
            question = _ids(tok, f"\n\nQuestion: What is the numeric value of VAR {chain[-1]}?\n"
                                 f"Answer: The numeric value of VAR {chain[-1]} is")
            body = filler.take(length - sum(len(p) for _, p in pieces) - len(question) - 8)
            cases.append(Case("vartrack", _with_inserts(body, pieces), question, [value],
                              length, 0.5, meta={"chain": chain}))
    return cases


def build_qa(tok, squad, lengths, depths, trials, rng) -> list:
    """SQuAD: the gold paragraph at `depth` among paragraphs from other
    articles, until the context reaches `length` tokens."""
    by_title: dict = {}
    for row in squad:
        by_title.setdefault(row["title"], []).append(row)
    titles = sorted(by_title)
    paragraphs = sorted({(row["title"], row["context"]) for row in squad})
    header = _ids(tok, "Answer the question using the documents below.\n\n")
    memo: dict = {}

    def para_ids(text):
        if text not in memo:
            memo[text] = _ids(tok, text)
        return memo[text]

    cases = []
    for length in lengths:
        for depth in depths:
            for _ in range(trials):
                title = rng.choice(titles)
                ex = rng.choice(by_title[title])
                question = _ids(tok, f"\n\nQuestion: {ex['question']}\nAnswer:")
                gold = para_ids(ex["context"])
                # Each document costs its label and separator too: budget them
                # at their longest, so the context cannot overshoot `length`.
                per_doc = len(_ids(tok, "Document 9999: ")) + len(_ids(tok, "\n\n"))
                budget = length - len(header) - len(gold) - per_doc - len(question) - 16
                others, used = [], 0
                pool = [p for t, p in paragraphs if t != title]
                rng.shuffle(pool)
                for p in pool:
                    ids = para_ids(p)
                    if used + len(ids) + per_doc > budget:
                        continue
                    others.append(ids)
                    used += len(ids) + per_doc
                    if used > budget - 64:
                        break
                docs = list(others)
                docs.insert(int(depth * len(docs)), gold)
                body = list(header)
                for i, d in enumerate(docs):
                    body += _ids(tok, f"Document {i + 1}: ") + d + _ids(tok, "\n\n")
                cases.append(Case("qa", body, question, list(ex["answers"]["text"]), length,
                                  depth, answer_tokens=16, meta={"question": ex["question"]}))
    return cases


# ------------------------------------------------------------- evaluation ---


def cfg_list(ratios, oracle=True) -> list:
    out = [("dense", 1.0)] + [("gpu", r) for r in ratios]
    return out + ([("oracle", r) for r in ratios] if oracle else [])


def circular(stream, start: int, n: int) -> list:
    start %= len(stream)
    out = stream[start:start + n]
    return out + stream[:n - len(out)] if len(out) < n else out


def eval_cases(ls, cases, cfgs, recent, tok, device, oracle_max_ctx, log=print,
               apply_fn=None) -> list:
    """Every case under every configuration: prefill once, rewind between."""
    import torch

    from benchmarks.runners.phase14_oracle import _apply, _step
    from model.attention import sparse as sp

    rows = []
    for ci, case in enumerate(cases):
        t0 = time.perf_counter()
        n = len(case.context)
        sp.install(ls, None)
        ls.allocate_cache(1, n + len(case.question) + case.answer_tokens + 16, paged=True,
                          block_size=16)
        ls.cache.reset()
        with torch.no_grad():
            ls.prefill(torch.tensor([case.context], device=device))
            for c in cfgs:
                row = {"task": case.task, "length": case.length, "depth": case.depth,
                       "policy": c[0], "ratio": c[1], "case": ci}
                if c[0] == "oracle" and case.length > oracle_max_ctx:
                    rows.append({**row, "correct": None, "output": None})
                    continue
                ls.cache.rewind(n)
                (apply_fn or _apply)(ls, c, recent)
                logits = None
                for t in case.question:
                    logits = _step(ls, t, device)
                gen = []
                for _ in range(case.answer_tokens):
                    nxt = int(logits.argmax())
                    gen.append(nxt)
                    logits = _step(ls, nxt, device)
                sp.install(ls, None)
                text = tok.decode(gen) if tok is not None else " ".join(map(str, gen))
                rows.append({**row, "correct": is_correct(text, case.answers), "output": text})
        log(f"      {case.task} {ci + 1}/{len(cases)} ({case.length // 1024}K, depth "
            f"{case.depth:.0%}) {time.perf_counter() - t0:.0f}s")
    return rows


def _apply_gen(ls, cfg, recent):
    """As phase14_oracle._apply, plus ("triton", 1.0): dense through the
    Triton kernel instead of the CUDA one. Two correct dense
    implementations — the control that says how fast greedy outputs drift
    apart from numerical noise alone on these prompts."""
    from benchmarks.runners.phase14_oracle import _apply
    from kernels.gqa.paged_decode import set_decode_backend
    from model.attention import sparse as sp

    if cfg[0] == "triton":
        sp.install(ls, None)
        if getattr(ls, "sparse_ratio", None) is not None:
            ls.set_sparse(None)
        set_decode_backend("triton")
        return None
    set_decode_backend("cuda")
    return _apply(ls, cfg, recent)


def eval_gen(ls, prompts, cfgs, recent, new_tokens, device, log=print) -> list:
    """Greedy generation under each budget against dense's own output."""
    import torch

    from benchmarks.runners.phase14_oracle import _apply, _step
    from model.attention import sparse as sp

    rows = []
    for pi, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        n = len(prompt)
        sp.install(ls, None)
        ls.allocate_cache(1, n + new_tokens + 16, paged=True, block_size=16)
        ls.cache.reset()
        outs = {}
        with torch.no_grad():
            first = ls.prefill(torch.tensor([prompt], device=device))[0, -1]
            for c in cfgs:
                ls.cache.rewind(n)
                _apply_gen(ls, c, recent)
                logits, gen = first, []
                for _ in range(new_tokens):
                    nxt = int(logits.argmax())
                    gen.append(nxt)
                    logits = _step(ls, nxt, device)
                sp.install(ls, None)
                outs[c] = gen
        ref = outs[("dense", 1.0)]
        for c, gen in outs.items():
            div = next((i for i, (a, b) in enumerate(zip(gen, ref)) if a != b), len(ref))
            rows.append({"policy": c[0], "ratio": c[1], "prompt": pi, "first_diff": div,
                         "identical": div == len(ref)})
        log(f"      gen {pi + 1}/{len(prompts)} {time.perf_counter() - t0:.0f}s")
    return rows


# ----------------------------------------------------------------- curves ---


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def paired(rows, policy, ratio) -> dict:
    """Against dense, case by case: failures (dense right, this wrong) and
    gains (dense wrong, this right)."""
    dense = {(r["task"], r["case"]): r["correct"] for r in rows if r["policy"] == "dense"}
    out = {"failures": 0, "gains": 0, "dense_correct": 0, "n": 0}
    for r in rows:
        if r["policy"] != policy or abs(r["ratio"] - ratio) > 1e-9 or r["correct"] is None:
            continue
        d = dense.get((r["task"], r["case"]))
        out["n"] += 1
        out["dense_correct"] += bool(d)
        out["failures"] += bool(d) and not r["correct"]
        out["gains"] += (not d) and r["correct"]
    return out


def verdicts(case_rows, text, ratios) -> list:
    """The pre-registered criteria, applied per budget on the GPU path."""
    out = []
    for ratio in ratios:
        pr = paired(case_rows, "gpu", ratio)
        allowed = max(1, int(MAX_PAIRED_FAILURE * pr["dense_correct"]))
        retrieval_ok = pr["failures"] <= allowed
        kls = {ctx: v["kl"] for ctx, by_cfg in (text or {}).items()
               for (p, r), v in by_cfg.items() if p == "gpu" and abs(r - ratio) < 1e-9}
        kl_ok = all(k <= MAX_KL for k in kls.values()) if kls else None
        reasons = []
        if not retrieval_ok:
            reasons.append(f"{pr['failures']} paired failures > {allowed} allowed")
        if kl_ok is False:
            reasons.append("KL " + ", ".join(f"{c // 1024}K {k:.4f}" for c, k in kls.items()
                                             if k > MAX_KL) + f" > {MAX_KL}")
        passed = retrieval_ok and kl_ok is not False
        out.append({"ratio": ratio, "pass": passed, "paired": pr, "allowed": allowed,
                    "kl": kls, "reasons": reasons, "kl_measured": kl_ok is not None})
    return out


def summarise_text(raw) -> dict:
    """{ctx: {(policy, ratio): {kl, agree, ppl}}} from eval_text output."""
    out = {}
    for ctx, by_cfg in raw.items():
        out[int(ctx)] = {}
        for key, v in by_cfg.items():
            policy, ratio = key.split("|")
            out[int(ctx)][(policy, float(ratio))] = {
                "kl": statistics.fmean(v["kl"]), "agree": statistics.fmean(v["agree"]),
                "ppl": math.exp(statistics.fmean(v["nll"]))}
    return out


def curves(results_dir: Path, ratios) -> int:
    case_rows = []
    for task in CASE_TASKS:
        f = results_dir / f"{task}.json"
        if f.exists():
            case_rows += json.loads(f.read_text())["rows"]
    text = summarise_text(json.loads((results_dir / "text.json").read_text())["raw"]) \
        if (results_dir / "text.json").exists() else None
    gen = json.loads((results_dir / "gen.json").read_text())["rows"] \
        if (results_dir / "gen.json").exists() else None
    lat = json.loads((results_dir / "latency.json").read_text())["rows"] \
        if (results_dir / "latency.json").exists() else None
    if not case_rows and not text:
        print("no Phase 15 results yet: run the tasks first")
        return 1

    lines = ["# Phase 15 results", ""]
    for f in sorted(results_dir.glob("*.json")):
        a = json.loads(f.read_text()).get("args", {})
        if "scoring" in a:
            lines += [f"GPU-path page scoring: **{a['scoring']}**, seed {a.get('seed', 0)}", ""]
            print(lines[-2])
            break

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("## Accuracy by task (correct / measured, 95% Wilson interval)\n")
    tasks = [t for t in CASE_TASKS if any(r["task"] == t for r in case_rows)]
    emit("| policy | pages | " + " | ".join(tasks) + " | paired fail / gain vs dense |")
    emit("| --- | ---: | " + " | ".join("---:" for _ in tasks) + " | ---: |")
    for policy, ratio in [("dense", 1.0)] + [(p, r) for p in ("gpu", "oracle") for r in ratios]:
        cells = []
        for t in tasks:
            rs = [r for r in case_rows if r["task"] == t and r["policy"] == policy
                  and abs(r["ratio"] - ratio) < 1e-9 and r["correct"] is not None]
            k, n = sum(r["correct"] for r in rs), len(rs)
            lo, hi = wilson(k, n)
            cells.append(f"{k}/{n} ({lo:.0%}-{hi:.0%})" if n else "n/a")
        pr = paired(case_rows, policy, ratio) if policy != "dense" else None
        tail = f"{pr['failures']} / {pr['gains']}" if pr else "-"
        emit(f"| {policy} | {ratio:.1%} | " + " | ".join(cells) + f" | {tail} |")

    lengths = sorted({r["length"] for r in case_rows if "length" in r})
    if lengths:
        emit("\n## Paired failures by context length (GPU path; dense right, budget wrong)\n")
        emit("| pages | " + " | ".join(f"{L // 1024}K" for L in lengths) + " |")
        emit("| ---: | " + " | ".join("---:" for _ in lengths) + " |")
        for ratio in ratios:
            cells = []
            for L in lengths:
                pr = paired([r for r in case_rows if r.get("length") == L], "gpu", ratio)
                cells.append(f"{pr['failures']}/{pr['dense_correct']}")
            emit(f"| {ratio:.1%} | " + " | ".join(cells) + " |")

    if text:
        emit("\n## Language modelling (KL from dense / top-1 agreement)\n")
        ctxs = sorted(text)
        emit("| policy | pages | " + " | ".join(f"{c // 1024}K" for c in ctxs) + " |")
        emit("| --- | ---: | " + " | ".join("---:" for _ in ctxs) + " |")
        for key in sorted({k for v in text.values() for k in v}, key=lambda k: (k[0] != "dense", k[0], -k[1])):
            cells = [f"{text[c][key]['kl']:.4f} / {text[c][key]['agree']:.1%}"
                     if key in text[c] else "n/a" for c in ctxs]
            emit(f"| {key[0]} | {key[1]:.1%} | " + " | ".join(cells) + " |")

    if gen:
        emit("\n## Ordinary generation (128 greedy tokens vs dense; reported, not judged)\n")
        emit("| pages | identical | median first difference |")
        emit("| ---: | ---: | ---: |")
        ctl = [r for r in gen if r["policy"] == "triton"]
        if ctl:
            emit(f"| dense, Triton kernel (control) | {sum(r['identical'] for r in ctl)}/{len(ctl)} | "
                 f"{statistics.median(r['first_diff'] for r in ctl):.0f} |")
        for ratio in ratios:
            rs = [r for r in gen if r["policy"] == "gpu" and abs(r["ratio"] - ratio) < 1e-9]
            if rs:
                emit(f"| {ratio:.1%} | {sum(r['identical'] for r in rs)}/{len(rs)} | "
                     f"{statistics.median(r['first_diff'] for r in rs):.0f} |")

    if lat:
        emit("\n## Whole decode steps (ms; speedup over dense)\n")
        shapes = sorted({(r["batch"], r["ctx"]) for r in lat})
        emit("| pages | " + " | ".join(f"b{b} / {c // 1024}K" for b, c in shapes) + " |")
        emit("| ---: | " + " | ".join("---:" for _ in shapes) + " |")
        dense = {(r["batch"], r["ctx"]): r["ms"] for r in lat if r["ratio"] is None}
        for ratio in [None] + list(ratios):
            cells = []
            for s in shapes:
                m = next((r["ms"] for r in lat if (r["batch"], r["ctx"]) == s
                          and r["ratio"] == ratio), None)
                cells.append("n/a" if m is None else
                             f"{m:.2f}" + ("" if ratio is None else f" ({dense[s] / m:.2f}x)"))
            emit(f"| {'dense' if ratio is None else f'{ratio:.1%}'} | " + " | ".join(cells) + " |")

    emit("\n## Verdicts (criteria fixed before any result)\n")
    for v in verdicts(case_rows, text, ratios):
        pr = v["paired"]
        state = "PASS" if v["pass"] else "FAIL"
        note = "" if v["kl_measured"] else " (text task not run: KL criterion unchecked)"
        emit(f"- **{v['ratio']:.1%} of pages: {state}**{note} - {pr['failures']} paired failures "
             f"of {pr['dense_correct']} dense-correct (allowed {v['allowed']})"
             + (f"; {'; '.join(v['reasons'])}" if v["reasons"] else ""))

    _plots(results_dir, case_rows, text, lat, ratios, tasks)
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwritten: {results_dir / 'summary.md'} and the plots beside it")
    return 0


def _plots(results_dir, case_rows, text, lat, ratios, tasks) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: tables only)")
        return
    xs = [r * 100 for r in ratios]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    for t in tasks:
        for policy, style in (("gpu", "-o"), ("oracle", "--x")):
            ys = []
            for r in ratios:
                rs = [x for x in case_rows if x["task"] == t and x["policy"] == policy
                      and abs(x["ratio"] - r) < 1e-9 and x["correct"] is not None]
                ys.append(100 * sum(x["correct"] for x in rs) / len(rs) if rs else float("nan"))
            ax.plot(xs, ys, style, label=f"{t} ({policy})")
    ax.set(xscale="log", xlabel="% of pages kept", ylabel="accuracy %", title="Retrieval and QA")
    ax.legend(fontsize=7)
    if text:
        ax = axes[1]
        for ctx in sorted(text):
            ys = [text[ctx].get(("gpu", r), {}).get("kl", float("nan")) for r in ratios]
            ax.plot(xs, ys, "-o", label=f"{ctx // 1024}K")
        ax.axhline(MAX_KL, color="grey", ls=":", label="pass line")
        ax.set(xscale="log", yscale="log", xlabel="% of pages kept", ylabel="KL from dense",
               title="Language modelling")
        ax.legend(fontsize=8)
    if lat:
        ax = axes[2]
        dense = {(r["batch"], r["ctx"]): r["ms"] for r in lat if r["ratio"] is None}
        for s in sorted(dense):
            ys = [dense[s] / next((x["ms"] for x in lat if (x["batch"], x["ctx"]) == s
                                   and x["ratio"] == r), float("nan")) for r in ratios]
            ax.plot(xs, ys, "-o", label=f"batch {s[0]} / {s[1] // 1024}K")
        ax.axhline(1.0, color="grey", ls=":")
        ax.set(xscale="log", xlabel="% of pages kept", ylabel="speedup over dense",
               title="Whole decode steps")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "curves.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------- main ---


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--task", required=True,
                   choices=list(CASE_TASKS) + ["text", "gen", "latency", "curves"])
    p.add_argument("--quick", action="store_true", help="fewer cases: checks it all works")
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    p.add_argument("--trials", type=int, default=None, help="per length x depth cell")
    p.add_argument("--recent", type=int, default=2)
    p.add_argument("--oracle-max-ctx", type=int, default=16384,
                   help="the oracle's fp32 reference is slow; skip it above this context")
    p.add_argument("--no-oracle", action="store_true")
    p.add_argument("--scoring", choices=["bounds", "mass"], default="bounds",
                   help="the GPU path's page scoring: max bound over the group (Phase 15) or "
                   "summed estimated mass (the follow-up hypothesis)")
    p.add_argument("--hops", type=int, default=2,
                   help="vartrack chain length. 4 hops left dense at 1/12 in the quick "
                   "pass: a test dense cannot do measures nothing about sparsity")
    p.add_argument("--distractors", type=int, default=2, help="vartrack distractor assignments")
    p.add_argument("--dense-only", action="store_true",
                   help="calibrate a task's difficulty on dense alone, before any sparse run")
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--results-dir", default="results/raw/phase15")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "curves":
        return curves(out_dir, args.ratios)

    import torch

    from config import load_config
    from kernels.gqa.paged_decode import set_decode_backend
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    if not torch.cuda.is_available():
        print("[ERROR] Phase 15 needs a GPU", file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    device = "cuda:0"
    rng = random.Random(args.seed)
    ref = QwenReference(model_name=cfg.model.name, dtype="fp16", device=device).load()
    tok = ref.tokenizer
    set_decode_backend("cuda")
    t0 = time.perf_counter()
    log = lambda m: print(m, flush=True)  # noqa: E731

    if args.task == "latency":
        from benchmarks.runners.phase14_fusion import time_graphed

        model = LatentServeQwen.from_reference(ref, max_seq_len_hint=32768 + 512,
                                               attn_impl="triton_paged", fuse_projections=True)
        model.set_elementwise(True)
        model.set_sparse(None, scoring=args.scoring)
        shapes = [(1, 8192), (8, 32768), (16, 16384)]
        rounds, steps = (1, 24) if args.quick else (3, 48)
        rows = []
        for batch, ctx in shapes:
            times = {}
            for _ in range(rounds):
                for ratio in [None] + args.ratios:
                    model.set_sparse(ratio, recent=args.recent)
                    times.setdefault(ratio, []).append(
                        time_graphed(model, batch, ctx, 16, steps, 8))
            model.set_sparse(None)
            for ratio, ts in times.items():
                rows.append({"batch": batch, "ctx": ctx, "ratio": ratio,
                             "ms": statistics.median(ts), "spread": max(ts) - min(ts)})
            d = statistics.median(times[None])
            log(f"  batch {batch} / {ctx // 1024}K: dense {d:.2f} ms; " + ", ".join(
                f"{r:.1%} {statistics.median(times[r]):.2f}" for r in args.ratios))
        (out_dir / "latency.json").write_text(json.dumps({"rows": rows, "args": vars(args)}))
        return 0

    from datasets import load_dataset

    ls = LatentServeQwen.from_reference(ref, attn_impl="triton_paged",
                                        max_seq_len_hint=max(args.lengths) + 1024)
    ls._gpu_mode = True
    ls.set_sparse(None, scoring=args.scoring)
    log(f"GPU-path page scoring: {args.scoring}")
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    stream = _ids(tok, "\n\n".join(t for t in wiki["text"] if t.strip()))
    filler = Filler(stream[100_000:] + stream[:100_000], offset=args.seed * 50_021)
    cfgs = [("dense", 1.0)] if args.dense_only else cfg_list(args.ratios, oracle=not args.no_oracle)

    if args.task in CASE_TASKS:
        depths = [0.1, 0.5, 0.9] if args.quick else [0.1, 0.25, 0.5, 0.75, 0.9]
        trials = args.trials or (1 if args.quick else 2)
        if args.task == "needle":
            cases = build_needles(tok, filler, args.lengths, depths, trials, rng)
        elif args.task == "multikey":
            cases = build_multikey(tok, filler, args.lengths, depths[::2] or depths, trials, rng)
        elif args.task == "vartrack":
            cases = build_vartrack(tok, filler, args.lengths, trials * 3, rng,
                                   hops=args.hops, distractors=args.distractors)
        else:
            squad = load_dataset("rajpurkar/squad", split="validation")
            cases = build_qa(tok, squad, args.lengths, depths[::2] or depths, trials + 1, rng)
        log(f"{args.task}: {len(cases)} cases x {len(cfgs)} configurations")
        rows = eval_cases(ls, cases, cfgs, args.recent, tok, device, args.oracle_max_ctx, log)
        dense = [r for r in rows if r["policy"] == "dense"]
        log(f"dense: {sum(r['correct'] for r in dense)}/{len(dense)} correct")
        (out_dir / f"{args.task}.json").write_text(json.dumps(
            {"rows": rows, "cases": [{k: v for k, v in asdict(c).items()
                                      if k not in ("context", "question")} for c in cases],
             "args": vars(args)}, default=str))
    elif args.task == "text":
        from benchmarks.runners.phase14_oracle import eval_text

        raw = {}
        cont = 64 if args.quick else 256
        # Context + continuation must stay inside the positions the model was
        # trained on (32,768 for Qwen2.5-1.5B), or the "32K" row would also be
        # measuring position extrapolation. The window shrinks to fit.
        limit = ref.shape.max_position_embeddings - cont - 16
        plan = [(8192, 1 if args.quick else 4), (min(32768, limit), 1 if args.quick else 2)]
        offset = args.seed * 100_003          # seed 0: the published windows
        for ctx, windows in plan:
            if ctx > max(args.lengths) + 1024:
                continue
            ws = []
            for _ in range(windows):
                ws.append(circular(stream, offset, ctx + cont))
                offset += ctx + cont
            text_cfgs = [c for c in cfgs if c[0] != "oracle"]   # ceiling already in Phase 14
            res = eval_text(ls, ws, text_cfgs, ctx, args.recent, device, log=log)
            raw[ctx] = {f"{k[0]}|{k[1]}": v for k, v in res.items()}
        (out_dir / "text.json").write_text(json.dumps({"raw": raw, "args": vars(args)}))
    elif args.task == "gen":
        n = 4 if args.quick else 16
        prompts = [filler.take(2048) for _ in range(n)]
        gen_cfgs = [c for c in cfgs if c[0] != "oracle"] + [("triton", 1.0)]
        rows = eval_gen(ls, prompts, gen_cfgs, args.recent, 128, device, log)
        (out_dir / "gen.json").write_text(json.dumps({"rows": rows, "args": vars(args)}))
    log(f"{args.task} done in {(time.perf_counter() - t0) / 60:.1f} min -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
