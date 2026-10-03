"""
Quality: does LatentServe compute the model as faithfully as vLLM?

    # in the vLLM session (torch 2.13), in this order:
    python -m benchmarks.runners.quality_vs_vllm --system latentserve
    python -m benchmarks.runners.quality_vs_vllm --system vllm
    python -m benchmarks.runners.quality_vs_vllm --compare

## What "quality" can mean here

Both systems run the *same weights* in fp16, so neither can know more than
the other. What differs is how faithfully each computes the model: fused
kernels, a different attention implementation, a different order of
rounding. So the question is **fidelity**, and it needs a yardstick:

  * **fp32 reference** — Hugging Face in fp32, the closest available stand-in
    for the model's true output;
  * **Hugging Face fp16** — the model as it is normally run. Every fp16
    system sits some distance from fp32; HF fp16's distance is the
    calibration point. Comparable distance is *on par*; clearly further is
    *degraded*; clearly closer is *better*.

LatentServe is measured twice — unfused and with elementwise fusion — so the
effect of fusion itself is separated from the comparison with vLLM.

## Three kinds of evidence

1. **Text (Wikitext-2 test):** perplexity, per-token agreement with the fp32
   argmax, and — for the systems that expose full logits — KL divergence
   from fp32 at every position. vLLM returns log-probabilities for the
   actual and top-1 tokens only, so it gets perplexity and agreement but no
   KL.
2. **A task (ARC-Easy, multiple choice):** each option scored by the
   log-probability of its tokens. Accuracy alone has a ±2-3% noise band at
   a few hundred items, so predictions are also compared *item by item*
   with the fp32 reference: the count of changed answers is what shows a
   numerical difference becoming a behavioural one.
3. **Generation (greedy, 128 tokens):** small differences compound here,
   so it is the most sensitive test, and the only one that exercises each
   system's *decode* path — for LatentServe, the CUDA graphs and the paged
   kernel. Reported as exact-match rate and the position of the first
   differing token, against HF fp16.

## Identical inputs

The LatentServe arm tokenizes everything once and saves the token ids; the
vLLM arm reads them back. Both systems score exactly the same tokens, and
the vLLM arm needs no dataset library.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

RESULTS = ("quality_data.json", "quality_latentserve.json", "quality_vllm.json")
SYSTEMS = ("hf_fp32", "hf_fp16", "latentserve_unfused", "latentserve_fused", "vllm")


# ----------------------------------------------------------------------
# Pure scoring helpers (unit-tested on CPU)
# ----------------------------------------------------------------------


def score_positions(logits, targets, ref_logp=None):
    """NLL of each target, the argmax, KL(ref || this), and the log-probs.

    logits [n, V] for n positions whose next tokens are `targets` [n]. Works
    on a slice of a sequence as well as a whole one, which is what keeps
    memory bounded: a full 1024-position chunk of fp32 logits over a
    152K vocabulary is 622 MB per copy, and the first version of this
    runner made several at once and ran out of memory on a T4.
    """
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    nll = (-logp.gather(-1, targets[:, None]).squeeze(-1)).tolist()
    top1 = logp.argmax(dim=-1).tolist()
    kl = None
    if ref_logp is not None:
        # KL is non-negative; float32 round-off can leave ~-1e-8 when two
        # distributions agree, which would print as a nonsensical negative.
        kl = (ref_logp.exp() * (ref_logp - logp)).sum(dim=-1).clamp_min(0).tolist()
    return nll, top1, kl, logp


def nll_top1_kl(logits, ids, ref_logp=None):
    """Whole-sequence form: position j predicts ids[j+1]."""
    nll, top1, kl, _ = score_positions(logits[:-1], ids[1:], ref_logp)
    return nll, top1, kl


def continuation_logprob(logits, ids, start):
    """Sum of log-probabilities of ids[start:] given what precedes them."""
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    positions = torch.arange(start - 1, len(ids) - 1)
    return float(logp[positions, ids[start:]].sum())


def mc_predict(scores, char_lens):
    """(acc prediction, acc_norm prediction) for one item.

    acc picks the highest total log-probability; acc_norm divides by the
    option's character length first, as lm-evaluation-harness does, so a
    long option is not penalised merely for having more tokens.
    """
    acc = max(range(len(scores)), key=lambda i: scores[i])
    norm = max(range(len(scores)), key=lambda i: scores[i] / max(1, char_lens[i]))
    return acc, norm


def first_divergence(a, b):
    """Index of the first differing token, or len if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def verdict(sys_dev, ref_dev, tol=0.25, floor=1e-12):
    """Compare a system's deviation from fp32 with HF fp16's.

    on par within +-25% of HF fp16's deviation (or both negligible);
    degraded beyond it; better below it.
    """
    if sys_dev is None or ref_dev is None:
        return "n/a"
    if max(sys_dev, ref_dev) < floor:
        return "on par"
    ratio = sys_dev / max(ref_dev, floor)
    if ratio > 1 + tol:
        return f"degraded ({ratio:.2f}x HF fp16's deviation)"
    if ratio < 1 - tol:
        return f"better ({ratio:.2f}x HF fp16's deviation)"
    return f"on par ({ratio:.2f}x)"


# ----------------------------------------------------------------------
# Data — built once, saved, reused by the vLLM arm
# ----------------------------------------------------------------------


def build_data(tokenizer, args) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "the LatentServe arm needs the `datasets` library to fetch Wikitext-2 and "
            "ARC-Easy: `python -m pip install datasets`, or pass --synthetic for a smoke "
            "test that measures nothing"
        ) from e

    def ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    stream = ids("\n\n".join(t for t in wiki["text"] if t.strip()))
    need = args.chunks * args.chunk_len
    if len(stream) < need + args.gen_prompts * args.gen_prompt_len:
        raise SystemExit("Wikitext-2 is shorter than the requested chunks + prompts")
    chunks = [stream[i * args.chunk_len:(i + 1) * args.chunk_len] for i in range(args.chunks)]
    # Generation prompts from text *after* the scored chunks, so the two
    # measurements do not share tokens.
    gen = [stream[need + i * args.gen_prompt_len: need + (i + 1) * args.gen_prompt_len]
           for i in range(args.gen_prompts)]

    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    items = []
    for row in arc.select(range(min(args.mc_items, len(arc)))):
        context = ids(f"Question: {row['question']}\nAnswer:")
        texts = row["choices"]["text"]
        label = row["choices"]["label"].index(row["answerKey"])
        items.append({"context": context,
                      "choices": [ids(" " + t) for t in texts],
                      "char_lens": [len(t) for t in texts],
                      "label": label})
    return {"chunks": chunks, "mc": items, "gen_prompts": gen,
            "gen_new_tokens": args.gen_new_tokens,
            "source": {"text": "wikitext-2-raw-v1/test", "task": "ARC-Easy/test"}}


def synthetic_data(vocab: int, args) -> dict:
    """Random tokens — for smoke tests only. Measures nothing about quality."""
    import torch

    g = torch.Generator().manual_seed(0)
    r = lambda n: torch.randint(0, vocab, (n,), generator=g).tolist()  # noqa: E731
    return {"chunks": [r(args.chunk_len) for _ in range(args.chunks)],
            "mc": [{"context": r(12), "choices": [r(3), r(4), r(2), r(5)],
                    "char_lens": [9, 12, 6, 15], "label": i % 4} for i in range(args.mc_items)],
            "gen_prompts": [r(args.gen_prompt_len) for _ in range(args.gen_prompts)],
            "gen_new_tokens": args.gen_new_tokens,
            "source": {"text": "SYNTHETIC", "task": "SYNTHETIC"}}


# ----------------------------------------------------------------------
# The LatentServe arm: fp32 and fp16 references, LatentServe both ways
# ----------------------------------------------------------------------


def _greedy_hf(model, prompt, n, device):
    """Greedy decode with Hugging Face's own cache and no stopping rule —
    every system generates exactly `n` tokens, end-of-text included."""
    import torch

    ids = torch.tensor([prompt], device=device)
    out = model(input_ids=ids, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    tokens = [int(nxt)]
    for _ in range(n - 1):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(int(nxt))
    return tokens


def _hf_hidden(model, t):
    """Final hidden states [S, H], after the final norm (Qwen2Model applies
    it), so logits for any slice are just `lm_head` of that slice."""
    return model.model(input_ids=t[None]).last_hidden_state[0]


def _ls_hidden(ls, ids, device):
    """LatentServe's hidden states [S, H], *before* the final norm, through
    its own layers. `_to_logits` applies the norm — fused or not — per slice."""
    import torch

    ls.allocate_cache(1, len(ids) + 8, paged=True, block_size=16)
    ls.cache.reset()
    return ls._forward_block(torch.tensor([ids], device=device), start_pos=0)[0]


def _ls_logits(ls, ids, device):
    import torch

    ls.allocate_cache(1, len(ids) + 8, paged=True, block_size=16)
    ls.cache.reset()
    return ls.forward_logits_all(torch.tensor([ids], device=device))[0]


def run_latentserve_arm(args, load_reference) -> dict:
    import torch

    from model.latentserve_qwen import LatentServeQwen
    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref32 = load_reference("fp32")
    ref16 = load_reference("fp16")
    device = ref16.device
    data = (synthetic_data(ref16.shape.vocab_size, args) if args.synthetic
            else build_data(ref16.tokenizer, args))
    (out_dir / RESULTS[0]).write_text(json.dumps(data))

    ls = LatentServeQwen.from_reference(ref16, max_seq_len_hint=args.chunk_len + 512,
                                        attn_impl="triton_paged", fuse_projections=True)

    def ls_variant(fused: bool):
        ls.set_fused(True)            # the fused MLP wrapper exists from here on
        ls.set_elementwise(fused)
        ls.set_fused(fused)

    res = {s: {"nll": [], "top1_agree": [], "kl": [], "mc_acc": [], "mc_norm": []}
           for s in SYSTEMS[:-1]}
    argmax32 = []

    with torch.no_grad():
        # --- text ---
        # Each system's network runs once per chunk; its hidden states are
        # turned into logits `args.score_slice` positions at a time and
        # scored before the next slice is made. Every position's score
        # depends only on its own logits, so slicing changes no number.
        print(f"[1/3] text: scoring {len(data['chunks'])} chunks x 4 systems", flush=True)
        for ci, chunk in enumerate(data["chunks"]):
            if ci and ci % 4 == 0:
                print(f"      chunk {ci}/{len(data['chunks'])}", flush=True)
            t = torch.tensor(chunk, device=device)
            hidden = {"hf_fp32": _hf_hidden(ref32.model, t),
                      "hf_fp16": _hf_hidden(ref16.model, t)}
            for name, fused in (("latentserve_unfused", False), ("latentserve_fused", True)):
                ls_variant(fused)
                hidden[name] = _ls_hidden(ls, chunk, device)
            heads = {"hf_fp32": ref32.model.lm_head, "hf_fp16": ref16.model.lm_head}

            def logits_of(name, a, b):
                if name in heads:
                    return heads[name](hidden[name][a:b])
                ls_variant(name == "latentserve_fused")
                return ls._to_logits(hidden[name][None, a:b])[0]

            chunk_top1 = []
            last = len(chunk) - 1                        # the final token predicts nothing
            for a in range(0, last, args.score_slice):
                b = min(a + args.score_slice, last)
                targets = t[a + 1 : b + 1]
                nll, top1, _, ref_logp = score_positions(logits_of("hf_fp32", a, b), targets)
                chunk_top1 += top1
                res["hf_fp32"]["nll"] += nll
                res["hf_fp32"]["top1_agree"] += [1.0] * len(top1)
                res["hf_fp32"]["kl"] += [0.0] * len(top1)
                for name in ("hf_fp16", "latentserve_unfused", "latentserve_fused"):
                    nll_s, top1_s, kl_s, _ = score_positions(logits_of(name, a, b), targets,
                                                             ref_logp)
                    res[name]["nll"] += nll_s
                    res[name]["kl"] += kl_s
                    res[name]["top1_agree"] += [float(x == y) for x, y in zip(top1_s, top1)]
                del ref_logp
            argmax32.append(chunk_top1)
            del hidden

        # --- task ---
        preds = {s: [] for s in SYSTEMS[:-1]}
        print(f"[2/3] ARC-Easy: {len(data['mc'])} questions x 4 systems", flush=True)
        for qi, item in enumerate(data["mc"]):
            if qi and qi % 50 == 0:
                print(f"      question {qi}/{len(data['mc'])}", flush=True)
            scores = {s: [] for s in SYSTEMS[:-1]}
            for choice in item["choices"]:
                seq = item["context"] + choice
                start = len(item["context"])
                t = torch.tensor(seq, device=device)
                scores["hf_fp32"].append(continuation_logprob(ref32.model(input_ids=t[None]).logits[0], t, start))
                scores["hf_fp16"].append(continuation_logprob(ref16.model(input_ids=t[None]).logits[0], t, start))
                for name, fused in (("latentserve_unfused", False), ("latentserve_fused", True)):
                    ls_variant(fused)
                    scores[name].append(continuation_logprob(_ls_logits(ls, seq, device), t, start))
            for s in scores:
                acc, norm = mc_predict(scores[s], item["char_lens"])
                preds[s].append([acc, norm])
                res[s]["mc_acc"].append(float(acc == item["label"]))
                res[s]["mc_norm"].append(float(norm == item["label"]))

        # --- generation ---
        n = data["gen_new_tokens"]
        print(f"[3/3] generation: {len(data['gen_prompts'])} prompts x {n} tokens "
              "(HF fp16 one token at a time, then LatentServe both ways)", flush=True)
        gens = {"hf_fp16": [_greedy_hf(ref16.model, p, n, device) for p in data["gen_prompts"]]}
        del ref32
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        for name, fused in (("latentserve_unfused", False), ("latentserve_fused", True)):
            ls_variant(fused)
            engine = ServingEngine(ls, max_running=args.gen_batch, block_size=16,
                                   max_seq_len=args.gen_prompt_len + n + 16,
                                   use_cuda_graphs=True)
            for i, p in enumerate(data["gen_prompts"]):
                engine.add_request(ServedRequest(request_id=i, prompt_ids=list(p),
                                                 max_new_tokens=n))
            done = {r.request_id: r.output_ids for r in engine.run()}
            gens[name] = [done[i][:n] for i in range(len(data["gen_prompts"]))]

    from kernels.gqa import paged_decode as _pd

    summary = {"argmax32": argmax32, "mc_preds": preds, "generations": gens,
               "metrics": {s: _metrics(res[s]) for s in res},
               "data_source": data["source"], "decode_backend": _pd.decode_backend()}
    (out_dir / RESULTS[1]).write_text(json.dumps(summary))
    _print_arm("latentserve arm", summary["metrics"])
    return summary


def _metrics(r: dict) -> dict:
    m = {"tokens": len(r["nll"]),
         "perplexity": math.exp(statistics.fmean(r["nll"])) if r["nll"] else None,
         "top1_agree": statistics.fmean(r["top1_agree"]) if r["top1_agree"] else None,
         "kl_mean": statistics.fmean(r["kl"]) if r["kl"] else None,
         "kl_max": max(r["kl"]) if r["kl"] else None,
         "mc_acc": statistics.fmean(r["mc_acc"]) if r["mc_acc"] else None,
         "mc_norm": statistics.fmean(r["mc_norm"]) if r["mc_norm"] else None,
         "mc_items": len(r["mc_acc"])}
    return m


def _print_arm(title, metrics):
    print(f"\n=== {title} ===")
    for s, m in metrics.items():
        kl = f"{m['kl_mean']:.2e}" if m.get("kl_mean") is not None else "  n/a  "
        print(f"  {s:<22} ppl {m['perplexity']:.4f}  top1 {m['top1_agree']:.4%}  "
              f"KL {kl}  ARC acc {m['mc_acc']:.3f} norm {m['mc_norm']:.3f}")


# ----------------------------------------------------------------------
# The vLLM arm
# ----------------------------------------------------------------------


def run_vllm_arm(args) -> dict:
    from comparisons.vllm.runner import VLLMRunner
    from config import load_config

    out_dir = Path(args.results_dir)
    data = json.loads((out_dir / RESULTS[0]).read_text())
    ls_side = json.loads((out_dir / RESULTS[1]).read_text())
    cfg = load_config(args.config)
    runner = VLLMRunner(cfg.model.name, max_model_len=args.chunk_len + 512,
                        max_num_seqs=args.gen_batch, seed=0)
    from vllm import SamplingParams

    try:
        from vllm.inputs import TokensPrompt

        prompt = lambda ids: TokensPrompt(prompt_token_ids=list(ids))  # noqa: E731
    except ImportError:  # pragma: no cover - older vLLM
        prompt = lambda ids: {"prompt_token_ids": list(ids)}  # noqa: E731

    score = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)

    def prompt_logprobs(seqs):
        """Per sequence: [(logprob of actual token j, top-1 token id)] for j >= 1."""
        outs = runner.llm.generate([prompt(s) for s in seqs], score, use_tqdm=False)
        rows = []
        for s, o in zip(seqs, outs):
            row = []
            for j in range(1, len(s)):
                entry = o.prompt_logprobs[j]
                actual = entry[s[j]].logprob
                best = min(entry.items(), key=lambda kv: kv[1].rank)[0]
                row.append((actual, best))
            rows.append(row)
        return rows

    r = {"nll": [], "top1_agree": [], "kl": [], "mc_acc": [], "mc_norm": []}
    for chunk_rows, ref_top1 in zip(prompt_logprobs(data["chunks"]), ls_side["argmax32"]):
        r["nll"] += [-lp for lp, _ in chunk_rows]
        r["top1_agree"] += [float(b == a) for (_, b), a in zip(chunk_rows, ref_top1)]

    preds = []
    for item in data["mc"]:
        seqs = [item["context"] + c for c in item["choices"]]
        start = len(item["context"])
        scores = [sum(lp for lp, _ in rows[start - 1:]) for rows in prompt_logprobs(seqs)]
        acc, norm = mc_predict(scores, item["char_lens"])
        preds.append([acc, norm])
        r["mc_acc"].append(float(acc == item["label"]))
        r["mc_norm"].append(float(norm == item["label"]))

    gen_params = SamplingParams(max_tokens=data["gen_new_tokens"], temperature=0.0,
                                ignore_eos=True)
    outs = runner.llm.generate([prompt(p) for p in data["gen_prompts"]], gen_params,
                               use_tqdm=False)
    gens = [list(o.outputs[0].token_ids) for o in outs]

    summary = {"mc_preds": preds, "generations": gens, "metrics": _metrics(r),
               "vllm": runner.describe()}
    (out_dir / RESULTS[2]).write_text(json.dumps(summary))
    _print_arm("vllm arm", {"vllm": summary["metrics"]})
    return summary


# ----------------------------------------------------------------------
# Compare
# ----------------------------------------------------------------------


def compare(args) -> int:
    out_dir = Path(args.results_dir)
    ls_side = json.loads((out_dir / RESULTS[1]).read_text())
    v = json.loads((out_dir / RESULTS[2]).read_text())
    metrics = dict(ls_side["metrics"], vllm=v["metrics"])
    preds = dict(ls_side["mc_preds"], vllm=v["mc_preds"])
    gens = dict(ls_side["generations"], vllm=v["generations"])
    if ls_side.get("data_source", {}).get("text") == "SYNTHETIC":
        print("[WARN] synthetic data: these numbers measure nothing about quality\n")

    ref = metrics["hf_fp16"]
    ppl32 = metrics["hf_fp32"]["perplexity"]
    print(f"LatentServe decode attention: {ls_side.get('decode_backend', 'triton (not recorded)')}"
          " — only the generation test runs through it\n")
    print(f"Text: {metrics['hf_fp32']['tokens']} tokens of "
          f"{ls_side.get('data_source', {}).get('text', '?')}; "
          f"fp32 perplexity {ppl32:.4f}\n")
    print(f"{'system':<22}{'perplexity':>12}{'vs fp32':>10}{'top-1 agree':>13}"
          f"{'KL vs fp32':>12}   verdict (deviation from fp32, vs HF fp16's)")
    ref_ppl_dev = abs(ref["perplexity"] - ppl32)
    ref_top1_dev = 1 - ref["top1_agree"]
    for s in SYSTEMS:
        m = metrics[s]
        dev = abs(m["perplexity"] - ppl32)
        kl = f"{m['kl_mean']:.2e}" if m.get("kl_mean") is not None else "n/a"
        v_top1 = verdict(1 - m["top1_agree"], ref_top1_dev) if s not in ("hf_fp32", "hf_fp16") else ""
        print(f"{s:<22}{m['perplexity']:>12.4f}{(m['perplexity'] / ppl32 - 1):>+10.3%}"
              f"{m['top1_agree']:>13.3%}{kl:>12}   {v_top1}")

    n_items = len(preds["hf_fp32"])
    print(f"\nTask: ARC-Easy, {n_items} items (accuracy noise ~±"
          f"{1.96 * math.sqrt(0.25 / max(1, n_items)):.1%} at 95%)\n")
    print(f"{'system':<22}{'acc':>8}{'acc_norm':>10}{'answers changed vs fp32':>26}")
    truth = [p[1] for p in preds["hf_fp32"]]
    for s in SYSTEMS:
        changed = sum(p[1] != t for p, t in zip(preds[s], truth))
        print(f"{s:<22}{metrics[s]['mc_acc']:>8.3f}{metrics[s]['mc_norm']:>10.3f}"
              f"{changed:>14} of {n_items}")

    base = gens["hf_fp16"]
    n_new = len(base[0]) if base else 0
    print(f"\nGeneration: {len(base)} prompts x {n_new} greedy tokens, against HF fp16\n")
    print(f"{'system':<22}{'identical':>11}{'median first diff':>19}{'mean':>8}")
    for s in ("latentserve_unfused", "latentserve_fused", "vllm"):
        div = [first_divergence(a, b) for a, b in zip(gens[s], base)]
        same = sum(d >= n_new for d in div)
        print(f"{s:<22}{same:>6}/{len(div):<4}{statistics.median(div):>19.0f}"
              f"{statistics.fmean(div):>8.1f}")
    div = [first_divergence(a, b) for a, b in zip(gens["latentserve_fused"], gens["vllm"])]
    print(f"{'fused vs vllm':<22}{sum(d >= n_new for d in div):>6}/{len(div):<4}"
          f"{statistics.median(div):>19.0f}{statistics.fmean(div):>8.1f}")
    print("\nGreedy decoding amplifies any difference: two correct fp16 implementations "
          "diverge\nonce a near-tie flips, so compare systems' divergence *positions*, "
          "not exact-match alone.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", choices=["latentserve", "vllm"])
    p.add_argument("--compare", action="store_true")
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--chunks", type=int, default=16, help="Wikitext chunks to score")
    p.add_argument("--chunk-len", type=int, default=1024)
    p.add_argument("--mc-items", type=int, default=300)
    p.add_argument("--gen-prompts", type=int, default=32)
    p.add_argument("--gen-prompt-len", type=int, default=128)
    p.add_argument("--gen-new-tokens", type=int, default=128)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--score-slice", type=int, default=128,
                   help="positions turned into logits at a time; bounds peak memory")
    p.add_argument("--synthetic", action="store_true", help="random tokens; smoke test only")
    p.add_argument("--decode-backend", default=None, choices=["triton", "cuda"],
                   help="LatentServe's decode attention kernel; recorded with the results")
    p.add_argument("--results-dir", default="results/raw/quality")
    args = p.parse_args()

    if args.compare:
        return compare(args)
    if args.system == "vllm":
        run_vllm_arm(args)
        return 0
    if args.system == "latentserve":
        from config import load_config
        from kernels.gqa import paged_decode as _pd

        if args.decode_backend:
            _pd.set_decode_backend(args.decode_backend)
        from model.qwen import QwenReference

        cfg = load_config(args.config)
        device = "cuda:0"

        def load_reference(dtype):
            return QwenReference(model_name=cfg.model.name, dtype=dtype, device=device).load()

        run_latentserve_arm(args, load_reference)
        return 0
    p.error("give --system latentserve, --system vllm, or --compare")


if __name__ == "__main__":
    sys.exit(main())