"""
Final sweep, stage 1: everything on one GPU, under identical conditions.

    python -m benchmarks.runners.sweep_stage1                  # everything; resumable
    python -m benchmarks.runners.sweep_stage1 --sections A B   # a subset
    python -m benchmarks.runners.sweep_stage1 --report         # tables from saved cells

Sections (docs/sweep_stage1.md has the plan and the predictions):

  A  decode steps — per-step latency at the engine level, batch 1-32 x
     context 2K-32K wherever it fits: LatentServe dense, sparse 50%, sparse
     37.5%, INT8; vLLM
  B  prefill — time to first token for one request, prompts 1K-32K:
     LatentServe dense and INT8; vLLM
  C  serving — burst; multi-turn chat with prefix caching on and off;
     Phase 17's varying traffic (quiet / burst / medium / quiet, in wall-clock
     time); and latency-versus-load curves: a capacity probe per
     configuration, then open-loop Poisson arrivals at nine rates from 10% to
     110% of that capacity

Every measurement is a cell, saved to its own JSON file the moment it
finishes; a rerun skips saved cells, so an interrupted session resumes where
it stopped (--force remeasures). Cells run in engine groups — LatentServe,
then vLLM with prefix caching on, then off — because one engine fits on the
GPU at a time and vLLM cannot switch prefix caching without reloading.
Within a group, rounds alternate configuration order. Because the groups run
at different times, a drift sentinel reruns one LatentServe cell at the end;
the report gives the drift as a percentage.

Ground rules, from the rest of the project: both engines get the same
concurrency limit, context limit, prompts and seeds; every worker warms up
before any timed cell; CUDA-graph capture is excluded from throughput;
vLLM does not detokenize; each open-loop point reports its request count.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
import zlib
from pathlib import Path

# ------------------------------------------------------------ the matrix ---

LS = {  # LatentServe configurations: worker options on top of the base
    "ls-dense": {},
    "ls-sparse50": {"sparse": 0.5},
    "ls-sparse37.5": {"sparse": 0.375},
    "ls-adaptive": {"tier": "relaxed"},
    "ls-int8": {"kv_dtype": "int8"},
}
A_LS = ["ls-dense", "ls-sparse50", "ls-sparse37.5", "ls-int8"]
A_BATCHES = (1, 4, 8, 16, 32)
# Section A's concurrency limit, for both engines: its largest batch. The
# serving workloads keep 16. (The first version applied 16 everywhere, so
# batch 32 could never run and was misreported as not fitting.)
A_MAX_RUNNING = max(A_BATCHES)
# Qwen2.5-1.5B's limit is 32,768 positions (max_position_embeddings) for the
# prompt *and* every generated token together. A "32,768" cell therefore fills
# the cache up to the limit: its prompt leaves room for the tokens decoded on
# top (a_prompt). The first version started the prompt at 32,768 and decoded
# past the end — LatentServe, which never checked, ran past the trained range;
# vLLM, which does, refused to start.
A_CONTEXTS = (2048, 8192, 16384, 32768)
B_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)   # 32,768: a 32,767-token prompt + 1 token
B_LS = ["ls-dense", "ls-int8"]
C_LS = ["ls-dense", "ls-sparse37.5", "ls-adaptive", "ls-int8"]
FRACTIONS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0, 1.1)
MAX_SEQ = 32768            # the model's own limit (max_position_embeddings), both engines
A_WARMUP, A_HEADROOM = 8, 16   # decoded beyond the timed steps (see _measure_steps)


def a_headroom(batch: int, ctx: int) -> int:
    """Tokens beyond warm-up and timed steps: enough for the slowest admission.
    vLLM admits a big batch over ~batch x ceil(ctx / 8192) steps while
    decoding the early requests; a fixed 16 let four vLLM cells run out of
    tokens mid-timing."""
    return A_HEADROOM + batch * -(-ctx // 8192)


def a_prompt(ctx: int, steps: int, batch: int = 1) -> int:
    """Section A's prompt for a context label: the label itself, unless the
    tokens decoded on top would pass the limit — then the prompt that ends
    exactly at it."""
    return min(ctx, MAX_SEQ - (A_WARMUP + steps + a_headroom(batch, ctx)))


def b_prompt(length: int) -> int:
    """Section B's prompt: the length itself, or 32,767 for 32,768 — the one
    output token needs a position too."""
    return min(length, MAX_SEQ - 1)


def cell_id(section: str, what: str, config: str, rnd: int) -> str:
    return f"{section}|{what}|{config}|r{rnd}"


def cell_path(results: Path, cid: str) -> Path:
    return results / (cid.replace("|", "__") + ".json")


# ------------------------------------------------------------- workloads ---

def _tok(rng, n):
    return [rng.randrange(1000, 100000) for _ in range(n)]


def sequential(length: int, repeats: int, seed: int) -> dict:
    """One request at a time, each prompt distinct (no prefix-cache hits)."""
    rng = random.Random(f"B{seed}-{length}")
    return {"kind": "sequential",
            "requests": [{"rid": i, "prompt": _tok(rng, length), "max_new": 1}
                         for i in range(repeats)]}


def openloop(seed: int, rate_rps: float, n: int, tag: str = "") -> dict:
    """Poisson arrivals at `rate_rps`; prompts 1K/2K/4K, 128 output tokens."""
    rng = random.Random(f"open{seed}-{tag}-{rate_rps:.6f}")
    t, reqs = 0.0, []
    for i in range(n):
        reqs.append({"rid": i, "prompt": _tok(rng, rng.choice((1024, 2048, 4096))),
                     "max_new": 128, "at": t})
        t += rng.expovariate(rate_rps)
    return {"kind": "open", "requests": reqs}


def probe(seed: int, n: int = 48) -> dict:
    """The open-loop request mix, all at once: saturation throughput."""
    rng = random.Random(f"probe{seed}")
    return {"kind": "burst",
            "requests": [{"rid": i, "prompt": _tok(rng, rng.choice((1024, 2048, 4096))),
                          "max_new": 128} for i in range(n)]}


def varying(seed: int) -> dict:
    """Phase 17's traffic in wall-clock time: quiet, a burst of long prompts,
    a medium stretch, quiet again."""
    rng = random.Random(f"vary{seed}")
    phases = (("quiet", 12, 6.0, (1024, 2048, 4096), (128,)),
              ("burst", 16, 0.0, (8192, 16384, 30000), (128, 256)),
              ("medium", 12, 1.5, (2048, 4096, 8192, 16384), (128,)),
              ("quiet", 8, 6.0, (1024, 2048, 4096), (128,)))
    t, i, reqs = 0.0, 0, []
    for name, n, every, lengths, outs in phases:
        for _ in range(n):
            reqs.append({"rid": i, "prompt": _tok(rng, rng.choice(lengths)),
                         "max_new": rng.choice(outs), "at": t, "phase": name})
            i += 1
            t += every
        t += 40.0 if name == "burst" else every
    return {"kind": "open", "requests": reqs}


def probe_capacity(res: dict) -> float:
    """Requests per second at saturation, graph capture excluded like every
    other throughput in the sweep."""
    span = res["makespan_s"] - max(res.get("capture_s") or [0.0])
    return res["requests"] / span


def points_for(capacity_rps: float) -> list:
    """(fraction, rate, request count) for the load curve: ~4 minutes of
    arrivals per point, at least 16 requests and at most 80."""
    out = []
    for f in FRACTIONS:
        rate = f * capacity_rps
        out.append((f, rate, int(min(80, max(16, round(rate * 240))))))
    return out


# ---------------------------------------------------------------- running ---

class Runner:
    def __init__(self, args):
        self.args = args
        self.results = Path(args.results_dir)
        self.results.mkdir(parents=True, exist_ok=True)
        self.ran = 0

    def done(self, cid: str) -> bool:
        return cell_path(self.results, cid).exists() and not self.args.force

    def save(self, cid: str, payload: dict) -> None:
        payload = {"cell": cid, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
        cell_path(self.results, cid).write_text(json.dumps(payload, indent=1, default=str))
        self.ran += 1

    def load(self, cid: str) -> dict:
        return json.loads(cell_path(self.results, cid).read_text())


def base_opts(args) -> dict:
    return {"config": args.config, "max_running": args.max_running, "max_seq_len": MAX_SEQ,
            "headroom_gb": args.headroom_gb, "kv_dtype": "fp16", "prefix_caching": False,
            "policy_table": args.policy_table}


def _alternate(items: list, rnd: int) -> list:
    return items if rnd % 2 == 0 else list(reversed(items))


def run_group(run: Runner, cluster, configs: dict, sections, base: dict, chat_only: bool,
              backend: str) -> None:
    """Every cell of one engine group, in a fixed, resumable order. `chat_only`
    is the vLLM prefix-caching-off group: it exists only for the chat cells."""
    from benchmarks.runners import phase18_replicas as pr
    from runtime.router import Router, percentiles

    a = run.args
    names = list(configs)

    def opts_for(name, **extra):
        return dict(base, **configs[name], **extra)

    def workload_cell(cid, name, wl, **extra):
        if run.done(cid):
            return
        o = opts_for(name, **extra)
        cluster.reset(o)
        res = pr.run_workload(cluster, Router(1), wl, 1)
        busy, capture = cluster.reset(o)
        res["busy_frac"] = [b / res["makespan_s"] for b in busy[:1]]
        res["capture_s"] = capture[:1]
        res["out_tok_s_ex_capture"] = pr.ex_capture(res)
        res.pop("tokens_per_gpu", None)
        run.save(cid, {"result": res, "config": name, "backend": backend})
        print(f"  {cid:<44} {res['out_tok_s_ex_capture']:8.1f} tok/s  TTFT p50/p99 "
              f"{res['ttft_ms']['p50']:.0f}/{res['ttft_ms']['p99']:.0f} ms  "
              f"n={res['requests']}", flush=True)

    if "A" in sections and not chat_only:
        for ctx in A_CONTEXTS:
            for b in A_BATCHES:
                for rnd in range(a.rounds):
                    for name in _alternate([n for n in names if n in A_LS or n == "vllm"], rnd):
                        cid = cell_id("A", f"b{b}-c{ctx}", name, rnd)
                        if run.done(cid):
                            continue
                        cluster.reset(opts_for(name, max_running=A_MAX_RUNNING))
                        m = cluster.measure_steps(0, batch=b, ctx=a_prompt(ctx, a.steps, b),
                                                  steps=a.steps, warmup=A_WARMUP,
                                                  headroom=a_headroom(b, ctx),
                                                  seed=zlib.crc32(cid.encode()) & 0xFFFF)
                        cluster.reset(opts_for(name))
                        if m.get("fits") and m.get("step_ms"):
                            m["pct"] = percentiles(m["step_ms"], (50, 90, 95, 99))
                            m["tok_s"] = b * 1000 / statistics.median(m["step_ms"])
                        run.save(cid, {"result": m, "config": name, "backend": backend,
                                       "batch": b, "ctx": ctx,
                                       "prompt_tokens": a_prompt(ctx, a.steps, b)})
                        note = (f"p50 {m['pct']['p50']:.2f} ms  p99 {m['pct']['p99']:.2f} ms"
                                if m.get("pct") else f"skipped: {m.get('reason', 'did not fit')}")
                        if m.get("timed_short"):
                            note += f"  TIMED SHORT ({len(m['step_ms'])} of {a.steps} steps)"
                        print(f"  {cid:<44} {note}", flush=True)

    if "B" in sections and not chat_only:
        for length in B_LENGTHS:
            for name in [n for n in names if n in B_LS or n == "vllm"]:
                # The A/B limit on both engines: vLLM's A/B engine is built
                # with it, so LatentServe's B cells use it too.
                workload_cell(cell_id("B", f"L{length}", name, 0), name,
                              sequential(b_prompt(length), a.b_repeats, a.seed),
                              max_running=A_MAX_RUNNING)

    if "C" in sections:
        cs = [n for n in names if n in C_LS or n == "vllm"]
        if not chat_only:
            for rnd in range(a.rounds):
                for name in _alternate(cs, rnd):
                    workload_cell(cell_id("C", "burst", name, rnd), name, pr.burst(a.seed))
                for name in _alternate(cs, rnd):
                    workload_cell(cell_id("C", "varying", name, rnd), name, varying(a.seed))
        chat_cfgs = []
        if backend == "latentserve":
            chat_cfgs = [("ls-dense-prefix-off", "ls-dense", {"prefix_caching": False}),
                         ("ls-dense-prefix-on", "ls-dense", {"prefix_caching": True})]
        else:
            tag = "vllm-prefix-on" if base.get("vllm_prefix", True) else "vllm-prefix-off"
            chat_cfgs = [(tag, "vllm", {})]
        for rnd in range(a.rounds):
            for tag, name, extra in _alternate(chat_cfgs, rnd):
                workload_cell(cell_id("C", "chat", tag, rnd), name, pr.chat(a.seed), **extra)
        if not chat_only and not a.skip_curves:
            for name in cs:                               # latency vs load: one round
                pid = cell_id("C", "probe", name, 0)
                workload_cell(pid, name, probe(a.seed))
                cap = run.load(pid)["result"]
                capacity = probe_capacity(cap)
                for f, rate, n in points_for(capacity):
                    workload_cell(cell_id("C", f"load{f:.2f}", name, 0), name,
                                  openloop(a.seed, rate, n, tag=name))


def ensure_policy_table(args) -> None:
    if Path(args.policy_table).exists():
        return
    print(f"no step table at {args.policy_table}: calibrating the adaptive policy first "
          "(phase17_calibrate, ~15 min)", flush=True)
    subprocess.run([sys.executable, "-m", "benchmarks.runners.phase17_calibrate",
                    "--results-dir", str(Path(args.policy_table).parent)], check=True)


def sentinel(run: Runner, cluster, base: dict) -> None:
    """Rerun one LatentServe cell after every group: drift between the groups."""
    from benchmarks.runners import phase18_replicas as pr
    from runtime.router import Router

    k = 0
    while cell_path(run.results, cell_id("S", "burst", "ls-dense", k)).exists():
        k += 1
    cluster.reset(dict(base))
    res = pr.run_workload(cluster, Router(1), pr.burst(run.args.seed), 1)
    busy, capture = cluster.reset(dict(base))
    res["capture_s"] = capture[:1]
    res["out_tok_s_ex_capture"] = pr.ex_capture(res)
    run.save(cell_id("S", "burst", "ls-dense", k), {"result": res, "config": "ls-dense",
                                                    "backend": "latentserve"})
    print(f"  sentinel {k}: {res['out_tok_s_ex_capture']:.1f} tok/s", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sections", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    p.add_argument("--backends", nargs="+", default=["latentserve", "vllm"],
                   choices=["latentserve", "vllm"])
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--steps", type=int, default=48, help="timed decode steps per section-A cell")
    p.add_argument("--b-repeats", type=int, default=8)
    p.add_argument("--max-running", type=int, default=16)
    p.add_argument("--headroom-gb", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-curves", action="store_true", help="omit the latency-vs-load curves")
    p.add_argument("--force", action="store_true", help="remeasure cells already saved")
    p.add_argument("--report", action="store_true", help="only print tables from saved cells")
    p.add_argument("--policy-table", default="results/raw/phase17/step_table.json")
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--results-dir", default="results/raw/sweep1")
    args = p.parse_args()
    if args.report:
        print(report(Path(args.results_dir)))
        return 0

    from benchmarks.runners import phase18_replicas as pr

    run = Runner(args)
    base = base_opts(args)
    t0 = time.perf_counter()
    if "latentserve" in args.backends:
        if "C" in args.sections:
            ensure_policy_table(args)
        print("LatentServe group: starting one worker and warming up...", flush=True)
        cluster = pr.Cluster(1, "latentserve", base)
        try:
            cluster.warmup(dict(base, prefix_caching=True))
            run_group(run, cluster, LS, args.sections, base, False, "latentserve")
        finally:
            cluster.stop()
    if "vllm" in args.backends:
        # vLLM fixes max_num_seqs at startup, so sections A/B (limit 32) and the
        # serving cells (limit 16, prefix caching on, then off) need separate
        # engines.
        groups = []
        ab = [s for s in args.sections if s in ("A", "B")]
        if ab:
            groups.append(("sections A/B", ab, dict(base, max_running=A_MAX_RUNNING), False))
        if "C" in args.sections:
            groups.append(("serving, prefix caching on", ["C"], dict(base, vllm_prefix=True), False))
            groups.append(("serving, prefix caching off", ["C"], dict(base, vllm_prefix=False), True))
        for label, secs, group_base, chat_only in groups:
            print(f"vLLM group ({label}): starting...", flush=True)
            cluster = pr.Cluster(1, "vllm", group_base)
            try:
                cluster.warmup(group_base)
                run_group(run, cluster, {"vllm": {}}, secs, group_base, chat_only, "vllm")
            finally:
                cluster.stop()
    if run.ran and "vllm" in args.backends and "latentserve" in args.backends \
            and "C" in args.sections:
        print("drift sentinel: rerunning LatentServe's dense burst...", flush=True)
        cluster = pr.Cluster(1, "latentserve", base)
        try:
            cluster.warmup(dict(base, prefix_caching=True))
            sentinel(run, cluster, base)
        finally:
            cluster.stop()
    print(f"\n{run.ran} cells measured in {(time.perf_counter() - t0) / 60:.0f} min; "
          f"results in {args.results_dir}\n")
    print(report(Path(args.results_dir)))
    return 0


# ----------------------------------------------------------------- report ---

def _cells(results: Path) -> list:
    out = []
    for f in sorted(results.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "cell" in d:
            sec, what, cfg, rnd = d["cell"].split("|")
            out.append({**d, "sec": sec, "what": what, "cfg": cfg, "rnd": int(rnd[1:])})
    return out


def unreliable_skip(cfg: str, reason: str) -> bool:
    """Was this skip an artifact of the sweep, not a capacity limit?

    A LatentServe pool was sized from whatever GPU memory happened to be free
    when its engine was built, so earlier cells could shrink it (pools of 64 to
    1,600 blocks were seen against ~15,000 for fp16 and ~29,000 for INT8). And
    skips written by the first version of the sweep carry the old message.
    Those cells were not measured; calling them "did not fit" would claim a
    hardware limit that was never established."""
    import re

    reason = reason or ""
    if reason == "the batch could not all decode at once":          # the first version's wording
        return True
    m = re.search(r"pool has (\d+)", reason)
    if m:
        return int(m.group(1)) < (26000 if cfg == "ls-int8" else 14000)
    return False


def _med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(xs) if xs else float("nan")


def report(results: Path) -> str:
    cells = _cells(results)
    if not cells:
        return f"(no saved cells in {results})"
    lines = [f"# Sweep stage 1 — {len(cells)} cells", ""]

    a = [c for c in cells if c["sec"] == "A"]
    if a:
        cfgs = [k for k in A_LS + ["vllm"] if any(c["cfg"] == k for c in a)]
        lines += ["## A. Decode step, engine level (median over rounds; p50 / p99 ms)", "",
                  "| batch | ctx | " + " | ".join(cfgs) + " |",
                  "| ---: | ---: | " + " | ".join("---:" for _ in cfgs) + " |"]
        for ctx in A_CONTEXTS:
            for b in A_BATCHES:
                row = []
                for k in cfgs:
                    rs = [c["result"] for c in a if c["cfg"] == k and c["what"] == f"b{b}-c{ctx}"]
                    ok = [r for r in rs if r.get("pct")]
                    if any(r.get("timed_short") for r in ok):
                        row.append("timed short")
                    elif ok:
                        row.append(f"{_med(r['pct']['p50'] for r in ok):.2f} / "
                                   f"{_med(r['pct']['p99'] for r in ok):.2f}")
                    elif rs and any(unreliable_skip(k, r.get("reason")) for r in rs):
                        row.append("not measured*")
                    else:
                        row.append("did not fit" if rs else "-")
                if any(x != "-" for x in row):
                    lines.append(f"| {b} | {ctx} | " + " | ".join(row) + " |")
        lines += ["", "The 32768 rows fill the cache to the model's limit: the prompt is "
                  "32,768 minus the tokens decoded on top (each cell records it).", ""]
        skipped = [c for c in a if not c["result"].get("fits")]
        if skipped:
            flagged = [c for c in skipped if unreliable_skip(c["cfg"], c["result"].get("reason"))]
            if flagged:
                lines += ["\\* **not measured**: skipped because of the sweep's own bugs, not a "
                          "hardware limit — a LatentServe pool sized from whatever memory was "
                          "free when its engine was built (earlier cells shrank it), or a skip "
                          "recorded by the first version. These cells show no result either way.", ""]
            lines += ["Skipped cells and their recorded reasons:", ""]
            for c in sorted(skipped, key=lambda c: (c["cfg"], c["what"])):
                tag = " (artifact)" if c in flagged else ""
                lines.append(f"- {c['what']} {c['cfg']} r{c['rnd']}: "
                             f"{c['result'].get('reason', 'did not fit')}{tag}")
            lines.append("")

    bcells = [c for c in cells if c["sec"] == "B"]
    if bcells:
        cfgs = [k for k in B_LS + ["vllm"] if any(c["cfg"] == k for c in bcells)]
        lines += ["## B. Time to first token, one request (p50 / p95 / p99 ms; n per cell)", "",
                  "| prompt | " + " | ".join(cfgs) + " |", "| ---: | " + " | ".join("---" for _ in cfgs) + " |"]
        for L in B_LENGTHS:
            row = []
            for k in cfgs:
                r = next((c["result"] for c in bcells if c["cfg"] == k and c["what"] == f"L{L}"), None)
                row.append(f"{r['ttft_ms']['p50']:.0f} / {r['ttft_ms']['p95']:.0f} / "
                           f"{r['ttft_ms']['p99']:.0f} (n={r['requests']})" if r else "-")
            lines.append(f"| {L} | " + " | ".join(row) + " |")
        lines += ["", "The 32768 row is a 32,767-token prompt: its one output token "
                  "needs the last position.", ""]

    c_cells = [c for c in cells if c["sec"] == "C" and c["what"] in ("burst", "varying", "chat")]
    if c_cells:
        lines += ["## C. Serving workloads (median over rounds)", "",
                  "| workload | config | tok/s | spread | TTFT p50 / p95 / p99 ms | TPOT p50 / p99 ms "
                  "| E2E p50 / p99 s | hit |",
                  "| --- | --- | ---: | ---: | --- | --- | --- | ---: |"]
        for what in ("burst", "varying", "chat"):
            for cfg in sorted({c["cfg"] for c in c_cells if c["what"] == what}):
                rs = [c["result"] for c in c_cells if c["what"] == what and c["cfg"] == cfg]
                tps = [r["out_tok_s_ex_capture"] for r in rs]
                spread = (max(tps) - min(tps)) / _med(tps) if len(tps) > 1 else float("nan")
                p = lambda key, q: _med(r[key][q] for r in rs)  # noqa: E731
                lines.append(
                    f"| {what} | {cfg} | {_med(tps):.1f} | "
                    f"{'-' if math.isnan(spread) else f'{spread:.1%}'} | "
                    f"{p('ttft_ms', 'p50'):.0f} / {p('ttft_ms', 'p95'):.0f} / {p('ttft_ms', 'p99'):.0f} | "
                    f"{p('tpot_ms', 'p50'):.1f} / {p('tpot_ms', 'p99'):.1f} | "
                    f"{p('e2e_ms', 'p50') / 1000:.1f} / {p('e2e_ms', 'p99') / 1000:.1f} | "
                    f"{_med(r['hit_rate'] for r in rs):.1%} |")
        lines.append("")

    curves = [c for c in cells if c["sec"] == "C" and (c["what"].startswith("load") or c["what"] == "probe")]
    if curves:
        lines += ["## C. Latency versus load (one round; p99 from fewer than 100 requests is "
                  "close to the maximum)", ""]
        for cfg in sorted({c["cfg"] for c in curves}):
            pr_ = next((c["result"] for c in curves if c["cfg"] == cfg and c["what"] == "probe"), None)
            cap = probe_capacity(pr_) if pr_ else float("nan")
            lines += [f"**{cfg}** — capacity {cap:.3f} req/s", "",
                      "| load | req/s | n | tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | E2E p50 / p99 s |",
                      "| ---: | ---: | ---: | ---: | --- | --- | --- |"]
            pts = sorted((c for c in curves if c["cfg"] == cfg and c["what"].startswith("load")),
                         key=lambda c: float(c["what"][4:]))
            for c in pts:
                r, f = c["result"], float(c["what"][4:])
                lines.append(f"| {f:.0%} | {f * cap:.3f} | {r['requests']} | {r['out_tok_s']:.1f} | "
                             f"{r['ttft_ms']['p50']:.0f} / {r['ttft_ms']['p99']:.0f} | "
                             f"{r['tpot_ms']['p50']:.1f} / {r['tpot_ms']['p99']:.1f} | "
                             f"{r['e2e_ms']['p50'] / 1000:.1f} / {r['e2e_ms']['p99'] / 1000:.1f} |")
            lines.append("")

    sent = [c for c in cells if c["sec"] == "S"]
    ref = [c for c in cells if c["sec"] == "C" and c["what"] == "burst" and c["cfg"] == "ls-dense"]
    if sent and ref:
        base = _med(c["result"]["out_tok_s_ex_capture"] for c in ref)
        lines += ["## Drift sentinel", ""]
        for c in sorted(sent, key=lambda c: c["rnd"]):
            v = c["result"]["out_tok_s_ex_capture"]
            lines.append(f"- sentinel {c['rnd']} ({c['saved']}): {v:.1f} tok/s, "
                         f"{(v - base) / base:+.1%} against the group's own dense burst")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
