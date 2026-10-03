"""
Which vLLM attention backends run on this GPU, and how fast do they decode?

    python -m benchmarks.runners.vllm_backend_probe
    python -m benchmarks.runners.vllm_backend_probe --backends auto FLASHINFER

The LatentServe-vs-vLLM comparison so far is against vLLM's *default*
configuration. On a T4 the attention backend is the setting that matters:
FlashAttention needs Ampere, so vLLM falls back to a Triton backend — and
Phase 12 found that Triton compiles attention without tensor cores on this
GPU. A different backend could close much of LatentServe's lead, so the
honest headline needs the best one that actually runs.

Each candidate runs in its own process (the backend is fixed when vLLM's
engine starts). For each, three facts:

  * whether it starts at all, and the error if not;
  * what vLLM's own log says it used — vLLM may substitute another backend,
    and the requested name is not evidence of what ran;
  * decode milliseconds per step, from running the same prompts for 16 and
    then 80 new tokens: the difference cancels the prefill.

Two points, kept small because vLLM's 8K prefill is slow on a T4: batch 16
at 2K (attention-heavy: Triton vs LatentServe's CUDA kernel was 36 vs 21 ms
there) and batch 1 at 8K.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time

CANDIDATES = [
    # attention backends, with vLLM's other defaults
    "auto", "TRITON_ATTN", "FLASHINFER", "FLEX_ATTENTION", "TORCH_SDPA", "XFORMERS", "FLASH_ATTN",
    # the default backend with the knobs that could move batch-1 latency
    "auto+piecewise", "auto+full_graphs", "auto+async", "auto+full_graphs+async",
]
POINTS = ((16, 2048), (1, 8192))
SHORT, LONG = 16, 80


def child(variant: str, config: str) -> int:
    from comparisons.vllm.runner import VLLMRunner, parse_variant
    from config import load_config

    cfg = load_config(config)
    backend, engine_args = parse_variant(variant)
    max_len = max(ctx for _, ctx in POINTS) + LONG + 16
    runner = VLLMRunner(cfg.model.name, max_model_len=max_len,
                        max_num_seqs=max(b for b, _ in POINTS),
                        attention_backend=backend, engine_args=engine_args)
    from vllm import SamplingParams

    try:
        from vllm.inputs import TokensPrompt

        wrap = lambda ids: TokensPrompt(prompt_token_ids=ids)  # noqa: E731
    except ImportError:  # pragma: no cover
        wrap = lambda ids: {"prompt_token_ids": ids}  # noqa: E731

    rng = random.Random(0)
    llm = runner.llm
    llm.generate([wrap([rng.randrange(1000, 100000) for _ in range(64)])],
                 SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True), use_tqdm=False)
    result = {"variant": variant, "decode_ms": {}}
    for batch, ctx in POINTS:
        prompts = [wrap([rng.randrange(1000, 100000) for _ in range(ctx)]) for _ in range(batch)]
        walls = {}
        for n in (SHORT, LONG):
            params = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
            t0 = time.perf_counter()
            llm.generate(prompts, params, use_tqdm=False)
            walls[n] = time.perf_counter() - t0
        result["decode_ms"][f"{batch}x{ctx}"] = (walls[LONG] - walls[SHORT]) / (LONG - SHORT) * 1000
    print("PROBE_RESULT " + json.dumps(result), flush=True)
    return 0


def reported_backends(log: str) -> list[str]:
    """What vLLM says it used, from its startup log: lines like
    'Using FlashInfer backend' or 'Using TRITON_ATTN backend'."""
    found = []
    for line in log.splitlines():
        m = re.search(r"Using\s+(.{1,60}?)\s+(?:attention\s+)?backend", line, re.IGNORECASE)
        if m:
            name = m.group(1).strip().strip(".:")
            if name and name not in found:
                found.append(name)
    return found


def reported_graph_mode(log: str) -> str | None:
    """The CUDA-graph mode vLLM printed in its startup config, if it did."""
    m = re.search(r"cudagraph_mode['\"]?\s*[:=]\s*[<'\"]?(?:CUDAGraphMode\.)?([A-Z_]+)", log)
    return m.group(1) if m else None


def failure_reason(log: str) -> str:
    for line in reversed(log.strip().splitlines()):
        if re.search(r"(Error|Exception|error:|not supported|Invalid)", line):
            return line.strip()[:110]
    tail = log.strip().splitlines()
    return tail[-1][:110] if tail else "(no output)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    p.add_argument("--backends", "--variants", dest="backends", nargs="+", default=CANDIDATES,
                   help="variants: BACKEND[+knob...], knobs: full_graphs, piecewise, async")
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--timeout", type=int, default=1200)
    args = p.parse_args()
    if args.child:
        return child(args.child, args.config)

    rows = []
    for backend in args.backends:
        print(f"[probe] {backend} ...", flush=True)
        env = dict(os.environ)
        env.pop("VLLM_ATTENTION_BACKEND", None)
        t0 = time.perf_counter()
        try:
            out = subprocess.run([sys.executable, "-m", "benchmarks.runners.vllm_backend_probe",
                                  "--child", backend, "--config", args.config],
                                 capture_output=True, text=True, timeout=args.timeout, env=env)
            log = out.stdout + "\n" + out.stderr
        except subprocess.TimeoutExpired as e:
            log = f"{e.stdout or ''}\n{e.stderr or ''}\nTIMEOUT after {args.timeout}s"
        result = None
        for line in log.splitlines():
            if line.startswith("PROBE_RESULT "):
                result = json.loads(line[len("PROBE_RESULT "):])
        rows.append({"backend": backend, "ok": result is not None,
                     "reported": reported_backends(log), "graph_mode": reported_graph_mode(log),
                     "result": result,
                     "reason": None if result else failure_reason(log),
                     "seconds": time.perf_counter() - t0})
        r = rows[-1]
        print(f"        {'ran' if r['ok'] else 'FAILED'} in {r['seconds']:.0f}s; vLLM reported: "
              f"{', '.join(r['reported']) or '(nothing)'}"
              + ("" if r["ok"] else f"\n        {r['reason']}"), flush=True)

    keys = [f"{b}x{c}" for b, c in POINTS]
    print(f"\n{'variant':<26}{'status':<8}" + "".join(f"{'decode ms ' + k:>18}" for k in keys)
          + f"{'graph mode':>20}   vLLM reported")
    for r in rows:
        ms = r["result"]["decode_ms"] if r["ok"] else {}
        print(f"{r['backend']:<26}{'ok' if r['ok'] else 'failed':<8}"
              + "".join(f"{(f'{ms[k]:.1f}' if k in ms else '-'):>18}" for k in keys)
              + f"{(r['graph_mode'] or '-'):>20}   {', '.join(r['reported']) or '-'}")
    ran = [r for r in rows if r["ok"]]
    if ran:
        # Per point: a knob that helps batch 1 need not help batch 16, and the
        # backend that wins at batch 16 need not win at batch 1.
        print()
        for k in keys:
            best = min(ran, key=lambda r: r["result"]["decode_ms"][k])
            auto = next((r for r in ran if r["backend"] == "auto"), None)
            gain = ""
            if auto and best is not auto:
                a, b = auto["result"]["decode_ms"][k], best["result"]["decode_ms"][k]
                gain = f"  ({(a - b) / a:.1%} faster than the default)"
            print(f"fastest at {k}: {best['backend']}{gain}")
        print("\nDifferences under ~2% are within this probe's noise (one run per point).\n"
              "For the full comparison: --vllm-variant <winner>")
    if any("flashinfer" in (r["reason"] or "").lower() for r in rows):
        print("\nFlashInfer failed on an import or build: `pip install flashinfer-python` and "
              "probe it again with --backends FLASHINFER.")
    return 0


if __name__ == "__main__":
    sys.exit(main())