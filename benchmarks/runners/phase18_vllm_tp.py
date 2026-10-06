"""
Phase 18, experiment B, measured on a real implementation: vLLM with the
model split across both T4s (tensor_parallel_size=2) against one T4.

    python -m benchmarks.runners.phase18_vllm_tp          # ~6 min, "GPU T4 x2"

`phase18_allreduce` estimates what tensor parallelism could gain from the
cost of the all-reduces it needs; this measures what vLLM's actual TP-2
achieves on the same hardware. The two cross-check each other: the recorded
prediction (docs/phase18_multigpu.md) is that TP-2 can cut batch-1 decode
latency only if one all-reduce costs under ~100-125 us.

Each TP setting runs in a fresh process (GPU memory from one cannot affect
the other), prefix caching off (repeated timings must not hit a cache), on
distinct random prompts of 2048 tokens. Per-token time is measured without
relying on any vLLM metrics field: generate 1 token and 257 tokens, and
divide the difference by 256. Time to first token is the 1-token run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path


def _measure(tp: int, batches, prompt_len: int, repeats: int, config: str, out_q) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    try:
        from vllm import LLM, SamplingParams

        from comparisons.vllm.runner import construct_engine
        from config import load_config

        try:
            from vllm.inputs import TokensPrompt
        except ImportError:  # pragma: no cover
            TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}  # noqa: E731

        cfg = load_config(config)
        kwargs = {"model": cfg.model.name, "dtype": "float16", "seed": 0,
                  "gpu_memory_utilization": 0.85, "max_model_len": prompt_len + 512,
                  "tensor_parallel_size": tp}
        llm, _ = construct_engine(LLM, kwargs, False, None)
        rng = random.Random(tp)

        def prompts(n):
            return [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 100000)
                                                   for _ in range(prompt_len)]) for _ in range(n)]

        def timed(n, max_tokens):
            params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
            ps = prompts(n)
            t0 = time.perf_counter()
            llm.generate(ps, params, use_tqdm=False)
            return time.perf_counter() - t0

        timed(1, 8)                                            # warm-up
        result = {}
        for b in batches:
            ttft, tpot = [], []
            for _ in range(repeats):
                t1 = timed(b, 1)
                t257 = timed(b, 257)
                ttft.append(t1 * 1000)
                tpot.append((t257 - t1) * 1000 / 256)
            result[b] = {"ttft_ms": statistics.median(ttft), "tpot_ms": statistics.median(tpot),
                         "tpot_spread_ms": max(tpot) - min(tpot)}
        out_q.put(("ok", tp, result))
    except Exception as e:  # noqa: BLE001 - report, do not hang the parent
        out_q.put(("error", tp, f"{type(e).__name__}: {e}"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    p.add_argument("--prompt-len", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--results-dir", default="results/raw/phase18")
    args = p.parse_args()

    import multiprocessing as mp

    import torch

    if torch.cuda.device_count() < 2:
        print("[ERROR] needs two GPUs: select the 'GPU T4 x2' accelerator", file=sys.stderr)
        return 1
    ctx = mp.get_context("spawn")
    results, errors = {}, {}
    for tp in (1, 2):
        q = ctx.Queue()
        proc = ctx.Process(target=_measure, args=(tp, args.batches, args.prompt_len,
                                                  args.repeats, args.config, q))
        proc.start()
        kind, _, payload = q.get(timeout=1800)
        proc.join(timeout=120)
        (results if kind == "ok" else errors)[tp] = payload
        print(f"TP={tp}: {'done' if kind == 'ok' else 'FAILED: ' + payload}", flush=True)

    if 1 in results and 2 in results:
        print(f"\n{'batch':>5}{'TP1 TTFT':>11}{'TP2 TTFT':>11}{'TP1 TPOT':>11}{'TP2 TPOT':>11}"
              f"{'TPOT TP2/TP1':>14}")
        for b in args.batches:
            a, c = results[1][b], results[2][b]
            print(f"{b:>5}{a['ttft_ms']:>9.1f}ms{c['ttft_ms']:>9.1f}ms{a['tpot_ms']:>9.2f}ms"
                  f"{c['tpot_ms']:>9.2f}ms{c['tpot_ms'] / a['tpot_ms']:>13.2f}x")
        print("\nTPOT TP2/TP1 below 1.00: splitting the model across both T4s made each token\n"
              "faster; above: the all-reduces over PCIe cost more than the halved work saved.")
    d = Path(args.results_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "vllm_tp.json").write_text(json.dumps({"results": results, "errors": errors,
                                                "args": vars(args)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
