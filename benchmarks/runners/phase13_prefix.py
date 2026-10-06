"""
Phase 13 — prefix caching: shared prefixes should not be recomputed.

    python -m benchmarks.runners.phase13_prefix               # ~25 min on a T4
    python -m benchmarks.runners.phase13_prefix --kv-dtype int8

The plan (methodology §20): measure cache hit rate, TTFT, VRAM, throughput
and cache overhead, then check prefix caching stacks with the
memory-efficient KV cache. Three workloads, each served with prefix caching
off and on, under the production setup (fused projections and elementwise
ops, CUDA decode kernel, CUDA graphs):

  shared   a 2048-token system prompt + a distinct 256-token message per
           request: the canonical case
  chat     conversations whose every turn resends the whole history (system
           prompt, earlier messages, earlier replies) plus a new message; the
           next turn is sent when the reply finishes
  none     prompts that share nothing: what prefix caching costs when it
           cannot help

Requests arrive over time, in decode steps (identical traffic for both
settings). Rounds alternate the order of off/on. Graph-capture time is
excluded from decode throughput, as in Phase 17.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path


def workload(name: str, seed: int, scale: float = 1.0) -> dict:
    """{"requests": [{id, arrival, prompt, max_new, conv, turn}], "chat": bool}."""
    rng = random.Random(seed)
    tok = lambda n: [rng.randrange(1000, 100000) for _ in range(n)]  # noqa: E731
    n = max(4, int(32 * scale))
    if name == "shared":
        system = tok(2048)
        reqs = [{"id": i, "arrival": 8 * i, "prompt": system + tok(256), "max_new": 64}
                for i in range(n)]
        return {"requests": reqs, "chat": False}
    if name == "none":
        reqs = [{"id": i, "arrival": 8 * i, "prompt": tok(2304), "max_new": 64} for i in range(n)]
        return {"requests": reqs, "chat": False}
    if name == "chat":
        system = tok(512)
        convs = max(2, int(8 * scale))
        turns = 5
        msgs = {(c, t): tok(128) for c in range(convs) for t in range(turns)}
        first = [{"id": c * turns, "arrival": 20 * c, "prompt": system + msgs[(c, 0)],
                  "max_new": 128, "conv": c, "turn": 0} for c in range(convs)]
        return {"requests": first, "chat": True, "msgs": msgs, "turns": turns}
    raise ValueError(name)


def serve(model, wl: dict, prefix_caching: bool, num_blocks: int, kv_dtype: str,
          max_running: int) -> dict:
    import torch

    from runtime.engine import ServingEngine
    from runtime.request import ServedRequest

    longest = max(len(r["prompt"]) for r in wl["requests"])
    if wl["chat"]:
        longest += wl["turns"] * (128 + 128) + 64
    engine = ServingEngine(model, max_running=max_running, max_seq_len=longest + 256,
                           block_size=16, num_blocks=num_blocks, use_cuda_graphs=True,
                           kv_dtype=kv_dtype, prefix_caching=prefix_caching)
    pending = sorted(wl["requests"], key=lambda r: (r["arrival"], r["id"]))
    by_id = {}

    def make(r):
        req = ServedRequest(request_id=r["id"], prompt_ids=list(r["prompt"]),
                            max_new_tokens=r["max_new"])
        req.arrival_time = time.perf_counter()
        by_id[r["id"]] = r
        return req

    if wl["chat"]:
        def next_turn(req, eng):
            meta = by_id[req.request_id]
            t = meta["turn"] + 1
            if t >= wl["turns"]:
                return
            prompt = meta["prompt"] + list(req.output_ids) + wl["msgs"][(meta["conv"], t)]
            eng.add_request(make({"id": req.request_id + 1, "prompt": prompt, "max_new": 128,
                                  "conv": meta["conv"], "turn": t, "arrival": 0}))
        engine.on_retire = next_turn

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while pending or engine.has_work:
        while pending and pending[0]["arrival"] <= step:
            engine.add_request(make(pending.pop(0)))
        if engine.has_work:
            engine.step()
            step += 1
        else:
            step = pending[0]["arrival"]
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    done = engine.finished
    ttft = [(r.first_token_time - r.arrival_time) * 1000 for r in done]
    prompt_tokens = sum(r.prompt_len for r in done)
    hit = sum(r.prefix_hit_tokens for r in done)
    capture = engine.decoder.capture_s if engine.decoder else 0.0
    gen = sum(len(r.output_ids) for r in done)
    out = {
        "requests": len(done), "prompt_tokens": prompt_tokens, "hit_tokens": hit,
        "hit_rate": hit / prompt_tokens if prompt_tokens else 0.0,
        "prefill_ms_per_request": 1000 * engine.prefill_s / max(1, len(done)),
        "ttft_mean_ms": statistics.fmean(ttft), "ttft_p50_ms": statistics.median(ttft),
        "decode_tok_s": gen / max(1e-9, engine.decode_s - capture), "wall_s": wall,
        # `peak_used` counts blocks the prefix cache keeps after requests
        # finish (evictable on demand); `peak_live` counts only blocks live
        # requests hold — the memory pressure that matters.
        "peak_blocks": engine.cache.allocator.peak_used,
        "peak_live_blocks": (engine.prefix.peak_live if engine.prefix
                             else engine.cache.allocator.peak_used),
        "prefix": engine.prefix.stats() if engine.prefix else None,
    }
    del engine
    model.cache = None
    torch.cuda.empty_cache()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--workloads", nargs="+", default=["shared", "chat", "none"])
    p.add_argument("--kv-dtype", choices=["fp16", "int8"], default="fp16")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--scale", type=float, default=1.0, help="workload size multiplier")
    p.add_argument("--max-running", type=int, default=8)
    p.add_argument("--headroom-gb", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--results-dir", default="results/raw/phase13")
    args = p.parse_args()

    import torch

    from benchmarks.runners.phase17_workload import pool_blocks
    from config import load_config
    from kernels.gqa.paged_decode import set_decode_backend
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    if not torch.cuda.is_available():
        print("[ERROR] needs a GPU", file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    ref = QwenReference(model_name=cfg.model.name, dtype="fp16", device="cuda:0").load()
    set_decode_backend("cuda")
    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=8192, attn_impl="triton_paged",
                                           fuse_projections=True)
    model.set_elementwise(True)
    blocks = pool_blocks(model, args.headroom_gb)
    print(f"pool {blocks} blocks; kv {args.kv_dtype}; max {args.max_running} running", flush=True)

    results = {}
    for name in args.workloads:
        wl = workload(name, args.seed, args.scale)
        runs = {False: [], True: []}
        for rnd in range(args.rounds):
            for on in ((False, True) if rnd % 2 == 0 else (True, False)):
                r = serve(model, wl, on, blocks, args.kv_dtype, args.max_running)
                runs[on].append(r)
                print(f"  {name:<7} round {rnd + 1} prefix {'on ' if on else 'off'}  "
                      f"hit {r['hit_rate']:6.1%}  TTFT p50 {r['ttft_p50_ms']:8.1f} ms  "
                      f"prefill/req {r['prefill_ms_per_request']:7.1f} ms  "
                      f"decode {r['decode_tok_s']:7.1f} tok/s", flush=True)
        results[name] = runs

    print(f"\n{'workload':<9}{'prefix':<8}{'hit rate':>9}{'TTFT p50':>11}{'TTFT mean':>11}"
          f"{'prefill/req':>13}{'decode tok/s':>14}{'peak live blocks':>18}")
    for name, runs in results.items():
        for on in (False, True):
            rs = runs[on]
            med = lambda k: statistics.median(x[k] for x in rs)  # noqa: E731
            print(f"{name:<9}{'on' if on else 'off':<8}{med('hit_rate'):>9.1%}"
                  f"{med('ttft_p50_ms'):>9.1f}ms{med('ttft_mean_ms'):>9.1f}ms"
                  f"{med('prefill_ms_per_request'):>11.1f}ms{med('decode_tok_s'):>14.1f}"
                  f"{int(med('peak_live_blocks')):>18}")
        off = statistics.median(x["ttft_p50_ms"] for x in runs[False])
        on_ = statistics.median(x["ttft_p50_ms"] for x in runs[True])
        print(f"{'':<9}TTFT p50 {(1 - on_ / off):+.1%} with prefix caching\n")
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"prefix_{args.kv_dtype}.json").write_text(json.dumps(
        {k: {str(a): b for a, b in v.items()} for k, v in results.items()}, default=str, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
