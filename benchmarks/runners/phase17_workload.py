"""
Phase 17: can an adaptive policy outperform a fixed backend? (methodology Q14)

    python -m benchmarks.runners.phase17_workload       # ~25 min on a T4

One fixed, seeded mix of requests — mostly short prompts, some long ones,
varied output lengths — served through the real engine under CUDA graphs.
As requests arrive and finish, the running batch and its context drift:
the conditions a policy has to handle. Five strategies serve the identical
mix:

  dense               no sparsity
  fixed-50, fixed-37.5   one budget on every step
  adaptive-balanced   policy: dense or 50%, chosen per step
  adaptive-relaxed    policy: dense, 50% or 37.5%, chosen per step

Reported: decode throughput, which budget generated each token, and the
expected share of answers lost implied by Phase 15's per-budget rates (an
estimate: those rates are descriptive and imprecise). Prefill is dense under
every strategy, so it is timed separately rather than diluting the
comparison. Strategies alternate order across rounds so drift favours none.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

STRATEGIES = {
    "dense": dict(fixed=None, tier=None),
    "fixed-50": dict(fixed=0.5, tier=None),
    "fixed-37.5": dict(fixed=0.375, tier=None),
    "adaptive-balanced": dict(fixed=None, tier="balanced"),
    "adaptive-relaxed": dict(fixed=None, tier="relaxed"),
}
# Prompt-length mix: mostly chat-sized, a tail of long documents.
MIX = ((2048, 0.40), (4096, 0.25), (8192, 0.15), (16384, 0.12), (30000, 0.08))
OUTPUTS = (64, 128, 256)


def workload(n: int, seed: int) -> list:
    rng = random.Random(seed)
    lengths, weights = zip(*MIX)
    reqs = []
    for i in range(n):
        plen = rng.choices(lengths, weights)[0]
        reqs.append({"id": i, "prompt_len": plen, "max_new": rng.choice(OUTPUTS),
                     "seed": rng.randrange(1 << 30)})
    return reqs


def pool_blocks(model, headroom_gb: float, block_size: int = 16) -> int:
    import torch

    free, _ = torch.cuda.mem_get_info()
    spec_layers = len(model.layers)
    kv_heads, head_dim = model.shape.num_key_value_heads, model.shape.head_dim
    per_block = spec_layers * block_size * kv_heads * head_dim * 2 * 2      # K and V, fp16
    per_block += spec_layers * 2 * kv_heads * head_dim * 2                  # page bounds
    return max(64, int((free - headroom_gb * 1024**3) // per_block))


def serve(model, reqs, strategy: dict, policy_table, num_blocks: int, max_running: int) -> dict:
    import torch

    from runtime.engine import ServingEngine
    from runtime.policy import SparsePolicy, expected_answer_loss
    from runtime.request import ServedRequest

    policy = (SparsePolicy.from_json(policy_table, tier=strategy["tier"])
              if strategy["tier"] else None)
    model.set_sparse(strategy["fixed"])
    engine = ServingEngine(model, max_running=max_running,
                           max_seq_len=max(r["prompt_len"] + r["max_new"] for r in reqs) + 64,
                           block_size=16, num_blocks=num_blocks, use_cuda_graphs=True,
                           policy=policy)
    for r in reqs:
        g = torch.Generator().manual_seed(r["seed"])
        ids = torch.randint(1000, 100000, (r["prompt_len"],), generator=g).tolist()
        engine.add_request(ServedRequest(request_id=r["id"], prompt_ids=ids,
                                         max_new_tokens=r["max_new"]))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    done = engine.run()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    tokens = sum(len(r.output_ids) for r in done)
    by_ratio = dict(engine.decoder.ratio_tokens) if engine.decoder else dict(engine.ratio_tokens)
    model.set_sparse(None)
    capture_s = engine.decoder.capture_s if engine.decoder else 0.0
    decode_s = wall - engine.prefill_s - capture_s     # graph capture is a one-off
    out = {"wall_s": wall, "prefill_s": engine.prefill_s, "decode_s": decode_s,
           "capture_s": capture_s, "graphs": len(engine.decoder.graphs) if engine.decoder else 0,
           "tokens": tokens, "decode_tok_s": tokens / decode_s if decode_s > 0 else float("nan"),
           "tokens_by_ratio": {("dense" if k is None else k): v for k, v in by_ratio.items()},
           "expected_loss": expected_answer_loss(by_ratio)}
    del engine
    model.cache = None
    torch.cuda.empty_cache()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--table", default="results/raw/phase17/step_table.json",
                   help="from phase17_calibrate")
    p.add_argument("--requests", type=int, default=40)
    p.add_argument("--max-running", type=int, default=16)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headroom-gb", type=float, default=2.5)
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--results-dir", default="results/raw/phase17")
    args = p.parse_args()

    import torch

    from config import load_config
    from kernels.gqa.paged_decode import set_decode_backend
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference

    if not torch.cuda.is_available():
        print("[ERROR] needs a GPU", file=sys.stderr)
        return 1
    if not Path(args.table).exists():
        print(f"[ERROR] no step table at {args.table}: run phase17_calibrate first",
              file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    ref = QwenReference(model_name=cfg.model.name, dtype="fp16", device="cuda:0").load()
    set_decode_backend("cuda")
    reqs = workload(args.requests, args.seed)
    longest = max(r["prompt_len"] + r["max_new"] for r in reqs) + 512
    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=longest,
                                           attn_impl="triton_paged", fuse_projections=True)
    model.set_elementwise(True)
    blocks = pool_blocks(model, args.headroom_gb)
    print(f"{len(reqs)} requests ({sum(r['prompt_len'] for r in reqs)} prompt tokens); "
          f"pool {blocks} blocks = {blocks * 16} tokens; max {args.max_running} running", flush=True)

    results = {s: [] for s in args.strategies}
    for rnd in range(args.rounds):
        order = args.strategies if rnd % 2 == 0 else list(reversed(args.strategies))
        for name in order:
            r = serve(model, reqs, STRATEGIES[name], args.table, blocks, args.max_running)
            results[name].append(r)
            print(f"  round {rnd + 1} {name:<18} decode {r['decode_tok_s']:7.1f} tok/s  "
                  f"(prefill {r['prefill_s']:.0f}s, decode {r['decode_s']:.1f}s, "
                  f"{r['graphs']} graphs captured in {r['capture_s']:.1f}s, excluded)", flush=True)

    base = statistics.median(x["decode_tok_s"] for x in results.get("dense", [])) \
        if results.get("dense") else None
    print(f"\n{'strategy':<20}{'decode tok/s':>13}{'vs dense':>10}{'dense':>8}{'50%':>7}"
          f"{'37.5%':>7}{'expected loss':>15}")
    summary = {}
    for name, rs in results.items():
        tps = statistics.median(x["decode_tok_s"] for x in rs)
        share = rs[0]["tokens_by_ratio"]
        total = sum(share.values()) or 1
        loss = rs[0]["expected_loss"]
        summary[name] = {"decode_tok_s": tps, "share": share, "expected_loss": loss,
                         "rounds": rs}
        rel = f"{tps / base:.2f}x" if base else "-"
        print(f"{name:<20}{tps:>13.1f}{rel:>10}"
              + "".join(f"{share.get(k, 0) / total:>7.0%}" if k != "dense" else
                        f"{share.get(k, 0) / total:>8.0%}" for k in ("dense", 0.5, 0.375))
              + f"{loss:>14.2%}")
    print("\n'expected loss': Phase 15's net answer-loss rates per budget, weighted by the\n"
          "tokens each budget generated — an estimate, since those rates are imprecise.")
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "workload.json").write_text(json.dumps(
        {"summary": summary, "requests": reqs, "args": vars(args)}, default=str, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
