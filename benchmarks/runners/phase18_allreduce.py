"""
Phase 18, experiment B — what would model parallelism across two T4s cost?

    python -m benchmarks.runners.phase18_allreduce        # ~2 min, "GPU T4 x2"

The plan (methodology §25) treats model parallelism as secondary — "do not
let this become a major dependency" — and asks about PCIe communication,
synchronisation and overhead. Rather than build tensor parallelism, this
measures what it would need: Megatron-style TP-2 does two all-reduces of the
hidden state per layer (after attention and after the MLP), so a decode step
costs 2 x 28 all-reduces of [batch, 1536] fp16.

It times NCCL all-reduce at those sizes (and a 2048-token prefill chunk),
then estimates a TP-2 decode step as half the measured single-GPU step plus
the communication. That is an *optimistic* bound: it assumes compute and
weight reads split perfectly and ignores the embedding and output layers. If
even the bound shows no gain, TP cannot pay on this hardware; if it does, it
is a ceiling, not a promise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HIDDEN, LAYERS = 1536, 28


def rank_main(rank: int, world: int, port: int, sizes, iters: int, out_q) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world)
    if rank == 0:
        out_q.put(("p2p", torch.cuda.can_device_access_peer(0, 1)))
    for label, rows in sizes:
        x = torch.randn(rows, HIDDEN, dtype=torch.float16, device="cuda")
        for _ in range(20):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            dist.all_reduce(x)
        end.record()
        torch.cuda.synchronize()
        if rank == 0:
            out_q.put(("size", label, rows, start.elapsed_time(end) / iters))
    dist.barrier()
    dist.destroy_process_group()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--port", type=int, default=29517)
    p.add_argument("--step-table", default="results/raw/phase17/step_table.json",
                   help="single-GPU step times from phase17_calibrate, for the estimate")
    p.add_argument("--results-dir", default="results/raw/phase18")
    args = p.parse_args()

    import multiprocessing as mp

    import torch

    if torch.cuda.device_count() < 2:
        print("[ERROR] needs two GPUs: select 'GPU T4 x2'", file=sys.stderr)
        return 1
    sizes = [(f"decode b{b}", b) for b in (1, 2, 4, 8, 16, 32)] + [("prefill chunk 2048", 2048)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=rank_main, args=(r, 2, args.port, sizes, args.iters, q))
             for r in range(2)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(timeout=600)
    p2p, lat = None, {}
    while not q.empty():
        m = q.get()
        if m[0] == "p2p":
            p2p = m[1]
        else:
            lat[m[1]] = (m[2], m[3])
    print(f"peer-to-peer access between the two GPUs: {p2p}\n")
    print(f"{'all-reduce':<22}{'bytes':>10}{'latency':>12}{'x56 per decode step':>22}")
    for label, (rows, ms) in lat.items():
        per_step = f"{ms * 2 * LAYERS:.2f}ms" if label.startswith("decode") else "-"
        print(f"{label:<22}{rows * HIDDEN * 2:>10}{ms * 1000:>10.1f}us{per_step:>22}")

    step = {}
    if Path(args.step_table).exists():
        for r in json.loads(Path(args.step_table).read_text())["rows"]:
            if r["ratio"] is None and r["ctx"] == 2048:
                step[r["batch"]] = r["ms"]
    if step:
        print(f"\nTP-2 estimate at 2K context (half the single-GPU step + "
              f"{2 * LAYERS} NCCL all-reduces):\n")
        print(f"{'batch':>6}{'1 GPU':>10}{'TP-2 est.':>12}{'speedup est.':>15}")
        for b in sorted(step):
            key = f"decode b{b}"
            if key not in lat:
                continue
            tp = step[b] / 2 + lat[key][1] * 2 * LAYERS
            print(f"{b:>6}{step[b]:>8.2f}ms{tp:>10.2f}ms{step[b] / tp:>14.2f}x")
        print("\nNot an upper bound. It assumes a perfect split of the compute (optimistic)\n"
              "but NCCL's all-reduce latency (pessimistic): engines with their own\n"
              "peer-to-peer all-reduce do better — vLLM's TP-2 measured 1.73x at batch 1\n"
              "against this 1.45x. See phase18_vllm_tp for the measured number.")
    else:
        print(f"\n(no step table at {args.step_table}: run phase17_calibrate for the estimate)")
    d = Path(args.results_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "allreduce.json").write_text(json.dumps(
        {"p2p": p2p, "latency_ms": {k: v[1] for k, v in lat.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
