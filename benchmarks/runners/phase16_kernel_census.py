"""
Phase 16: where do INT8's kernels go? A census of one decode step.

    python -m benchmarks.runners.phase16_kernel_census

The A/B runner reported `int8 709` after the fused write had cut INT8's
end-to-end penalty from 6-19% to 1-9%. The explanation offered — that its
counter ran an eager step without the deferred mode the graph decoder enables,
so it measured INT8's old write path — was reasoning, not evidence. This
counts, and names, the kernels of one decode step under:

  fp16                          the baseline
  int8, eager                   what the old counter measured
  int8, deferred, fused off     the timed path before Phase 16
  int8, deferred, fused on      the timed path now

and checks by name that `int8_decode_write` runs where it should and the
torch ops it replaced do not. Then it profiles one replay of a captured CUDA
graph — the path the timings run — if the profiler can see inside it.
"""

from __future__ import annotations

import argparse
import collections
import sys


def _profile(fn):
    import torch

    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return names


def _short(name: str) -> str:
    for cut in ("(", "<"):
        name = name.split(cut)[0]
    return name.replace("void ", "").replace("at::native::", "")[:60]


def eager_step(model, kv_dtype, deferred, fused, batch, ctx):
    import torch

    from kernels.cuda import int8_write as iw

    model.allocate_cache(batch, ctx + 64, paged=True, block_size=16, kv_dtype=kv_dtype)
    if deferred and hasattr(model.cache, "enable_deferred_finalize"):
        model.cache.enable_deferred_finalize()
    model.cache.reset()
    model.cache.advance(ctx, batch_size=batch)
    slots = list(range(batch))
    ids = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
    pos = torch.full((batch, 1), ctx, dtype=torch.long, device="cuda")
    before = iw.ENABLED
    iw.ENABLED = fused
    try:
        model.cache.advance(1, slots=slots)
        model.decode_forward_static(ids, pos, max_position=ctx + 63)          # warm
        model.cache.advance(1, slots=slots)
        names = _profile(lambda: model.decode_forward_static(ids, pos + 1, max_position=ctx + 63))
    finally:
        iw.ENABLED = before
    model.cache = None
    torch.cuda.empty_cache()
    return names


def graph_replay(model, kv_dtype, batch, ctx):
    """One replay of a captured decode step, as the timed runs execute it."""
    import torch

    from runtime.cuda_graph import GraphedDecoder

    model.allocate_cache(batch, ctx + 64, paged=True, block_size=16, kv_dtype=kv_dtype)
    model.cache.reset()
    model.cache.advance(ctx, batch_size=batch)
    decoder = GraphedDecoder(model)
    ids = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
    slots = list(range(batch))
    for i in range(3):                                        # capture, then replays
        decoder.step(ids, torch.full((batch, 1), ctx + i, dtype=torch.long, device="cuda"), slots)
    names = _profile(lambda: decoder.step(
        ids, torch.full((batch, 1), ctx + 3, dtype=torch.long, device="cuda"), slots))
    model.cache = None
    torch.cuda.empty_cache()
    return names


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    args = p.parse_args()

    import torch

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
    model = LatentServeQwen.from_reference(ref, max_seq_len_hint=args.ctx + 512,
                                           attn_impl="triton_paged", fuse_projections=True)
    model.set_elementwise(True)
    layers = len(model.layers)

    runs = {
        "fp16": eager_step(model, "fp16", False, True, args.batch, args.ctx),
        "int8, eager (old counter)": eager_step(model, "int8", False, True, args.batch, args.ctx),
        "int8, deferred, fused off": eager_step(model, "int8", True, False, args.batch, args.ctx),
        "int8, deferred, fused on": eager_step(model, "int8", True, True, args.batch, args.ctx),
    }
    print(f"\nKernels in one eager decode step (batch {args.batch}, ctx {args.ctx}, {layers} layers):\n")
    for name, ks in runs.items():
        fused = sum("int8_decode_write" in k for k in ks)
        print(f"  {name:<28}{len(ks):>5}   int8_decode_write x{fused}")

    base = collections.Counter(_short(k) for k in runs["fp16"])
    for name in ("int8, deferred, fused off", "int8, deferred, fused on"):
        c = collections.Counter(_short(k) for k in runs[name])
        extra = sorted(((k, c[k] - base.get(k, 0)) for k in c if c[k] != base.get(k, 0)),
                       key=lambda kv: -abs(kv[1]))
        print(f"\n  {name} vs fp16 — kernels whose count differs:")
        for k, d in extra[:14]:
            print(f"    {d:+5d}  {k}")

    print("\nOne replay of a captured CUDA graph (the path the timings run):\n")
    for kv in ("fp16", "int8"):
        ks = graph_replay(model, kv, args.batch, args.ctx)
        fused = sum("int8_decode_write" in k for k in ks)
        note = "" if ks else "   (profiler sees no kernels inside graph replays here)"
        print(f"  {kv:<6}{len(ks):>5} kernels   int8_decode_write x{fused}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())