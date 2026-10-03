"""
Phase 14 — does the GPU actually benefit from sparse attention?

    python -m benchmarks.runners.phase14_sparse_bench

The plan's central question (methodology §21): *does the GPU benefit from
the reduced computation, or does irregular memory access destroy the
theoretical speedup?* — answered with measured runtime, the indexer's cost
included, never a FLOP count.

Per layer-call, under CUDA-graph replay (production decodes under graphs,
and replay charges no Python launch overhead to anyone):

  dense     the fp16 CUDA kernel over every page
  sparse    indexer + top-k + sparse attention + merge, at 50/25/12.5/6.25%
  indexer   indexer + top-k alone: the overhead sparsity has to pay back

and the write-side cost every decode step pays: updating the page bounds
for the token just written, per layer.

The oracle study put the safe operating point at 25% of pages (KL ~0.009,
every needle found) and 12.5% as borderline.
"""

from __future__ import annotations

import argparse
import sys

RATIOS = (0.5, 0.25, 0.125, 0.0625)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--context-lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    args = p.parse_args()

    import torch

    from benchmarks.runners.phase12_diag import build_inputs, time_in_graph
    from kernels.cuda import paged_decode_cuda as pdc
    from kernels.cuda import paged_sparse as ps

    if not torch.cuda.is_available():
        print("[ERROR] a GPU benchmark", file=sys.stderr)
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}\n", flush=True)

    for name, r in ps.kernel_resources().items():
        print(f"  {name:<20} {r['regs']:>3} registers, {r['local_bytes']} B spill memory, "
              f"{r['shared_bytes']} B shared")

    # Correctness before any timing.
    (q, k, v, tables, lens), _, _ = build_inputs(torch, 2, 2048, False, ragged=[2048, 700], seed=7)
    kmin = torch.empty(k.shape[0], k.shape[2], k.shape[3], dtype=k.dtype, device=k.device)
    kmax = torch.empty_like(kmin)
    ps.rebuild_bounds(k, kmin, kmax)
    dense = pdc.paged_decode_cuda(q, k, v, tables, lens, 2048).float().clone()
    full = ps.sparse_attention(q, k, v, tables, lens, kmin, kmax, ratio=1.0).float()
    err = float((full - dense).abs().max())
    print(f"\n  correctness: sparse at 100% vs dense, max abs error {err:.2e} -> "
          + ("OK" if err < 2e-3 else "WRONG - timings below mean nothing"), flush=True)

    head = (f"\n  {'batch':>5}{'ctx':>7}{'dense ms':>10}"
            + "".join(f"{f'{r:.1%} ms (x)':>17}" for r in args.ratios)
            + f"{'indexer @25%':>14}")
    print(head)
    for batch in args.batch_sizes:
        for ctx in args.context_lengths:
            try:
                (q, k, v, tables, lens), _, _ = build_inputs(torch, batch, ctx, False)
                kmin = torch.empty(k.shape[0], k.shape[2], k.shape[3], dtype=k.dtype,
                                   device=k.device)
                kmax = torch.empty_like(kmin)
                ps.rebuild_bounds(k, kmin, kmax)
                d_ms = time_in_graph(torch, lambda: pdc.paged_decode_cuda(q, k, v, tables, lens, ctx))
                cells, idx25 = [], None
                for ratio in args.ratios:
                    s_ms = time_in_graph(torch, lambda r=ratio: ps.sparse_attention(
                        q, k, v, tables, lens, kmin, kmax, ratio=r))
                    cells.append(f"{s_ms:>9.3f} ({d_ms / s_ms:>4.2f}x)")
                    if abs(ratio - 0.25) < 1e-9:
                        kk = ps.budget(0.25, tables.shape[1])
                        idx25 = time_in_graph(torch, lambda: ps.select(
                            ps.page_scores(q, kmin, kmax, tables, lens), kk))
                print(f"  {batch:>5}{ctx:>7}{d_ms:>10.3f}" + "".join(f"{c:>17}" for c in cells)
                      + (f"{idx25:>11.3f} ms" if idx25 is not None else ""), flush=True)
                del q, k, v, tables, lens, kmin, kmax
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  {batch:>5}{ctx:>7}  out of memory: skipped", flush=True)
                torch.cuda.empty_cache()

    # The write side: page bounds updated for one token per sequence, per layer.
    b, h, d = 16, 2, 128
    kmin = torch.zeros(4096, h, d, dtype=torch.float16, device="cuda")
    kmax = torch.zeros_like(kmin)
    k_new = torch.randn(b, h, d, dtype=torch.float16, device="cuda")
    slots = torch.arange(b, device="cuda", dtype=torch.long) * 16 + 5
    u_ms = time_in_graph(torch, lambda: ps.update_bounds(k_new, slots, kmin, kmax))
    print(f"\n  write side: page-bound update, batch 16, per layer: {u_ms * 1000:.1f} us "
          f"(x28 layers = {u_ms * 28:.3f} ms per decode step)")
    print("\n  '(x)' is the speedup over dense for the whole sparse pipeline, indexer\n"
          "  included. The oracle study: 25% is safe, 12.5% borderline. Where the 25%\n"
          "  column is below 1.00x, sparsity costs more than it saves at that size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
