# Phase 12, resumed — what limits the decode kernel

> **Outcome.** The first Triton decode kernel was arithmetic-bound, not memory-bound: it compiled its matrix multiplies to scalar fp32 math (no tensor-core instructions) at 12% occupancy, and ran at 69 GB/s against 243 GB/s for a loads-only version. A hand-written CUDA-core kernel reached 194 GB/s and was 2.8× faster (1.93 ms to 0.69 ms); the whole decode step at batch 16 / 8K went from 79.0 ms to 34.2 ms.

Nsight Compute is refused in this container (`ERR_NVGPUCTRPERM`), so the
kernel was taken apart instead: compiler resource counts, ablation kernels,
and a census of the compiled machine code (`benchmarks/runners/phase12_diag.py`).
T4, torch 2.13, Triton 3.7, batch 16 / ctx 8192 unless stated.

## Hypotheses that died

| hypothesis | test | result |
| --- | --- | --- |
| shared memory from pipeline buffering caps occupancy | stages 1 vs 2 | identical times, 40 KB either way |
| the dependent block-table load serialises the loop | no-lookup ablation | 1.927 vs 1.933 ms |
| prefetching the next page hides that latency | prefetch kernel | 2.79 ms, slower than the 4-page tile |
| INT8 is slow because of fp32 dequant tiles | fp16-tile fix | correct, ~4% faster, registers unchanged |

## What the ablations show

| | ms | GB/s |
| --- | ---: | ---: |
| full kernel | 1.93 | 69 |
| loads only | **0.55** | **243** |

The memory system delivers this exact access pattern at 243 GB/s, 76% of
the T4's peak. The remaining ~1.4 ms is the arithmetic. (The compute-only
ablation, 1.34 ms, is not clean: holding its reused tiles across the loop
spilled 510 values, so it includes local-memory traffic. The loads-only
result and the census below are the firmer evidence.)

## What the machine code shows

`HMMA = 0` in every variant. The kernel's matrix multiplies are compiled
to scalar multiply-adds on the CUDA cores, with operands staged through
shared memory (271 shared-memory loads in the production kernel). On this
GPU, Triton is not using tensor cores. That explains, at once:

* **the ceiling** — the kernel is compute-bound, not memory-bound;
* **the 40 KB of shared memory** — operand staging, allowing one program per
  SM, hence 12% occupancy;
* **255 registers and 90 spills** — scalar dot products held in registers;
* **INT8 losing** — it adds conversion work to an arithmetic-bound kernel,
  and halving the bytes saves nothing when bytes are not the limit;
* and the waste: each KV head has 6 query rows, padded to 16 for `tl.dot`,
  so 62% of the arithmetic is on padding rows.

To be confirmed before redesigning around it: the census now also counts
FFMA/HFMA2 (scalar multiply-adds) and checks Triton's PTX for any `mma`
instruction. `python -m benchmarks.runners.phase12_diag --census-only`.

## What it points to

Decode attention here is a matrix-*vector* problem: 6 query rows per KV
head. The kernel for that on Turing is CUDA-core arithmetic with no
padding — dot products reduced across a warp, 16-byte vectorised loads —
the design of vLLM's original CUDA paged-attention kernel. Its target is
the loads-only speed: ~0.55 ms per layer-call at batch 16 / 8K instead of
1.93. With attention ~54 ms of the 81 ms step, that would put the step
near 40-45 ms against vLLM's 107, and would let INT8 finally win, since a
memory-bound kernel benefits from half the bytes.

## Confirmed

The census with scalar-instruction counts: `HMMA = 0` and `PTX mma = 0` in
every variant, `FFMA` 1,540 of 2,800 instructions in the production kernel.
Triton never asks for tensor cores on sm_75; the matrix multiplies are fp32
scalar FMA. `IMAD` is only 20, so per-element address arithmetic is not
where the registers go — that guess was wrong.

## The CUDA-core kernel

`kernels/cuda/paged_decode_fp16.cu`: one warp per (sequence, split, KV
head); the two half-warps take alternate tokens; each of 16 lanes owns 8 of
the 128 dimensions, so every K and V row is one coalesced 256-byte read.
Six query rows, no padding. The same lanes own the same dimensions for K
and V, so nothing moves between the two phases. Partials use the Triton
layout, so the Triton merge kernel finishes the job.

Compiled with NVRTC through CuPy (`pip install cupy-cuda13x` on vLLM's
stack): torch's extension builder refuses the system nvcc 12.8 against
torch's CUDA 13.0, and NVRTC uses torch's own CUDA 13 headers instead.
Launched on torch's current stream, so CUDA graphs capture it.

| | registers | spills | shared memory | resident warps / SM |
| --- | ---: | ---: | ---: | ---: |
| Triton production (fp16, 4 pages/iter) | 255 | 90 | 40 KB | 4 |
| CUDA-core kernel | 168 | 0 | 0 | ~12 |

Those are the compiler's numbers for sm_75; the speed is for the GPU to say.
`python -m benchmarks.runners.phase12_diag --cuda` checks correctness and
times both backends under CUDA-graph replay, so neither is charged for
Python launch overhead. `phase14_fusion --toggle cuda_decode` measures
whole decode steps.

## Result (T4, graph-replay timing, ragged-batch correctness OK, max error 2.4e-4)

| batch | ctx | Triton | CUDA | speedup | CUDA GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 0.122 ms | 0.092 | 1.3x | 23 |
| 1 | 8192 | 0.389 | 0.101 | 3.8x | 83 |
| 4 | 8192 | 0.606 | 0.195 | 3.1x | 172 |
| 16 | 8192 | 1.953 | 0.690 | 2.8x | 194 |
| 16 | 2048 | 0.553 | 0.205 | 2.7x | 164 |

The driver reports 168 registers, 0 bytes of local (spill) memory and 0
bytes of shared memory for the loaded kernel. At batch 16 / 8K it reaches
194 GB/s, 80% of the loads-only ceiling: memory-bound once the staging,
padding and spills were gone. Batch 1 / 2K has too little work (2 KV heads x 128 pages) to fill 40 SMs.

The automatic split count was the fastest tried at batch 16 but not at
batch 1 (64 beat it) or 4 (32 did), so end-to-end numbers using the default
slightly understate the kernel at small batch.

The first census of this kernel printed a row of zeros: nvdisasm 12.8
evidently cannot read NVRTC 13.0's cubin and printed nothing. It now says so.

## Whole decode steps (phase14_fusion --toggle cuda_decode)

Full model, projections and elementwise fused, CUDA graphs; only decode
attention differs. Bare decode step, 4 alternating rounds.

| batch | ctx | Triton | CUDA | saved | spread |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 16.80 ms | 16.65 | 0.15 (0.9%) | 0.05 |
| 1 | 8192 | 20.08 | 17.75 | 2.33 (11.6%) | 0.17 |
| 4 | 2048 | 20.65 | 18.47 | 2.19 (10.6%) | 0.08 |
| 4 | 8192 | 32.42 | 21.65 | 10.79 (33.3%) | 0.20 |
| 16 | 2048 | 36.24 | 21.28 | 14.97 (41.3%) | 0.25 |
| 16 | 8192 | 78.99 | 34.20 | 44.79 (56.7%) | 1.23 |

Scaling the isolated kernel timings by 28 layers matches batch 4 / 8K
(11.5 ms), understates batch 16 (35 vs 45) and overstates batch 1 / 8K (8.1
vs 2.3): inside the model, Triton's batch-1 attention cost less than in
isolation, so there was less to save. With the
CUDA kernel, batch 1 grows only 1.1 ms from 2K to 8K — the step is now
almost all weight reads — and at batch 16 the 2K-to-8K growth fell from
43 ms to 13: attention's per-token cost dropped 3.3x.

## INT8 on the CUDA kernel

`kernels/cuda/paged_decode_int8.cu`: the fp16 kernel's structure line for
line, differing only in how rows are read. K is int8 with a per-(block, head,
channel) scale and optional zero point: one 8-byte load per lane per row,
the lane's eight channel scales loaded once per page and applied as each
token converts (8 multiplies against the dot products' 48). V's per-token
scale folds into the softmax weight, so V is never dequantized. The last
page's K comes from the fp16 residual through `res_rows`, one uniform branch
per page. Symmetric/asymmetric and residual/no-residual are compile-time.

Compiled for sm_75, unbounded, every variant is spill-free but heavier than
fp16's 168 registers: 199 for the production variant (symmetric, residual),
up to 233 — 8-10 resident warps instead of 12. Bounding the launch to 12
blocks per SM holds them to 168 registers at the cost of small spills (12 B
stored / 24 B loaded for the production variant, up to 148 B for others).
`--maxrregcount` cannot do this: it is ignored for kernels that declare
launch bounds. Which bound is faster is measured, not assumed:
`phase12_diag --int8` times both, against Triton INT8 and CUDA fp16.

### Result (T4, graph replay; correctness OK, max error 1.5e-5)

| batch | ctx | INT8 Triton | INT8 CUDA b1 | INT8 CUDA b12 | fp16 CUDA | INT8 vs fp16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 0.143 ms | 0.135 | 0.137 | 0.129 | 0.96x |
| 1 | 8192 | 0.326 | 0.168 | 0.171 | 0.147 | 0.87x |
| 4 | 8192 | 0.877 | 0.280 | 0.248 | 0.244 | 0.99x |
| 16 | 8192 | 2.981 | 0.837 | 0.724 | 0.690 | 0.95x |
| 16 | 2048 | 0.818 | 0.240 | 0.207 | 0.200 | 0.97x |

INT8 is 1-13% slower than fp16 everywhere, even where attention is
memory-bound. The goal of the step was met: INT8
on the CUDA kernel is 3.4-4.1x faster than on Triton, so INT8 goes from
roughly 3x slower than fp16 decode to a few percent, for ~1.8x the capacity.

Why halving the bytes bought no time: both kernels issue the same number of
load instructions per token (one per K row, one per V row, per lane); INT8's
carry 8 bytes instead of 16. A kernel limited by memory latency, with a
fixed number of loads in flight, moves half the bytes in the same time:
INT8 reaches ~106 GB/s against fp16's 194. The per-token V-scale load and
the conversions then make it slightly slower. Consistent with this, the
bounded variant (b12) — more resident warps, more loads in flight — is
11-13% faster at batch 4 and 16 despite its small spills, and is now the
default. To make INT8 *faster* than fp16, each load must carry more: e.g.
16 INT8 channels per lane, which costs registers this design does not have.

### Whole decode steps, fp16 vs INT8 cache, both on the CUDA kernel

| batch | ctx | fp16 | INT8 | INT8 cost |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 16.60 ms | 17.67 | +6.5% |
| 1 | 8192 | 17.70 | 19.04 | +7.5% |
| 4 | 2048 | 18.42 | 19.53 | +6.0% |
| 4 | 8192 | 21.51 | 23.97 | +11.4% |
| 16 | 2048 | 20.98 | 23.61 | +12.5% |
| 16 | 8192 | 33.05 | 39.40 | +19.4% |

The attention kernel's extra cost is only ~0.03 ms per layer at batch 16 /
8K, about 1 ms per step; the step is 6.4 ms slower. The rest is INT8's write side: 709 kernels per step
against 429, the 280 extra being the per-token V quantization and residual
copy, about ten small operations per layer — the cost Phase 14c identified,
whose fix (one fused quantize-and-write kernel) was never built.

Status: INT8 is correct, runs on the CUDA kernel (3.4-4.1x faster than on
Triton), and gives ~1.8x capacity for 6-19% decode time. Parked in favour of
Phase 14. The remaining cost is known, and so is its fix.

A caution on reading kernel timings: fp16 at batch 1 / 8K measured 0.147 ms
and 0.095 ms in two diagnostic runs. Batch-1 kernel timings in isolation are
too noisy to rank variants; the batch 4 and 16 rows agree between runs to
within 1%.
