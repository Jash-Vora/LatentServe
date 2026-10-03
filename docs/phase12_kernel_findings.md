# Phase 12, resumed — what limits the decode kernel

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
