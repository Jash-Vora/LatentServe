# Phase 16 — INT8 with sparse decode

The plan's Phase 16 combined the memory-efficient technique (MLA, closed in
Phase 7, replaced by INT8) with sparse attention. What it can buy on a T4 is
capacity, not per-step speed: fp16 cannot hold 16 x 32K or 32 x 16K; INT8
(~1.8x capacity) can. Sparsity makes capacity worth more — with attention cut
to 37.5%, doubling the batch no longer doubles the step, so the weights are
shared across twice the requests. INT8 has been slower than fp16 end to end
in every run so far (6-19%), so the phase is built to stop early.

## Stages, with a gate after the first

1. **Fuse INT8's decode write** (`kernels/cuda/int8_write.cu`): V's
   per-token quantize and scatter and K's scatter into the fp16 residual as
   one kernel per layer, instead of ~a dozen ops (~280 launches per step) —
   the known bulk of INT8's end-to-end cost since Phase 14c. Byte-identical
   to the torch path (tested: every pool, scale, zero point and residual,
   symmetric and asymmetric).
2. *(only past the gate)* A sparse kernel reading INT8 pages, page bounds
   kept from the fp16 keys before quantization.
3. *(only past the gate)* The decisive measurement: decode throughput of
   INT8 + sparse at the larger batches it allows, against the best fp16 +
   sparse configuration that fits, at 16K and 32K. Success: >= 15% more.
4. *(only past the gate)* A quality spot-check of the combination.

## Gate — fixed before measuring

Whole decode steps, fp16 against INT8, both on the CUDA kernel
(`phase14_fusion --toggle int8 --decode-backend cuda`). **INT8 must be within
8% of fp16 at every shape with batch >= 8 and context >= 8K** — where the
capacity it buys would be used. Slower than that, its per-step tax eats the
capacity gain, Phase 16 stops, and INT8 is recorded as a capacity-only
option: for requests fp16 cannot hold at all.

**Prediction:** the fused write brings INT8 within ~5% of fp16 at large
shapes (its attention kernel measured 2-7% slower than fp16's); small
batches stay slower.
