# Phase 16 — INT8 with sparse decode

> **Outcome.** The fused INT8 write cut INT8's end-to-end penalty from 6–19% to 1–9% and issues fewer kernels per step than fp16 (401 against 429), but INT8 still failed its gate at batch 16 / 16K (9.3% slower than fp16), so the INT8 + sparse kernel was not built. INT8 is a capacity option: the final sweep shows it running batch 32 at 16K and batch 16 at 32K, which fp16 cannot.

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

## The first fused write was not byte-identical

The end-to-end check failed: symmetric — layer 0's INT8 values identical,
layer 1's different; asymmetric — layer 0's already different. Identical
values with a downstream difference pointed at the *scales*: one bit off
barely changes a rounded value but changes the V that layer attends to, which
changes the next layer's input.

Cause: PyTorch does not divide by a Python number on the GPU. `t / 127` is
computed as `t * (1/127)`, the reciprocal rounded once in fp32 on the host;
its source says this "may lose one bit of precision". The kernel divided
properly. Measured in fp32: the two disagree by one ulp in 4.7% of cases at
/127 (symmetric) and 74.3% at /255 (asymmetric) — rare enough that values
survive and scales do not, and common enough that asymmetric values flip.
Fixed by computing scales as the torch path does: times the fp32 reciprocal.

Test design, also corrected: the end-to-end state comparison is a strong
check but a poor diagnostic — one wrong bit spreads through every later
layer and step. Added a unit test (kernel against the torch formula on
identical inputs, 256 tokens over five decades of magnitude) and a
determinism control (the torch path end to end must reproduce itself, or a
mismatch against the fused path proves nothing).

## Gate result — FAIL; Phase 16 stops

Whole decode steps, INT8 against fp16, both on the CUDA kernel, fused write:

| batch | ctx | fp16 | INT8 | INT8 vs fp16 | gate (within 8% at b >= 8, ctx >= 8K) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2048 | 16.71 ms | 16.98 | -1.5% | |
| 1 | 16384 | 19.27 | 20.14 | -4.5% | |
| 4 | 8192 | 21.94 | 22.21 | -1.2% | |
| 8 | 8192 | 26.39 | 27.46 | -4.0% | pass |
| 8 | 16384 | 35.94 | 38.76 | -7.4% | pass |
| 16 | 2048 | 24.47 | 24.36 | +0.5% | |
| 16 | 8192 | 38.24 | 39.29 | -2.7% | pass (spread 1.71 ms) |
| 16 | 16384 | 53.80 | 58.89 | **-9.3%** | **fail** |

By the rule fixed beforehand, Phase 16 stops: INT8 is a **capacity-only**
option, for requests fp16 cannot hold at all. INT8's penalty grows with
batch x context — where attention dominates the step and INT8's attention
kernel is slower than fp16's (the Phase 12 latency finding) — so its tax is
heaviest exactly where its capacity would be used.

**The fused write is a clear win on its own:** INT8 end to end went from
6.5-19.4% slower than fp16 to 1.2-9.3%, and at batch 16 / 2K from -12.5% to
even. As a capacity-only mode it is much cheaper than it was.

**Untested hypothesis, not a result:** the gate measured dense INT8, because
the INT8 sparse kernel was stage 2. Under sparsity attention is a smaller
share of each step, so INT8's attention tax would be smaller too — the gate
may have been conservative. That argument was made after the gate failed, so
it does not override it.

**Instrument fix:** the A/B runner's kernel counter ran its eager step
without the deferred mode GraphedDecoder enables for INT8 caches, so it
measured INT8's old write path — one the timed runs never take — and still
reported 709 kernels after the fused write. It now enables the same mode.

## Kernel census (`phase16_kernel_census`) — the 709, measured

| one decode step, batch 1 / 2K | kernels | int8_decode_write |
| --- | ---: | ---: |
| fp16 | 429 | 0 |
| INT8, eager (the old counter) | 709 | 0 |
| INT8, deferred, fused off (timed path before Phase 16) | 709 | 0 |
| INT8, deferred, fused on (timed path now) | 401 | 28 |
| captured graph, replayed: fp16 | 433 | 0 |
| captured graph, replayed: INT8 | 407 | 28 |

The explanation is confirmed: the old counter and the pre-fusion timed path
both had 709 kernels, which is why the count never looked wrong until the
fusion changed only one of them; the replay shows the timed path running the
fused kernel once per layer. With fusion off, INT8's extra kernels were 280
elementwise, reduce and index ops (+308) less fp16's attention kernel (-28).
With fusion on, INT8 issues **28 fewer kernels than fp16** (one fused write
per layer against fp16's two scatters).

**Consequence for the gate:** INT8's remaining penalty (up to 9.3% at batch
16 / 16K) cannot be launch overhead — it now launches fewer kernels than
fp16. It is INT8's attention kernel, latency-bound as Phase 12 found, which is
why the penalty grows with batch x context. That makes the gate's verdict
firmer, not weaker.

(The census's first version listed differing kernels only over the INT8
run's own names, so fp16's attention kernel — absent from INT8 runs — never
showed its -28; fixed to use the union.)

## What the final sweep added

The final sweep (`sweep_stage1_results.md`) ran INT8 end to end in the real engine, with its pool sized for INT8's real bytes per block. Four things came out of it.

**INT8 runs shapes fp16 can't.** Batch 32 at 16K takes 114.7 ms per step and batch 16 at 32K takes 107.3 ms. Neither fp16 LatentServe nor vLLM fits either shape at the sweep's memory settings.

**Where both fit, INT8 is level to 12% slower.** It is within 1% of dense at four shapes (batch 4 / 2K, batch 8 / 32K, batch 16 / 8K and batch 16 / 16K) and slower elsewhere, by up to 12.1% at batch 32 / 8K (66.1 against 58.9 ms) and 9.6% at batch 1 / 32K. On the burst workload it was 4.3% slower (245 against 256 tok/s). Its tail is worse by one step in sixteen: INT8's p99 sits 5.7–9.9 ms above its p50 at every shape, while dense's sits under 2 ms.

**Capacity alone doesn't beat sparsity.** Decode throughput, from the step times (batch divided by step time, prefill excluded):

| Context | fp16 dense | fp16 sparse 37.5% | INT8 dense, twice the batch |
| --- | ---: | ---: | ---: |
| 16K | 253 tok/s (batch 16) | 409 tok/s (batch 16) | 279 tok/s (batch 32) |
| 32K | 129 tok/s (batch 8) | 214 tok/s (batch 8) | 149 tok/s (batch 16) |

Doubling the batch with INT8 buys 10–16% over fp16 dense, while fp16 sparse buys 62–66%. The combination this phase was meant to test was never built, so whether INT8 with sparse beats sparse alone is still untested; any gain would have to come from that combination, not from the capacity.

**One real-traffic gain, from a single round.** On the varying workload INT8 matched dense (32.6 against 31.7 tok/s) and cut p99 first-token time 11% (92 against 104 s), plausibly because its larger pool admits more long requests at once, at the price of a worse tail per-token time (p99 655 against 459 ms).

**The gate's number and the sweep's disagree at batch 16 / 16K, and the cause wasn't found.** The gate measured INT8 9.3% slower than fp16 (58.89 against 53.80 ms, graphs only). The sweep has them level (62.70 against 63.27 ms, whole engine). Dense's step is 18% longer in the sweep than in the gate, while INT8's is 6.5% longer, so most of the difference is in dense's step, not INT8's. The two harnesses differ (the sweep runs the full engine, with its scheduler and admission; the gate only the captured step). The gate's verdict used the graph-only number and stands as recorded.
