# Final sweep, stage 1 — single GPU

`benchmarks/runners/sweep_stage1.py`. One T4 ("GPU T4" accelerator). Resumable:
every cell is saved on completion and skipped on rerun.

## What is measured

| section | measure | configurations |
| --- | --- | --- |
| A. decode step | engine-level step time once every request decodes; batch 1, 4, 8, 16, 32 x context 2K, 8K, 16K, 32K, where it fits; 48 timed steps; 2 rounds | LS dense, sparse 50%, sparse 37.5%, INT8; vLLM |
| B. prefill | time to first token, one request at a time; prompts 1K-32K; 8 distinct prompts each | LS dense, INT8; vLLM |
| C. serving | burst (96 requests at once); chat (16 conversations x 5 turns); Phase 17's varying traffic in wall-clock time; 2 rounds each | LS dense, sparse 37.5%, adaptive, INT8; vLLM. Chat: prefix caching on and off, both engines |
| C. load curves | capacity probe, then Poisson arrivals at 10, 25, 40, 55, 70, 80, 90, 100, 110% of it; one round | same as serving |

## Ground rules

* The same concurrency limit (16), context limit (33,280), prompts and seeds
  for both engines; vLLM gets `max_num_seqs` (it ran at its default in Phase
  18) and does not detokenize (LatentServe never does).
* Every worker warms up before any timed cell; CUDA-graph capture is excluded
  from throughput, capacity included.
* Two rounds in alternating order where affordable; spreads are reported.
* Engines run in groups (one fits on the GPU at a time); a drift sentinel
  reruns LatentServe's dense burst at the end, and the report gives the drift.
* Section A is engine-level — scheduling overhead included, as users see it.
  LatentServe's graph-only step times are in Phase 17's calibration.
* Open-loop points report their request count: p99 from fewer than 100
  requests is close to the maximum, and the report says so.
* A batch that cannot all decode at once is reported as not fitting, never
  timed with fewer requests than intended.

## Predictions — written before the run

Each against the earlier measurement it extends:

* **A.** LS dense vs vLLM, per step: near parity at batch 1 / 2K, widening to
  ~2x at batch 4 / 8K and ~3x at batch 16 / 8K (Phase 12: 16.65 vs 17.6 ms,
  21.65 vs 42.1, 34.2 vs 107.3). Engine-level adds LatentServe's host
  overhead, so its gap may be a little smaller than Phase 12's. Sparse 37.5%
  over dense: ~1.3-1.55x at large shapes, ~0.96-1.03x at batch 1 (Phase 17's
  recalibration). INT8: 1-9% slower than dense where both fit (Phase 16) —
  and INT8 alone fits batch 16 / 32K and batch 32 / 16K.
* **B.** LS faster than vLLM by 2.7-6x, the ratio growing with prompt length
  (Phase 6). INT8 prefill within a few percent of dense.
* **C, burst.** LS dense ~2.5-3x vLLM (Phase 18: 2.8x) — vLLM now capped at
  16 concurrent requests instead of its default, which may move it either way.
  Sparse 37.5% ~5-15% over dense (contexts 1-4K, batch up to 16: Phase 17
  measured 1.05-1.17x there); adaptive close to fixed 37.5%, since it picks
  sparse wherever that clears 5%; INT8 2-5% below dense.
* **C, chat.** LS +40-60% over vLLM on throughput (Phase 18, one GPU: +54%);
  prefix caching cuts TTFT ~75% for both engines (Phase 13: -77%).
* **C, varying.** Adaptive between dense and fixed 37.5% on throughput, as in
  Phase 17.
* **C, load curves.** LS's knee at a higher absolute rate than vLLM's, by
  roughly its burst-throughput ratio. Below the knee, TTFT is mostly prefill,
  so LS's median TTFT is lower by roughly section B's ratio; per-token time
  at light load near parity.
