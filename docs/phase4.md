# Phase 4 — Serving Runtime

Goal (docs/methodology.md Phase 4): turn the inference backend into a
minimal serving engine — request lifecycle, continuous batching, and a
scheduler — with scheduler overhead measured explicitly.

## Why this could not have been built before Phase 3

A ragged decode batch (sequences at positions 512, 9000 and 17 in the
same step) cannot be represented by a contiguous cache: it has one
shared fill length. Paging makes each sequence's KV independent, so
**Phase 3 is a prerequisite for Phase 4, not an optimization of it.**
Worth stating plainly in the report, since "paged KV" and "continuous
batching" are usually presented as two separate features.

Two other things had to change for ragged execution, and both were
latent bugs waiting for it:

* **RoPE per sequence.** `cos_sin_at(positions)` takes a `[B, S]` tensor
  of absolute positions. A single scalar offset — all Phases 1-3 ever
  needed — silently gives every sequence the first one's positional
  phase.
* **The cache reports its own length.** Attention used to derive
  `kv_len = start_pos + s`, which describes nobody in a ragged batch.
  It now asks the cache. (This one bit during development: the ragged
  path read exactly one key position and attention quietly attended to
  a single token.)

## What landed

| File | Role |
| --- | --- |
| `runtime/request.py` | `ServedRequest`, lifecycle states, per-request metrics |
| `runtime/scheduler.py` | FIFO, length-aware, fair (SJF + aging), SLO-aware; overhead instrumentation |
| `runtime/batching.py` | slot and position assembly for a ragged decode batch |
| `runtime/engine.py` | `ServingEngine` (continuous) and `run_static_batching` (baseline) |
| `benchmarks/runners/phase4_serving.py` | continuous vs. static, and the scheduler comparison |
| `tests/test_phase4_serving.py` | Gate 3, including output equivalence under concurrency |
| `configs/phase4_serving.yaml` | held constant across the sweep |

## One engine step

    1. retire finished sequences, freeing their blocks immediately
    2. ask the scheduler which waiting requests to admit
    3. prefill each admitted request into its own slot
    4. one ragged decode step across every resident sequence

**Prefill runs one request at a time, and blocks decoding.** Phase 2
measured prefill as compute-bound (650 ms at 4K against a 31 ms decode
step), so batching prefills buys little, while mixing a prefill into a
decode batch needs the combined causal+padding mask Phase 3 deferred.
The cost is real and should be reported rather than hidden: an 8K prompt
arriving mid-flight adds its whole ~2 s prefill to every resident
sequence's inter-token latency. Expect it in p99 TPOT. Chunked mixed
batching is the fix, and it belongs with the adaptive runtime in
Phase 17.

## Gate 3 — "can the runtime serve multiple requests?"

The test that matters is not that it runs but that **batching changes
nothing about what each request receives**. Continuous batching is a
scheduling optimization; if a request's tokens depend on who it shared a
step with, the engine is broken in a way no throughput number reveals.
`test_continuous_batching_matches_solo_generation` generates four
ragged requests alone and then through the engine at concurrency 1, 2
and 4, and demands identical output ids. `test_static_batching_matches
_continuous_outputs` does the same across the two batching policies, so
the throughput comparison between them is meaningful.

## Run it

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH

pytest tests/test_phase4_serving.py -v        # no GPU, no download

python -m benchmarks.runners.phase4_serving --config configs/phase4_serving.yaml \
    --workload mixed --num-requests 32 --max-prompt 8192 --max-output 256 \
    --batch-sizes 1 4 8 --max-running 8
```

`--max-prompt` clamps the workload's long tail and is recorded in the
results. The `mixed` family reaches 64K prompts; a single 64K prefill is
~70 s on a T4, which is fine as a capacity experiment (Phase 3 simulated
it for free) and ruinous as a serving experiment where the point is to
watch many requests interact.

## Predictions (written before measuring)

**P1 — continuous beats static by roughly the occupancy ratio, not by
per-step speed.** Phase 2 showed batching is nearly free at 4K: batch 1
to 8 left TPOT at ~31 ms for 8x the throughput, because decode is
weight-bound and weights are read once per step regardless of batch.
So continuous batching cannot win on per-step cost — it wins by keeping
the batch full. Static holds every slot until the group's slowest member
finishes, so with the `mixed` family's 128/256 output split its mean
occupancy should sit near 0.6-0.7 of nominal, and continuous should lead
throughput by about 1/0.65 ≈ 1.5x at batch 8.
*Falsified if* static's `mean_batch_occupancy` is close to its batch size
— in which case the workload has too little output-length variance to
distinguish the policies, and the experiment, not the engine, is wrong.

**P2 — p99 TPOT will be two orders of magnitude above p50.** Prefill
blocks decoding, so any resident sequence unlucky enough to be decoding
when an 8K prompt is admitted waits ~2 s for one token. Predict p50 TPOT
~35 ms and p99 above 1 s.
*This is a design consequence, not a bug*, and the number is the argument
for chunked mixed batching later.

**P3 — scheduler overhead is irrelevant.** A sort over a queue of tens
of requests should cost single-digit microseconds against a ~35 ms step,
i.e. under 0.05%. Measured anyway, because methodology Phase 4 asks for
it and "negligible" is a claim.

**P4 — length-aware improves p50 TTFT and worsens p99.** SJF minimises
mean waiting time, and on `mixed` the short requests are the majority.
The long prompts pay for it by being overtaken repeatedly. `fair`
(SJF + aging) should land between the two, which is the entire point of
having it.

**P5 — continuous batching's TPOT is slightly *worse* than static's at
matched nominal batch.** Higher occupancy means more resident KV, so more
bytes per decode step. Throughput up, per-token latency mildly down: the
trade should be stated rather than buried.

## Gate 3 checklist

- [ ] `pytest tests/test_phase4_serving.py` green (all tiers, CPU)
- [ ] continuous vs. static run at batch 1, 4, 8 with occupancy recorded
- [ ] all four schedulers compared at fixed concurrency
- [ ] scheduler overhead reported as microseconds per call *and* as a
      fraction of step time
- [ ] TTFT reported split into queue and prefill — they are different
      quantities from Phases 1-3's TTFT and respond to different fixes
- [ ] every prediction above marked confirmed or falsified, in writing

## Deliberately deferred

* **Chunked mixed batching** (prefill and decode in one step) — Phase 17.
* **Preemption and swapping.** The engine raises a clear `deadlock`
  error if the shortest waiting request cannot fit an empty pool, and
  admits nothing it cannot seat. Evicting a running sequence to admit a
  higher-priority one needs a recompute-or-swap policy; that is a
  scheduler research question of its own.
* **Real arrival times.** Every request is queued at t=0 (a burst),
  which maximises queueing pressure and makes the scheduler comparison
  sharp. `generate_requests(arrival_rate=...)` already supports Poisson
  arrivals for a steady-state study.
