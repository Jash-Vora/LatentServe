# Phase 4 — Serving Runtime

> **Outcome.** Continuous batching served ragged requests with outputs identical to generating them alone (Gate 3). It bought only 0–3% throughput over static batching, not the predicted ~1.5×, because the paged gather makes a bigger batch costlier per step. Scheduler overhead is negligible (5–10 µs per call, about 0.2% of wall time). Under open-loop arrivals, shortest-job-first cut median first-token time 3.3× but worsened p99 2.3×, and `fair` (shortest-job-first with aging) beat every policy on p99 and throughput. Three predictions first read as falsified; two of those were the harness measuring the wrong thing, which led to Phase 5. The re-measured inter-token p99s are quoted in `phase5.md`.

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

## Measured (mixed workload, 32 requests, prompts clamped at 8192, burst arrivals)

| policy | batch | tok/s | occupancy | TPOT p50 |
| --- | ---: | ---: | ---: | ---: |
| static | 1 | 24.81 | 1.00 | 33.43 ms |
| continuous | 1 | 24.70 | 1.00 | 33.27 ms |
| static | 4 | 50.85 | 3.58 | 50.19 ms |
| continuous | 4 | 52.18 | 3.84 | 51.51 ms |
| static | 8 | 67.42 | 7.15 | 66.79 ms |
| continuous | 8 | 67.49 | 7.21 | 65.27 ms |

Batch 1 static and continuous agree to 0.4%, which is the sanity check:
at `max_running=1` they are the same algorithm.

**P1 falsified — the premise, not the mechanism.** P1 assumed batching
stays free, as Phase 2 measured on the contiguous cache (batch 1 to 8
left TPOT at ~31 ms). It does not here: TPOT goes 33.3 -> 51.5 -> 65.3 ms,
roughly 2x from batch 1 to 8. The cause is Phase 3's gather, which costs
2x the *resident* KV per step, and resident KV is proportional to batch:
~455 MiB at batch 4, ~854 MiB at batch 8, i.e. ~9 ms and ~16 ms of pure
gather at the 110 GB/s this card sustains. Add the attention read itself
and most of the gap is accounted for.

So continuous batching's occupancy advantage is real but small, and it
is partly cancelled by its own success: running 3.84 sequences instead
of 3.58 means more resident KV and a more expensive step. Decode-only
throughput (occupancy / TPOT) is 74.5 vs 71.3 tok/s at batch 4 — a 4.5%
edge that prefill time dilutes to the measured 2.6%.

**The honest headline is that continuous batching bought ~0-3% here, and
the reason is the paged gather.** That is the third independent argument
for the Phase 11 paged-attention kernel: without the gather, batching
returns to nearly free (Phase 2's contiguous result) and the occupancy
advantage would actually pay. Worth re-running this sweep after Phase 11
as a direct before/after.

**P2 falsified by a bug in the harness, not by the system.** Measured
p99/p50 TPOT was 1.0-1.2, against a predicted 100x. The percentiles were
taken over each *request's mean* inter-token latency rather than over
individual decode steps, so a 2 s prefill stall spread across a request's
229 steps added 9 ms to its mean and disappeared. Phases 1-3 pooled
per-step latencies precisely to avoid this; this runner did not. Fixed:
`tpot_p50/p95/p99` now come from pooled steps, with per-request means
kept separately under `tpot_mean_per_request_*`. **The P2 numbers above
should be re-measured.**

**P3 confirmed.** Scheduler cost is 6-10 us/call and total non-GPU
runtime overhead is 0.20-0.23% of wall time. Negligible, as claimed, and
now measured rather than asserted.

**P4 untestable as run, and the reason is interesting.** length_aware
improved TTFT p50 by 39% over FIFO (29.3 s vs 47.9 s) and improved p99
too (88.9 s vs 92.4 s) — it was supposed to trade the tail away. `fair`
was best on everything, including throughput (73.2 tok/s).

Nothing starved because nothing *can* starve in a finite burst: with all
32 requests queued at t=0, the last one finishes when the total work
does, in whatever order you serve it. Starvation needs a stream of
newcomers to keep overtaking the waiting long request. Use
`--arrival-rate 0.5` (Poisson, open-loop) to test P4 properly.

**slo_aware was a FIFO clone** — it matched FIFO to four decimal places,
because no request carried an `slo_ttft_ms`, so every sort key reduced
to arrival order. Fixed: `assign_slo_ms()` now gives tiered targets by
prompt size (2 s interactive / 10 s document / 30 s batch). A policy that
cannot be distinguished from the baseline has not been tested.

**TTFT here is 99.6% queue time** (queue p50 47.7 s of a 47.9 s TTFT).
Under a burst these numbers measure queueing, not prefill speed. Worth
stating explicitly wherever they appear, and another reason to re-run
with arrivals.

## Measured with open-loop arrivals — P4 confirmed

Poisson arrivals, max_running 8. This is the run where the scheduler
comparison becomes meaningful: in a burst nobody can starve, because the
last request finishes when the total work does.

| policy | tok/s | TTFT p50 | TTFT p99 |
| --- | ---: | ---: | ---: |
| fifo | 65.12 | 10.06 s | 18.20 s |
| length_aware | 64.60 | **3.06 s** | **42.19 s** |
| fair | 68.69 | 5.79 s | **13.39 s** |
| slo_aware | 65.38 | 10.30 s | 21.57 s |

**P4 confirmed.** Shortest-job-first buys a 3.3x better median TTFT and
pays with a 2.3x worse p99 — the starvation the burst could not show.

**`fair` dominates, which was not predicted.** SJF-with-aging has the
best p99 of any policy (13.4 s, better than FIFO's 18.2 s), the second
best p50, and the highest throughput. It was included as a compromise
between FIFO and SJF and turned out to beat both on the tail. Worth
investigating in the writeup rather than just reporting: aging plausibly
helps throughput too by clearing long requests before they accumulate.

P3 holds under arrivals: 4.6-5.5 us per scheduler call, 0.20-0.22%
runtime overhead.

## Two more metric bugs, both found by predictions that refused to die

**Inter-token latency was measuring kernel time, not client experience.**
After pooling per-step latencies (the first fix), P2 still came back at
1.1-1.2 with a worst gap of 119 ms — nowhere near the seconds predicted.
The remaining error: `decode_step_ms` recorded the *duration of the
decode call*, while prefill runs between two decode calls. A 2 s prompt
admitted mid-flight leaves every decode call at a healthy 65 ms and the
incumbent's client waiting two seconds.

Now recorded as the wall gap between consecutive tokens for that request,
which is what a user experiences. A regression test admits a large prompt
mid-flight and asserts the incumbent's inter-token latency spikes;
without the fix it measured 1.2 ms where the gap was 102 ms, an 89x
understatement. **P2 needs re-measuring a third time.**

The general lesson is worth keeping for the report: three predictions in
this phase were "falsified", and two of those were the harness being
wrong rather than the system. A prediction specific enough to be wrong is
also specific enough to catch a measurement that quietly answers a
different question.

**slo_aware was being judged on metrics it does not optimise.** It looked
strictly worse than FIFO (p99 21.6 s vs 18.2 s) — but its job is meeting
deadlines, and nothing measured whether deadlines were met.
`slo_attainment` is now reported overall and per tier (2 s interactive /
10 s document / 30 s batch), since a policy that saves the interactive
tier by sacrificing the batch tier is working as designed and the
aggregate alone would hide it.

## Gate 3 checklist

- [x] `pytest tests/test_phase4_serving.py` green (all tiers, CPU)
- [x] continuous vs. static run at batch 1, 4, 8 with occupancy recorded
- [x] all four schedulers compared at fixed concurrency
- [x] scheduler overhead reported as microseconds per call *and* as a
      fraction of step time
- [x] TTFT reported split into queue and prefill — they are different
      quantities from Phases 1-3's TTFT and respond to different fixes
- [ ] every prediction above marked confirmed or falsified, in writing

*Reviewed at project close: ticked where this note's results show the item done. Left open: P5 was never marked, and P2's re-measured numbers appear in `phase5.md`, not here.*

## Deliberately deferred

* **Chunked mixed batching** (prefill and decode in one step). Not built: Phase 17 became the adaptive runtime. Prefill is chunked at 4,096 tokens, but runs in its own pass before the step's decode.
* **Preemption and swapping.** The engine raises a clear `deadlock`
  error if the shortest waiting request cannot fit an empty pool, and
  admits nothing it cannot seat. Evicting a running sequence to admit a
  higher-priority one needs a recompute-or-swap policy; that is a
  scheduler research question of its own.
* **Real arrival times.** Every request is queued at t=0 (a burst),
  which maximises queueing pressure and makes the scheduler comparison
  sharp. `generate_requests(arrival_rate=...)` already supports Poisson
  arrivals for a steady-state study.
