# Phase 5 — Benchmark Harness

> **Outcome.** The shared measurement harness: nearest-rank percentiles, bootstrap confidence intervals, a stall rate for bimodal latency, and comparisons that raise when two rows differ on a control. Repeating Phase 4's scheduler sweep reproduced first-token times within 1.2%, which sets the noise floor for the rest of the project. No new measurements.

Goal (docs/methodology.md Phase 5): "Before introducing MLA, make
measurement trustworthy."

That instruction earned itself in Phase 4. Two results there came from **the
harness answering a different question than the one asked**:

* inter-token percentiles taken over per-request *means*, so a 2 s stall
  spread across 229 steps vanished into a 9 ms bump;
* then over the decode *call* duration rather than the wall gap between
  tokens, so prefill blocking stayed invisible a second time.

Neither was visible by reading the code. Both showed up when a result
contradicted simple arithmetic. `benchmarks/harness.py` exists so those
mistakes are made once, centrally.

## What landed

| File | Role |
| --- | --- |
| `benchmarks/harness.py` | percentiles, bootstrap CIs, bimodal-aware latency summary, fairness guards, repeatability, environment-drift detection |
| `tests/test_phase5_harness.py` | the harness's own failure modes, including the two Phase 4 measurement bugs |

`benchmarks/runners/phase4_serving.py` now imports the shared percentile
rather than defining its own.

## Three decisions worth defending in the report

**1. Percentile definition: nearest-rank.** Three were available.
numpy's default interpolates between neighbours, inventing values that
never occurred — wrong for a latency distribution whose interesting mode
is a real 2.3 s stall, not an average of one. The Phase 1-4 runners used
`round(p * (n - 1))`, which inherits Python's banker's rounding
(`round(4.5) == 4` but `round(5.5) == 6`), so p50 tilts in a direction
that depends on index parity. Nearest-rank is textbook, always returns an
observed value, and has no rounding-mode surprises. The switch moves an
index by at most one position, so Phase 1-4 headline numbers are
unaffected in any way that matters.

**2. Bimodal latency gets a stall rate, not just percentiles.**
Inter-token latency under a serving runtime has two populations: a normal
one around the decode step, and a stall population at the length of
whatever prefill was admitted. Phase 4 had ~1% stalls, which put p99
exactly on the boundary — `fair`'s 1872 ms against FIFO's 2257 ms was
sampling luck, not a scheduler difference.
`test_p99_sits_on_the_boundary_when_stalls_are_one_percent` pins this
down: shifting the stall count from 8 to 12 in 1000 flips p99 from 30 ms
to 2300 ms, while `stall_rate` moves smoothly. Where a distribution is
bimodal, report how often the second mode fires.

**3. Unfair comparisons raise rather than warn.** `assert_comparable`
refuses to produce a ratio between rows that differ on model, dtype,
batch size, context length, output length, GPU, sampling, workload or
arrival rate. A key present in one row and absent from the other counts
as a mismatch, because that usually means one side quietly defaulted.
The failure mode being prevented — LatentServe at 256 output tokens
against vLLM at 128 — yields a plausible number, and a plausible wrong
number is worse than a crash.

## Repeatability

`repeatability_report` turns repeated identical runs into a coefficient
of variation. Phase 4's scheduler sweep, run twice, reproduced TTFT
within 1.2% (FIFO 10056 -> 9992 ms, fair 5786 -> 5903, length_aware
3059 -> 3063). That is the noise floor, and it is what licenses calling a
3.3% throughput difference real and a 0.4% one noise. Without it, every
small delta in the final report is an assertion.

Bootstrap CIs return `(None, None)` below three samples rather than a
number that would look more authoritative than the data supports.

## Not done here

No new measurements. Phase 5 is instrumentation; the numbers it produces
belong to the phases that use it. Nothing from Phases 1-4 was re-run,
because the percentile change is sub-index and the two metric fixes were
already applied and re-measured in Phase 4.
