# Phase 17 — adaptive runtime

> **Outcome.** On varying traffic, the adaptive policies came within 2% of fixed sparse attention's throughput at about half its expected answer loss, so by the definition fixed in advance both tiers outperform. Their speed gain over dense is small (1.08–1.12×) because much of real traffic runs where sparsity buys nothing. The note also records a first workload that tested nothing, and why.

> Can an adaptive policy outperform a fixed backend? (methodology Q14)

## The idea

Phase 15 measured what sparse decode costs — ~0.4% of answers at 50% of
pages, ~1.2% at 37.5% — and Phase 14 what it buys: nothing at batch 1, up to
~1.5x at batch 8 / 32K. A fixed sparse setting pays its cost on every step,
including those where it buys nothing. `runtime/policy.py` pays it only where
measured step times say it is bought:

    choose(batch, mean context) -> the fastest budget the tier allows, if it
                                   beats dense by min_gain (5%); else dense

Tiers: strict (dense only), balanced (dense or 50%), relaxed (dense, 50% or
37.5%). Step times come from a table measured on the target GPU
(`phase17_calibrate`), interpolated in log space — thresholds derived from
data, as the plan requires. INT8 stays a configuration-time choice (only when
fp16 cannot hold the working set, since it is slower); combining INT8 with
sparsity is Phase 16.

## What had to change first

* **The sparse budget decayed inside a CUDA-graph bucket.** It was computed
  from the context at capture, and the graph reused for every longer length
  in its bucket; buckets double, so "37.5%" fell toward ~19% as a sequence
  grew. Earlier results were unaffected (Phase 15 recomputed per step; the
  latency runs captured at the length they timed). Now the selection is
  sized for the top of the bucket and returned sorted, and the kernel reads
  a per-sequence count, computed on the GPU every step, of how many of the
  best pages to attend: the nominal ratio of each sequence's *current*
  length. This also fixes ragged batches, where a short sequence used to get
  a budget sized for the longest.
* **Graphs were keyed (batch, bucket) only**, so a set_sparse between steps
  replayed a graph captured under another budget. Now (batch, bucket, ratio).
* **Admission checked only the prompt.** Two requests whose prompts fit
  exhausted a 12-block pool mid-decode (`OutOfBlocks`). Admission now
  reserves prompt + maximum output, after what running requests are owed.
* **Page bounds are kept for the whole run** whenever a policy may sparsify,
  dense steps included: the policy can switch a running batch to sparse at
  any step.

## Evaluation — fixed before it is run

`phase17_workload`: one seeded mix of 40 requests (prompts 40% 2K, 25% 4K,
15% 8K, 12% 16K, 8% 30K; outputs 64/128/256), served through the engine under
CUDA graphs, so the running batch and context drift as requests arrive and
finish. Strategies: dense, fixed-50, fixed-37.5, adaptive-balanced,
adaptive-relaxed; two rounds in alternating order. Graph-capture time is
excluded (an adaptive strategy captures more graphs; that one-off cost would
count against exactly the strategy being tested). Prefill is dense
everywhere and timed apart.

**"Outperform"**, defined now: an adaptive tier outperforms the fixed budget
it can reach if its decode throughput is within 5% of that budget's while
its expected answer loss is lower — Pareto-better: nearly as fast, cheaper
in quality. Expected loss is an estimate, from Phase 15's descriptive
per-budget rates weighted by the tokens each budget generated.

**Prediction:** adaptive-relaxed within 5% of fixed-37.5's throughput with
lower expected loss, because it runs dense where sparsity buys nothing;
adaptive-balanced likewise against fixed-50; all sparse strategies faster
than dense, by less than Phase 14's peak, since this mix is mostly short
prompts.

## First workload: adaptive matched fixed exactly — a test that tested nothing

| strategy | decode tok/s | vs dense | tokens dense / 50% / 37.5% | expected loss |
| --- | ---: | ---: | --- | ---: |
| dense | 275.4 | 1.00x | 100 / 0 / 0 | 0.00% |
| fixed-50 | 349.4 | 1.27x | 0 / 100 / 0 | 0.40% |
| fixed-37.5 | 388.5 | 1.41x | 0 / 0 / 100 | 1.20% |
| adaptive-balanced | 348.8 | 1.27x | 0 / 100 / 0 | 0.40% |
| adaptive-relaxed | 388.2 | 1.41x | 0 / 0 / 100 | 1.20% |

The prediction (adaptive cheaper in quality) was wrong, and the reason is the
test: all 40 requests (~9K-token prompts on average) were submitted at once
with up to 16 running, so the batch stayed large and contexts long
throughout — 6 graphs in the whole run. Calibration says sparse wins in that
regime, so the policy chose it on every step: correct, and indistinguishable
from fixed. Adaptivity can only beat a fixed setting when traffic passes
through conditions where sparse loses.

## Calibration (after the bucket fix)

At large shapes sparse matches or beats Phase 15 — 37.5%: 1.47x at b8 / 32K,
1.51x at b16 / 16K, 1.49x at b32 / 8K. **At small batches it regressed:**
0.90-0.97x at batch 1, against Phase 15's ~0.97-1.0x. The per-sequence budget
was computed with several torch ops per layer, ~200 extra launches per step,
which only matters when steps are short. Fixed: the sparse kernel computes
`max(recent + 1, ceil(ratio x pages))` inline from the sequence length it
already reads (float32 `ceilf`; the reference `budgets()` now rounds the same
way — at 0.3 x 10 pages, float32 gives 3 where float64 gives 4). Calibration
is re-run after the fix.

## Second workload — fixed before it is run

Requests arrive over time, in decode steps (identical traffic relative to
each strategy's progress), in four phases: quiet (12 requests, 1-4K prompts,
one every 140 steps — usually alone), burst (16 requests, 8-30K, all at
once), medium (12 requests, 2-16K, one every 20 steps), quiet again (8).
Reported additionally: tokens generated at batch <= 2, the regime where
sparse is slower than dense. The definition of "outperform" is unchanged.

**Prediction:** in the quiet phases adaptive runs dense — faster than fixed
sparse there and free in quality; in the burst it matches fixed sparse.
Overall each adaptive tier is within 5% of its fixed counterpart's
throughput, or ahead, at lower expected loss. If quiet traffic is a small
share of tokens the difference will be small — adaptivity pays exactly as
much as traffic spends at low load.

## Result: varying traffic — both adaptive tiers outperform, by the fixed definition

Recalibrated after the kernel fix: at batch 1, 37.5% runs at 0.96-1.03x of
dense (0.92-1.00x before); 50% still dips to 0.93x at batch 1 / 4K — where the
policy runs dense. Large shapes unchanged (37.5%: 1.55x at b8 / 32K, 1.50x at
b16 / 16K, 1.49x at b32 / 8K).

48 requests, 386K prompt tokens, quiet / burst / medium / quiet; two rounds:

| strategy | decode tok/s | vs dense | dense / 50% / 37.5% tokens | expected loss |
| --- | ---: | ---: | --- | ---: |
| dense | 95.6 | 1.00x | 100 / 0 / 0 | 0.00% |
| fixed-50 | 103.7 | 1.08x | 0 / 100 / 0 | 0.40% |
| fixed-37.5 | 107.2 | 1.12x | 0 / 0 / 100 | 1.20% |
| adaptive-balanced | 103.0 | 1.08x | 54 / 46 / 0 | 0.18% |
| adaptive-relaxed | 105.4 | 1.10x | 46 / 0 / 54 | 0.65% |

37% of tokens were generated at batch <= 2 (identical traffic for every
strategy). Adaptive-balanced is 0.7% slower than fixed-50 at 0.18% expected
loss against 0.40%; adaptive-relaxed 1.7% slower than fixed-37.5 at 0.65%
against 1.20%. **Both within 5% at lower expected loss: both outperform.**

**Answer to Q14: yes — not by being faster, by being nearly as fast at about
half the quality cost.** The prediction was half right: lower cost, yes; but
adaptive is 1-2% *slower* than fixed, not faster. The 5% threshold forgoes
small real gains (37.5% is 3-5% faster at batch 4) to save quality; it is the
price of quality, and `min_gain` is the knob.

Also shown: on realistic mixed traffic sparsity buys 1.08-1.12x, against up to
1.5x at peak shapes — 37% of tokens ran where it buys nothing; and prefill
(~142 s, dense under every strategy) dominates decode (~67 s): for this
traffic, prefill is where further speed would come from.

**Caveat:** expected loss is Phase 15's per-budget rates weighted by tokens,
not quality measured on this traffic (random-token prompts, chosen for
timing). The quality advantage is inferred, and as precise as those rates —
which is not very.
