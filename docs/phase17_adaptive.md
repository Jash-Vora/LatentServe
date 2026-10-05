# Phase 17 — adaptive runtime

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
