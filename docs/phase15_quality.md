# Phase 15 — sparse attention quality study

> **Outcome.** No sparse budget passed a confirmatory test. Pooled over the seeds run, net answers lost were about 0.4% at 50% of pages, 1.2% at 37.5% and 5.5% at 25% (where the KL criterion also failed on every seed), all in question answering. The oracle shows that page selection, not sparsity, is the limit. This note is chronological: earlier verdicts were revised by later runs, and the conclusion is near the end.

> How sparse can attention become before useful information is lost?
> (methodology §22)

`benchmarks/runners/phase15_quality.py`, one `--task` per measurement, each
saving its own results file; `--task curves` applies the criteria below and
draws the trade-off plots.

## Pass criteria — fixed before any result was seen

Applied to each budget on the real GPU path (CUDA indexer + sparse kernel):

1. **Retrieval and QA.** Paired failures — cases dense answers correctly and
   the budget does not — summed over needle, multikey, vartrack and qa, at
   most **2% of dense's correct answers**, with at least one allowed so a
   small sample is not failed by a single case.
2. **Language modelling.** Mean KL from dense **<= 0.01** at both 8K and 32K.
3. **Ordinary generation** is reported, not judged: no threshold was
   proposed for it in advance, and inventing one after seeing data is what
   fixing criteria first exists to prevent.

## Tasks

| task | what | why |
| --- | --- | --- |
| needle | passkey at 10/25/50/75/90% depth, 4K-32K | retrieval, up to the model's full context |
| multikey | four named passkeys, asked for one | distractor pages resemble the right one: hard for a bound-based indexer |
| vartrack | `VAR A = 41823 ... VAR D = VAR C` across the context | multi-hop: one dropped page breaks the chain |
| qa | SQuAD paragraph + question among other articles' paragraphs | real questions, distant evidence |
| text | continuation at 8K and ~32K: KL, top-1 vs dense | language modelling, thousands of tokens |
| gen | ~2K-token prompts, 128 greedy tokens vs dense | ordinary use |
| latency | whole decode steps, every budget, b1/8K, b8/32K, b16/16K | the latency axis of the curves |

Policies: the real GPU path at every budget, and the oracle (perfect page
selection) on retrieval and QA up to 16K as the ceiling — its fp32
reference is several times slower, and Phase 14 already measured the ceiling
on text. Answers count as correct when a gold answer appears in the output
after normalising case, punctuation and articles (RULER-style matching).
Accuracies carry 95% Wilson intervals; the comparison that decides is the
*paired* one against dense.

Every case keeps context + question + answer within its stated length, and
the "32K" text windows shrink to fit their continuation: Qwen2.5-1.5B was
trained on 32,768 positions, and going past them would mix position
extrapolation into the sparsity measurement.

## Quick pass (`--quick`, ~20 min): what it showed, and two instrument fixes

Verdicts on the quick sample: **50% PASS, 25% PASS (1 paired failure of 29
dense-correct — exactly the limit), 12.5% FAIL (4 of 29, ~14% against 2%),
6.25% and 3.1% FAIL.** The oracle failed once at 12.5%: perfect selection
would still pass there, so 12.5%'s failures belong to the indexer, not to
sparsity itself. Multikey matched dense down to 6.25% (7/8) — the prediction
that it would break first was wrong; needle and QA broke first.

Whole steps (batch 8 / 32K): 1.32x at 50%, 1.70x at 25%, 1.97x at 12.5%,
2.14x at 6.25%, 2.24x at 3.1%. At batch 1 / 8K, 0.97x-1.08x.

Two instruments needed fixing before the full run:

1. **vartrack: dense scored 1/12.** A 1.5B model could not follow a four-hop
   chain with distractors, so the task contributed almost no dense-correct
   cases and multi-hop was untested. It now defaults to two hops
   (`--hops`), and its difficulty is calibrated on **dense alone**
   (`--dense-only`) before any sparse run. These quick results include
   sparse numbers, so the change rests only on dense's failure; the pass
   criteria are untouched.
2. **gen had no control.** Even 50% matched dense on 1 of 4 outputs, first
   difference at a median of 12 tokens — uninterpretable without knowing how
   fast two *correct* dense implementations drift apart on the same
   high-entropy Wikitext prompts. The gen task now also runs dense through
   the Triton kernel, as that control.

`curves` now also breaks paired failures down by context length.

## vartrack: dropped

Calibrated on dense alone at two hops (`--dense-only --quick`): **0/12**. By
the rule set before calibrating — if dense still fails at two hops, drop the
task rather than simplify it into a second needle test — it is out of the
full run. The pass criteria are unaffected: vartrack supplied one of the
quick pass's 29 dense-correct cases. **Limitation:** sparsity's effect on
multi-hop retrieval is unmeasured here, because Qwen2.5-1.5B cannot do
multi-hop variable tracking even with full attention; answering it needs a
larger model.

## Full run: results

Needle 40 cases, multikey 24, QA 36; dense correct on 39, 22 and 22 (83
total). Text: 4 windows x 256 tokens at 8K, 2 at ~31K. Gen: 16 prompts.

| budget | paired failures / gains (of 83; 1 allowed) | KL 8K | KL 31K | verdict |
| ---: | ---: | ---: | ---: | --- |
| 50% | 0 / 0 | 0.0032 | 0.0007 | **PASS** |
| 25% | 4 / 1 | 0.0146 | 0.0051 | **FAIL** (both criteria) |
| 12.5% | 11 / 3 | 0.0453 | 0.0163 | FAIL |
| 6.25% | 25 / 0 | 0.0943 | 0.0381 | FAIL |
| 3.1% | 54 / 1 | 0.1804 | 0.0701 | FAIL |

Paired failures by context at 25%: 1/21 at 4K, 1/23 at 8K, 0/19 at 16K,
2/20 at 32K — not confined to long contexts. With ~20 cases per cell, 0/19
is consistent with true failure rates up to ~15%; choosing a per-length
policy from these cells after seeing them would be the post-hoc slicing the
fixed criteria exist to prevent.

**The indexer, not sparsity, is what fails.** Up to 16K the oracle loses
nothing at 25% (0 failures) and one case at 12.5%; the GPU path loses two at
25% over the same lengths. Perfect page selection would pass at 25%.

**Generation control:** dense on the Triton kernel matches dense on the CUDA
kernel on 14/16 outputs for all 128 tokens. Sparse divergence — median first
difference 19 tokens even at 50% — is a real change in behaviour, not
numerical noise. Whether the divergent text is worse is not measured.

Whole steps (batch 8 / 32K, batch 16 / 16K): 1.32x at 50%, 1.71x at 25%,
1.97x at 12.5%, 2.16x at 6.25%, 2.27x at 3.1%. Batch 1 / 8K: 0.97x at 50%.

**Phase 15 result: sparse decode is validated at 50% of pages — ~1.3x where
attention dominates the step, nothing at batch 1. 25% buys 1.7x but loses
retrieval and language-modelling quality with this indexer; the oracle
shows the loss is in page selection.** The prediction that 25% would pass
was wrong.

## Indexer bake-off — protocol fixed before any result

`benchmarks/runners/phase15_bakeoff.py`. At 25% of pages, in reference math:

| candidate | change | build cost |
| --- | --- | ---: |
| bounds | today's indexer (baseline) | 0 |
| bounds+window8 | 8 recent pages kept, not 2 | 0 |
| bounds+dense2 | first 2 layers dense (Quest) | 0 |
| mass | bounds as estimated attention mass, summed over the group (oracle-like ranking) | 1 |
| mean | the same estimate from q . mean(K): an estimate, one vector per page | 2 |
| rerank | bounds pick 2x the budget; exact key scores keep the best | 3 |
| mass / mean / rerank + dense2 + window8 | each with both cheap tweaks | 1 / 2 / 3 |
| oracle | the ceiling; never selected | - |

**Rule.** A development set from a new seed (default 1) — new text, new
questions; the starting points of filler and text windows depend on the
seed, so a new seed is genuinely new data (seed 0 keeps Phase 15's). The
winner is the *cheapest* buildable candidate whose mean development KL (8K
and ~31K) is within 10% of the best. Paired failures are reported, not
ranked on: ~25 dense-correct cases are too few to rank by. The winner is
then built on the GPU path and judged on a fresh seed with the unchanged
Phase 15 criteria. The 83 Phase 15 cases are never used for selection.

## Bake-off result (seed 1, 25% of pages)

| candidate | mass kept | KL 8K | KL ~31K | mean KL | paired fail / gain (of 32) |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 0.957 | 0.0022 | 0.0008 | 0.0015 | 0 / 1 |
| rerank+dense2+window8 | 0.954 | 0.0034 | 0.0009 | 0.0022 | 0 / 1 |
| rerank | 0.947 | 0.0034 | 0.0014 | 0.0024 | 0 / 1 |
| mass+dense2+window8 | 0.942 | 0.0059 | 0.0020 | 0.0039 | 5 / 2 |
| mass | 0.935 | 0.0059 | 0.0023 | 0.0041 | 3 / 1 |
| bounds+dense2 | 0.923 | 0.0078 | 0.0039 | 0.0059 | 1 / 1 |
| bounds+window8 | 0.916 | 0.0075 | 0.0043 | 0.0059 | 3 / 2 |
| bounds | 0.914 | 0.0080 | 0.0045 | 0.0062 | 1 / 2 |
| mean | 0.949 | 0.0204 | 0.0012 | 0.0108 | 3 / 1 |
| mean+dense2+window8 | 0.956 | 0.0205 | 0.0012 | 0.0109 | 3 / 1 |

Selected by the rule: **rerank+dense2+window8**, near the oracle. The
prediction (mass+dense2+window8) was wrong: the two cheap tweaks barely
help (-5% each), summed mass does real work (-34%), exact reranking most of
it (-61%). Mean keys keep 95% of the mass yet fail badly at 8K: averaging a
page erases the one spiky key that matters, so mass kept alone is not a
sufficient measure.

**A flaw in the rule:** "cost" meant build effort, not runtime. Reranking
reads the keys of twice the budget to score them. Built simply, 25% with
reranking reads ~56% of dense's bytes — the same as 50% with plain bounds,
which already passes. Fused (scoring keeps its exact scores, attention
reads only the selected pages' V), ~44%: about 1.45x at batch 8 / 32K
against 50%'s 1.32x. Cost should have been bytes read.

## Follow-up hypothesis — fixed before it is run

Not the bake-off's selection, so tested on its own terms:

* **Hypothesis:** summed-mass scoring makes 25% of pages pass the Phase 15
  criteria, unchanged.
* **Fresh data:** seed 2 — neither Phase 15's seed 0 nor the bake-off's seed
  1 — at Phase 15's full sample sizes.
* **Control on the same cases:** bounds scoring at 25%, seed 2. Without it a
  pass could mean seed 2 is easier.
* **Accepted either way;** no other seeds tried.

GPU path: `page_index_heads` writes each query head's bound; summed mass is
computed from them in torch, inside the CUDA graph
(`set_sparse(..., scoring="mass")`, `phase15_quality --scoring mass`).

## Follow-up result: summed mass at 25% — FAIL

Seed 2, Phase 15's full sample sizes, with bounds on the same cases as the
control:

| scoring | paired failures / gains (of 87; 1 allowed) | KL 8K | KL ~31K | verdict |
| --- | ---: | ---: | ---: | --- |
| bounds (control) | 5 / 0 | 0.0101 | 0.0043 | FAIL |
| summed mass | 4 / 1 | 0.0077 | 0.0034 | FAIL |

Mass is genuinely better on identical cases — KL ~23% lower at both lengths,
enough to pass the KL criterion that bounds narrowly fails — but retrieval
improves by one case where the limit is one. Failures sit at 4K-16K (none at
32K); a better *estimate* still misses the pages that matter, which only
exact reranking avoided in the bake-off. Seed 2 is easier than seed 0
(bounds' KL at 8K 0.0101 against 0.0146), which is why the control was run.

**Mass scoring also costs speed:** converting per-head bounds to estimated
mass adds about a dozen small operations per layer, ~2-3 ms per step. At
25%: 1.56x at batch 8 / 32K (bounds 1.71x), and 0.90x at batch 1 / 8K —
slower than dense. Even a pass would have bought 1.56x, not the 1.7x the
follow-up was framed around; that cost was not considered when proposing it.

Outcome accepted as fixed beforehand: no further seeds.

## Second follow-up — fixed before it is run: 37.5% with bounds, and a 50% replication

* **Hypothesis:** 37.5% of pages with plain bounds scoring passes the Phase 15
  criteria, unchanged. Motivation: interpolating measured steps at batch 8 /
  32K (37.9 ms at 50%, 29.4 at 25%) puts it near 33-34 ms, ~1.5x — about
  what fused reranking was estimated to buy, with no new kernel.
* **Replication:** 50% on the same cases. It is the only passing result, on
  one seed; if it fails here, it is not validated.
* **Fresh data:** seed 3 (seeds 0, 1, 2 have all been seen). Phase 15's full
  sample sizes. Both outcomes accepted; no further seeds.
* **Prediction:** KL passes (~0.007 at 8K by interpolation); retrieval is a
  coin flip (0 failures at 50%, 4-5 at 25%; the limit is 1); no gain at
  batch 1.

Choosing 37.5% after seeing 25% fail and 50% pass is a new hypothesis, which
is legitimate only because it is judged on unseen data and every attempt —
this one included — is reported.

## Second follow-up result (seed 3), and a revised retrieval rule

| budget | paired failures / gains (of 85) | KL 8K | KL ~31K | original rule | step, b8 / 32K |
| ---: | ---: | ---: | ---: | --- | ---: |
| 50% | 2 / 3 | 0.0028 | 0.0010 | FAIL | 38.13 ms (1.32x) |
| 37.5% | 2 / 3 | 0.0062 | 0.0020 | FAIL | 33.63 ms (1.50x) |
| 25% | — | — | — | — | 29.51 ms (1.71x) |

By the criteria fixed in advance, both fail — **including the 50%
replication.** Needle and multikey were perfect at both budgets; every flip
was in QA, where dense itself is right only 61% of the time, and the flips
went both ways: 2 lost, 3 gained, net +1 for sparse. KL predicted ~0.007 at
37.5%, measured 0.0062; 1.5x as predicted.

**Why the original rule was unreliable.** "At most 2% of dense-correct" with
85 cases rounded down to 1 allowed — in practice stricter than 2% — and
counted failures only. The chance of failing a configuration purely by luck,
with 1 allowed of 85, by its true failure rate:

| true failure rate | chance of 2+ failures (FAIL) |
| ---: | ---: |
| 0.5% | 7% |
| 1% | 21% |
| 2% (the stated tolerance) | 51% |

A configuration within the stated tolerance fails half the time: the rule
cannot separate noise flips from damage at this sample size, which is how
50% passed on seed 0 and failed on seed 3. Raising the limit to 2 — the
count 37.5% happened to get — would fit the rule to the result; it is not
done.

**Revised retrieval rule** (KL criterion unchanged): *net* paired loss —
failures minus gains — at most 2% of dense-correct, minimum 1. Noise flips
answers both ways and cancels; damage goes one way. The justification is
independent of 37.5%: the original rule failed its own replication, a
configuration at KL 0.0028.

Re-scored with the revised rule — **exploratory**, because these runs
informed the revision:

| run | budget | failures / gains | net loss | revised |
| --- | ---: | ---: | ---: | --- |
| seed 0 | 50% | 0 / 0 | 0 | pass |
| seed 0 | 25% | 4 / 1 | 3 | fail (KL fails too) |
| seed 0 | 12.5% | 11 / 3 | 8 | fail |
| seed 2, bounds | 25% | 5 / 0 | 5 | fail |
| seed 2, mass | 25% | 4 / 1 | 3 | fail |
| seed 3 | 50% | 2 / 3 | -1 | pass |
| seed 3 | 37.5% | 2 / 3 | -1 | pass |

## Confirmatory run — fixed before it is run

* **Seed 4**, unseen; Phase 15's full sample sizes; bounds scoring.
* **Budgets:** 50%, 37.5%, and 25% as a negative control — if the revised
  rule passes 25% too, it is too lenient to mean anything.
* **Noise floor in the same run:** dense on the Triton kernel
  (`--dense-control`), numerically different but not sparser, on retrieval,
  QA and text. Its paired flips and KL are what noise alone produces.
* **Rule:** the revised one; the original printed beside it. Accepted
  either way.
* **Prediction:** 50% and 37.5% pass, 25% fails; the Triton control shows
  some flips both ways with net loss near 0, and KL near 1e-5.

## Confirmatory result (seed 4) — every budget fails

| configuration | failures / gains (of 84) | net loss | KL 8K | KL ~31K | revised rule |
| --- | ---: | ---: | ---: | ---: | --- |
| dense on Triton (noise floor) | 0 / 0 | 0 | 0.0001 | 0.0001 | — |
| 50% | 2 / 0 | 2 | 0.0020 | 0.0007 | FAIL |
| 37.5% | 3 / 0 | 3 | 0.0048 | 0.0015 | FAIL |
| 25% | 6 / 0 | 6 | 0.0108 | 0.0036 | FAIL (KL too) |

The prediction (50% and 37.5% pass, 25% fails) was wrong. **The noise floor
is zero:** dense through a numerically different kernel flips no answers.
The flips at 50% and 37.5% are therefore real effects of sparsity, not
numerical noise. Seed 3's "2 lost, 3 gained" was read as symmetric churn
that cancels; seed 4 (all losses) and the control refute that reading — it
was a story fitted to one seed.

A second error: the revised rule kept the round-down that makes "2% of
dense-correct" allow only 1 of 84 — the very defect shown, with numbers,
when the rule was revised. It is not corrected and re-scored here: that
would be a third rule written after seeing data.

## Phase 15 conclusion

**No sparse budget passed a confirmatory test.** Pooled over every seed run
(descriptive — several were exploratory):

| budget | net answers lost (of dense-correct) | KL 8K across seeds | step, b8 / 32K |
| ---: | ---: | --- | ---: |
| 50% | 1 of 252 (0.4%) | 0.0020-0.0032 | 1.32x |
| 37.5% | 2 of 169 (1.2%) | 0.0048-0.0062 | 1.50x |
| 25% | 14 of 254 (5.5%) | 0.0101-0.0146: fails on every seed | 1.71x |
| 12.5% | 8 of 83 (9.6%) | 0.0453 | 1.97x |

Every answer lost at 50% and 37.5% was a QA question: needle and multikey
retrieval were perfect at both budgets on every seed. Exact facts survive
sparsity; borderline comprehension answers move.

Sparse decode carries a small, real quality cost at every budget tested,
traded for speed — the sparsity / quality / latency curve the plan asks for,
not a single safe point. A pass/fail test at ~84 cases per seed cannot
resolve differences this small: distinguishing 0.4% from 0% reliably needs
well over a thousand cases. No further seeds are run to chase a pass. The
curve is the input to Phase 17's adaptive runtime: dense where every answer
matters, 37.5-50% where ~1% of borderline answers is a fair price for
1.3-1.5x.

Limitations: one model (Qwen2.5-1.5B), one GPU (T4), training-free
page-level indexers only; multi-hop retrieval untested (the model cannot do
it densely).
