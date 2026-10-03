# Phase 15 — sparse attention quality study

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
