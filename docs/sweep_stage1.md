# Final sweep, stage 1 — single GPU

> **Outcome.** The sweep produced about 260 cells, and `sweep_stage1 --report` prints their tables, which are saved in `sweep_stage1_results.md`; the README summarises them. This note is chronological: the plan and predictions come first, then the problems found along the way (the context limit, concurrency limits, INT8 pool sizing, vLLM's gradual admission, drifting pool sizes). Where the plan and a later section disagree, the later section is right.

`benchmarks/runners/sweep_stage1.py`. One T4 ("GPU T4" accelerator). Resumable:
every cell is saved on completion and skipped on rerun.

## What is measured

| section | measure | configurations |
| --- | --- | --- |
| A. decode step | engine-level step time once every request decodes; batch 1, 4, 8, 16, 32 x context 2K, 8K, 16K, 32,768 (filled to the limit), where it fits; 48 timed steps; 2 rounds | LS dense, sparse 50%, sparse 37.5%, INT8; vLLM |
| B. prefill | time to first token, one request at a time; prompts 1K-32,768 (the last: 32,767 + 1 output token); 8 distinct prompts each | LS dense, INT8; vLLM |
| C. serving | burst (96 requests at once); chat (16 conversations x 5 turns); Phase 17's varying traffic in wall-clock time; 2 rounds each | LS dense, sparse 37.5%, adaptive, INT8; vLLM. Chat: prefix caching on and off, both engines |
| C. load curves | capacity probe, then Poisson arrivals at 10, 25, 40, 55, 70, 80, 90, 100, 110% of it; one round | same as serving |

## Ground rules

* The same concurrency limits, context limit (32,768 — the model's own), prompts and seeds for
  both engines: **32 for sections A and B** (A's largest batch), **16 for the
  serving workloads**. vLLM fixes `max_num_seqs` at startup, so it gets one
  engine for A/B and two for serving (prefix caching on, then off). vLLM
  does not detokenize (LatentServe never does).
* *Fixed after the first minutes of the run:* the first version applied the
  serving limit of 16 to section A as well, so batch 32 could never run and
  was reported as "could not all decode at once" — a configuration error
  presented as memory. Skips now state their cause: concurrency limit, or
  memory.
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

## An engine bug found by section B

Section B asks for exactly one output token per request, so a request is
finished the moment prefill produces it. The engine appended every prefilled
request to the decode batch and *then* retired it if finished — and
retirement frees the slot without removing the request from the batch, so the
next decode step met a request with no slot and crashed. Real traffic would
hit the same path whenever a request's first token is end-of-sequence; fixed
lengths had hidden it all project. Prefill now retires a finished request
*or* adds it to the batch, never both. Tests cover one-token requests alone
and alongside others, end-of-sequence as the first token, and one-token
requests with prefix caching — all three fail on the old code. Sections A and
C are unaffected: their requests generate dozens of tokens and never finish
at prefill.


## Contexts kept within the model's limit

Qwen2.5-1.5B's limit of 32,768 positions covers the prompt *and* every
generated token together. The first version started section A's 32,768 cells
with a 32,768-token prompt and then decoded on top — past the end. LatentServe,
which never checked, ran past the trained range; vLLM, which does, refused to
start. Now a 32,768 cell **fills the cache to the limit**: its prompt is
32,768 minus the tokens decoded on top (32,696 with 48 timed steps), so the
timed steps run in the last stretch before the limit. Section B's longest
prompt is 32,767, its one output token taking the last position. Both
engines, no overrides; each cell records the prompt it actually used, and a
test checks every cell fits.

## Why vLLM's section A took hours, and what was waste

Timing decode at a long context first requires prefilling every request's
prompt: vLLM's 32K row alone is ~4 million prompt tokens (batches 1-32, two
rounds), at roughly 0.7K tokens/s — about a third of LatentServe's rate at
that length (section B). That part is inherent to measuring decode at long
contexts on an engine with slow long-prompt prefill.

The waste was in batches that cannot fit. LatentServe checks its pool before
starting; vLLM had no such check, so a non-fitting batch ran until vLLM proved
it could not hold it all — and the code then *drained* every remaining
request, prefilling each for nothing (~25 minutes per 32K cell). Now: a KV
capacity pre-check where the vLLM version exposes it (config or
`cache_config_info` metric), and otherwise the remaining requests are
*aborted* the moment the batch proves too big. Tests cover both paths.

## First full pass: three measurement bugs, found from the report

**INT8 never got its capacity.** Every pool was sized with fp16's bytes per
block, so INT8 received the same number of blocks as fp16 in half the memory:
it fit nothing fp16 did not, and the sweep measured INT8's cost without its
benefit. Pools are now sized from each cache type's real bytes per block
(fp16 17,408 / INT8 9,344 per layer per block: ~1.86x the blocks), tested
against the caches' actual tensors. fp16's sizing is unchanged. Affected:
section A's INT8 cells that did not fit, and section C's *varying* INT8 runs
(the one workload whose long burst fills the pool).

**Four vLLM cells reported "did not fit" fitted.** vLLM's capacity was fine —
325,440 tokens, more than LatentServe's 258,720-token fp16 pool, and the
pre-check read it correctly. But vLLM admits a big batch over many steps (its
8,192-token step budget is shared with the tokens being decoded) while already
decoding the early requests; with only 16 tokens of headroom, the first
request ran out before the 48 timed steps ended. And a cell timed short was
reported as "did not fit". Now the headroom grows with the admission stagger
(16 + batch x ceil(context / 8,192)), the 32,768 row's prompt shrinks by the
same amount so the cache still ends at the limit, and a cell timed short is
reported as such — never as a capacity result. A fake engine with vLLM's
shared step budget reproduces the failure at the old headroom and passes at
the new.

**LatentServe's batch 32 / 2K "did not fit"** — about 68K tokens, which
fits easily — were stale cells from the concurrency-limit bug, kept by resume.

The batch-1 and batch-4 cells of the 32,768 row (both engines) fitted and
timed fully with the old headroom; they are kept, measured with
32,696-token prompts — 4-16 tokens more than the new rule gives, under 0.05%.

## Decision: report the measured cells; mark the rest honestly

LatentServe's KV pool is sized from whatever GPU memory is free when its
engine is built, so cells that follow memory-hungry ones (large INT8 batches,
big CUDA-graph pools) can get a far smaller pool — 64 to 1,600 blocks were
seen, against ~15,000 (fp16) and ~29,000 (INT8). A cell skipped on such a
pool is **not measured**; it is not evidence that the batch does not fit. The
report marks these "not measured\*", lists every skipped cell with its recorded
reason, and keeps "did not fit" for skips that a correctly sized pool would
also have made. Cells that ran are valid timings either way — a smaller pool
does not change the speed of a batch that fits.

Known gaps this leaves in section A: INT8's extra capacity (batch 8 at 32K,
for instance) is not demonstrated for the cells that were skipped; the
write-up says so rather than claiming it. A deterministic fix — sizing every
pool from the free memory recorded once, after the model loads — is a few
lines and is left for a future run.

## Pool sizes that varied from 64 to 15,000 blocks

After the INT8 sizing fix, cells began skipping with "pool has 64" or "pool
has 1598" for pools that elsewhere held ~15,000 blocks. Not a GPU
out-of-memory: the sweep's own pre-check. The worker built each new engine
with `engine = factory(...)`, which keeps the *old* engine alive while the
factory runs; the pool is sized from free memory at that moment, so it saw the
old pool and its CUDA-graph memory still allocated. A pool's size depended on
its predecessor's, and which cells had been skipped by resume decided the
pattern — the first full pass, with no skips, happened to be regular. Probable
cause, not confirmed on the GPU; the fix does not depend on it being the only
one. The worker now drops the old engine and returns its memory before
building, every pool is sized from the free memory measured once after the
model loads, a pool under 256 blocks is an error, and section-A cells record
the pool they had. The new test fails without the one-line change.

No timing is affected: a cell either ran, with the batch it asked for, or was
skipped. Only skipped cells were ever wrong.
