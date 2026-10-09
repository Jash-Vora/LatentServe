# Phase 13 — prefix caching

> **Outcome.** Time to first token fell 84% on a shared 2,048-token system prompt and 77% on multi-turn chat (hit rate 86–87%), live KV blocks fell 37% and 22%, and the overhead without sharing was about 1%, within noise. It works unchanged with the INT8 cache. Shared-prompt decoding was also 4–5% faster; the L2-cache explanation for that is a hypothesis.

> Shared prefixes should not be recomputed. Measure cache hit rate, TTFT,
> VRAM, throughput, cache overhead; then investigate whether prefix caching
> stacks with the memory-efficient KV cache. (methodology §20)

## Design (`cache/prefix_cache.py`)

* **Full 16-token blocks, chained hashes** — each block's key covers its own
  tokens and everything before it; two prompts share a block only if they are
  identical up to and including it (vLLM's automatic prefix caching).
  Matches are verified against the stored tokens and parent hash, so a hash
  collision cannot reuse the wrong block.
* **Admission** finds the longest cached run at the start of the prompt,
  attaches it (`attach_prefix`, a reference per block) and prefills only the
  rest (`prefill_slot(..., start=reused)`); at least the last prompt token is
  always prefilled, for the first output's logits.
* **Only full blocks are shared**; the cache is append-only, so a shared
  block is never written again — no copy-on-write.
* **The cache holds its own reference.** A block is in use while a sequence
  also holds it, evictable once only the cache does; evictable blocks stay
  until memory is needed (LRU), via the allocator's reclaim hook. Admission
  counts them as available — except blocks the incoming request reuses.
* **Registration** happens after prefill (prompt blocks, so concurrent
  requests can share them) and on retirement (every full block, generated
  tokens included — a conversation's next prompt contains this reply).
* Decode under CUDA graphs is unchanged; sparse page bounds live per block
  and are reused with it.

## Bugs caught while building it

* **INT8 under graphs finalizes a block one step late** — its K lives only
  in the sequence's fp16 residual until the next step. Sharing such a block at
  retirement would hand a later request K never written to the pool. Each
  cache now reports `shareable_blocks()`: all full blocks for fp16, only
  finalized ones for INT8 under deferred mode (eager INT8 quantizes at once).
* **An edit to `BlockAllocator.free()`** captured the pool return inside a new
  callback branch: with prefix caching on, shared blocks went back while still
  in use (a double free); with it off, nothing went back at all (a leak). The
  tests failed at once; a dedicated test now guards `free()`'s semantics.
* **A planned O(n) count** of reclaimable blocks, asked several times per
  step, would have cost milliseconds per step with a full pool — overhead the
  plan wants measured, but caused by laziness. It is kept incrementally from
  reference-count changes, and tested equal to the scan.

## Tests (`tests/test_phase13_prefix.py`)

Greedy outputs identical with prefix caching on and off (sequential and
concurrent); exact hit accounting (64 reused tokens of a 70-token system
prompt); an identical prompt still prefills its last token; a second turn
reuses blocks reaching into the first turn's reply; eviction under a
12-block pool stays exact; after all requests finish every block is free or
held only by the cache; collision-safe matching; INT8 stacks exactly; under
graphs, INT8 never shares an unfinalized block (GPU).

## Benchmark — fixed before running

`phase13_prefix`: three workloads served with prefix caching off and on —
*shared* (2048-token system prompt + distinct 256-token messages, 32
requests), *chat* (8 conversations x 5 turns, each resending its history),
*none* (32 unrelated 2304-token prompts). Arrivals in decode steps; two
rounds, alternating order; production setup.

## Result (T4; two rounds, alternating order; median of rounds)

| workload | kv | hit rate | TTFT p50 off -> on | prefill / request | decode off -> on |
| --- | --- | ---: | --- | --- | --- |
| shared | fp16 | 86.1% | 318.8 -> 51.1 ms (-84%) | 344.4 -> 58.5 ms | 337.0 -> 353.9 tok/s |
| chat | fp16 | 86.7% | 182.5 -> 41.6 ms (-77%) | 183.2 -> 42.1 ms | 351.2 -> 351.9 |
| none | fp16 | 0% | 374.9 -> 378.5 ms (+1.0%) | 373.5 -> 376.4 ms | 322.6 -> 321.4 |
| shared | int8 | 86.1% | 345.7 -> 60.2 ms (-83%) | 370.7 -> 67.8 ms | 315.0 -> 327.6 |
| chat | int8 | 86.7% | 192.3 -> 58.7 ms (-70%) | 193.5 -> 59.1 ms | 308.0 -> 316.5 |
| none | int8 | 0% | 383.8 -> 381.7 ms (-0.6%) | 382.6 -> 380.3 ms | 307.0 -> 307.1 |

Live KV blocks at peak (fp16): shared 1172 -> 738 (-37%), chat 856 -> 665
(-22%).

**What it shows.** Hit rates track the shared share of each prompt: 86.1% on the shared
system prompt, 86.7% on chat. With nothing to share the overhead is +1.0% (fp16) and
-0.6% (int8), against ~3% round-to-round noise, so indistinguishable from zero.
Shared-prompt decoding is also 4-5% faster with prefix caching, in both rounds and in
INT8. Hypothesis, not measured: every sequence in a batch reads the same physical pages
for the 2048-token prefix (~2 MB per layer, within the T4's 4 MB L2), so later sequences
hit L2. Chat shares less per batch and shows no gain, which is consistent.

**It stacks with INT8:** identical hit rates, TTFT -83% (shared) and -70%
(chat), no overhead without sharing. Chat gains less than fp16 because the
uncached part of each prompt still attends over the reused prefix, and
prefill reads INT8 pages more slowly. (Correctness: the CPU stacking test and
the GPU test that INT8 never shares an unfinalized block.)

**Instrument correction:** the first run reported 4706 peak blocks for
*none* with caching on (1172 off). That is not pressure — `peak_used`
counted blocks the cache keeps after requests finish, which a 16K-block pool
never needed to evict. The benchmark now reports peak *live* blocks
(held by running requests), from the prefix cache's own accounting.
