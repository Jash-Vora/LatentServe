# Phase 13 — prefix caching

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

**Prediction:** shared — hit rate ~86% (2048 of 2304 tokens, all but the
first request), prefill per request and TTFT down ~80-85%; chat — hit rate
rising turn by turn, ~70-85% overall; none — hit rate ~0 and overhead under
1% on every metric; decode throughput unchanged within ~2% everywhere.
