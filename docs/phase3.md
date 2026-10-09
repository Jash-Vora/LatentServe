# Phase 3 — Paged KV Cache

> **Outcome.** Paging's capacity gain equals the workload's max/mean sequence length: 6.6× over an oracle-sized contiguous cache on the mixed workload, and about 1× on uniform traffic. Block size barely matters (98.8–100% capacity efficiency from 1 to 256 tokens, under 1% TPOT difference), so 16 was chosen, which later became the granularity of prefix sharing in Phase 13. Paging's decode cost is exactly the cost of its gather, 10–70% from batch 1 / 4K to batch 4 / 16K, which the paged-attention kernel later removed.

Goal (docs/methodology.md Phase 3): a PagedAttention-style memory
manager, compared against the contiguous cache under variable sequence
lengths, concurrent requests, request termination and high utilization.
Metrics: fragmentation, usable cache capacity, allocation overhead,
latency impact.

## Set expectations first

**Paging is a capacity optimization and should make decode slightly
slower.** Attention can no longer read a contiguous tensor, so scattered
blocks are gathered into one each step — a real copy of the live KV, per
layer, per step. Phase 2 calibrates the cost: `repeat_kv` was 13x the
necessary traffic and dominated TPOT past 4K; this is 2x (one read, one
write) on a term that was 4-13% of decode bytes at batch 1.

Any result showing paging *faster* on a uniform-length benchmark is
measuring a mistake, not a win.

## What landed

| File | Role |
| --- | --- |
| `cache/block_allocator.py` | `BlockAllocator` (free list, refcounts, typed `OutOfBlocks`) and `BlockTable` (logical -> physical, grows on demand) |
| `cache/paged_cache.py` | `PagedKVCache`: block pool, scatter write, gather read, fragmentation accounting |
| `benchmarks/workloads/ragged.py` | request streams for Workloads A-E; reused by Phase 4 |
| `benchmarks/runners/phase3_paged.py` | capacity simulation (CPU) and paging latency cost (GPU) |
| `tests/test_phase3_paged.py` | allocator behaviour + paged/contiguous equivalence |
| `configs/phase3_paged.yaml` | held constant across the sweep |

`ContiguousKVCache` and `PagedKVCache` expose the same read/write
contract, so `GQAAttention` cannot tell them apart and
`allocate_cache(..., paged=True)` is the only difference between the two
arms. That interchangeability is what makes the comparison controlled —
and it is also how Phase 7's latent cache will drop in.

## Layouts differ, on purpose

Contiguous is head-major `[B, kv_heads, S, head_dim]`: SDPA wants
`[B, H, S, D]` and no transpose is needed on the read path.

Paged is `[num_blocks, block_size, kv_heads, head_dim]`, viewed flat as
`[num_blocks * block_size, kv_heads, head_dim]`. A token's KV has to be
one contiguous run so scattered blocks can be gathered by a single index
operation; the transpose moves to the read.

Two caches, two layouts, each following from how it is accessed. "Which
layout is faster" has no context-free answer, which is worth saying
plainly in the report.

## Run it

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH

pytest tests/test_phase3_paged.py -v          # no GPU, no download

# Experiment A — capacity. CPU, seconds.
python -m benchmarks.runners.phase3_paged --experiment capacity \
    --workloads mixed long_generation short_interactive \
    --block-sizes 1 8 16 32 64 128 256 --budget-gb 8 --num-requests 200

# Experiment B — latency cost. T4.
python -m benchmarks.runners.phase3_paged --experiment latency \
    --context-lengths 4096 8192 16384 --batch-sizes 1 4 \
    --latency-block-sizes 16 128
```

## Why most of Phase 3 needs no GPU

Fragmentation, usable capacity and allocation overhead are properties of
the allocation policy, not of the T4. Simulating 200 requests across a
seven-point block-size sweep and three workloads takes seconds on CPU
and produces the Gate 4 numbers directly. Only the latency comparison
needs the GPU. Spending T4 hours to rediscover arithmetic would be a
poor trade on a 30-hour weekly quota.

## Two contiguous baselines

The comparison depends entirely on what `max_model_len` a contiguous
cache was sized for, so the runner reports both:

* **generic** — sized for the longest request the server will accept
  (65536). What you get without workload knowledge.
* **oracle** — sized to the longest sequence this workload actually
  produces. Unachievable in practice, since it requires knowing the
  future. Included so paging has to beat the best possible contiguous
  configuration rather than a strawman.

Indicative simulation output (8 GiB budget, 60 requests, seed 0):

| workload | contiguous | oracle | paged (bs=128) |
| --- | ---: | ---: | ---: |
| mixed | 5,375 tok/GiB | 6,705 | **31,928** |
| short_interactive | 534 | 7,483 | 7,483 |

Paging's benefit is **heterogeneity**, not paging. On `mixed`
(max/mean sequence ratio 6.3) it serves ~4.8x the concurrency of even an
oracle-tuned contiguous cache. On `short_interactive`, where every
request is ~1K, it ties the oracle exactly — all it buys there is not
having to know the workload in advance. Reporting only the mixed number
would overclaim; reporting only the uniform one would miss the point.

Allocation overhead is ~0.01-0.05 us/token, i.e. nothing next to a
~31 ms decode step. The block-size trade is real but mild: smaller
blocks waste less tail (internal fragmentation is bounded by one block
per sequence) and cost more bookkeeping.

## Measured (8 GiB budget, 200 requests, seed 0; T4 for latency)

### Capacity gain equals the workload's max/mean length ratio

| workload | max/mean sequence | paged vs oracle |
| --- | ---: | ---: |
| mixed | 7.40 | 6.6x |
| long_prompt | 1.47 | 1.4x |
| long_generation | 1.10 | 1.1x |
| short_interactive | 1.10 | 1.0x |

This is not a coincidence, and it is derivable rather than empirical: a
contiguous cache reserves the *longest* sequence for every slot, paging
reserves roughly the *mean*, so the capacity ratio is max/mean. The
simulation confirms the prediction to within 12% across a 6.7x range of
heterogeneity.

Stated as a rule for the report: **paging buys you exactly the
heterogeneity of your workload, and nothing otherwise.** On uniform
traffic it ties an oracle-tuned contiguous cache. Its real-world value
on uniform traffic is that you do not have to *be* an oracle — the
generic contiguous baseline, sized for the longest request the server
accepts, is 46x worse on `short_interactive`.

### Block size barely matters — prediction falsified

Capacity efficiency stays between 98.8% and 100% from block_size 1 to
256, and TPOT differs by under 1% between 16 and 128. The expectation
that finer scattering would cost more locality was wrong: one token's KV
is 2 heads x 128 dims x 2 bytes = 512 contiguous bytes even at
block_size 1, which is already enough for a coalesced read. Scatter
granularity is irrelevant; total bytes moved is everything.

Allocation overhead falls with block size (0.27 -> 0.05 us/token) but
never approaches relevance against a ~30 ms decode step.

**Choose 16.** Capacity and latency are indifferent, so the tiebreaker
is Phase 13: block size is the granularity at which prefixes can be
shared, and 16 tokens is a far more likely common prefix boundary than
256. (This is also what vLLM uses, for the same reason.)

### Paging costs exactly what the gather costs

| batch / ctx | contiguous | paged (bs=16) | overhead | gather | implied |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 4096 | 29.34 ms | 32.39 | +10.4% | 231 MiB | 79 GB/s |
| 1 / 16384 | 43.78 | 52.55 | +20.0% | 903 MiB | 108 GB/s |
| 4 / 8192 | 32.88 | 48.91 | +48.7% | 1820 MiB | 119 GB/s |
| 4 / 16384 | 46.40 | 78.87 | +70.0% | 3612 MiB | 117 GB/s |

The last column is the extra bytes divided by the extra time. It lands
at 79-135 GB/s — the same band Phase 2 measured for every other
memory-bound operation on this card (85-146 GB/s). So the whole of
paging's latency cost is explained by one sentence: **the gather moves
2x the live KV per step, at the machine's bandwidth.** Nothing is
unaccounted for, and no profiling is needed to attribute it.

TTFT is unchanged within a few percent, which is the control: prefill is
compute-bound at O(S^2), so the same gather disappears into it.

### What this hands to Phase 11

A paged-attention kernel that walks the block table with an online
softmax removes the gather entirely, and the table above says exactly
what that is worth: up to 32 ms/token at batch 4 / 16K, ~70% of TPOT.
Phase 11 now has a target with a number attached rather than a hunch,
and a pre-registered prediction — a correct kernel should recover nearly
all of that gap and land within noise of contiguous.

## Gate 4 checklist — "Can paged KV improve memory utilization?"

- [x] `pytest tests/test_phase3_paged.py` green, including
      `test_paged_matches_contiguous_logits` at block_size 1, 4 and 16
- [x] capacity simulation run across all three workloads and the full
      block-size sweep
- [x] paging's advantage stated against the **oracle** baseline, not
      only the generic one
- [x] measured paging latency cost recorded at batch 1 and batch 4
- [x] a block size chosen, with the fragmentation/overhead numbers that
      justify it

*Reviewed at project close: every item is shown by the measurements above.*

## Deliberately deferred

* **Ragged execution.** The cache stores ragged batches and builds the
  padding mask, and the allocator frees one sequence at a time, but the
  GPU path still benchmarks uniform batches. Ragged *scheduling* is
  Phase 4's subject; doing it here would mean building a scheduler to
  test an allocator.
* **A real paged-attention kernel.** Walking the block table inside the
  kernel with an online softmax removes the gather entirely. That is
  Phase 11. Phase 3's job is to measure what the gather costs so that
  kernel is motivated by a number rather than a hunch.
* **Preemption and swapping.** The simulation records a capacity failure
  when a running sequence cannot grow; real systems preempt. The policy
  belongs to the scheduler, so it lands in Phase 4.
