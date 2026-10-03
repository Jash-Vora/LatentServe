# Phase 14 — sparse decode attention

The plan's Phase 14 (methodology §21): an indexer scores relevance, the top
pages are kept, retention ratios from 100% down to 3.125%, the indexer's own
cost counted, a real GPU sparse kernel — and the central question: *does the
GPU benefit, or does irregular access destroy the theoretical speedup?*

## One adaptation

DeepSeek's sparse attention uses a *trained* indexer. Qwen2.5 was never
trained with one, and training one is beyond this project. The indexer here
is training-free and works at page granularity (Quest-style): each page's
per-channel min and max of K bound q.k for every key in it,
`sum_d max(q_d min_d, q_d max_d)`. It reads two 128-value vectors per page
instead of sixteen K and sixteen V rows — about 6% of dense attention's
bytes — and because it selects whole pages, a sparse kernel reads them
exactly as it reads any page: no irregular gather. One page set per KV head,
scored across the group's six query heads, so each page is still read once
for all six. The first page (attention sink) and the two most recent pages
are always kept.

## Stages

1. Long-context evaluation harness — needle retrieval and long-text quality.
2. **Oracle study — the gate.** If even perfect page selection degrades
   quality at a budget, no kernel can make that budget safe.
3. The indexer on the GPU, page min/max maintained on write in one fused
   operation (the lesson of INT8's 280 extra kernels per step).
4. The sparse kernel: the fp16 CUDA kernel reading a selected page list.
5. Measurements per retention ratio: indexer, selection, attention, whole
   steps, and quality.

## Stages 1-2: the oracle study

`benchmarks/runners/phase14_oracle.py`, with the reference math in
`model/attention/sparse.py`. Prompts are prefilled densely and every decode
step attends only to selected pages — sparse attention is a decode-time
technique. Sixteen configurations: dense; and the **oracle** (the pages that
truly carry the most attention: the ceiling any indexer can reach), the
**bounds** indexer, and a **sink + recent window** (no query awareness: the
control) at 50%, 25%, 12.5%, 6.25% and 3.125% of pages.

* **Long text:** 4 windows of Wikitext; 8K tokens prefilled, the next 128
  fed one at a time as sparse decode steps. Perplexity, KL and top-1
  agreement against dense, attention mass kept, pages kept.
* **Needles:** a passkey at 10/50/90% depth in 4K/8K/16K of Wikitext. The
  question is fed as decode steps too: prefilled densely, the answer's first
  token would see full attention and the test would say nothing about
  sparsity.

Each context is prefilled once; the paged cache is rewound between
configurations (`PagedKVCache.rewind`). Dense's own needle accuracy is in
the table: where the model fails with full attention, that column is about
the model, not sparsity.

**How to read it.** Where the oracle degrades, no indexer can make that
budget safe. Where the bounds indexer trails the oracle, the indexer is the
problem. Where the window matches the bounds indexer, query awareness buys
nothing.
