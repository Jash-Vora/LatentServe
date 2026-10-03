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

## Oracle study result (T4, `--quick`: 2 windows x 64 tokens, 3 needles per length)

| policy | pages | ppl | KL vs dense | top-1 agree | mass kept | needles 4K / 8K / 16K |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense | 100% | 6.372 | 0 | 100% | 1.000 | 3/3 3/3 3/3 |
| oracle | 25% | 6.384 | 0.0025 | 95.2% | 0.946 | 3/3 3/3 3/3 |
| oracle | 12.5% | 6.377 | 0.0099 | 93.7% | 0.896 | 3/3 3/3 3/3 |
| oracle | 3.1% | 6.748 | 0.0597 | 88.1% | 0.767 | 3/3 3/3 3/3 |
| bounds | 50% | 6.380 | 0.0014 | 98.4% | 0.959 | 3/3 3/3 3/3 |
| bounds | 25% | 6.409 | 0.0086 | 97.6% | 0.895 | 3/3 3/3 3/3 |
| bounds | 12.5% | 6.515 | 0.0339 | 93.7% | 0.825 | 3/3 3/3 2/3 |
| bounds | 3.1% | 7.750 | 0.1766 | 81.7% | 0.681 | 2/3 2/3 1/3 |
| window | 50% | 6.337 | 0.0065 | 98.4% | 0.950 | 1/3 1/3 1/3 |
| window | 25% | 6.389 | 0.0249 | 94.4% | 0.910 | 1/3 1/3 1/3 |
| window | 6.2% | 8.205 | 0.2360 | 82.5% | 0.724 | 0/3 0/3 0/3 |

**The gate passes.** Qwen's decode attention is concentrated: with perfect
selection 12.5% of pages keep ~90% of the attention mass and every needle is
found, even at 3.1%. The bounds indexer needs about twice the oracle's
budget for the same quality — bounds at 25% and oracle at 12.5% keep 0.895
and 0.896 of the mass, with KL 0.0086 and 0.0099. Query awareness is
essential, and only the needles show it: by perplexity the sink+recent
window looks fine at 25-50% (at 50% even below dense — noise over 126
tokens), but it finds one needle in three at every length.

Operating points: **25% safe** (4x fewer pages; KL ~0.009, every needle),
**12.5% borderline** (one miss at 16K, KL 0.034). With three needles per
length, single misses are noise-level — bounds at 6.25% scored better than
at 12.5% — so the full run (`phase14_oracle` without `--quick`) settles 12.5%.

## Stages 3-4: the kernels

`kernels/cuda/paged_sparse_fp16.cu`, `kernels/cuda/paged_sparse.py`:

* `page_bounds_update` — page min/max kept on write: one launch per layer per
  step (INT8's write side cost 280 extra kernels per step; this costs 28).
* `page_index` — the indexer: each page's Quest bound, maximised over the
  six query heads of the group; sink and recent pages forced in (+inf),
  pages past the end forced out (-inf). 63 registers.
* `torch.topk` — a fixed number of pages per call, `ceil(ratio x capacity)`,
  because a captured graph needs fixed shapes.
* `paged_sparse_fp16` — the dense fp16 kernel with its page loop over the
  selected list. 168 registers and no spills, identical to the dense kernel:
  a selected page is read exactly as any page is, so selection adds no
  irregular gather.

`benchmarks/runners/phase14_sparse_bench.py` times dense against the whole
sparse pipeline at 50/25/12.5/6.25%, batch 1/4/16, 4K-32K, under graph
replay, with the indexer's share and the write-side cost reported on their
own.
