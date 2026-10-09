# Phase 2 — GQA + KV Cache

> **Outcome.** LatentServe's own GQA and KV-cache path matches the Hugging Face oracle and cut peak memory at 8K by 7.2 GB (3,463 MB against 10,654 MB), so 16K runs where the reference runs out of memory. Decode at batch 1 is weight-bound, at about 30% of the T4's bandwidth. The sweep found two things: `repeat_kv` handed GQA's whole saving back (13× the necessary traffic), fixed by folding query heads into the query axis; and GQA is a capacity feature, not a speed one, since halving the cache made decode slightly slower and a 6× larger cache cost only 2.8% at batch 1.

Goal (docs/methodology.md Phase 2): build the conventional modern
inference baseline around Qwen, and make KV bytes/token, VRAM growth,
memory bandwidth and decode latency measurable. Everything MLA (Phase 7)
and DSA (Phase 14) later claim is a ratio against numbers produced here.

## What landed

| File | Role |
| --- | --- |
| `model/rope.py` | LatentServe's own RoPE (rotate_half convention, float32 tables, cast at apply time) |
| `cache/kv_cache.py` | `ContiguousKVCache`: preallocated `[B, kv_heads, max_seq, head_dim]` per layer, exact byte accounting, `KVHeadsMode` |
| `model/attention/gqa.py` | `GQAAttention`: borrowed q/k/v/o projections, RoPE, cache write, SDPA or explicit-math attention |
| `model/latentserve_qwen.py` | `LatentServeQwen`: our own decoder layer loop; prefill (optionally chunked), decode step, greedy generation, timing |
| `benchmarks/runners/phase2_gqa.py` | the sweep: context length x batch size x `kv_heads_mode`, with bandwidth accounting |
| `tests/test_phase2_gqa.py` | Gate 2 correctness, two tiers (fast CPU/random-weights, and the real checkpoint vs. the Phase 1 oracle) |
| `configs/phase2_gqa.yaml` | what's held constant across the sweep |

`config.py` gains `attention.kv_heads_mode`, `attention.impl` and
`runtime.prefill_chunk_size`. `benchmarks/schema.py` gains
`BenchmarkResult.extra`, a free-form dict so each phase can record what
only it measures without a schema migration; existing
`phase1_reference.jsonl` rows stay valid (the field defaults to `{}`).

From here on LatentServe is no longer Hugging Face with a stopwatch
attached. HF becomes purely the correctness oracle.

## Run it

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH

# Gate 2, fast, no GPU and no 3 GB download (tiny random Qwen2):
pytest tests/test_phase2_gqa.py -v

# Gate 2 on the real checkpoint (T4):
LATENTSERVE_REAL_MODEL_TESTS=1 pytest tests/test_phase2_gqa.py -v

# The sweep:
python -m benchmarks.runners.phase2_gqa --config configs/phase2_gqa.yaml \
    --context-lengths 1024 4096 8192 16384 --batch-sizes 1 4 8 \
    --kv-heads-modes native mha_sim --include-hf-baseline
```

## The GQA-configuration experiment

Qwen2.5-1.5B-Instruct is 12 query heads over 2 KV heads — group size 6,
fixed by the weights. "Vary GQA configuration" therefore cannot mean
retraining; it means varying what the *cache* stores:

* **`native`** — 2 KV heads. 28,672 B/token (2 x 28 layers x 2 heads x
  128 dim x 2 bytes).
* **`mha_sim`** — expand to 12 KV heads before caching, one per query
  head. Identical arithmetic, so logits match `native` to tolerance
  (`test_mha_sim_is_numerically_equivalent`), while KV memory and KV
  traffic go up 6x to 172,032 B/token. Model quality is held *exactly*
  constant, so any latency difference is a pure memory-system effect.
  That is a cleaner isolation than comparing an MHA checkpoint against a
  GQA checkpoint, where the weights differ too.
* **`mqa_sim`** — mean-pool to 1 KV head, 14,336 B/token. This changes
  the numerics and is a bandwidth probe only. Never quote quality from
  an `mqa_sim` run.

## Measured: TPOT rises past 4K, and why (first sweep, batch 1, native)

| ctx | LatentServe TPOT | HF TPOT | LatentServe peak VRAM | HF peak VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 33.0 ms | 40.7 ms | 3079 MB | — |
| 4096 | 33.0 ms | 45.7 ms | 3340 MB | — |
| 8192 | 48.9 ms | 72.4 ms | 3463 MB | 10654 MB |
| 16384 | 79.1 ms | OOM | 3794 MB | OOM |

Decode at batch 1 reaches 94-97 GB/s at short context, 29-30% of the T4's peak.
Peak memory is 7.2 GB lower at 8K, and 16K runs where the HF
reference cannot allocate at all.

TPOT is flat from 1024 to 4096, then rises 48% at 8K and
140% at 16K. The slope above 4K is ~3.8 us per token of context per
decode step; reading the KV cache once at the measured ~100 GB/s would
cost 0.28 us/token. That is ~13x the necessary traffic, and 13 is
exactly what `repeat_kv` costs on the decode path: 1x read of the cache,
6x written by the copy the reshape forces, 6x read back by SDPA. GQA
stores 2 KV heads instead of 12, and the attention path handed the
saving straight back every step, in every one of the 28 layers.

Below 4K it is invisible because it is smaller than the 33 ms weight-read
floor — which is itself the weight-bound finding above. Two independent findings interacting is the
kind of thing the three-layer evidence rule (math, benchmark, profile)
exists to catch.

Fix: on the decode path a single query attends to everything, so no mask
is needed and the group of query heads sharing a KV head can be folded
into the query axis instead — [B, 12, 1, D] -> [B, 2, 6, D] against an
un-expanded [B, 2, S, D] cache. Identical arithmetic
(`test_fold_and_materialize_agree`), cache read exactly once. Prefill
keeps `repeat_kv`: it is compute-bound at O(S^2), so the copy is a much
smaller share there, and the folded mask would be [n_rep * q_len, kv_len]
booleans.

`--kv-expansion materialize` reproduces the slow path on purpose. The
before/after belongs in the report: a theoretical memory saving erased
by an implementation detail, visible only past the context where it
clears the weight-read floor, is the project thesis in miniature.

## Measured: Runs B and C — GQA is capacity, not speed

Batch scaling (native, fold, output 128):

| batch / ctx | TPOT | KV share of decode bytes | achieved BW |
| --- | ---: | ---: | ---: |
| 1 / 4096 | 31.3 ms | 3.8% | 103 GB/s |
| 8 / 4096 | 30.6 ms | 23.9% | 132 GB/s |
| 1 / 16384 | 42.4 ms | 13.3% | 85 GB/s |
| 8 / 16384 | 49.4 ms | 55.1% | 139 GB/s |

Batching is nearly free at 4K — batch 1 to 8 leaves TPOT unchanged for
8x the throughput (32 -> 261 tok/s), because decode is weight-bound and
the weights are read once per step regardless of batch. That is the
Phase 4 argument, measured rather than asserted.

The GQA-configuration experiment (identical arithmetic, 6x the KV):

| batch / ctx | native | mha_sim | delta | KV cache |
| --- | ---: | ---: | ---: | ---: |
| 1 / 4096 | 31.30 | 32.17 | +2.8% | 115 -> 693 MB |
| 1 / 16384 | 42.44 | 47.41 | +11.7% | 452 -> 2709 MB |
| 4 / 8192 | 32.30 | 44.45 | +37.6% | 910 -> 5459 MB |

And mqa_sim, which *halves* KV bytes versus native:

| ctx | native TPOT | mqa_sim TPOT | KV bytes/token |
| --- | ---: | ---: | ---: |
| 8192 | 30.26 ms | 31.13 ms | 28,672 -> 14,336 |
| 16384 | 42.06 ms | 43.04 ms | 28,672 -> 14,336 |

**Halving the cache made decode slightly slower.** Taken with mha_sim
(6x the cache, +2.8% at batch 1), the conclusion is that at batch 1 the
number of bytes in the KV cache barely affects decode latency at all.
What it determines is what fits: mha_sim at batch 4 / 16K needs 10.8 GB
and does not run on a T4, while native does.

The mechanism is parallelism, and achieved bandwidth tracks
`batch x kv_heads` almost monotonically: 2 blocks (B1 native) reaches
83-103 GB/s, 12 blocks (B1 mha_sim) 119-133, 48 blocks (B4 mha_sim)
198 GB/s — 62% of peak, the best number in the sweep. A smaller cache is
read by fewer thread blocks on 40 SMs, so it is read less efficiently,
and the byte saving is handed back.

**Consequence for Phase 7.** MLA compresses toward one latent vector per
token: fewer head-like axes than GQA's 2, not more. A bigger byte
reduction *and* a worse occupancy problem, partly cancelling. That is
Question 6 (do MLA and DSA move the bottleneck rather than remove it)
with Phase 2 evidence behind it.

**Consequence for the Phase 10 sweep.** GQA already caches only
2 x 128 x 2 = 512 numbers per token per layer. MLA caches
`latent_dim + rope_dim`, so with rope_dim 64 it only saves memory at all
when `latent_dim < 448`. Three of the five values in the methodology
doc's sweep (512, 768, 1024) make the cache *larger* than the GQA
baseline. MLA's headline memory win is stated against MHA; against
2-head GQA the headroom is much narrower. Re-centre on roughly
96 / 128 / 192 / 256 / 384, and measure latency at batch 4-8 where KV is
38-55% of traffic rather than at batch 1 where it is 4-13%.

## Learnings

* Decode at batch 1 is weight-bound: the card reads weights at about 30% of its peak bandwidth, so changes to the KV cache barely move it.
* `repeat_kv` handed GQA's saving straight back, at 13x the necessary traffic; folding the query heads into the query axis fixed it.
* GQA is a capacity feature, not a speed one. Halving the cache made decode slightly slower, because fewer blocks spread over 40 SMs; a 6x larger cache cost only 2.8% at batch 1.
* Test cache and compression ideas at batch 4-8 and long context, where KV is 38-55% of traffic, not at batch 1.

## Gate 2 checklist — "Can LatentServe execute cached decoding?"

- [x] `pytest tests/test_phase2_gqa.py` green (tier 1, CPU)
- [x] `LATENTSERVE_REAL_MODEL_TESTS=1 pytest` green (tier 2, T4): logits
      match `QwenReference.forward_teacher_forced`, greedy output ids
      identical, cache accounting equals HF's measured `past_key_values`
- [x] `results/raw/phase2_gqa.jsonl` populated across the sweep
- [x] `kv_bytes_per_token` = 28,672 for `native` and matches
      `ModelShape.kv_bytes_per_token()`
- [x] each finding of the sweep stated in writing

*Reviewed at project close: ticked where this note's results show the item done. The real-checkpoint (tier 2) run was rechecked by the maintainer after this note was written and isn't recorded here, so that box is ticked on their confirmation; the project's later real-model results corroborate it (LatentServe agrees with the fp32 reference to about 1.7×10⁻⁵ KL; see `benchmark_vs_vllm.md`).*

## Known limitations, deliberately

* **Uniform sequence lengths per batch.** The contiguous cache pads to
  the longest sequence, so a ragged batch wastes capacity. That waste is
  the thing Phase 3's paged allocator exists to remove, so the fair
  contiguous-vs-paged comparison needs this baseline to exist first.
* **Head-major layout** (`[B, kv_heads, seq, head_dim]`) is chosen for
  SDPA's benefit on the read path. Seq-major writes one contiguous run
  per token instead of one per head, which suits a block allocator
  (Phase 3) and gather-based sparse attention (Phase 14) better. Phase
  11 should measure this rather than settle it by argument.
* **No sliding-window support.** `LatentServeQwen` raises if a
  checkpoint sets `use_sliding_window`, rather than silently computing
  different math than the model was trained for.
* **`throughput_tokens_sec` is aggregate across the batch** in Phase 2
  rows (per-sequence rate x batch size). Phase 1 rows are batch 1, so
  the two are directly comparable there.

## Note on the Phase 1 `attention="mha"` label

`configs/phase1_reference.yaml` documents this as a label meaning
"unmodified HF, no custom attention path yet". It is worth relabelling
before the final report: Qwen2.5-1.5B-Instruct is GQA in every run,
including Phase 1's, and a results table containing both `mha` and `gqa`
rows for the same checkpoint invites a reader to think an architectural
variable changed when only the runtime did. The paired HF rows written
by `phase2_gqa.py --include-hf-baseline` use `attention="gqa"` with
`system="huggingface_reference"`, which keeps the architecture column
honest and puts the real distinction in the system column.
