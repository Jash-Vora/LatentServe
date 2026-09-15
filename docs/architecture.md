# Architecture

TODO (fill in as each phase lands):
- System diagram: model -> cache -> kernels -> runtime -> serving (see docs/methodology.md, Section 34).
- Package layout and how each top-level dir maps to a phase.
- Data flow for a single request: ARRIVED -> QUEUED -> PREFILL -> DECODING -> FINISHED.

Full architecture documentation is deferred until Phase 4 (serving
runtime) exists — no point documenting an architecture that's still
mostly directory stubs. In the meantime:

## Phase 1 — what exists today

`model/qwen.py` provides `QwenReference`, an instrumented wrapper
around `transformers.AutoModelForCausalLM` for Qwen2.5-1.5B-Instruct:

- **`QwenReference.load()`** — loads weights/tokenizer, records load
  time, introspects `ModelShape` (layer/head/kv-head/head-dim counts —
  the numbers Phase 2's GQA implementation is built from).
- **`prefill()` / `decode_step()`** — the two primitives every later
  phase's serving path (Phase 4 onward) is built on top of.
- **`forward_teacher_forced()` / `forward_incremental()`** — the
  correctness-check pair: a full-sequence forward pass with no cache
  (ground truth) vs. token-by-token forward passes through the KV
  cache. Every custom attention/cache implementation from Phase 2
  onward should be checked against `forward_teacher_forced()` using
  this same pattern, not against `forward_incremental()`'s HF cache
  directly, since the whole point of later phases is to replace that
  cache.
- **`generate_with_timing()`** — the benchmark primitive: runs prefill
  + N decode steps, returns a `TimedGenerationResult` with TTFT, TPOT,
  E2E latency, throughput, peak VRAM, and measured KV-cache bytes.
  `benchmarks/runners/phase1_reference.py` sweeps this across context
  lengths and writes `BenchmarkResult` rows.
- **`kv_cache_bytes()`** — measures actual `past_key_values` tensor
  sizes; checked against `ModelShape.kv_bytes_per_token()`'s theoretical
  estimate in `tests/test_phase1_correctness.py`.

No custom attention kernel, KV-cache layout, or batching exists yet —
that's Phases 2 through 4.

## Phase 2 — what exists today

From Phase 2 onward LatentServe drives the model itself. `model/qwen.py`
(`QwenReference`) stays exactly as it was and becomes purely the
correctness oracle; the serving path is:

```
        input_ids
            |
   LatentServeQwen                 model/latentserve_qwen.py
   (own decoder layer loop,
    borrowed HF weight modules)
            |
      +-----+------------------------------+
      |                                    |
  GQAAttention                     RMSNorm / MLP
  model/attention/gqa.py           (HF modules, untouched)
      |
   RotaryEmbedding  model/rope.py
      |
   ContiguousKVCache  cache/kv_cache.py
   preallocated [B, kv_heads, max_seq, head_dim] per layer
```

Three design decisions worth restating here because later phases depend
on them:

1. **Weights are borrowed, not copied.** `GQAAttention` holds references
   to the loaded checkpoint's `q_proj`/`k_proj`/`v_proj`/`o_proj`
   modules. The model is fixed and only the execution system changes —
   sharing the same tensor objects makes that literally true, and avoids
   a second 3.1 GB of weights on a 16 GB card.
2. **We own the layer loop, not a patched HF attention module.**
   Chunked prefill, continuous batching (Phase 4), prefix caching
   (Phase 13) and adaptive backend selection (Phase 17) are decisions
   made above attention, not inside it. Owning the loop also means the
   code does not depend on HF's attention signature, which has moved
   repeatedly across releases (`model/qwen.py::kv_cache_bytes` already
   carries three compatibility branches for the cache alone).
3. **The KV cache is preallocated and accounts for itself exactly.**
   `kv_cache_mb` is read off the cache, not estimated by walking HF
   tensors, and `bytes_read_per_decode_step()` is the numerator of every
   bandwidth claim from here to Phase 18.

See `docs/phase2.md` for the sweep, the measurement design, and the
predictions registered before running it.

## Phase 3 — what exists today

`ContiguousKVCache` and `PagedKVCache` implement the same read/write
contract, so `GQAAttention` is unchanged and
`allocate_cache(..., paged=True)` is the only difference between the two
arms of the comparison:

```
   LatentServeQwen
         |
   GQAAttention  ---- cache.advance(n, batch_size)
         |             cache.write(layer, k, v, start_pos)
         |             cache.read(layer, batch, length)
         |             cache.padding_mask()
         +---> ContiguousKVCache   [B, kv_heads, S, D] per layer
         |     preallocated, head-major, read in place
         |
         +---> PagedKVCache        [num_blocks, block_size, kv_heads, D]
               block pool + BlockTable per sequence, gathered per read
```

The same contract is how Phase 7's latent cache drops in without
touching the executor.

One ordering constraint worth knowing: `advance()` now runs *before* the
layer loop, not after. A paged cache must have block tables and slot
indices built before any layer scatters into the pool; the contiguous
cache only tracks a fill counter, so the earlier position is harmless
there.
