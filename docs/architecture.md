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
