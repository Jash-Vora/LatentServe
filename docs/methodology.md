# Methodology

This file is the running summary of the full project plan (phases, gates,
benchmark matrix, fairness rules for the vLLM comparison, ablations). The
complete original plan lives outside this repo for now — copy or link it
here once Phase 1 is underway and this doc needs to become the living
reference instead of a static snapshot.

Key things every experiment must respect (see Section 32 of the plan):
- control model weights, tokenizer, precision, GPU, input/output token
  counts, batch/concurrency, sampling params, context length across
  every comparison
- exclude warm-up runs from steady-state latency numbers
- multiple repetitions; report median, p95, p99, and CIs where practical
- every result comes from `benchmarks/schema.py::ResultWriter` — never
  hand-typed into this doc or a notebook
