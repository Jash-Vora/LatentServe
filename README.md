# LatentServe

Memory-efficient and sparse long-context LLM inference on commodity GPUs
(1–2× NVIDIA T4), using **Qwen2.5-1.5B-Instruct as a fixed real-model
substrate** — no scratch Transformer. The experimental variable is the
execution system (GQA → paged/latent KV → MLA-inspired attention →
DSA-inspired sparse attention → adaptive runtime), not the model. DSA is
a core research axis alongside MLA, not an optional late-stage add-on.
See `docs/methodology.md` for the full research plan.

This README covers **Phases 0-2**: environment + experimental
infrastructure, the Qwen2.5-1.5B-Instruct reference implementation and
correctness harness, and LatentServe's own GQA + KV-cache execution
path. Phase 3 (paged KV cache) is next. See `docs/phase2.md` for what
Phase 2 measures and the predictions registered before measuring.

## What's in this scaffold

```
config.py                              # experiment config schema (pydantic) + YAML loader
configs/                               # YAML experiment definitions (edit these, not code)
benchmarks/schema.py                   # BenchmarkResult schema + JSONL writer (auto reproducibility metadata)
benchmarks/runners/check_env.py        # Phase 0 gate: verifies CUDA/GPU/torch actually work
benchmarks/runners/phase1_reference.py # Phase 1: context-length sweep -> results/raw/phase1_reference.jsonl
benchmarks/runners/phase2_gqa.py       # Phase 2: GQA/KV sweep -> results/raw/phase2_gqa.jsonl
model/qwen.py                          # Phase 1: instrumented QwenReference wrapper (now the correctness oracle)
model/latentserve_qwen.py              # Phase 2: LatentServe's own decoder layer loop over the fixed weights
model/attention/gqa.py                 # Phase 2: GQA attention against our KV cache (baseline for MLA/sparse)
model/rope.py                          # Phase 2: our RoPE (Phase 8 decouples it, Phase 11 fuses it)
cache/kv_cache.py                      # Phase 2: preallocated contiguous KV cache + byte accounting
runtime/ kernels/                      # empty package stubs for Phases 4+
evaluation/ comparisons/vllm/          # empty package stubs for Phases 6, 15
profiling/                             # where .nsys-rep / .ncu-rep artifacts go (Phase 12)
results/{raw,processed,figures}/       # results/raw is machine-written only, never hand-edited
tests/test_phase0_infra.py             # proves config loading + result writing work
tests/test_phase1_correctness.py       # Gate 1: HF teacher-forced vs incremental logits/cache/determinism
tests/test_phase2_gqa.py               # Gate 2: LatentServe's execution path vs the HF oracle
```

This is a **flat layout**: `model`, `cache`, `runtime`, etc. are top-level
Python packages. Run everything with this repo root on your `PYTHONPATH`
(or `pip install -e .` from this directory).

## Setup

```bash
# Local machine or a fresh Kaggle/cloud box with a T4:
pip install -r requirements.txt

# Or, editable install (also picks up the package layout for `import model`, etc.):
pip install -e .
```

On Kaggle, torch is normally preinstalled — skip the torch line if so.

## Step 1 — verify the environment (do this first, every new machine)

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python -m benchmarks.runners.check_env
```

This checks:
- PyTorch sees the GPU(s) and reports compute capability (T4 = 7.5, Turing — **use fp16, not bf16**, Turing has no bf16 tensor cores)
- a real fp16 matmul actually runs on the device (catches broken driver/CUDA installs that `torch.cuda.is_available()` alone can miss)
- library versions for the reproducibility record
- whether `triton`, `vllm`, `nsys`, `ncu` are present (fine if not yet — they're needed starting Phase 11, 6, and 12 respectively, not now)

If this doesn't print "Environment looks ready for Phase 1", fix it before writing any model code.

## Step 2 — confirm the config + result infrastructure works

```bash
pytest tests/test_phase0_infra.py -v
```

This proves, before any model exists:
- YAML configs load into validated `ExperimentConfig` objects (`configs/baseline_gqa4.yaml`, `configs/mla_512.yaml`)
- invalid configs are rejected (e.g. MLA config missing `latent_dim`, GQA grouping that doesn't divide evenly)
- `BenchmarkResult` → `ResultWriter` produces valid JSONL in `results/raw/` with git commit, library versions, GPU info, and timestamp attached automatically

## Defining a new experiment

Don't edit code to change hyperparameters — add or edit a YAML file in
`configs/` following the schema in `config.py`:

```yaml
tag: my_experiment_name   # groups results under results/raw/my_experiment_name.jsonl

model: {...}
attention: {type: gqa | mla | sparse | mla_sparse, ...}
runtime: {...}
generation: {...}
hardware: {devices: [0], dtype: fp16}
```

Load it in code with:

```python
from config import load_config
cfg = load_config("configs/my_experiment_name.yaml")
```

## Writing a benchmark result

Every benchmark run should end with exactly this pattern (see
`benchmarks/schema.py` for the full field list — TTFT, TPOT, throughput,
peak VRAM, KV-cache memory, quality metrics, percentiles):

```python
from benchmarks.schema import BenchmarkResult, ResultWriter

result = BenchmarkResult(
    system="latentserve", tag=cfg.tag, attention=cfg.attention.type,
    model=cfg.model.name, batch_size=cfg.runtime.batch_size,
    context_length=cfg.generation.input_tokens,
    output_length=cfg.generation.output_tokens, num_gpus=len(cfg.hardware.devices),
    ttft_ms=..., tpot_ms=..., throughput_tokens_sec=..., peak_vram_mb=...,
)
ResultWriter().write(result)
```

This appends one JSON line to `results/raw/<tag>.jsonl`, self-tagged with
git commit, library versions, GPU model, and timestamp — never type a
benchmark number into a doc or notebook by hand.

## Phase 1 — Qwen reference + correctness baseline

`model/qwen.py` loads **Qwen2.5-1.5B-Instruct** via `transformers` and
wraps it in `QwenReference`: an instrumented prefill/decode-step API
plus `ModelShape` (layer/head/kv-head introspection that Phase 2's GQA
work needs) and `kv_cache_bytes()` (measured KV-cache size from a real
`past_key_values`, checked against the theoretical estimate). At this
phase "LatentServe" *is* the Hugging Face model — no custom attention or
KV-cache layout exists yet; that starts Phase 2. The point of Phase 1 is
ground truth + a trustworthy measurement harness for everything after it.

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH

# Correctness harness (Gate 1): incremental decode vs. teacher-forced
# logits, prefill+decode-step agreement, deterministic greedy generation,
# KV-cache memory vs. theoretical estimate. Skips cleanly without
# torch/transformers/CUDA/network instead of failing.
pytest tests/test_phase1_correctness.py -v

# Full context-length sweep (1/16/1K/4K/8K/16K), needs a real GPU:
LATENTSERVE_LONG_CONTEXT_TESTS=1 pytest tests/test_phase1_correctness.py -v

# Reference benchmark: load time, TTFT, TPOT, E2E, throughput, peak
# VRAM, KV-cache memory, written to results/raw/phase1_reference.jsonl
python -m benchmarks.runners.phase1_reference --config configs/phase1_reference.yaml
```

## Next: Phase 2

Build LatentServe's own GQA + KV-cache execution path (`model/attention/gqa.py`,
`cache/kv_cache.py`) around the same fixed Qwen weights, using
`QwenReference.shape` for head/layer counts and `test_phase1_correctness.py`'s
teacher-forced-vs-incremental pattern as the template for checking the
custom path against ground truth. See `docs/methodology.md` Phase 2.
