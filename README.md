# LatentServe

Memory-efficient and sparse long-context LLM inference on commodity GPUs
(1–2× NVIDIA T4), using **Qwen2.5-1.5B-Instruct as a fixed real-model
substrate** — no scratch Transformer. The experimental variable is the
execution system (GQA → paged/latent KV → MLA-inspired attention →
DSA-inspired sparse attention → adaptive runtime), not the model. DSA is
a core research axis alongside MLA, not an optional late-stage add-on.
See `docs/methodology.md` for the full research plan.

This README covers **Phase 0 only**: environment + experimental
infrastructure. Nothing here trains or runs a model yet — that's Phase 1.

## What's in this scaffold

```
config.py                          # experiment config schema (pydantic) + YAML loader
configs/                           # YAML experiment definitions (edit these, not code)
benchmarks/schema.py               # BenchmarkResult schema + JSONL writer (auto reproducibility metadata)
benchmarks/runners/check_env.py    # Phase 0 gate: verifies CUDA/GPU/torch actually work
model/ cache/ runtime/ kernels/    # empty package stubs for Phases 1+
evaluation/ comparisons/vllm/      # empty package stubs for Phases 6, 15
profiling/                         # where .nsys-rep / .ncu-rep artifacts go (Phase 12)
results/{raw,processed,figures}/   # results/raw is machine-written only, never hand-edited
tests/test_phase0_infra.py         # proves config loading + result writing work
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

## Next: Phase 1

There is no scratch model to write. Load **Qwen2.5-1.5B-Instruct** via
`transformers` in `model/qwen.py` and treat the Hugging Face reference
implementation as the correctness oracle. Then build LatentServe's own
GQA + KV-cache execution path around those same fixed weights, and write
the correctness harness comparing Hugging Face vs. LatentServe logits,
attention masking, RoPE, and greedy decode output at 1 / 16 / 1K / 4K /
8K / 16K+ tokens. See `docs/methodology.md` Phase 1 for details.
