# LatentServe vs vLLM — speed and quality

The full comparison, after Phase 14's elementwise fusion. One session on
vLLM's stack (torch 2.13 + vLLM 0.30), both systems on CUDA graphs, same
weights, same prompts, same machine.

## Speed

`benchmarks/runners/phase6_vllm.py` with every safeguard the project has
accumulated:

* `--settle-seconds 30` (default) — sustained GPU load before anything is
  timed, so no run starts at boost clock;
* `--throwaway-first` — one unrecorded run, so no measured run is the first
  after loading;
* `--abba` — each configuration runs short, long, long, short; a steady
  drift cancels, and the two per-pair estimates are the repeatability check;
* per-host *and* per-stack tables — rows are never paired across machines
  or torch versions.

Reported separately, because they answer different questions: **decode
latency** (batch 1), **decode throughput** (largest batch), and end-to-end
throughput, flagged wherever prefill dominates it.

## Quality

`benchmarks/runners/quality_vs_vllm.py`. Same weights, so the question is
**fidelity**: how close each system's computation is to the model's.

* **Yardstick:** Hugging Face in fp32.
* **Calibration:** Hugging Face in fp16 — the model as normally run. A
  system is *on par* if its distance from fp32 is within 25% of HF fp16's,
  *degraded* if further, *better* if closer.
* **LatentServe twice**, unfused and fused, so the effect of fusion is
  separated from the comparison with vLLM.

| evidence | what it shows |
| --- | --- |
| Wikitext-2: perplexity, top-1 agreement with fp32, KL from fp32 | numerical fidelity per token. vLLM exposes top-k log-probs only, so no KL |
| ARC-Easy, 300 items, log-prob scored | a real task. Accuracy has a ±~5.7% band at 300 items, so changed answers are counted item by item against fp32 |
| greedy generation, 32 prompts x 128 tokens | the most sensitive test, and the only one through each system's decode path — for LatentServe, graphs and the paged kernel |

The LatentServe arm tokenizes once and saves token ids; the vLLM arm reads
them, so both score identical tokens.

**What to expect.** Fidelity differences between correct fp16
implementations are small: perplexity within a few hundredths of a
percent, top-1 agreement in the high 99s, ARC answers changed in single
digits or none, and greedy generations diverging somewhere in the tens of
tokens once a near-tie flips. A system clearly outside that band has a
numerical problem, not a "quality" one.

## Run

```bash
!python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"   # 2.13.0+cu130 0.30.0
!python -m pip install -q datasets                                              # for the quality data
```
```python
SPEED = ("--abba --throwaway-first --num-requests 16 --batch-sizes 1 4 16 "
         "--context-lengths 2048 8192 --output-lengths 128 256 "
         "--results-dir results/raw/final_speed")
```
```bash
!python -m benchmarks.runners.phase6_vllm --system latentserve --cuda-graphs --fuse-projections --fuse-elementwise $SPEED
!python -m benchmarks.runners.phase6_vllm --system vllm $SPEED
!python -m benchmarks.runners.phase6_vllm --compare --results-dir results/raw/final_speed

!python -m benchmarks.runners.quality_vs_vllm --system latentserve
!python -m benchmarks.runners.quality_vs_vllm --system vllm
!python -m benchmarks.runners.quality_vs_vllm --compare
```

Rough time on a T4: speed ~30 min for LatentServe and ~60 for vLLM (its 8K
prefill is ~3x slower, and it builds an engine per run); quality ~10 min in
total. To halve the speed half, drop batch 4 and use `--num-requests 8`.
