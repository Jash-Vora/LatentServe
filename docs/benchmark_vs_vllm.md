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

## A bug the quality benchmark caught

The first quality run returned NaN for every LatentServe metric — perplexity,
ARC (24%, i.e. always choosing the first option), and generations differing
at the first token — in both the unfused and fused variants. vLLM's column
was on par with HF fp16, which showed the harness itself was sound.

**Cause.** `GQAAttention._attend` sent only `attn_impl == "sdpa"` to PyTorch's
fused attention; anything else fell through to the explicit math path, the
reference implementation. Phase 11 added `"triton_paged"`, whose Triton
kernel handles *decode* and returns early — but *prefill* fell through. The
math path forms q·k in fp16 before scaling; Qwen2.5's massive activations
(measured in Phase 7) overflow that to inf, and the softmax turns inf into
NaN. Phase 11's design said prefill stays on SDPA; the code did not.

**Why nothing caught it for three phases.**

* Every test since Phase 11 used a tiny random model, whose activations
  never overflow, or ran in fp32 on CPU, where nothing does. On CPU the only
  trace was a 6e-7 difference from Hugging Face where `sdpa` gave exactly 0.
* The speed harness never checked outputs, and kernel time barely depends on
  the values it computes.
* Prefill throughput fell from 5,238 tok/s at 8K (Phase 6, SDPA) to ~2,400
  in every run since — the math path materialising score matrices — and
  that drop was never questioned.

**Fix.** Only `"math"` takes the explicit path. Two guards so the class of bug
cannot hide again: a test at fp16 scale where q·k overflows (mutation-checked:
reintroducing the bug makes it fail), and the speed harness now refuses to
time a model whose logits are not finite.

**What survives.** Decode timings are unaffected: decode runs the Triton
kernel, which never reaches `_attend`, and NaN arithmetic takes the same time
as any other. LatentServe's prefill and end-to-end figures since Phase 11
measured the math path and are understated. vLLM's quality results are
unaffected and reused.

## Results (T4, torch 2.13, vLLM 0.30, same host per table)

### Speed — decode per step, short-long-long-short, both on CUDA graphs

| batch | ctx | LatentServe | vLLM | |
| ---: | ---: | ---: | ---: | --- |
| 1 | 2048 | 18.3 ms | 17.6 | 3% slower — LatentServe's own halves differ by 0.6 ms |
| 1 | 8192 | 21.3 | 23.3 | 9% faster |
| 4 | 2048 | 22.2 | 23.9 | 7% faster |
| 4 | 8192 | 35.1 | 42.1 | 17% faster |
| 16 | 2048 | 38.5 | 42.2 | 9% faster |
| 16 | 8192 | 81.4 | 107.3 | 24% faster |

Prefill was 2.0-3.3x vLLM's, measured on the math-path bug above, so the
true figure is higher; it needs one rerun of the LatentServe speed arm to
state. Decode is unaffected by that bug.

Elementwise fusion is what moved the short-context rows: batch 1 / 2K went
from 12% behind to 3%, and batch 4 / 2K from 18% behind to 7% ahead.

### Quality — fidelity to the fp32 model

| system | perplexity | KL vs fp32 | top-1 agree | ARC answers changed | gens identical to HF fp16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF fp32 | 9.7015 | 0 | 100% | 0 / 300 | — |
| HF fp16 | 9.7019 | 1.74e-5 | 99.71% | 0 / 300 | — |
| LatentServe, unfused | 9.7022 | 1.77e-5 | 99.65% | 0 / 300 | 26 / 32 |
| LatentServe, fused | 9.7015 | 1.76e-5 | 99.69% | 0 / 300 | 26 / 32 |
| vLLM | 9.7022 | n/a | 99.66% | 1 / 300 | 28 / 32 |

16,368 tokens of Wikitext-2 test; 300 ARC-Easy questions (73.3% acc,
74.7% acc_norm for every system); 32 prompts x 128 greedy tokens.

**Verdict: on par on every measure, and elementwise fusion costs nothing.**
KL — the most sensitive measure — is the same across systems to two
significant figures. The fused variant's perplexity landing exactly on
fp32's is not evidence of extra accuracy: deviations cancel in an average,
which is why KL is the number to read. 26 versus 28 identical generations
out of 32 is within what 32 prompts can resolve, and a divergence between
two correct fp16 implementations is a near-tie flipping, not an error.
