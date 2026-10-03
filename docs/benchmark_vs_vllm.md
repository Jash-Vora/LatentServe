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

## Default vLLM, and the best vLLM that runs

The comparison above is against vLLM's defaults, with two fairness pins:
`max_num_seqs` equal to the batch size, and prefix caching off. Everything
else — scheduler, compilation, CUDA graphs, memory utilisation, and the
attention backend — is vLLM's own choice. On a T4 that last one matters most:
FlashAttention needs Ampere, so vLLM falls back to a Triton backend, and
Phase 12 found Triton compiles attention without tensor cores on this GPU.
LatentServe's high-batch, long-context lead is largely its CUDA-core kernel
against that fallback, so a better vLLM backend could shrink it.

`benchmarks/runners/vllm_backend_probe.py` starts vLLM once per candidate
backend, records whether it runs and what vLLM's log says it used, and times
decode at two points. The fastest that runs becomes a second vLLM arm:

    --vllm-attention-backend NAME     rows labelled vllm_<name>
    --compare --baseline vllm_<name>  LatentServe measured against that arm

A tuned arm must be compared with LatentServe rows from the same machine and
session: rows from different hosts are never paired.

### Knobs besides the backend

Most of vLLM's settings cannot move per-step decode time in this benchmark:
memory utilisation sizes a KV pool that never fills (the largest case needs
~3.8 GB of ~8-9 GB), `max_num_seqs` is pinned to the batch, prefix caching
has nothing to share, and chunked-prefill limits shape prefill, which the
decode measurement cancels. Two could move batch-1 latency, so the probe
tests them as variants — `BACKEND[+knob...]`:

| knob | engine setting | why it might matter |
| --- | --- | --- |
| `full_graphs` | `cudagraph_mode=FULL_DECODE_ONLY` | whole decode step in one graph, attention included |
| `piecewise` | `cudagraph_mode=PIECEWISE` | attention eager between graph pieces: shows what the default does |
| `async` | `async_scheduling=True` | next step prepared on the CPU while the GPU runs this one |

The probe also reads the graph mode vLLM printed at startup, so the default
is reported rather than assumed. The winner runs in the full comparison as
`--vllm-variant <variant>`, labelled `vllm_<variant>`.

## Results with the CUDA-core decode kernel (same session, default vLLM)

| | vs vLLM |
| --- | --- |
| decode latency, batch 1 / 2K | parity (0%) |
| decode latency, batch 1 / 8K | 22% faster |
| decode throughput, batch 16 / 2K | 1.91x |
| decode throughput, batch 16 / 8K | 2.78x |
| prefill | 2.7-6.0x (on the corrected SDPA path; was 2.0-3.3x on the math path) |
| end-to-end, batch 16 | 2.29x at 2K, 5.16x at 8K — mostly a prefill result |

Quality with the CUDA kernel: unchanged on text and ARC, which do not run
through decode; greedy generation matches HF fp16 on 26 of 32 prompts (26
before, with Triton; vLLM 28). On par.

## What the first backend probe showed

vLLM's default on a T4 is TRITON_ATTN, as its log reports. The probe asked
for FLASHINFER, FLEX_ATTENTION, TORCH_SDPA and XFORMERS through the
`VLLM_ATTENTION_BACKEND` variable, every one ran, and vLLM's log reported
TRITON_ATTN for all of them: this version no longer reads the variable.
Without reading the log, those rows would have been reported as four other
backends. They are repeat measurements of Triton.

The runner now tries each route this vLLM might accept — an
`attention_backend` engine option, then `attention_config`, then the
variable — skips a route only when the engine says that option is unknown,
and records every rejection. Rows where vLLM still runs something other than
what was asked are marked IGNORED and can never be chosen as a winner. The
old prefix-caching fallback, which retried on *any* TypeError and would have
dropped an unknown attention option silently, now fires only for its own
argument.

## The first probe's table, read correctly

All eleven rows ran TRITON_ATTN (the environment variable was ignored), so
six of them were the identical configuration measured six times: 44.2-48.0
ms at batch 16 / 2K and 22.4-24.5 ms at batch 1 / 8K, an 8-9% spread. The
`auto` row's 27.0 ms at batch 16 / 2K was the same configuration again — an
outlier, not a winner. The rows that measured fastest were the ones whose
startup took longest (221 s and 173 s, compiling while the GPU idled and
cooled): the boost-clock artifact once more, in a tool that had none of the
main benchmark's protections.

What holds: vLLM's default on a T4 is TRITON_ATTN with FULL_AND_PIECEWISE
graphs, which already captures whole decode steps, so the graph knobs have
nothing to add, and async scheduling showed nothing beyond the noise. The
probe now settles the GPU before each variant, measures each point
short-long-long-short, brackets the run with the default first and last,
and reports the noise it measured instead of assuming one.

## Backend probe, with settling and bracketing (T4, vLLM 0.30)

| variant | batch 16 / 2K | batch 1 / 8K | vLLM's log |
| --- | ---: | ---: | --- |
| default (first) | 42.7 ms | 22.8 | TRITON_ATTN |
| FLASHINFER | failed at engine start | | — |
| FLEX_ATTENTION | 290.4 | 84.2 | FLEX_ATTENTION (via `attention_backend`) |
| default (last) | 39.0 | 23.6 | TRITON_ATTN |

Measured noise: a row's two halves differ by up to 6.6%; the default drifted
8.7% between its first and last run. The `attention_backend` engine option
is the route this vLLM accepts. FLEX_ATTENTION runs and is about 7x slower
at batch 16 / 2K. FlashInfer (0.6.18.post1, the version vLLM pins) failed
during engine start-up; its root cause was not captured by that probe run,
which now saves every variant's full log to results/probe_logs/.

FlashInfer's root cause, from the engine log: `ValueError: Selected backend
AttentionBackendEnum.FLASHINFER is not valid for this configuration. Reason:
['compute capability not supported']` — vLLM's own support check, raised
while building the attention layers, before any kernel is compiled. vLLM
0.30 does not offer FlashInfer on Turing; there is nothing to fix.

On a T4, then: TRITON_ATTN (the default) runs; FLEX_ATTENTION runs about 7x
slower; FLASHINFER is refused; FLASH_ATTN needs Ampere. XFORMERS and
TORCH_SDPA, and the graph and async knobs, remain to be measured through the
working route and the settled probe.
