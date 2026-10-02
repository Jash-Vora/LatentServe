# Phase 14 — Closing the batch-1 gap, then INT8 at graph speed

## Where Phase 13 left things

Against vLLM, same host, same stack, both on CUDA graphs (host `a886…`):

| | vs vLLM |
| --- | --- |
| decode, batch 1 | 2-5% slower |
| decode, batch 16 / 2K | parity |
| decode, 8K, batch 4 / 8 / 16 | 13% / 22% / 24% faster |
| prefill | 1.9-3.1x faster (Turing-specific) |

The batch-1 gap is ~1-1.5 ms, outside attention. A replication on a
second host matched LatentServe to 0.1 ms. The 15% gap first reported
came from a single outlying vLLM measurement on one host; two later
hosts disagree with it.

Two pieces, built and measured one at a time so each gain is
attributable:

* **14a — projection fusion**, for the batch-1 gap.
* **14b — a graph-capturable INT8 cache**, for capacity at graph speed.

## 14a — projection fusion

vLLM computes q/k/v as one projection and gate/up as another; LatentServe
ran three and two. At batch 1 each projection is a matrix-vector product
that streams its weights once with little arithmetic per byte, so a
separate launch's fixed cost is a real share of it. CUDA graphs removed
the CPU cost of launching; they did not remove the launches.

Fusing them is 84 fewer kernels per decode step (two fewer per layer in
attention, one in the MLP, 28 layers).

### No extra memory, and both paths stay live

Keeping the originals plus a concatenated copy would cost ~1.7 GB on this
model. Instead the concatenated tensor is built once and the checkpoint's
own parameters are re-pointed at slices of it — one copy of every weight,
every existing reader (the HF oracle, Phase 7's hooks) unaffected.

Because the unfused path still works on the same weights, `set_fused()`
switches between them exactly and for free, which is what makes a
**same-process A/B** possible.

### Measuring it

Phase 13 measured host-to-host noise at 4-8%, larger than the 5-7% being
sought. So `phase14_fusion.py` loads the model once and alternates
unfused / fused rounds for each configuration; slow drift lands on both
sides equally, and the reported saving is the median per-round
difference with its spread. A saving smaller than its spread is not one.

It also counts CUDA kernels per decode step for both paths first. If
that count does not fall by about 84, nothing downstream is measuring
fusion.

```bash
python -m pytest tests/test_phase14_fusion.py -v
python -m benchmarks.runners.phase14_fusion --context-lengths 2048 8192 --batch-sizes 1 4 16
```

### Prediction

Batch 1 / 2K recovers most of the ~1.5 ms gap. The saving is per-launch
fixed cost, so it should be roughly constant in milliseconds across
context lengths and shrink as a *fraction* as the step grows. If batch 1
barely moves, the gap is not launch count and Phase 13's explanation was
incomplete.

### Then the side-by-side

Once the A/B says it helps, the claim against vLLM still has to be made
side by side:

```bash
ARGS="--num-requests 32 --batch-sizes 1 4 8 16 --context-lengths 2048 8192 \
      --output-lengths 128 256 --results-dir results/raw/phase14_vs_vllm"
python -m benchmarks.runners.phase6_vllm --system latentserve --cuda-graphs --fuse-projections $ARGS
python -m benchmarks.runners.phase6_vllm --system vllm $ARGS
python -m benchmarks.runners.phase6_vllm --compare --results-dir results/raw/phase14_vs_vllm
```

## 14b — INT8 cache at graph speed

Not started. Needs three things the INT8 cache does not have: block
finalisation moved into host-side `advance()` (it currently runs on a
host-side condition inside the forward), a persistent fp16 residual
buffer, and a kernel that reads finished blocks as INT8 and the partial
block as fp16 and merges the two.
