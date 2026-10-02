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

### Measured (same process, alternating rounds)

| batch | ctx | unfused | fused | saved | spread |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 19.14 ms | 18.78 | **0.36** | 0.05 |
| 1 | 8192 | 22.61 | 22.25 | 0.39 | 0.08 |
| 4 | 2048 | 23.60 | 23.36 | 0.26 | 0.11 |
| 4 | 8192 | 36.09 | 35.93 | 0.14 | 0.14 |
| 16 | 2048 | 42.30 | 40.74 | **1.55** | 0.14 |
| 16 | 8192 | 88.46 | 87.21 | 1.15 | 0.45 |

Repeated in a second session at batch 1 / 2K: 0.36 ms saved, spread 0.01.

**Kernels fell by 140, not the predicted 84.** At batch 1 a projection
with a bias is two kernels — one copies the bias into the output, one
multiplies into it. Qwen's q, k and v all have biases, so they were six
kernels per layer and are now two; gate/up have none and save one. Five
per layer, 28 layers.

**The batch-1 prediction was falsified.** 0.36 ms is about a quarter of
the ~1.5 ms gap. It does give a useful number: ~2.6 us per kernel under
graph replay, so the 1,191 kernels still in each step can account for at
most ~3 ms between them.

**Past batch 1 the gain is efficiency, not launches.** At batch 16 the
projections are real matrix multiplications, and one large one uses the
GPU better than several small ones — hence 1.55 ms at 16 / 2K, the
largest saving measured. Batch 4 / 8K (0.14 ms, spread 0.14) is not
distinguishable from zero.

### The rest of the gap is in the serving loop

Same host, same session, batch 1 / 2K, fused:

| | ms per token |
| --- | ---: |
| model step alone (graphed) | 18.91 |
| through the engine, median gap between tokens | 19.6-20.1 |
| through the engine, differenced mean | 20.7 |

The engine adds 0.7-1.8 ms per token over the model — the same size as
the remaining gap to vLLM (1.1-1.6 ms). The other batch sizes, compared
across hosts and so only supporting, put the engine 1.6-3.2 ms above the
bare step, growing with batch: the shape of per-request Python work.

## 14b — the serving loop

After every replay the engine waited for the GPU, then launched the
greedy argmax as a separate kernel outside the graph, copied the result
to the host, updated every request in Python, ran the scheduler, built
the next step's inputs on the host and copied them back up. The GPU sat
idle throughout, and the token it had just produced made a round trip
through the CPU only to come straight back as the next input.

### Built

* **Token selection inside the graph** (`GraphedDecoder(greedy=True)`):
  the argmax becomes one more recorded kernel. The engine only ever
  samples greedily, so this changes no output.
* **On-device token feedback**: when a step serves exactly the same
  requests in the same order as the previous one, its inputs are the
  previous step's output buffer and the previous positions plus one —
  nothing built on the host, nothing copied up.
* **A loop profile** (`profile_loop=True`): every decode step split into
  inputs, decoder host work, GPU wait, sampling, bookkeeping and
  scheduling.

### The trap

"Same requests" is keyed on request ids, not slot numbers. When a request
finishes and the next is admitted into the slot it freed, the slot list
is unchanged — a slot-keyed check would feed the newcomer the previous
occupant's last token. `test_a_new_request_in_a_freed_slot_is_not_fed_the_old_token`
guards it, and was mutation-checked: switching the key to slots makes it
fail. The general ragged-workload equivalence test *passed under that
mutation*, because its workload happens to reorder the batch whenever a
slot is reused — so the dedicated test is the only guard.

### Deferred: overlapping host work with the next step

The remaining option is to launch step t+1 before processing step t's
results, so the Python bookkeeping runs while the GPU computes. It is
deliberately not built yet:

* every small host-to-device copy in `cache.advance()` currently comes
  from pageable memory, and CUDA performs a full stream sync before such
  a copy — so as written, the first copy would wait for the running
  replay and the overlap would silently not happen. It needs pinned,
  double-buffered staging throughout the cache;
* it has to speculate that every request continues, and unwind when one
  stops at an end-of-sequence token;
* its failure mode is a race between host and device, which no CPU test
  can reproduce.

It should be built only if the loop profile shows bookkeeping and
scheduling — host work the GPU waits through — still holding a material
share of the per-token cost after in-graph sampling.

```bash
python -m pytest tests/test_phase14_loop.py -v
python -m benchmarks.runners.phase14_loop --context-lengths 2048 --batch-sizes 1 4 16
```

## 14c — INT8 cache at graph speed

Not started. Needs three things the INT8 cache does not have: block
finalisation moved into host-side `advance()` (it currently runs on a
host-side condition inside the forward), a persistent fp16 residual
buffer, and a kernel that reads finished blocks as INT8 and the partial
block as fp16 and merges the two.
