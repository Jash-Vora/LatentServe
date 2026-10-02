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

### Measured: the loop is cheap

`phase14_loop.py`, vLLM's stack, one process:

| batch | floor (model step) | engine, host sampling | engine, in-graph | in-graph saves |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 18.60 ms | 18.86 (+0.27) | 18.81 (+0.21) | 0.06 |
| 4 | 22.65 | 23.04 (+0.39) | 23.02 (+0.37) | 0.02 |
| 16 | 37.92 | 39.42 (+1.50) | 39.70 (+1.78) | -0.28 (no spread recorded) |

The engine costs only **0.2-0.4 ms per token over the model at batch 1-4**,
and bookkeeping plus scheduling is ~0.03 ms of that. The 0.7-1.8 ms
inferred earlier came from comparing the bare step against the long
`phase6` runs — two different harnesses — and does not survive being
measured in one. The serving-loop hypothesis was mostly wrong, and by the
rule set in advance, **overlapping host work with the next step is not
justified**: there is almost nothing left for it to hide.

In-graph sampling saved 0.06 ms at batch 1. The batch-16 figure cannot be
judged — this runner first shipped without a round spread, now fixed.

### A correction, and a harness discrepancy

Host `291dd30e574b`, torch 2.13, fused, in-graph, against vLLM:

| | LatentServe | vLLM |
| --- | ---: | ---: |
| batch 1 / 2K | 21.3 | **17.7** |
| batch 1 / 8K | 23.2 | 23.1 |
| batch 4 / 8K | 37.0 | 42.0 |
| batch 16 / 8K | 83.4 | 107.1 |

**vLLM's 17.7 ms at batch 1 / 2K is not an outlier.** It was called one
when only the original host showed it; this host reproduces it. Across
four hosts vLLM measures ~17.7 on two and ~21 on two, while LatentServe
stays at 21-22 on all of them. On machines like these, vLLM leads
batch-1 short-context decode by ~17%, and "parity" would be the wrong
conclusion.

**And the same engine measures 2.5 ms apart in two harnesses.** On this
host and stack, batch 1 / 2K: 18.81 ms in `phase14_loop`, 19.1-20.1 ms as
the median token gap in `phase6`, 21.3 ms as `phase6`'s differenced mean.
vLLM is only ever measured in `phase6`, so whatever inflates LatentServe
there feeds directly into the comparison. The median sitting ~1.5 ms below
the mean also says part of it is a tail of slow steps, not every step.

Two candidates, distinguishable with the instrumentation now in `phase6`
(`--profile-loop`, plus a GPU sampler on both arms):

* **the GPU runs slower under sustained load** — `phase6` is minutes of
  back-to-back prefill and decode on a passively cooled 70 W card, while
  `phase14_loop` measures short bursts with pauses. Shows up as higher
  `gpu_wait`, a lower SM clock, and power or thermal throttle reasons;
* **the waiting queue** — at batch 1 `phase6` keeps up to 31 requests
  queued, `phase14_loop` none. Shows up as higher `schedule`.

### Resolved from existing data: a measurement artifact

The 2.5 ms discrepancy needed no new run. Every batch-1 / 2K result from
Phases 13-14, side by side:

| run | median gap, 128-token run | median gap, 256-token run | differenced |
| --- | ---: | ---: | ---: |
| host `6a3a…` | 19.2 | 19.9 | 20.8 |
| host `dc4b…`, torch 2.10 | 20.5 | 22.0 | 24.0 |
| host `291d…`, default stack | 19.6 | 20.1 | 20.7 |
| host `291d…`, torch 2.13 | 19.1 | 20.1 | 21.3 |

On every host the long run is slower than the short one — same code, same
machine, minutes apart — and the differenced "decode" sits above both.

Differencing subtracts the short run from the long one and assumes they
went at the same speed. If the short run averaged *a* ms per step and the
long one *b*, it returns **2b − a**: on `291d…`, 2 × 20.1 − 19.1 = 21.1,
against 21.3 measured. The other three hosts agree to within 0.5 ms.

Batch 1 / 2K is the first configuration of every run, measured straight
after model loading and graph capture — idle time, after which a T4 runs
at boost clock before settling lower under sustained load. Hence the gap
is largest there, and at batch 1 / 8K — never first — the differenced
figure and the median agree exactly (23.2, 23.2). The same inflation
leaves too little time for prefill, which is why batch 1 / 2K alone
reported prefill rates of 11,755-38,176 tokens/s against ~4,300 everywhere
else. Those were flagged as broken at the time; what they implied about
the decode figure was missed.

**LatentServe's batch-1 / 2K decode on `291d…` is ~19.1-20.1 ms, not
21.3.** The bimodal vLLM figure (17.7 on some hosts, ~21 on others) may
have the same cause, depending on run order in a session; the median for
vLLM is already in the results files.

What changed:

* `--compare` tests the assumption directly: if a configuration's two runs
  differ by more than 2.5% in median token gap, differencing is discarded
  for that row — for either system — and the long run's median is used,
  with the prefill estimate withheld. Steady configurations measured so
  far differ by at most 1.8%; every batch-1 / 2K case by 2.6-7.3%.
* The comparison harness sustains GPU load for 30 s before the first timed
  configuration (`--settle-seconds`), so no run starts at boost clock.

### Settling batch 1 / 2K: short, long, long, short

vLLM's V1 engine reports no per-token timing, so its batch-1 / 2K figure
(17.7 ms) cannot be checked for the same artifact — and it is the first
configuration of vLLM's run too. The artifact can push either way: a first
run *slower* than the second, which vLLM's first real request may be,
makes the subtraction come out *below* the true per-step time.

The comparison harness now supports `--abba`: each configuration runs
short, long, long, short, and the compare averages each pair. A drift that
is linear in time hits both lengths equally and cancels; the two per-pair
estimates give a repeatability check that needs no per-token timing from
either system. `--throwaway-first` adds one unrecorded run before the
first configuration, so no measured run is the first after loading, and
the 30-second GPU settle still runs before everything. Repeated lengths
are averaged only within one invocation's run id, so a new run never
blends with an older session's rows.

### Settled: vLLM leads batch 1 / 2K by 12%

Host `6e340272fefa`, torch 2.13, settle + throwaway + short-long-long-short:

| | ABBA pair estimates | decode |
| --- | --- | ---: |
| LatentServe | 20.4 / 20.5 | 20.4 ms |
| vLLM | 17.9 / 18.0 | 18.0 ms |

Both systems' two half-run estimates agree to 0.1 ms, and the prefill
rates are back in line with every other configuration (4,076 and 2,131
tokens/s) — the impossible ones were the drift. vLLM's 18.0 matches the
17.7 from two earlier hosts.

## 14a, step 2 — elementwise fusion

The 2.4 ms is not attention (level or ahead at 8K) and not the loop
(~0.2 ms). It is the fixed per-token work outside attention, which vLLM's
compiler fuses: per layer, two RMSNorms at eight kernels each, two
residual adds, RoPE at ten kernels across q and k, and SiLU-and-multiply
at two — about 30 kernels, 840 per token. Fused (kernels/fused_elementwise.py):
one kernel per norm with its residual add folded in, one for RoPE on q
and k together, one for SiLU-and-multiply. About 730 fewer per token; at
~2.6 us each under graphs, ~1.9 ms.

Each kernel reproduces Hugging Face's rounding step by step — the RMSNorm
rounds to fp16 before applying the weight, RoPE rounds each product and
then the sum — so it matches the model to fp16 round-off, not
approximately. The layer loop is restructured so each residual add folds
into the *next* norm and the last into the final norm; that is the part
most likely to be wrong, and it is checked on CPU against the original
loop with zero tolerance. `--toggle elementwise` measures it alone, with
projections fused on both sides.

**Prediction:** batch 1 / 2K from 20.4 toward ~18.5 ms, roughly constant
in milliseconds across contexts.

## 14c — INT8 cache at graph speed

The INT8 cache quantized each completed block inside `write()`, on a
Python decision, per layer — not capturable. But the block that just
filled is still whole in the fp16 residual, so its quantization can wait
for the next `advance()`, which runs outside the graph. Until then the
kernel reads that page from the residual.

**Invariant, in both modes:** every page of a sequence except its last is
in the INT8 pool; the last is in the residual, exact. The kernel derives
"last page" from the sequence length on the device. Immediate mode
already satisfied this, so the INT8 cache now uses the kernel path
without graphs too — the tail merge Phase 11 deferred.

Deferral changes *when* quantization happens, never *what* it computes,
so it is tested exactly on CPU: the deferred pool is byte-identical to the
immediate one; the invariant holds at every step; decode logits are
identical through the kernel path and through the gather path.

### A bug the exact test caught

The first version flushed pending blocks for **every layer** whenever the
gather path read. But writes happen layer by layer within a step: when
layer 0 reads, layers 1+ have not written this step's token yet — so
their blocks were quantized one token short. The kernel path never hit
it, because it only flushes in `advance()`, before any layer writes.
Quantization is now tracked per layer, and `read()` flushes only the
layer it reads.

### Prediction

Per step, INT8 reads half the KV bytes, and the kernel no longer pays a
gather or a dequantize pass. At batch 1 / short context KV is a small
share of the step, so expect little change. At long context and high
batch — B16/8K, where KV is ~60 of ~83 ms — expect a large drop if the
kernel's bytes-per-second holds on INT8 data. Deferred quantization costs
an eager burst every 16 tokens per sequence, ~0.1 ms per step amortized.

## Run

```bash
pytest tests/test_phase14_elementwise.py tests/test_phase14_int8_graph.py -v
python -m benchmarks.runners.phase14_fusion --toggle elementwise --context-lengths 2048 8192 --batch-sizes 1 4 16
python -m benchmarks.runners.phase14_fusion --toggle int8 --context-lengths 2048 8192 --batch-sizes 1 4 16
```

Not started. Needs three things the INT8 cache does not have: block
finalisation moved into host-side `advance()` (it currently runs on a
host-side condition inside the forward), a persistent fp16 residual
buffer, and a kernel that reads finished blocks as INT8 and the partial
block as fp16 and merges the two.
