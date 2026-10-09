# Phase 13 — CUDA Graph Decode

> **Outcome.** CUDA graphs removed the CPU-dispatch bottleneck. At batch 1 the step fell from about 40 ms to 20–28 ms (29–49% less), because the GPU had been idle 39% of the time waiting for about 900 Python-launched kernels. Batch 4 at long context gained nothing, since it is bound by the attention kernel, not launches. A GPU-only bug, where a CPU model was "captured" and replayed frozen outputs, was found and fixed. Against vLLM with graphs on both sides, decode was within 8% except at batch 1 with short context, and prefill was 1.9–3.3× faster. These numbers predate the CUDA-core decode kernel and the prefill routing fix, so they understate LatentServe; the final sweep supersedes them.

## Why this phase exists

Phase 12's profiler, batch 1 / ctx 8192, per decode step:

| | |
| --- | ---: |
| wall | 40.0 ms |
| GPU busy | 24.6 ms |
| **CPU busy** | **42.6 ms** |
| GPU idle | 15.4 ms (39%) |

CPU time exceeds wall time. The step is limited by Python dispatching
~900 kernel launches — 630 of them small elementwise ops — not by the GPU
executing them. Attention is 5.7 ms of the 24.6 ms of GPU work, so even a
free attention kernel would leave wall near 40 ms. That is why three
rounds of kernel tuning in Phase 11 barely moved batch 1.

A CUDA graph records the launch sequence once and replays it in a single
call. The per-launch cost disappears.

## What a graph freezes

A graph records memory **addresses** and **shapes**. On replay it reads
the same addresses with the same sizes regardless of what the Python
objects now say. Anything that moves or resizes between steps replays
stale data — silently, with fluent output and no error.

The audit found three things on the decode path that broke this.

**The cache reassigned its tensors every step.** `advance()` built fresh
`_write_slots`, `_block_tables` and `_seq_lens_tensor`, and rebuilt
`_read_slots` with a per-sequence loop of host-to-device copies that the
kernel path never reads. Now: persistent buffers allocated once at full
capacity, written into with `copy_()`; the block table held at
*capacity* width so one graph serves a growing sequence; `_read_slots`
built lazily by the gather path only; and `reset()` clears contents
without reallocating, so a graph captured before a reset survives it.

**RoPE synced the host every step.** `positions.max().item()` to decide
whether to grow its tables. Callers that know the bound now pass it.

**`advance()` ran inside the forward.** A graph replays GPU work only, so
the bookkeeping that decides *where* this step's KV goes would run once
at capture and never again — every replay writing the same slot. The
forward is now split: host bookkeeping outside, `decode_forward_static`
— no host state, no syncs, no data-dependent branches — inside.

Two more, found while designing the capture rather than by the audit:

* **Kernel scratch was keyed by name only.** Capturing a second graph
  with a different split count *reallocated* the shared buffer, leaving
  the first graph pointing at freed memory. Now keyed by shape.
* **RoPE rebuilds reallocate.** A position past the tables would rebuild
  them and orphan every earlier graph's cos/sin pointers. The decoder
  pre-builds RoPE to the cache's capacity; `advance()` refuses positions
  past capacity, so no rebuild can happen after a capture.

## A side effect worth reporting

The kernel branch in attention used to require a uniform batch
(`key_mask is None`), which kept continuous batching off the kernel
entirely. That was caution, not necessity: the kernel masks each
sequence by its own length from `seq_lens`. The branch now runs before
`padding_mask()` and accepts ragged batches, and
`test_kernel_path_handles_ragged_batches_without_a_padding_mask` checks
it against the gather path on a genuinely ragged batch.

## Design

`runtime/cuda_graph.py`. Graphs are keyed by (batch size, context
bucket). The bucket is a **performance** refinement, not a correctness
one — any graph whose bucket covers the current length is correct. It
buys a split count suited to the length, since a long-context split
count on a short sequence launches mostly-empty programs.

Capture happens *during* a real decode step using that step's real
inputs. That writes this token's KV into its slot during warm-up and
capture, which is harmless because it is idempotent: same token, same
position, same values, same slot. Capture records kernels without
running them, so the graph is replayed once immediately afterwards to
produce the step's logits.

Out of scope, and refused with `GraphUnsupported` so the engine can fall
back to eager rather than crash:

* the contiguous cache — it reads a slice whose length changes each step;
* the INT8 cache — it finalises blocks on a host-side condition, and its
  residual writes are not idempotent.

## Run it

```bash
pytest tests/test_phase13_graph_buffers.py tests/test_phase13_cuda_graph.py -v

python -m benchmarks.runners.phase13_graphs --experiment compile
python -m benchmarks.runners.phase13_graphs --experiment latency \
    --context-lengths 4096 8192 16384 --batch-sizes 1 4
```

Run `--experiment compile` first. If `torch.compile(mode="reduce-overhead")`
captures the decode core without graph breaks, the hand-rolled path is
unnecessary. Breaks are expected given how stateful the cache is, but
it is a thirty-minute check.

## Predictions

* **Batch 1 / 8K falls from ~40 ms toward ~25–28 ms** — the GPU floor
  plus one replay call and the host-side `advance()`.
* **Batch 4 at long context gains much less.** It was already GPU-bound;
  the kernel was within 2.4% of SDPA at 8K.
* If batch 4 gains as much as batch 1, Phase 12's reading of where the
  time went was wrong.

## Measured (T4, Qwen2.5-1.5B fp16, paged cache, Triton kernel path)

| batch | ctx | eager | graphed | change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 40.29 ms | **20.45 ms** | -49.2% |
| 1 | 8192 | 39.80 | **24.05** | -39.6% |
| 1 | 16384 | 39.73 | **28.34** | -28.7% |
| 4 | 4096 | 39.22 | **27.05** | -31.0% |
| 4 | 8192 | 40.22 | 39.74 | -1.2% |
| 4 | 16384 | 53.89 | 53.56 | -0.6% |

All 21 Phase 13 tests pass on the T4, including
`test_different_inputs_give_different_outputs`. That matters for reading
the table: a graph replaying stale inputs would be exactly this fast, so
the latency numbers alone could not have ruled it out.

**`torch.compile(mode="reduce-overhead")`: 13 graph breaks and per-layer
recompilation.** `PagedKVCache.max_len` uses `max(..., default=0)` over a
generator, which Dynamo cannot trace, splitting the graph at every layer;
and Dynamo specialises on `layer_idx` as a static integer, compiling a
separate attention per layer until it hits its recompile limit at layer
8. Fixable in principle, unnecessary in practice — the hand-rolled
capture records kernel launches rather than Python, so neither applies.

### Batch 1: prediction confirmed, overhead eliminated

Phase 12 measured 24.6 ms of GPU work at B1/8K under 42.6 ms of CPU
dispatch. Graphed B1/8K is **24.05 ms**: wall time now *is* GPU time.

At 4K it beat the assumed floor. 20.45 ms for 3.09 GB of weights plus
~117 MB of KV is ~157 GB/s, against the ~130 GB/s this document had been
assuming the card sustains. With no gaps between launches the memory
system never idles, so achieved bandwidth rises as well as dispatch
falling — the 130 GB/s figure was partly an artefact of eager execution.

### Batch 4 at long context: no gain, and the graph shows why

B4/8K and B4/16K are GPU-bound, as predicted. With overhead gone, the
incremental cost of batch 4 over batch 1 is directly readable: at 8K it
adds 15.7 ms for ~705 MB more KV, i.e. **~45 GB/s** — matching the
~47 GB/s ceiling Phase 12's isolated sweep found for the kernel. Two
independent measurements, same number.

So the bottleneck has moved. At batch 1, dispatch is solved. At batch 4,
the attention kernel reads KV at ~47 GB/s on a card that reads weights
at ~150. Re-tuning `num_splits` will not close that: the Phase 12 sweep
showed batch 4 nearly flat across split counts (45-48 GB/s).

### Against earlier targets

* Phase 11's target was ~33 ms at B4/8K. Graphed: 39.7 ms. The remaining
  gap is the kernel's KV read rate, not overhead.
* Phase 6 measured vLLM's differenced decode at 42.9 ms at B4/8K. Graphed
  LatentServe is 39.7 ms — *suggesting* parity or better on decode.
  Those numbers come from different sessions and library versions; rerun
  them side by side before claiming it.

## The test that matters most

`test_different_inputs_give_different_outputs`. The classic graph bug is
forgetting to copy new inputs into the static buffers, after which the
graph replays the previous step and the output for token B is the output
for token A. Nothing raises, and every other test can pass while it is
broken.

## Gate 13 checklist

- [x] CPU suites green (address stability, static core, orchestration)
- [x] GPU suite green, including the stale-input test
- [x] `--experiment compile` run and its break count recorded (13 breaks)
- [x] batch 1 and batch 4 latency measured, eager vs graphed
- [x] each prediction marked: both confirmed
- [x] `num_splits` re-tuned per bucket **under replay** — Phase 12 showed
      the isolated optimum did not survive the real model, and replay
      changes the environment again
- [x] graphs wired into `ServingEngine` (`use_cuda_graphs=True`), with
      `warmup_graphs()` to move capture cost to start-up
- [x] side-by-side vLLM comparison with both systems on CUDA graphs
- [x] `num_splits` re-tuned under replay: capped at 16

*Reviewed at project close: the first `num_splits` item duplicates the last one, which was already ticked: done, capped at 16.*

## Serving integration

`ServingEngine(..., use_cuda_graphs=True)` decodes through
`GraphedDecoder`; prefill stays eager. Enabling graphs sets the Triton
kernel path on every layer, since the gather path cannot be captured,
and falls back to eager with a warning if the configuration is not
capturable — a missing graph is a slowdown, not a failure.

Continuous batching changes the batch size as requests arrive and finish,
and graphs are keyed by batch size, so the first step at each new size
pays a capture. `warmup_graphs(batch_sizes, context_length)` moves that
cost to start-up. It runs throwaway sequences through real cache slots
and then resets the cache; reset keeps storage, so the captured addresses
stay valid.

All graphs share one memory pool. Warming every batch size at two context
points is 2B graphs — 32 at batch 16 — each of which would otherwise hold
a private pool sized to its own peak. Sharing is safe because graphs
never run concurrently and each graph's output stays referenced, so no
later capture can reuse it.

`test_engine_with_graphs_matches_engine_without` is Gate 3 for the graph
path: ragged prompts and output lengths, so the batch size moves and
slots are recycled mid-run — exactly the conditions under which a graph
keyed by batch size could replay the wrong rows.

## The vLLM comparison, redone properly

The Phase 6 comparison built LatentServe with the default
`attn_impl="sdpa"` — the Phase 3 **gather** path — while vLLM ran its
paged kernel under CUDA graphs. That was never a like-for-like decode
comparison. `phase6_vllm.py` now takes `--attn-impl triton_paged` and
`--cuda-graphs`, and labels each LatentServe variant as its own system so
all of them can be compared against the same vLLM rows.

`--compare` reports two things separately, because they can disagree:

* **latency** — decode ms/step from differenced output lengths, read at
  batch 1, which is what a single user feels between tokens;
* **throughput** — output tokens/s, read at the largest batch, which is
  what a server can sustain and is vLLM's home ground.

## A bug the GPU found that the CPU suite could not

`test_engine_with_graphs_matches_engine_without` builds its model on the
CPU. In a CPU-only environment it passed, because the decoder never
enabled. On a T4 it failed — the graphed engine emitted the same token on
every step — and CUDA reported "The CUDA Graph is empty".

`GraphedDecoder` decided whether to capture by checking
`torch.cuda.is_available()`, which is about the *machine*. A CPU model on
a GPU machine therefore "captured": the CPU ops executed once during
capture and produced one real set of logits, the graph recorded no CUDA
kernels, and every replay afterwards was a no-op. The output buffer froze
on the first step's logits.

That is the stale-replay failure this phase was most worried about,
arriving by a route nobody wrote a test for, and visible only on hardware
the default test run does not use. Capture is now keyed on the model's
device, `check_capturable` refuses a CPU model, and
`test_cpu_model_is_never_captured_even_when_a_gpu_exists` fakes a GPU so
the condition reproduces on any machine.

## `num_splits` under replay

16 splits was best at three of four points (b1/4K 20.33 ms vs 21.19 at
32; b1/16K 26.89 vs 28.08 at 64; b4/4K 27.23 vs 31.49 at 64) and within
noise at b4/16K. The isolated sweep had said 64; that optimum survived
neither eager execution (Phase 12) nor replay. `MAX_SPLITS` is now 16.

## Against vLLM, same session, both on CUDA graphs

32 uniform requests, contexts 2K and 8K, decode isolated by differencing
128 vs 256 output tokens.

| batch | ctx | LS decode | vLLM decode | decode | prefill ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 20.8 ms | 17.7 ms | **-15%** | 3.3x |
| 1 | 8192 | 24.7 | 23.0 | -7% | 3.1x |
| 4 | 2048 | 23.8 | 22.2 | -7% | 1.9x |
| 4 | 8192 | 39.5 | 42.1 | +7% | 3.1x |
| 8 | 2048 | 28.7 | 29.1 | +1% | 1.9x |
| 8 | 8192 | 59.4 | 63.9 | +8% | 3.1x |
| 16 | 2048 | 39.9 | 42.2 | +6% | 1.9x |
| 16 | 8192 | 99.6 | 106.9 | +7% | 3.1x |

### The prediction was wrong both ways

The prediction was: lead on batch-1 latency, fall behind on throughput as
batch grows. Measured: **behind on batch-1 latency** (15% at 2K) and
**slightly ahead on decode at batch >= 4 and long context** (5-8%).

### The first verdict reported the wrong throughput

The compare's first version read throughput from end-to-end output
tokens/s and reported LatentServe "147% ahead" at b16/8K. At that point
prefill is 86% of vLLM's run and 68% of LatentServe's, and LatentServe's
prefill is 3.1x faster — the Turing-specific Triton fallback Phase 6
already found. So the headline lead was a **prefill** result presented
as a throughput one. The compare now reports decode latency, decode
throughput and end-to-end throughput separately, and annotates the last
with the prefill ratio wherever prefill dominates.

That is the fourth time in this project a measurement answered a
different question than the one asked, and the second time the cause was
blending prefill into a per-token figure (Phase 6 caught the first).

### What the decode column says

One consistent reading fits every row: a roughly **constant ~3 ms deficit
outside attention**, offset by an **attention advantage that grows with
context and batch**.

* At b1/2K attention is a small share of the step, and the full deficit
  shows: 3.1 ms.
* At b1/8K the deficit narrows to 1.7 ms, because the attention kernel
  is now doing more of the work and is slightly better than vLLM's.
* From b4/8K on, the attention advantage outweighs the deficit.

The attention side is Phase 6's finding again: on sm75 vLLM falls back to
TRITON_ATTN, which degrades with context. The deficit side is **fusion**.
vLLM's startup log (Phase 6) shows inductor compilation, and its Qwen2
implementation merges `q/k/v` into one projection and `gate/up` into
another. LatentServe runs three projections where vLLM runs one, two
where vLLM runs one, and roughly 630 small elementwise kernels per step
(Phase 12: 1.80 ms of GPU time on its own).

CUDA graphs removed the CPU cost of launching those kernels. They did not
remove the kernels: each still has a minimum GPU execution time, and the
boundaries between them serialise. **Graphs remove launch overhead, not
kernel count** — which is why the batch-1 gap survived Phase 13.

### The honest headline

On a T4, under CUDA graphs on both sides:

* **decode is at parity with vLLM** — within 8% everywhere except batch 1
  at short context, where vLLM leads by 15%;
* **prefill is 1.9-3.3x faster**, a Turing-specific result (no FA2 on
  sm75) that should be expected to invert on an A100;
* end-to-end throughput therefore ranges 0.96x-2.47x vLLM, and should be
  quoted alongside the prefill ratio, never on its own.
