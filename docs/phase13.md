# Phase 13 — CUDA Graph Decode

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
- [ ] `num_splits` re-tuned per bucket **under replay** — Phase 12 showed
      the isolated optimum did not survive the real model, and replay
      changes the environment again
- [x] graphs wired into `ServingEngine` (`use_cuda_graphs=True`), with
      `warmup_graphs()` to move capture cost to start-up
- [ ] side-by-side vLLM comparison with both systems on CUDA graphs

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
