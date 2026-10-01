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

## The test that matters most

`test_different_inputs_give_different_outputs`. The classic graph bug is
forgetting to copy new inputs into the static buffers, after which the
graph replays the previous step and the output for token B is the output
for token A. Nothing raises, and every other test can pass while it is
broken.

## Gate 13 checklist

- [ ] CPU suites green (address stability, static core, orchestration)
- [ ] GPU suite green, including the stale-input test
- [ ] `--experiment compile` run and its break count recorded
- [ ] batch 1 and batch 4 latency measured, eager vs graphed
- [ ] each prediction above marked confirmed or falsified
- [ ] `num_splits` re-tuned per bucket **under replay** — Phase 12 showed
      the isolated optimum did not survive the real model, and replay
      changes the environment again
