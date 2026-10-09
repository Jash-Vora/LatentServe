# Phase 18 — multi-GPU (two T4s)

> **Outcome.** Two replicas scaled burst throughput 1.7–1.8× (efficiency 0.85–0.88). vLLM's tensor parallelism cut batch-1 per-token latency 1.7× on two T4s. Least-loaded routing beat round-robin by 8–20% on chat. The first replica run was invalidated by cold-start compilation; it is kept, with the reason, in the sections below.

## Experiment A — replicated serving (`phase18_replicas`)

A router in front of one engine per GPU, each in its own worker process
(two engines in one Python process would contend for the interpreter lock
during prefill). Workers see only their own GPU; a `reset` rebuilds a
worker's engine without reloading the model, so every configuration starts
with clean KV and prefix caches.

Routing (`runtime/router.py`): round-robin; least-loaded (fewest outstanding
requests); prefix-aware (the replica that last served the deepest matching
prefix, unless it is more than 2 requests busier than the least loaded — pure
affinity would send every conversation to whichever replica first saw the
shared system prompt; the slack keeps deep matches at home and balances
shallow ones).

Workloads: *burst* — 96 requests at once (1-4K prompts, 128-256 outputs),
throughput-bound, for scaling efficiency = 2-GPU throughput / (2 x 1-GPU);
*chat* — 16 conversations x 5 turns sharing a 512-token system prompt, each
turn sent when its reply arrives, prefix caching on, for routing. Per request:
time to first token, time per output token and end-to-end time at p50/p95/p99;
plus throughput, hit rate, tokens and busy time per GPU. The worker backend is
pluggable so the final sweep can run vLLM replicas through the same harness.

## Experiment B — model parallelism, measured not built (`phase18_allreduce`)

Secondary in the plan. Tensor parallelism across two GPUs needs two
all-reduces of the hidden state per layer — 56 per decode step. The runner
times NCCL all-reduce at those sizes and estimates a TP-2 step as half the
single-GPU step plus the communication: an optimistic bound (perfect split,
embedding and output layers ignored). At batch 1 a step mostly reads weights, which
TP-2 halves (about 7 ms at 2K), so TP can cut batch-1 latency only if one all-reduce
costs under ~100-125 us.

## Additions before the first run

* **vLLM worker backend** (`--backend vllm`): vLLM's incremental engine
  (`llm.llm_engine`: add_request / step) behind the interface the worker loop
  drives, so vLLM replicas run through the same router and workloads. Prefix
  caching on throughout (vLLM cannot toggle it per configuration); a reset
  clears it. Hits come from `num_cached_tokens` where the version reports it.
* **Graph capture excluded from throughput.** Each configuration rebuilds its
  engine, so LatentServe recaptured CUDA graphs inside the timed run — the
  same seconds per GPU whatever the GPU count, so on two GPUs the share
  doubled and would have biased scaling efficiency down; and vLLM captures
  at startup, so including it would bias every cross-engine comparison.
  Throughput is reported raw and with the longest per-GPU capture excluded
  (the GPUs capture in parallel); scaling efficiency uses the latter.
* **A test that passed for the wrong reason.** The harness test's fake
  cluster started its clock at a fixed 100.0 while submissions are stamped
  with the real `perf_counter` (system uptime): on any machine up for more
  than ~100 s the makespan was *negative*, throughput negative, and an
  exact float comparison passed only when two negatives multiplied back to
  the right integer — flaky, and wrong when it passed. The fake now starts
  from the real clock, the comparison has a tolerance, and the test asserts
  makespan and throughput are positive.
* **`phase18_vllm_tp`:** vLLM's actual tensor parallelism, TP=2 against TP=1,
  batch 1 and 8, 2048-token prompts — per-token time from (257-token run −
  1-token run) / 256. It cross-checks the all-reduce estimate.

## First results (Kaggle T4 x2)

### Experiment B — tensor parallelism: works, and beat the "bound"

NCCL all-reduce, peer-to-peer access on: 46-56 us at decode sizes (batch
1-16), 70 us at batch 32 — 2.6-3.9 ms per decode step for 56 of them.
Under the recorded ~100-125 us threshold, so TP-2 should cut batch-1
latency. vLLM's real TP-2 against TP-1 (2048-token prompts):

| | TP-1 | TP-2 | TP-2 / TP-1 |
| --- | ---: | ---: | ---: |
| per-token, batch 1 | 16.76 ms | 9.66 ms | 0.58x |
| per-token, batch 8 | 28.31 ms | 16.18 ms | 0.57x |
| first token, batch 1 | 840 ms | 532 ms | 0.63x |

TP-2 cut batch-1 latency, and by more than the "optimistic upper bound"
allowed: at most 1.45x at batch 1, against the 1.73x vLLM achieved. The estimate
assumed a perfect compute split (optimistic) but NCCL's all-reduce latency
(pessimistic); vLLM's log shows it uses its own peer-to-peer all-reduce
(`['CUSTOM', 'PYNCCL']`), evidently several times cheaper at these sizes.
The runner no longer calls it a bound. **On two PCIe T4s with peer access,
splitting the model gives ~1.7x faster decode** — measured, not estimated.

### Experiment A — LatentServe replicas: an artifact, and chat findings

**Burst scaling efficiency came out 1.44 — impossible for independent
replicas.** Super-linear scaling means a handicapped baseline: "burst 1 GPU"
ran first on cold workers and paid one-off compilation (Triton prefill
kernels per prompt shape, NVRTC, cuBLAS setup — vLLM's log shows the same
"JIT compilation during inference"); round-robin, run after least-loaded,
was faster too. Graph capture was excluded, compilation was not, and every
configuration ran once in a fixed order.

Chat (16 closed-loop conversations x 5 turns, 2 GPUs):

| routing | hit rate | TTFT p50 / p95 / p99 | tok/s |
| --- | ---: | --- | ---: |
| round-robin | 74.1% | 137 / 479 / 546 ms | 703.8 |
| least-loaded | 86.7% | 282 / 490 / 528 ms | 757.3 |
| prefix-aware | 86.7% | 265 / 488 / 572 ms | 725.3 |

* Round-robin hit 74%: the GPU a turn alternates to
  still holds the conversation's history from two turns earlier.
* Least-loaded matched prefix-aware: with
  closed-loop clients a returning conversation tends to go to the GPU that
  just freed its slot — the one holding its history. Affinity emerges.
* Round-robin had the *lowest* median TTFT, half the others'.
  Strict balance beats cache hits for median
  latency; the cache-friendly policies win 3-8% throughput. Tails similar.
* The "chat scaling efficiency 0.59" was a meaningless number to print:
  closed-loop with fixed concurrency, a second GPU halves each GPU's batch.
  Chat measures latency; only the burst can measure scalability.

**vLLM replicas crashed at startup:** `daemonic processes are not allowed to
have children` — vLLM starts its own engine process; the workers were
daemons. The parent then waited out a 15-minute timeout, the dead workers
having reported nothing.

### Fixes before the rerun

* Workers are not daemons (torn down explicitly), report any failure with
  its traceback, and the parent checks their liveness while waiting.
* A warm-up workload on every worker, every prompt shape, before any timed
  run.
* Two rounds, the second in reverse order; medians and the throughput spread
  are reported, so order effects are visible rather than decisive.
* Scaling efficiency is reported for the burst only.

## Rerun (warm-up, two rounds in alternating order) — Phase 18 results

The warm-up alone moved LatentServe's single-GPU burst from 161.7 to 276.1
tok/s: the first run's baseline had paid one-off compilation.

| configuration | LatentServe tok/s | vLLM tok/s | LS TTFT p50 / p99 | vLLM TTFT p50 / p99 | hit (LS / vLLM) |
| --- | ---: | ---: | --- | --- | --- |
| burst 1 GPU | 276.1 | 98.7 | 30.3 / 61.6 s | 75.1 / 148.7 s | — |
| burst 2 GPU least-loaded | 472.1 (spread 11.2%) | 173.3 | 15.5 / 34.6 s | 38.5 / 83.8 s | — |
| burst 2 GPU round-robin | 466.7 | 173.0 | 15.8 / 35.0 s | 38.8 / 83.8 s | — |
| chat 1 GPU | 648.1 | 419.8 | 421 / 793 ms | 1069 / 1436 ms | 87.2% / 87.2% |
| chat 2 GPU round-robin | 701.8 | 484.8 | 212 / 553 ms | 438 / 983 ms | 73.6% / 73.2% |
| chat 2 GPU least-loaded | 759.3 | 581.7 | 261 / 537 ms | 487 / 782 ms | 86.7% / 86.7% |
| chat 2 GPU prefix-aware | 752.9 | 552.7 | 259 / 557 ms | 516 / 856 ms | 86.7% / 86.7% |

**Burst scaling efficiency: 0.85 (LatentServe), 0.88 (vLLM).** Main cause, from busy time: one GPU idles for the last 14% (LS) /
20% (vLLM) of the run. Both routers balance request *counts*, and the burst's
prompts are 1K, 2K or 4K at random, so equal counts are unequal work. That
explains nearly all of vLLM's loss and about half of LatentServe's; the rest
is plausibly the shorter per-GPU tail or the two workers sharing the host's
few CPUs — this run cannot separate them. Balancing outstanding *tokens*
instead of requests should recover much of it. Least-loaded's two burst
rounds differed by 11%; with all requests arriving at once it behaves like
round-robin (stable, 1.8%), so its medians are not a difference.

**Routing, replicated on both engines:** round-robin gives the lowest
*median* TTFT (strict balance); least-loaded the highest throughput (+8% LS,
+20% vLLM over round-robin) and the best tails; least-loaded matches
prefix-aware's hit rate because closed-loop clients produce affinity on
their own. A property of the routing, not of one engine.

**Engines on the same router and workloads:** LatentServe 2.8x vLLM on the
burst — ~220K prompt tokens, so mostly Phase 6's prefill advantage (2.7-6x) —
and +54% (1 GPU) / +31% (2 GPUs) on chat. **Fairness caveat:** LatentServe ran
with at most 16 concurrent requests; vLLM with its default `max_num_seqs`
(not passed — likely 256). On a prefill-bound burst it probably does not
change the conclusion, but the final sweep sets the same limit for both.
