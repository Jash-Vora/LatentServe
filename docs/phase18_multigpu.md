# Phase 18 — multi-GPU (two T4s)

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
embedding and output layers ignored).

## Predictions — fixed before running

* Burst scaling efficiency ~0.95: the GPUs are independent; the host CPU is
  the only shared resource.
* Chat hit rate ~86% with prefix-aware routing, ~40-60% with round-robin,
  which sends a conversation's turns to alternating GPUs; least-loaded in
  between. Prefix-aware's TTFT correspondingly lowest.
* Experiment B: at batch 1 a step mostly reads weights, which TP-2 halves
  (~7 ms saved at 2K), so TP *could* cut batch-1 latency if one all-reduce
  costs under ~100-125 us; above that it cannot. Revised from an earlier
  flat "TP will not pay", before measuring.

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
  1-token run) / 256. It cross-checks the all-reduce estimate: by the
  prediction above, TP-2 cuts batch-1 latency only if one all-reduce costs
  under ~100-125 us.
