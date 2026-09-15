# Phase 6 — vLLM Baseline

Goal (docs/methodology.md Phase 6), and the rule that governs the whole
phase:

> Do **not** make "Beat vLLM" the objective. Investigate where a
> specialized runtime can approach or exceed a mature serving system and
> where it cannot. If vLLM wins, **explain why**.

## What landed

| File | Role |
| --- | --- |
| `comparisons/vllm/runner.py` | offline vLLM engine configured to match LatentServe's conditions, with every control recorded |
| `benchmarks/runners/phase6_vllm.py` | drives either system over identical prompts; `--compare` pairs rows and refuses unfair ones |
| `configs/phase6_vllm.yaml` | the controls, as config rather than as flags to remember |

## The controls, and why each one matters

vLLM is not a drop-in peer. At its defaults it will, depending on
version, enable automatic prefix caching, chunked prefill and CUDA
graphs, and stop early on EOS. Each changes the work done.

| Control | Setting | Why |
| --- | --- | --- |
| Prompts | `prompt_token_ids`, not text | no tokenizer difference can creep in |
| `ignore_eos` / `min_tokens` | forced | on random synthetic prompts vLLM stops early; the comparison would measure who gave up sooner |
| Sampling | greedy, `temperature=0` | matches LatentServe's argmax |
| `enable_prefix_caching` | False | LatentServe has none until Phase 13; the `mixed` workload shares no prefixes, so leaving it on is an unearned win on a workload where it cannot help |
| dtype / GPU / `max_model_len` | fp16, 1x T4, matched | Turing has no bf16 tensor cores |
| Warm-up | one pass, discarded | vLLM's first call pays graph capture and allocator warm-up |
| Chunked prefill | **recorded, not forced** | see below |

Chunked prefill is deliberately left alone. vLLM's scheduler can mix
prefill and decode in one step; LatentServe's prefill blocks decoding,
which Phase 4 measured as ~2.3 s inter-token stalls with a stall rate
near 1%. That is a real architectural difference and one of the most
interesting things to report, so it is recorded in `extra` rather than
configured away.

## Run it

**Separate invocations, not one process.** vLLM takes a large persistent
share of VRAM at construction, so LatentServe's cache would be sized
against the leftovers.

```bash
python -m benchmarks.runners.phase6_vllm --config configs/phase6_vllm.yaml \
    --system latentserve --workload mixed --num-requests 32 \
    --max-prompt 8192 --max-output 256 --batch-sizes 4 8

# ... restart the session, install vLLM, then:
python -m benchmarks.runners.phase6_vllm --system vllm   [same arguments]

python -m benchmarks.runners.phase6_vllm --compare
```

Install vLLM in its own session: it pins its own torch build and will
replace the one the rest of LatentServe runs against. On a T4 (sm75),
check that the version you get still supports Turing — if not, pin an
older release rather than changing the model or dtype, since those are
controlled variables.

## What a gap means

Phase 3 measured LatentServe's paged gather at 2x resident KV per step
(~16 ms at batch 8 in isolation), and Phase 4 showed it makes batching
non-free: TPOT rose 33 -> 65 ms from batch 1 to 8, where Phase 2's
contiguous cache had stayed flat. vLLM has the paged-attention kernel
that removes that gather — the same kernel Phase 11 is aiming at.

So **the decode-throughput gap at large batch is a measurement of what
the Phase 11 kernel is worth**, not a verdict. That framing is the whole
value of running this phase before Phase 11 rather than after.

Two other gaps are expected and should be attributed rather than
lamented: CUDA graphs (LatentServe runs eager Python per layer, and
Phase 2's P1 found decode at 24-32% of peak bandwidth, i.e. overhead-
bound) and chunked prefill (Phase 4's stall behaviour).

## One asymmetry that cannot be controlled away

vLLM exposes a per-request mean inter-token latency, not per-token gaps.
LatentServe records the wall gap between consecutive tokens. So **p50 ITL
is comparable and p99 is not** — one is a percentile over request means,
the other over individual gaps, and Phase 4 showed those differ by 30x
under prefill blocking. The runner labels this with
`itl_measurement: per_request_mean`. Saying so is the difference between
a measurement and a claim.

## Gate 6 checklist — "can we fairly benchmark against vLLM?"

- [ ] `pytest tests/test_phase5_harness.py` green
- [ ] both systems run from the same prompt token ids and output lengths
- [ ] `--compare` produces no `SKIPPED — UnfairComparison` lines
- [ ] vLLM version, chunked-prefill and prefix-caching settings recorded
      in every vLLM row
- [ ] each gap attributed to a mechanism (gather, CUDA graphs, chunked
      prefill), not just reported
- [ ] p99 ITL explicitly excluded from the comparison
