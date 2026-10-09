# Phase 6 — vLLM Baseline

> **Outcome.** An early comparison, partly superseded. It found vLLM's decode faster (26.0 against 34.5 ms per step at 2K) and its prefill collapsing at long context (744 against 5,238 tok/s at 8K), because a T4 can't run FlashAttention-2 and vLLM falls back to Triton attention. The decode deficit was LatentServe's own paged gather, worth about 17 ms per step at 8K; the paged-attention kernel (Phases 11–12) closed it, and the final sweep shows LatentServe decoding 1.05–3.3× faster than vLLM. The prefill finding stands, and widens at longer prompts.

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
Phase 2 found decode at 24-32% of peak bandwidth, i.e. overhead-
bound) and chunked prefill (Phase 4's stall behaviour).

## One asymmetry that cannot be controlled away

vLLM exposes a per-request mean inter-token latency, not per-token gaps.
LatentServe records the wall gap between consecutive tokens. So **p50 ITL
is comparable and p99 is not** — one is a percentile over request means,
the other over individual gaps, and Phase 4 showed those differ by 30x
under prefill blocking. The runner labels this with
`itl_measurement: per_request_mean`. Saying so is the difference between
a measurement and a claim.

## Measured (Qwen2.5-1.5B fp16, 1x T4, 8 requests, batch 4, burst)

### End to end, uniform contexts

| ctx | LatentServe | vLLM | ratio |
| ---: | ---: | ---: | ---: |
| 1024 | 39.2 ms/step | 27.9 | 0.71x |
| 2048 | 48.0 | 52.5 | 1.09x |
| 4096 | 65.3 | 128.2 | 1.96x |
| 8192 | 120.6 | 387.1 | 3.21x |

LatentServe crosses over at ~1.8K tokens (batch 4; ~1.2K at batch 8) and
is 3.2x faster by 8K. These are wall/steps and therefore blend prefill
into a per-step figure — quote the isolated numbers below instead.

### Isolated by differencing 128 vs 256 output tokens

Identical prompts mean identical prefill, so it cancels in the
difference and decode falls out as a slope, prefill as the intercept.
The only way to separate them for vLLM, whose V1 engine reports no TTFT.

| ctx | decode ms/step | | prefill tok/s | |
| ---: | ---: | ---: | ---: | ---: |
| | **LatentServe** | **vLLM** | **LatentServe** | **vLLM** |
| 2048 | 34.5 | **26.0** | **5535** | 2563 |
| 8192 | 57.5 | **42.9** | **5238** | 744 |

**vLLM's decode is faster; its prefill collapses.** Two separate
findings that the end-to-end numbers had blended into one.

### Decode: the gap is our own gather

| ctx | weights | attn read | gather | model | measured | vLLM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 28.1 | 2.3 | 4.5 | 34.9 | 34.5 | 26.0 |
| 8192 | 28.1 | 8.7 | 17.3 | 54.1 | 57.5 | 42.9 |

Weights at 3.09 GB and KV at 28,672 B/token, both over the ~110 GB/s
this card sustains; the gather moves 2x resident KV (Phase 3). The model
closes to within 3.4 ms. Drop the gather term and LatentServe lands at
~37 ms against vLLM's 42.9 — i.e. **the whole decode deficit is the
gather, and Phase 11's paged-attention kernel is worth ~17 ms/step at
8K/batch 4.** That is a target with a number attached.

An earlier draft of this document claimed the LatentServe-vs-vLLM gap
*was* an estimate of Phase 11's value. It is not: vLLM's decode kernel
already beats ours, so there is no decode deficit to recover from them.
The correct statement is the one above — our decode deficit is our own
gather, measured against a system that does not pay it.

### Prefill: the TRITON_ATTN fallback is quadratic-dominated

4x the context (2048 -> 8192) costs LatentServe 1.06x prefill throughput
and vLLM 3.44x. A cost that scales with context length is the signature
of the O(S^2) attention term dominating: LatentServe's prefill is still
GEMM-dominated at these lengths, vLLM's is not.

The mechanism is stated in vLLM's own startup log: `Cannot use FA version
2 ... FA2 is only supported on devices with compute capability >= 8`,
followed by `Using TRITON_ATTN attention backend`. A T4 is sm75, so vLLM
falls back to Triton, while LatentServe reaches PyTorch SDPA's
memory-efficient CUDA kernel. **This is a Turing-specific result and must
be labelled as one** — on an A100 vLLM would use FA2 and the prefill
comparison would likely invert.

### End to end, reconstructed

8 requests x 8192 ctx x 128 output tokens:

| | prefill | decode | total | measured |
| --- | ---: | ---: | ---: | ---: |
| LatentServe | 12.5 s | 14.7 s | 27.2 s | 27.2 s |
| vLLM | 88.1 s | 11.0 s | 99.1 s | 99.1 s |

Both reconstruct exactly. vLLM loses 77 s in prefill and wins 3.7 s back
in decode, and the crossover at ~1.8K needs no extra mechanism: short
contexts are decode-dominated (vLLM's advantage), long contexts are
prefill-dominated (ours). Neither system changes between those points —
only the mix of work does.

### Gate 6 verdict

The honest headline is **not** "LatentServe beats vLLM". It is:

> On a Turing GPU where FA2 is unavailable, vLLM's Triton attention
> fallback makes prefill 7x slower at 8K context, which dominates
> long-context serving and reverses an otherwise consistent decode
> advantage of 1.3x. LatentServe's remaining decode deficit is entirely
> its paged gather, worth ~17 ms/step at 8K.

## Gate 6 checklist — "can we fairly benchmark against vLLM?"

- [x] `pytest tests/test_phase5_harness.py` green
- [x] both systems run from the same prompt token ids and output lengths
- [ ] `--compare` produces no `SKIPPED — UnfairComparison` lines
- [x] vLLM version, chunked-prefill and prefix-caching settings recorded
      in every vLLM row
- [x] each gap attributed to a mechanism (gather, CUDA graphs, chunked
      prefill), not just reported
- [x] p99 ITL explicitly excluded from the comparison

*Reviewed at project close: ticked where this note or the runner shows the item done. Left open: that `--compare` produced no SKIPPED lines isn't recorded here.*
