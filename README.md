# LatentServe

A from-scratch LLM inference engine, built to learn how serving works and measured against vLLM on the same hardware. It serves one model, Qwen2.5-1.5B-Instruct, on commodity NVIDIA T4s, with a paged KV cache, continuous batching, hand-written CUDA kernels, CUDA graphs, prefix caching and multi-GPU routing.

On one T4 it decodes 1.05–3.3× faster than vLLM, prefills up to 9.6× faster, and produces matching outputs.

## Faster than vLLM on a T4, and why that's partly the T4

Both engines got the same prompts, concurrency limits and seeds, were warmed up before timing, and agree with an fp32 reference to about 1.7×10⁻⁵ KL. vLLM was probed across every attention backend and setting it supports on this GPU; its default turned out to be its fastest.

| What was measured | LatentServe | vLLM | Gap |
| --- | ---: | ---: | ---: |
| Decode step, batch 1, 2K context | 16.7 ms | 17.5 ms | 1.05× |
| Decode step, batch 16, 16K context | 63.3 ms | 190.6 ms | 3.0× |
| Decode step, batch 8, 32K context | 62.2 ms | 198.0 ms | 3.2× |
| First token, 1K-token prompt | 131 ms | 271 ms | 2.1× |
| First token, 32K-token prompt | 16.9 s | 161.8 s | 9.6× |
| Burst throughput | 256 tok/s | 90 tok/s | 2.8× |
| Multi-turn chat throughput, prefix caching on | 629 tok/s | 415 tok/s | 1.5× |
| Saturation capacity, open-loop requests | 1.58 req/s | 0.50 req/s | 3.2× |

The pattern explains itself. At batch 1 a decode step is mostly reading the model's weights, which both engines do equally well, so they tie. The gap opens as attention grows, with more sequences and longer contexts, and in prefill, which at long lengths is mostly attention.

That points at the attention kernel. A T4 can't run FlashAttention-2 or FlashInfer, so vLLM falls back to a Triton attention backend, while LatentServe's CUDA kernel is written for this chip. vLLM's kernel wasn't profiled, so this is the likely cause rather than a measured one. On mixed traffic with a burst of long prompts the gap compounds: vLLM's median first-token time was 12 minutes against LatentServe's 59 seconds.

## Where the speed comes from

**The decode kernel.** The first attention kernel, written in Triton, ran at 47–69 GB/s on a GPU that can do about 320. A stripped-down version that only loaded data reached 243 GB/s, which showed memory wasn't the limit: the math was running on the slow scalar path, with no tensor-core instructions and one program per SM. Rewriting it in CUDA made it 2.8× faster (1.93 ms to 0.69 ms) and cut the whole decode step at batch 16 / 8K from 79 ms to 34 ms.

**CUDA graphs.** Each decode step is captured once per batch-size and context bucket and replayed, which removes the launch overhead that otherwise leaves the GPU idle between kernels at small batches.

**A prefill bug that hid for four phases.** Prefill took a slow path that overflowed in fp16 on Qwen's real activations and produced NaNs. Every test used small models where nothing overflows. Fixing the routing made prefill about 2× faster, and a guard now refuses to time a model whose outputs aren't finite.

## Ideas that paid off

- **Prefix caching.** Shared prompt prefixes are computed once. First-token time fell 72–84% across a shared system prompt and multi-turn chat, and live cache memory fell 22–37%. With nothing to share it costs about 1%. It was small to build because the block allocator had been reference-counted since the paged cache, and it works unchanged with the INT8 cache.
- **An adaptive policy for sparse attention.** Sparse attention costs a little accuracy everywhere and saves time only where attention dominates. A policy that picks dense or sparse for each decoding step, from a table of measured step times, came within 2% of fixed sparse attention's speed at about half its expected answer loss (0.65% against 1.2%).
- **Two ways to use two GPUs.** Two independent replicas behind a router raised throughput 1.7–1.8×, about 85–88% of ideal; one GPU idled at the end because the router balanced request counts, not work. Splitting vLLM's model across both T4s cut per-token latency 1.7× at batch 1, because the T4s have peer-to-peer access and an all-reduce takes about 50 µs. Replicas suit throughput and splitting suits latency. Sending each request to the least-loaded GPU beat round-robin by 8% (LatentServe) to 20% (vLLM) on chat.

## Ideas that cost more than they gave

- **Sparse attention below 50% of pages.** Dense decoding reads every cached key and value; sparse reads only the pages most likely to matter. At 50% of pages it made decoding 1.40× faster at batch 8 / 32K and lost about 0.4% of answers. At 37.5% it was 1.66× faster and lost about 1.2%. All the losses were in question answering, none in exact-fact retrieval. At 25% it lost 5.5% of answers and failed the quality check on every seed, and no budget passed the confirmatory run. Dense stays the default; 50% and 37.5% are opt-in with their costs stated, and 25% and below aren't recommended.
- **Why sparsity fell short.** A perfect page chooser lost nothing at 25% up to 16K tokens, so sparsity isn't the problem; picking the pages is. The training-free indexer, which scores each page by its min and max keys, is too crude. Scoring by estimated attention mass cut the error 23% and still failed. Re-checking candidates exactly reached near-perfect quality but read as much data as the 50% budget, so it gained nothing. A learned indexer, as DeepSeek trains, would likely close the gap and was out of scope.
- **INT8 caching.** It holds 1.86× as many cache blocks in the same memory, enough to run batch 32 at 16K and batch 16 at 32K, which fp16 can't fit. It is never faster: where both fit it is level with dense or up to 12% slower, and its worst-case latency is worse (p99 of 25 ms against 17 ms at batch 1) because one step in sixteen pays to quantize a finished block. At 16K and 32K, doubling the batch with INT8 buys 10–16% more decode throughput than fp16 dense, against 62–66% for sparse attention. Its attention kernel is latency-bound, so halving the bytes it reads didn't shorten the step. Fusing its write path into one kernel removed about 300 launches per step and cut its penalty from 6–19% to 1–9%, which makes it a capacity option, not a speed option.
- **MLA.** The track meant to shrink the cache with a compressed latent representation (Phases 7–10) was closed after the compressibility study in `docs/phase7.md`, and INT8 took its place. The study found that the 2.24× compression an energy threshold suggested was an illusion: values carry 3–5× the reconstruction error of keys, so a latent at 10% value error compresses only about 1.6×, while INT8 gives exactly 2× with no reconstruction compute. The final go/no-go wasn't written down at the time, but the numbers point one way.

## Measuring it honestly

Most of the project's time went into making measurements trustworthy. The phase notes record their predictions before the results, including the ones that turned out wrong: "25% sparse is on par with dense" (it wasn't), "adaptive will be faster than fixed" (it was 1–2% slower and cheaper in quality), and scaling efficiency of 0.95 (it was 0.85–0.88).

Setting criteria in advance overturned two of our own conclusions. "25% sparse is on par" rested on nine needles and fell to an 83-case study. "The answer flips are just noise" fell to a control showing that a numerically different dense kernel flips none.

Real-scale runs exposed bugs that small tests couldn't:

- Admission checked only the prompt and ran out of cache blocks mid-decode.
- A sparse budget fixed at CUDA-graph capture decayed to half its ratio as sequences grew.
- The engine crashed when a request finished during prefill, which happens with a one-token request or an immediate end-of-sequence.
- The final sweep's problems were assumptions about vLLM and the GPU: vLLM enforces the model's 32,768-token limit and admits large batches gradually, INT8's memory pool was sized like fp16's, and a pool's size depended on the engine built before it.

The cheapest fix for most of these is a 15-minute smoke run on the real GPU, covering the largest shapes on both engines, before any multi-hour run.

## What the numbers don't say

Within the conditions tested, the LatentServe-versus-vLLM comparison is like for like. It is not a claim about vLLM in general.

- **The gap is partly specific to the T4.** This GPU can't run FlashAttention-2 or FlashInfer, so vLLM runs a fallback. On a newer GPU the gap should shrink; that was not measured.
- **LatentServe is specialised.** It serves one model family with greedy decoding and has no API server or sampling options. That narrowness is what allowed a model-specific kernel.
- **The setup was narrow.** One model, one GPU type, random-token prompts, fixed output lengths and closed-loop chat clients.
- **The sparse-attention quality costs are rough.** They come from about 85 questions per seed over four seeds, so treat 0.4% and 1.2% as plus or minus a point.
- **The load curves are single runs** of 16–80 requests per point. The overall shape is reliable and individual points are noisy.
- **Multi-hop retrieval went untested.** The 1.5B model scored 0 of 12 on it even with dense attention, so it couldn't show what sparsity does to it.

## Where it could go next

1. Rerun the vLLM comparison on an A10 or L4 to separate real engineering gains from the T4 effect.
2. Train a small page indexer against dense attention's own choices, to see whether 25% sparsity can pass. It matters most at 64K tokens and beyond.
3. Try a larger model, so multi-hop retrieval can finally be measured.
4. Build tensor parallelism into LatentServe itself. vLLM's version shows 1.7× lower per-token latency is available on two T4s.
5. Route replicas by outstanding tokens instead of request counts, to close the gap between 85–88% scaling and ideal.

## Using it

A minimal serving loop, with the production settings the benchmarks use:

```python
from kernels.gqa.paged_decode import set_decode_backend
from model.latentserve_qwen import LatentServeQwen
from model.qwen import QwenReference
from runtime.engine import ServingEngine
from runtime.request import ServedRequest

set_decode_backend("cuda")                       # the CUDA-core decode kernel
ref = QwenReference(model_name="Qwen/Qwen2.5-1.5B-Instruct", dtype="fp16",
                    device="cuda:0").load()
model = LatentServeQwen.from_reference(ref, max_seq_len_hint=8192,
                                       attn_impl="triton_paged", fuse_projections=True)
model.set_elementwise(True)

engine = ServingEngine(model, max_running=16, max_seq_len=8192, block_size=16,
                       use_cuda_graphs=True, prefix_caching=True)
engine.add_request(ServedRequest(request_id=0, prompt_ids=token_ids, max_new_tokens=128))
finished = engine.run()                          # completed requests, with per-request timings
```

Dense fp16 is the default. The other options are opt-in:

| Option | How | What it costs |
| --- | --- | --- |
| Prefix caching | `ServingEngine(..., prefix_caching=True)` | about 1% when nothing is shared |
| Sparse attention | `model.set_sparse(0.5)` or `0.375`, before building the engine | about 0.4% / 1.2% of answers; fp16 cache only |
| Adaptive sparsity | `ServingEngine(..., policy=SparsePolicy.from_json(table, tier="relaxed"))` from `runtime.policy`; build the table with `phase17_calibrate` | the "relaxed" tier allows 50% and 37.5% |
| INT8 KV cache | `ServingEngine(..., kv_dtype="int8")` | up to 12% slower, 1.86× the capacity |

## Reproducing the results

```bash
pip install -r requirements.txt        # on Kaggle, torch is preinstalled
pip install cupy-cuda13x               # CUDA kernels compile at runtime with NVRTC; match your CUDA version
pip install vllm                       # only for the vLLM comparisons
export PYTHONPATH=$(pwd)
python -m benchmarks.runners.check_env # verifies the GPU and library versions first
python -m pytest tests/ -q             # 485 pass on a CPU-only machine; the 104 GPU tests skip
```

T4s are compute capability 7.5, so everything runs in fp16, not bf16.

| What | Command | Time and hardware |
| --- | --- | --- |
| Everything against vLLM: decode, prefill, serving, load curves | `python -m benchmarks.runners.sweep_stage1`, then `--report` | several hours, one T4, resumable |
| Prefix caching | `python -m benchmarks.runners.phase13_prefix` | about 25 min |
| Why the first decode kernel was slow | `python -m benchmarks.runners.phase12_diag` | one T4 |
| Sparse attention quality study | `python -m benchmarks.runners.phase15_quality --task needle`, then `multikey`, `qa`, `text`, `gen`, `latency`, and `--task curves` | about 70 min |
| Adaptive runtime | `phase17_calibrate`, then `phase17_workload` | about 15 min and 40 min |
| INT8 kernel count per step | `python -m benchmarks.runners.phase16_kernel_census` | about 2 min |
| Two GPUs | `phase18_allreduce`, `phase18_vllm_tp`, `phase18_replicas` (add `--backend vllm`) | needs two T4s; about 2 min, 6 min, 1 hour |

Experiments are defined in YAML under `configs/` and validated by `config.py`. Every result is a JSON line under `results/raw/`, stamped with the git commit, library versions and GPU, so no number is ever typed in by hand.

## Repository map

```
model/             Qwen reference wrapper, LatentServe's own decoder loop, GQA and sparse attention
cache/             refcounted block allocator, paged KV cache, INT8 cache, prefix cache
kernels/cuda/      fp16, INT8 and sparse decode kernels and the fused INT8 write (NVRTC via CuPy)
kernels/gqa/       Triton decode kernels and backend dispatch
runtime/           serving engine, schedulers, CUDA-graph decoder, adaptive sparsity policy, replica router
benchmarks/        result schema and harness, one runner per phase, the final sweep
comparisons/vllm/  vLLM engine construction with every control recorded
compression/       KV compressibility studies
configs/           YAML experiment definitions
docs/              one note per phase
tests/             485 tests; GPU-only tests skip on CPU
```

## Notes by phase

| Note | What it covers |
| --- | --- |
| `docs/methodology.md`, `docs/architecture.md` | the original research plan, gates and fairness rules (the phase notes win where they differ); how the pieces fit |
| `docs/phase2.md` to `docs/phase6.md` | GQA and KV cache, paged cache, serving runtime, benchmark harness, the vLLM baseline |
| `docs/phase7.md` | KV compressibility study |
| `docs/phase12_kernel_findings.md` | why the first decode kernel was slow |
| `docs/phase13.md`, `docs/phase13_prefix.md` | CUDA graph decode; prefix caching |
| `docs/phase14.md`, `docs/phase14_sparse.md` | closing the batch-1 gap and INT8 at graph speed; sparse decode attention |
| `docs/phase15_quality.md` | the sparse-attention quality study, including the claims it overturned |
| `docs/phase16_int8_sparse.md` | the fused INT8 write, and why INT8 with sparse stopped at its gate |
| `docs/phase17_adaptive.md`, `docs/phase18_multigpu.md` | the adaptive policy; replicas, routing and tensor parallelism |
| `docs/benchmark_vs_vllm.md`, `docs/sweep_stage1.md`, `docs/sweep_stage1_results.md` | the earlier vLLM comparison; the final single-GPU sweep's setup and the bugs found along the way; its tables |
| `docs/kernels.md`, `docs/profiling.md`, `docs/mla.md`, `docs/dsa.md` | short pointers to where each topic is covered |
