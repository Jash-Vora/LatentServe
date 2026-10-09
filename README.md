# LatentServe

A from-scratch LLM inference engine for Qwen2.5-1.5B-Instruct on NVIDIA T4s, built to learn how serving works and measured against vLLM on the same hardware.

On one T4 it decodes 1.05–3.3× faster than vLLM, prefills up to 9.6× faster, and matches its outputs.

## Against vLLM

| | LatentServe | vLLM | Gap |
| --- | ---: | ---: | ---: |
| Decode step, batch 1, 2K context | 16.7 ms | 17.5 ms | 1.05× |
| Decode step, batch 16, 16K context | 63.3 ms | 190.6 ms | 3.0× |
| Decode step, batch 8, 32K context | 62.2 ms | 198.0 ms | 3.2× |
| First token, 1K-token prompt | 131 ms | 271 ms | 2.1× |
| First token, 32K-token prompt | 16.9 s | 161.8 s | 9.6× |
| Burst throughput | 256 tok/s | 90 tok/s | 2.8× |
| Multi-turn chat, prefix caching on | 629 tok/s | 415 tok/s | 1.5× |
| Saturation, open-loop requests | 1.58 req/s | 0.50 req/s | 3.2× |

Same prompts, limits and seeds; vLLM in its best configuration for this GPU. The gap comes from attention: a T4 can't run FlashAttention-2, so vLLM falls back to Triton. At batch 1 the step is mostly weight reads, so the two tie.

## What worked

- **Decode kernel.** A hand-written CUDA kernel replaced a Triton one that was arithmetic-bound: 2.8× faster, and decode at batch 16 / 8K went from 79 ms to 34 ms.
- **CUDA graphs.** Removed launch overhead: batch-1 steps fell from about 40 ms to 20–28 ms.
- **Prefix caching.** First-token time down 72–84% on shared prompts and chat, about 1% overhead when nothing is shared.
- **Adaptive sparsity.** Choosing dense or sparse per step came within 2% of fixed sparse speed at about half the expected answer loss.
- **Two GPUs.** Replicas raised throughput 1.7–1.8×. Splitting vLLM's model across both T4s cut batch-1 latency 1.7×.

## What didn't

- **Sparse attention below 50% of pages.** 50% was 1.40× faster at batch 8 / 32K and lost about 0.4% of answers; 37.5% was 1.66× and 1.2%; 25% lost 5.5% and failed. A perfect page chooser loses nothing at 25%, so page selection is the limit. A learned indexer would likely fix it; it was out of scope.
- **INT8 KV cache.** Holds 1.86× the blocks, but is never faster: level with dense to 12% slower. Doubling the batch with INT8 buys 10–16% over fp16 dense, while sparse attention buys 62–66%.
- **MLA latent cache.** Closed after Phase 7: a latent at 10% value error compresses only about 1.6×, while INT8 gives 2× for free.

## Caveats

- The gap is partly the T4. A newer GPU, where vLLM gets FlashAttention, wasn't tested.
- LatentServe is specialised: one model, greedy decoding, no API server.
- One model, random-token prompts, fixed output lengths. Sparse quality costs are about ±1 point; load curves are single runs.

## Lessons

- Tiny test models hid a prefill bug for four phases (NaNs, 2× slower). A guard now refuses to time non-finite outputs.
- Real-scale runs found engine bugs: admission ignoring generated tokens, a sparse budget decaying inside graph buckets, a crash on one-token requests.

## Next

Rerun on an A10 or L4. Train a learned page indexer. Try a larger model. Build tensor parallelism into LatentServe. Route replicas by tokens, not request counts.

## Using it

```python
from kernels.gqa.paged_decode import set_decode_backend
from model.latentserve_qwen import LatentServeQwen
from model.qwen import QwenReference
from runtime.engine import ServingEngine
from runtime.request import ServedRequest

set_decode_backend("cuda")
ref = QwenReference(model_name="Qwen/Qwen2.5-1.5B-Instruct", dtype="fp16", device="cuda:0").load()
model = LatentServeQwen.from_reference(ref, max_seq_len_hint=8192,
                                       attn_impl="triton_paged", fuse_projections=True)
model.set_elementwise(True)

engine = ServingEngine(model, max_running=16, max_seq_len=8192, block_size=16,
                       use_cuda_graphs=True, prefix_caching=True)
engine.add_request(ServedRequest(request_id=0, prompt_ids=token_ids, max_new_tokens=128))
finished = engine.run()
```

Dense fp16 is the default. Opt-in: `kv_dtype="int8"` on the engine, `model.set_sparse(0.5)` before building it, or `policy=SparsePolicy.from_json(table, tier="relaxed")` from `runtime.policy`.

## Run it

```bash
pip install -r requirements.txt cupy-cuda13x vllm   # vllm only for the comparisons
export PYTHONPATH=$(pwd)
python -m benchmarks.runners.check_env
python -m pytest tests/ -q                          # GPU tests skip on CPU
python -m benchmarks.runners.sweep_stage1           # the full vLLM comparison; --report for tables
```

Other experiments are in `benchmarks/runners/`, one per phase (`phase13_prefix`, `phase15_quality`, `phase17_*`, `phase18_*`). The T4 is compute capability 7.5, so everything runs fp16.

## Layout

```
model/        Qwen wrapper, decoder loop, GQA and sparse attention
cache/        paged KV cache, block allocator, INT8 and prefix caches
kernels/      Triton and CUDA decode kernels (CUDA via NVRTC and CuPy)
runtime/      engine, schedulers, CUDA-graph decoder, sparsity policy, router
benchmarks/   harness and one runner per phase
docs/         one note per phase, each opening with its outcome
```

The final sweep's tables are in `docs/sweep_stage1_results.md`; the original plan is `docs/methodology.md`.
