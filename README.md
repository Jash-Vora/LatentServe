# LatentServe

An LLM inference engine built from scratch to learn how serving works. It runs Qwen2.5-1.5B-Instruct on NVIDIA T4s, and everything was benchmarked against vLLM on the same hardware.

On one T4 it decodes 1.05–3.3× faster than vLLM, prefills up to 9.6× faster, and gives the same outputs.

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

Both engines got the same prompts, concurrency limits and seeds, and vLLM was in its fastest setup on this GPU. Most of the gap is attention. A T4 can't run FlashAttention-2, so vLLM falls back to a Triton kernel. At batch 1 with a short context a step is mostly reading weights, so the two tie. Add batch size or context and the gap opens up.

![Decode speedup over vLLM by batch size and context](docs/figures/decode_speedup.png)

![First-token time by prompt length](docs/figures/prefill_ttft.png)

![Latency under load](docs/figures/latency_vs_load.png)

## Two GPUs

| Two T4s | LatentServe | vLLM |
| --- | ---: | ---: |
| Burst throughput, one GPU | 276 tok/s | 99 tok/s |
| Burst throughput, two replicas | 472 tok/s (1.71×) | 173 tok/s (1.76×) |
| Chat throughput, round-robin routing | 702 tok/s | 485 tok/s |
| Chat throughput, least-loaded routing | 759 tok/s (+8%) | 582 tok/s (+20%) |
| Per-token time at batch 1, model split across both GPUs | not built | 9.7 ms against 16.8 ms on one (1.73×) |

Two replicas means one engine per GPU behind a router. The other way to use two GPUs is to split a single model across them (tensor parallelism), which only vLLM does here. Replicas scaled to 85% of ideal for LatentServe and 88% for vLLM. One GPU sits idle at the end of a run because the router balances request counts, not work.

## What worked

- The decode kernel did most of the work. The first version, written in Triton, only reached 69 GB/s on a card that can do about 320, because its math was running on the slow scalar path. The CUDA rewrite reaches 194 GB/s and is 2.8× faster, which took decode at batch 16 / 8K from 79 ms to 34 ms.
- CUDA graphs removed the launch overhead. A batch-1 step went from about 40 ms to 20–28 ms.
- Prefix caching cut first-token time by 72–84% on shared prompts and chat. With nothing to share it costs about 1%.
- Choosing dense or sparse attention for each step (the adaptive policy) came within 2% of always-sparse speed, at about half the expected answer loss.

## What didn't

- Sparse attention below 50% of pages. At 50% it's 1.40× faster at batch 8 / 32K and loses about 0.4% of answers. At 37.5% it's 1.66× and 1.2%. At 25% it loses 5.5% and fails the quality check. A perfect page picker loses nothing at 25%, so the trouble is choosing pages, not sparsity itself. A learned indexer would probably fix that, but it wasn't tried.
- The INT8 KV cache. It holds 1.86× as many blocks but is never faster: level with dense at best, up to 12% slower at worst. Doubling the batch with INT8 buys 10–16% more throughput than fp16 dense, while sparse attention buys 62–66%.
- The MLA latent cache, dropped after Phase 7. A latent that keeps value error at 10% only compresses about 1.6×, and INT8 gets 2× for free.

![Decode throughput: INT8 capacity vs sparse attention](docs/figures/capacity_vs_sparsity.png)

## Lessons

- Small test models hid a prefill bug for four phases (NaNs, and prefill twice as slow). A guard now refuses to time a model whose outputs aren't finite.
- Running at real scale found engine bugs the tests never hit. Admission ignored generated tokens, a sparse budget decayed inside graph buckets, and one-token requests crashed the engine.

## To do

- [ ] Rerun the vLLM comparison on an A10 or L4. Part of the gap is the T4 (no FlashAttention-2), and how much isn't known.
- [ ] Try a larger model. The 1.5B one can't do multi-hop retrieval even with full attention, so sparse attention's effect on that is untested.
- [ ] Train a learned page indexer so sparse attention can go below 50%.
- [ ] Add tensor parallelism to LatentServe. vLLM's gets 1.7× lower per-token latency on two T4s.
- [ ] Route replicas by outstanding tokens instead of request counts.
- [ ] Repeat the load curves. Each point is a single run of 16–80 requests, so individual points are noisy.
- [ ] Benchmark with real prompts and varied output lengths. Everything here uses random tokens and fixed lengths.
- [ ] Measure sparse attention's accuracy cost on more questions. The figures come from about 85 per seed, so they're good to roughly ±1 point.
- [ ] Support sampling beyond greedy decoding, and put an API server in front.

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
python -m benchmarks.make_figures                   # redraw docs/figures from the saved tables
```

Other experiments are in `benchmarks/runners/`, one per phase (`phase13_prefix`, `phase15_quality`, `phase17_*`, `phase18_*`). The T4 is compute capability 7.5, so everything runs fp16.

## Layout

```
model/        Qwen wrapper, decoder loop, GQA and sparse attention
cache/        paged KV cache, block allocator, INT8 and prefix caches
kernels/      Triton and CUDA decode kernels (CUDA via NVRTC and CuPy)
runtime/      engine, schedulers, CUDA-graph decoder, sparsity policy, router
benchmarks/   harness, one runner per phase, and the figure script
docs/         one note per phase, each opening with its outcome
```

The final sweep's tables are in `docs/sweep_stage1_results.md`; the original plan is `docs/methodology.md`.
