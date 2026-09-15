# Methodology

This is the full LatentServe research plan, landed here now that Phase 1
is underway (see the Phase 0 note this replaced). It is the living
reference for phases, gates, the benchmark matrix, fairness rules for
the vLLM comparison, and ablations — update it as phases land, don't
let it drift into a stale snapshot.

**Status:** Phase 0 (environment + experimental infrastructure) and
Phase 1 (Qwen reference + correctness baseline) are implemented — see
`model/qwen.py`, `benchmarks/runners/phase1_reference.py`,
`tests/test_phase1_correctness.py`, and `configs/phase1_reference.yaml`.
Phase 2 (GQA + KV cache) is next.

Key things every experiment must respect (see Section 28, "Fairness
Rules for vLLM", and Section 12, "Phase 5 — Benchmark Harness"):
- control model weights, tokenizer, precision, GPU, input/output token
  counts, batch/concurrency, sampling params, context length across
  every comparison
- exclude warm-up runs from steady-state latency numbers
- multiple repetitions; report median, p50, p90, p95, p99, and CIs
  where practical
- every result comes from `benchmarks/schema.py::ResultWriter` — never
  hand-typed into this doc or a notebook

---

# LatentServe

## Memory-Efficient and Sparse Long-Context LLM Inference on Commodity GPUs

**Project type:** GPU systems / LLM inference / research engineering
**Primary hardware:** 1x-2x NVIDIA T4, 16 GB VRAM each
**Primary tooling:** Python, PyTorch, Triton, CUDA C++, Nsight Systems, Nsight Compute
**Model:** Qwen2.5-1.5B-Instruct
**Industry baseline:** vLLM
**Research inspiration:** GQA, PagedAttention, DeepSeek MLA, DeepSeek Sparse Attention (DSA)

---

## 1. Executive Summary

LatentServe is a research-oriented LLM inference and serving runtime designed to investigate how modern attention, KV-cache, memory-management, sparse-attention, and GPU-kernel techniques affect long-context inference on constrained GPU hardware.

Rather than implementing a Transformer from scratch, LatentServe uses **Qwen2.5-1.5B-Instruct as a fixed real-world model substrate**.

The project progressively replaces and optimizes its inference path:

1. Qwen2.5-1.5B-Instruct reference implementation
2. GQA instrumentation and inference baseline
3. KV caching
4. Paged KV caching
5. Continuous batching
6. Serving scheduler
7. vLLM comparison
8. MLA-inspired latent KV compression
9. Decoupled positional representation
10. MLA matrix absorption
11. MLA compression studies
12. Custom Triton/CUDA kernels
13. Nsight profiling and optimization
14. Prefix caching
15. DSA-inspired token selection
16. Sparse attention
17. Sparse GPU kernels
18. MLA + DSA
19. Adaptive attention/runtime policies
20. Single-GPU and two-GPU serving

The central objective is not simply to build a faster inference server.

The objective is to answer:

> **How do KV-cache representation, attention sparsity, GPU kernels, memory management, scheduling, and workload characteristics interact to determine real-world LLM inference performance?**

The project explicitly separates:

* theoretical complexity
* memory savings
* compute reduction
* kernel-level performance
* end-to-end serving performance
* model-quality tradeoffs

---

## 2. Core Research Question

### Primary question

> **Can latent KV compression and token-level sparse attention jointly improve long-context LLM inference on commodity GPUs, and what hardware/runtime bottlenecks determine whether the theoretical gains translate into real performance?**

### Secondary questions

#### Model / Attention

* What is the actual attention and KV-cache structure of Qwen2.5-1.5B-Instruct?
* How does GQA affect memory and decode performance?
* How does latent KV compression change the memory/compute tradeoff?
* How does latent dimension affect quality and latency?

#### MLA

* How much KV-cache memory can latent compression eliminate?
* Does reduced KV memory translate into lower decode latency?
* When does latent attention become compute-bound?
* How much does matrix absorption matter?
* What is the interaction between latent dimension and GPU utilization?

#### DSA

* How much attention computation can token selection eliminate?
* What is the cost of the indexer?
* What sparsity level preserves useful context?
* At what context length does sparse attention become worthwhile?
* Does irregular memory access eliminate theoretical speedups?
* How does sparse attention behave on T4 GPUs?

#### MLA + DSA

* Are memory and compute reductions complementary?
* Does MLA make DSA more or less useful?
* Does DSA shift the bottleneck from compute to memory?
* Does MLA + DSA produce an actual end-to-end improvement?

#### Runtime

* How important is continuous batching?
* How much does prefix caching improve TTFT?
* How should requests with different context lengths be scheduled?
* Can an adaptive runtime choose between dense GQA, MLA, and MLA + DSA?

#### Systems

* Where does vLLM outperform the custom runtime?
* Where can a specialized runtime approach vLLM?
* Which theoretical optimizations survive contact with actual GPU hardware?

---

## 3. Project Philosophy

The central principle:

> **Don't merely report that an optimization is faster. Explain why.**

Every major performance claim should have three layers of evidence.

### Layer 1 - Mathematical

What does the optimization theoretically reduce?

### Layer 2 - Benchmark

Does the measured latency, memory, or throughput improve?

### Layer 3 - Profiling

What actually happened inside the GPU?

For example: "MLA reduces KV-cache memory" is useful, but the desired result is closer to: "MLA reduced KV-cache memory by X%, but decode latency only improved by Y% because latent projection introduced additional memory operations. Nsight showed the kernel was memory-bound at Z% of achievable bandwidth. Kernel fusion reduced memory traffic and produced an additional improvement."

**Measured results should always replace hypothetical numbers.**

---

## 4. Non-Goals

LatentServe is **not** intended to become a complete replacement for vLLM.

Explicitly avoid:

* Kubernetes
* authentication
* multi-node serving
* cloud orchestration
* production observability
* dozens of model architectures
* dozens of quantization formats
* 70B+ models
* complete OpenAI-compatible API coverage
* production-grade fault tolerance
* frontend/dashboard work
* generic web infrastructure

The focus is:

> **Deep inference engineering rather than broad production infrastructure.**

---

## 5. Hardware Strategy

### Primary hardware

```
1 x NVIDIA T4
2 x NVIDIA T4
```

Each T4: 16 GB VRAM. The primary environment is Kaggle.

### Why T4?

The T4 provides meaningful constraints: limited VRAM, limited memory bandwidth relative to newer GPUs, older GPU architecture, PCIe-based multi-GPU communication. These constraints make memory pressure, cache behavior, kernel efficiency, and communication overhead visible.

The project therefore asks:

> **How much inference efficiency can we extract from constrained hardware?**

---

## 6. Model Strategy

### Fixed model

**Qwen2.5-1.5B-Instruct**

The model weights and tokenizer remain fixed across the primary experiments. The experimental variable is the **execution system**:

```
Qwen2.5-1.5B-Instruct
        |
        v
Hugging Face Reference
        |
        v
  LatentServe GQA
        |
        v
Paged KV + Batching
        |
        v
      MLA
        |
        v
Matrix Absorption
        |
        v
Custom GPU Kernels
        |
        v
  DSA Sparse Path
        |
        v
    MLA + DSA
        |
        v
 Adaptive Runtime
```

> **Keep the model fixed. Change the inference mechanism. Measure the consequences.**

### Important methodological distinction

Qwen2.5-1.5B-Instruct should **not** be described as natively implementing DeepSeek MLA or DSA. The project instead implements:

> **MLA-inspired latent KV attention and DSA-inspired sparse attention as experimental inference variants on the Qwen2.5-1.5B-Instruct substrate.**

That distinction should be maintained throughout the documentation.

---

## 7. Phase 0 - Environment and Experimental Infrastructure [DONE]

**Goal:** create reproducible infrastructure before implementing optimizations.

**Deliverables:** Git repository, Python package, CUDA environment, PyTorch, Triton, CUDA C++, Nsight Systems, Nsight Compute, Kaggle environment, configuration system, benchmark harness, result schema, experiment logging.

Implemented as `config.py` (pydantic `ExperimentConfig`), `configs/*.yaml`, `benchmarks/schema.py` (`BenchmarkResult` + `ResultWriter`), `benchmarks/runners/check_env.py`, `tests/test_phase0_infra.py`.

Experiments are configuration-driven - no manually edited benchmark scripts for individual experiments.

---

## 8. Phase 1 - Qwen Reference + Correctness Baseline [DONE]

**Goal:** establish Hugging Face Qwen2.5-1.5B-Instruct as the correctness oracle.

Implemented as `model/qwen.py` (`QwenReference`, `ModelShape`, `kv_cache_bytes`), `benchmarks/runners/phase1_reference.py`, `tests/test_phase1_correctness.py`, `configs/phase1_reference.yaml`.

Measured: model load time, prefill latency, decode latency, TTFT, TPOT, E2E latency, throughput, peak VRAM, KV-cache memory.

### Correctness

Compared: Hugging Face (unmodified) vs. the instrumented wrapper's incremental (cached) execution path. At this phase "LatentServe" *is* the Hugging Face model - the wrapper adds timing/memory instrumentation and a prefill/decode-step API but no custom attention or KV-cache layout yet (that starts Phase 2), so the meaningful check is internal: does token-by-token cached decode agree with a full-sequence teacher-forced forward pass, which exercises attention masking, RoPE position ids, and KV-cache updates together.

Verified: logits (teacher-forced vs. incremental), attention masking + RoPE (indirectly, via that same agreement), KV-cache updates, multi-token prefill, single-token decode, deterministic generation, KV-cache memory (measured vs. theoretical).

Tested token counts: 1, 16, 128, 1K (fast/default subset) and 4K, 8K, 16K+ (gated behind `LATENTSERVE_LONG_CONTEXT_TESTS=1`, meant for the T4 target rather than every local test run).

**Deliverable:** a reproducible reference benchmark for Qwen2.5-1.5B-Instruct (`results/raw/phase1_reference.jsonl`).

---

## 9. Phase 2 - GQA + KV Cache

**Goal:** build the conventional modern inference baseline around Qwen.

Implement/instrument:

### GQA

Understand the actual Qwen query/KV head configuration (already introspected in Phase 1 via `QwenReference.shape` - `num_attention_heads`, `num_key_value_heads`, `gqa_group_size`).

### KV cache

Represent:

```
K[layer][sequence][kv_heads][head_dim]
V[layer][sequence][kv_heads][head_dim]
```

Measure: KV bytes/token, VRAM growth, memory bandwidth, decode latency.

### Experiments

Vary: GQA configuration, batch size, context length. Determine: memory scaling, decode scaling, bandwidth pressure.

This becomes the baseline for MLA.

---

## 10. Phase 3 - Paged KV Cache

**Goal:** implement a simplified PagedAttention-style memory manager.

Logical sequence `Token 0 ... Token N` maps onto physical blocks (e.g. Block 17, Block 3, Block 41, Block 8, ...).

### Components

* **Block allocator** - allocation, freeing, reuse
* **Block table** - maps logical positions to physical blocks
* **Cache manager** - tracks free blocks, used blocks, sequence blocks, reference counts

### Experiments

Compare contiguous KV vs. paged KV under variable sequence lengths, concurrent requests, request termination, high utilization. Measure: fragmentation, usable cache capacity, allocation overhead, latency impact.

---

## 11. Phase 4 - Serving Runtime

**Goal:** turn the inference backend into a minimal serving engine.

Request lifecycle: `ARRIVED -> QUEUED -> PREFILL -> DECODING -> FINISHED`.

### Continuous batching

New requests can join while existing requests decode:

```
Step 1: A B C D
Step 2: A B   D
Step 3: A B
Step 4: A
```

### Scheduler

Implement progressively: FIFO (baseline), length-aware (reduce wasted work), fair (prevent starvation), SLO-aware (prioritize requests approaching latency targets). Measure scheduler overhead explicitly.

---

## 12. Phase 5 - Benchmark Harness

Before introducing MLA, make measurement trustworthy.

### Metrics

* **Latency:** TTFT, TPOT, E2E latency
* **Throughput:** tokens/sec, requests/sec
* **Memory:** peak VRAM, KV-cache memory, cache utilization, fragmentation
* **Runtime:** scheduler overhead, CPU overhead, GPU utilization

### Percentiles

Always report p50, p90, p95, p99 - mean latency alone is insufficient.

### Repetition

Use warm-up runs and repeated trials. Record median, p95, p99, confidence intervals where practical.

---

## 13. Phase 6 - vLLM Baseline

Compare Hugging Face vs. LatentServe vs. vLLM, using the same model, weights, tokenizer, precision, GPU, prompt distribution, input length, output length, concurrency, sampling parameters.

Measure: TTFT, TPOT, throughput, VRAM, concurrency scaling, p95/p99.

### Important rule

Do **not** make "beat vLLM" the objective. Instead:

> **Investigate where a specialized runtime can approach or exceed a mature serving system and where it cannot.**

If vLLM wins, explain why. That is a successful experiment.

---

## 14. Phase 7 - MLA-Inspired Latent KV Attention

The first major research component. Implement a latent KV representation:

```
c_t^KV = W^DKV h_t
```

Instead of storing full K/V for every token, investigate storing `c_t^KV`.

```
hidden state
     |
     +---------------> query path
     |
     v
KV compression
     |
     v
latent KV
     |
     +---------------> key path
     |
     +---------------> value path
```

Compare full KV vs. latent KV. Measure: bytes/token, total VRAM, memory bandwidth, TTFT, TPOT, throughput, quality.

---

## 15. Phase 8 - Decoupled Positional Representation

Investigate separating positional information from the compressed content representation:

```
                  +-- content --> latent KV
hidden state -----+
                  +-- position --> RoPE component
```

Vary: latent dimension, positional dimension, compression ratio. Measure: perplexity, long-context retrieval, memory, latency, throughput.

---

## 16. Phase 9 - MLA Matrix Absorption

Implement the algebraic optimization that avoids reconstructing full K/V where possible. Instead of `c^KV -> K -> Attention`, investigate absorbed computation:

```
q^T K = q^T W^UK c^KV
```

and reorganize the execution.

### Three implementations

* **MLA-1** - materialized K/V
* **MLA-2** - latent cache + reconstruct K/V
* **MLA-3** - latent cache + absorbed projections

> **Does the mathematical optimization translate into actual GPU performance?**

---

## 17. Phase 10 - MLA Compression Study

Systematic latent-dimension sweep, e.g. `latent_dim: 256, 384, 512, 768, 1024`.

Measure:

* **Memory** - KV bytes/token, total VRAM, cache capacity
* **Performance** - TTFT, TPOT, throughput, memory bandwidth
* **Quality** - perplexity, long-context retrieval, long-context QA

Produce Compression Ratio -> Quality and Compression Ratio -> Decode Latency curves.

> **Where is the useful compression frontier?**

---

## 18. Phase 11 - Custom GPU Kernels

Optimize actual hot paths. Potential targets: GQA decode, MLA decode, latent-cache attention, RoPE operations, KV-cache operations, sparse attention, fused operations.

### Technology progression

Start with **Triton**, then **CUDA C++** when lower-level control is useful.

### Investigate

Coalesced access, memory bandwidth, register pressure, shared memory, occupancy, warp divergence, L2 behavior, launch overhead, tensor-core utilization.

Every kernel optimization gets a before/after benchmark.

---

## 19. Phase 12 - Nsight Profiling

### Nsight Systems

CPU/GPU timeline, scheduler behavior, kernel launch gaps, synchronization, batching, prefill/decode behavior, GPU idle periods, request-level execution.

### Nsight Compute

Achieved occupancy, DRAM bandwidth, L2 hit rate, registers/thread, shared memory, warp stalls, instruction throughput, tensor-core utilization.

### Optimization loop

```
Implement -> Benchmark -> Nsight Systems -> Identify bottleneck
   -> Nsight Compute -> Modify kernel -> Benchmark -> Compare
```

The goal is not merely optimization. The goal is **causal understanding**.

---

## 20. Phase 13 - Prefix Caching

Shared prefixes should not be recomputed unnecessarily:

```
              System Prompt
                    |
          +---------+---------+
          v         v         v
          A         B         C
```

Measure: cache hit rate, TTFT, VRAM, throughput, cache overhead. Then investigate whether prefix caching + latent KV caching stack cleanly.

---

## 21. Phase 14 - DSA-Inspired Sparse Attention

The second major research component. Reduce the number of tokens receiving expensive attention.

Dense: `Attention(q, K_1:T, V_1:T)`. Sparse: `S_t subset {1,...,T}` with `|S_t| << T`, then perform expensive attention only over `K_S_t, V_S_t`.

### DSA Component 1 - Indexer

```
Query -> Indexer -> Relevance scores -> Top-K selection
```

The indexer must be significantly cheaper than the attention computation it replaces - otherwise **DSA loses**. This overhead must be included in end-to-end measurements.

### DSA Component 2 - Sparse Selection

Experiment with retention ratios: 100%, 50%, 25%, 12.5%, 6.25%, 3.125%. For every level measure: attention compute, indexer compute, selection overhead, latency, throughput, GPU utilization, quality, retrieval accuracy. Don't report theoretical FLOP reduction without measuring actual runtime.

### DSA Component 3 - Sparse Attention Kernel

Implement an actual GPU sparse-attention path. Do **not** simply do `K[selected_indices]` and assume it is fast - investigate irregular memory access, gather efficiency, cache behavior, coalescing, occupancy, L2 behavior, indexer overhead, launch overhead.

> **Does the GPU benefit from the reduced computation, or does irregular memory access destroy the theoretical speedup?**

---

## 22. Phase 15 - Sparse Attention Quality Study

> **How sparse can attention become before useful information is lost?**

Evaluate: long-context retrieval (needle-in-a-haystack), long-context QA (questions requiring distant context), language modeling (perplexity), general generation (a small standardized evaluation set).

Produce Sparsity <-> Quality <-> Latency tradeoff curves.

---

## 23. Phase 16 - MLA + DSA

The **flagship experiment**. Combine MLA + latent KV cache + DSA-style token selection + sparse attention + optimized kernels.

* **MLA attacks:** memory per cached token
* **DSA attacks:** number of tokens receiving expensive attention

Hypothesis (to test, not assume): `MLA + DSA = less memory + less attention computation`.

### Critical MLA + DSA Experiment

Full factorial comparison:

| System     | KV representation | Attention |
| ---------- | ------------------ | --------- |
| GQA Dense  | Full KV             | Dense     |
| GQA Sparse | Full KV             | Sparse    |
| MLA Dense  | Latent KV           | Dense     |
| MLA Sparse | Latent KV           | Sparse    |

Evaluate across 4K, 8K, 16K, 32K, 64K, 128K where feasible. Tells you whether MLA works independently, DSA works independently, their effects are additive, one optimization makes the other less useful, and whether the bottleneck shifts after combining them.

---

## 24. Phase 17 - Adaptive Attention Runtime

Turn the individual techniques into an actual runtime policy. Backends: GQA, MLA, MLA + DSA.

Initial policy (thresholds to be derived from benchmark data, not assumed):

```
short context      -> GQA
medium context      -> MLA
very long context    -> MLA + DSA
```

Potential policy inputs: context length, batch size, available VRAM, prompt/output ratio, GPU utilization, latency SLO, KV-cache pressure.

Compare fixed strategy vs. adaptive strategy. This transforms the project from a collection of optimizations into an actual inference runtime.

---

## 25. Phase 18 - Multi-GPU

Use two T4s in two different ways.

### Experiment A - Replicated serving

```
                 Router
                /      \
            T4 #0      T4 #1
           Engine A   Engine B
```

Measure Scaling Efficiency = Throughput(2GPU) / (2 x Throughput(1GPU)). Investigate request routing, load balancing, aggregate throughput, latency, GPU utilization.

### Experiment B - Model parallelism (secondary)

Investigate PCIe communication, synchronization, communication/computation overlap, scaling overhead. Do not let this become a major dependency.

---

## 26. Benchmark Matrix

* **Context lengths:** 1K, 4K, 8K, 16K, 32K, 64K, 128K (not every system needs every length)
* **Batch/concurrency:** 1, 2, 4, 8, 16, 32
* **Output lengths:** 32, 128, 512, 1024
* **Systems:** Hugging Face, vLLM, LatentServe GQA, LatentServe GQA + Paged KV, LatentServe MLA, LatentServe MLA + optimized kernels, LatentServe DSA, LatentServe MLA + DSA, LatentServe Adaptive

---

## 27. Workload Families

Don't benchmark only fixed-length prompts.

* **A - Short interactive:** 1K input, 128 output, batch 1-4 (tests responsiveness)
* **B - Long prompt:** 32K-128K input, 128 output (tests prefill and memory)
* **C - Long generation:** 8K input, 1K output (tests decode)
* **D - Mixed serving:** requests of 1K/4K/16K/32K/64K mixed together (tests scheduling)
* **E - Shared prefixes:** many requests share a large prefix (tests prefix caching)
* **F - Sparse-relevant context:** synthetic long-context workloads where only a small subset of tokens contains relevant information (tests DSA quality/performance)

---

## 28. Fairness Rules for vLLM

Control: model weights, tokenizer, precision, GPU, CUDA environment where relevant, input tokens, output tokens, concurrency, sampling, context length. Warm-up runs excluded. Multiple repetitions. Report median, p95, p99, confidence intervals where practical. No cherry-picked workloads.

---

## 29. Metrics Dashboard

* **Performance:** TTFT, TPOT, E2E latency, tokens/sec, requests/sec
* **Memory:** peak VRAM, KV bytes/token, cache utilization, fragmentation
* **GPU:** SM utilization, memory utilization, achieved bandwidth, kernel time, occupancy, L2 hit rate
* **Runtime:** scheduler overhead, batching efficiency, queue time, allocation overhead
* **Quality:** perplexity, retrieval accuracy, long-context QA accuracy

---

## 30. Core Graphs

1. Peak VRAM vs. context length (GQA, MLA, MLA + DSA)
2. Decode latency vs. context length (vLLM, GQA, MLA, DSA, MLA + DSA)
3. Throughput vs. concurrency (vLLM, LatentServe)
4. MLA compression ratio vs. quality
5. Attention sparsity vs. quality
6. Attention sparsity vs. latency
7. Indexer overhead vs. sparse-attention speedup (particularly important for DSA)
8. GPU memory bandwidth before/after kernel optimization
9. 1 GPU vs. 2 GPU throughput
10. Quality vs. performance frontier - should become one of the project's central visualizations

---

## 31. Ablation Studies

* **MLA ablations:** remove latent compression, matrix absorption, decoupled positional path; compare each component.
* **DSA ablations:** remove indexer, sparse selection, sparse kernel optimization; measure the effect.
* **Runtime ablations:** remove continuous batching, prefix caching, paging, adaptive policy.
* **Joint ablation:** GQA dense, GQA sparse, MLA dense, MLA sparse - the most important MLA/DSA ablation.

---

## 32. Expected Failure Modes

Actively look for failure.

* **MLA saves memory but isn't faster** - possible causes: extra projections, additional memory operations, poor kernel utilization, insufficient KV-cache pressure.
* **DSA reduces FLOPs but isn't faster** - possible causes: indexer overhead, irregular memory access, gather overhead, poor GPU utilization, kernel launch overhead.
* **Sparse attention hurts quality** - expected at sufficiently aggressive sparsity.
* **MLA + DSA isn't additive** - potentially very interesting; one optimization may change the bottleneck that makes the other useful.
* **vLLM beats LatentServe** - expected for some workloads; explain why.
* **Two T4s don't scale linearly** - expected because of PCIe, synchronization, routing, workload imbalance.

These are **results**, not project failures.

---

## 33. Success Criteria

### Minimum viable

Qwen2.5-1.5B-Instruct inference, correctness harness, GQA baseline, KV cache, continuous batching, paged KV cache, vLLM comparison, MLA, memory measurements, Nsight profiling. At this point the project is already strong.

### Strong project

Add: matrix absorption, custom Triton/CUDA kernels, prefix caching, extensive vLLM comparison, long-context experiments, quality evaluation, multi-GPU experiments.

### Standout project

Add: DSA-inspired sparse attention, real sparse GPU kernel, MLA + DSA, quality/performance frontier, adaptive attention selection, automated profiling, reproducible Kaggle benchmark environment.

### Exceptional project

At least one genuinely novel contribution: better sparse-selection strategy, better MLA kernel, improved sparse memory layout, adaptive scheduling policy, adaptive attention backend selection, new cache-management technique, previously undocumented T4 bottleneck, novel empirical finding about MLA/DSA interaction.

The project **does not require a novel algorithm** to be impressive. A rigorous systems characterization can already be excellent.

---

## 34. Software Architecture

```
model/
|-- qwen.py                # Phase 1: instrumented QwenReference wrapper (done)
|-- attention/
|   |-- reference.py
|   |-- gqa.py              # Phase 2
|   |-- mla.py               # Phase 7+
|   `-- sparse.py            # Phase 14+
`-- rope.py

cache/
|-- kv_cache.py             # Phase 2
|-- paged_cache.py          # Phase 3
|-- latent_cache.py         # Phase 7+
`-- prefix_cache.py         # Phase 13

runtime/
|-- engine.py                # Phase 4
|-- scheduler.py
|-- batching.py
|-- request.py
`-- policy.py                # Phase 17

kernels/
|-- gqa/                      # Phase 11
|-- mla/
|-- sparse/
`-- common/

benchmarks/
|-- schema.py                 # Phase 0 (done)
|-- latency.py
|-- throughput.py
|-- memory.py
|-- workloads/
`-- runners/
    |-- check_env.py          # Phase 0 (done)
    `-- phase1_reference.py   # Phase 1 (done)

profiling/
|-- nsight_systems/           # Phase 12
`-- nsight_compute/

evaluation/
|-- perplexity.py             # Phase 15
|-- retrieval.py
`-- long_context.py

comparisons/
`-- vllm/                     # Phase 6

configs/                      # Phase 0 (done)

notebooks/
`-- kaggle/

results/
|-- raw/
|-- processed/
`-- figures/

docs/
|-- architecture.md
|-- mla.md
|-- dsa.md
|-- kernels.md
|-- profiling.md
`-- methodology.md            # this file
```

---

## 35. Testing Strategy

Three levels.

### Unit tests

KV allocation, page mapping, cache freeing, attention masking, RoPE, latent projections, top-K selection, sparse index mapping.

### Numerical tests

Compare PyTorch reference vs. Triton vs. CUDA within explicit tolerances.

### End-to-end tests

Same prompt -> same/near-identical logits -> same deterministic output, where architectural equivalence permits. (Phase 1's `test_phase1_correctness.py` is the first instance of this pattern - teacher-forced vs. incremental decode, and deterministic greedy generation.)

---

## 36. Reproducibility

Every benchmark records: git commit, model identifier/hash, configuration, GPU model, number of GPUs, CUDA version, PyTorch version, Triton version, driver version, random seed. Results are generated automatically - **no manually typed benchmark numbers** (enforced by `benchmarks/schema.py::ResultWriter`).

---

## 37. Kaggle Strategy

Kaggle is the primary experimental environment. Suggested notebooks:

```
01_qwen_reference.ipynb
02_gqa_kv.ipynb
03_paged_kv.ipynb
04_continuous_batching.ipynb
05_vllm_comparison.ipynb
06_mla.ipynb
07_mla_absorption.ipynb
08_mla_kernels.ipynb
09_dsa.ipynb
10_sparse_kernel.ipynb
11_mla_dsa.ipynb
12_final_benchmarks.ipynb
```

Heavy benchmark results should be exported to machine-readable files. Notebooks are **experiments**, not the architecture itself.

---

## 38. Nsight Artifact Strategy

Store representative profiles under `profiling/systems/*.nsys-rep` and `profiling/compute/*.ncu-rep`. Document profiling commands, configurations, hardware, representative workloads. The final report should include representative Nsight screenshots/statistics.

---

## 39. Proposed Timeline

| Weeks | Work |
| --- | --- |
| 1 | Qwen environment + benchmark infrastructure (Phase 0) |
| 2-3 | Qwen reference + GQA + KV instrumentation (Phase 1-2) |
| 4 | Paged KV cache (Phase 3) |
| 5-6 | Continuous batching + scheduler (Phase 4) |
| 7 | vLLM comparison (Phase 6) |
| 8-10 | MLA-inspired latent KV implementation (Phase 7) |
| 11 | Decoupled positional path + matrix absorption (Phase 8-9) |
| 12-13 | MLA compression study + Triton/CUDA kernels + Nsight (Phase 10-12) |
| 14 | Prefix caching (Phase 13) |
| 15-17 | DSA indexer + sparse attention (Phase 14) |
| 18-19 | Sparse GPU kernel + quality evaluation (Phase 14-15) |
| 20-21 | MLA + DSA (Phase 16) |
| 22 | Adaptive runtime (Phase 17) |
| 23 | 2x T4 experiments (Phase 18) |
| 24 | Final benchmarking, analysis, documentation |

This is modular. If time runs short, Phases 0-10 (through the MLA compression study) already give a seriously strong project.

---

## 40. Milestone Gates

1. Can Qwen generate correctly? [DONE - Phase 1]
2. Can LatentServe execute cached decoding? [DONE - Phase 1, via the instrumented wrapper; a custom cache implementation is Phase 2's job]
3. Can the runtime serve multiple requests? (Phase 4)
4. Can paged KV improve memory utilization? (Phase 3)
5. Can continuous batching improve utilization? (Phase 4)
6. Can we fairly benchmark against vLLM? (Phase 6)
7. Does MLA work numerically? (Phase 7)
8. Does MLA actually reduce measured memory? (Phase 7)
9. Can Nsight explain MLA performance? (Phase 12)
10. Can a custom kernel produce a measurable improvement? (Phase 11)
11. Does DSA reduce end-to-end attention cost? (Phase 14)
12. Does sparse attention preserve useful context? (Phase 15)
13. Do MLA and DSA provide complementary gains? (Phase 16)
14. Can an adaptive policy outperform a fixed backend? (Phase 17)

---

## 41. Final Evaluation

Five (or six) major questions:

1. **Memory** - How much does MLA reduce KV-cache memory? `Reduction = (M_GQA - M_MLA) / M_GQA`
2. **Latency** - Does that memory reduction actually improve decode latency?
3. **Compute** - Does DSA actually reduce end-to-end attention computation (not theoretical FLOPs alone)?
4. **Quality** - What quality is sacrificed at different compression/sparsity ratios?
5. **Serving** - Can the resulting system approach or exceed vLLM on any meaningful workload?
6. **Interaction** - Do MLA and DSA produce complementary improvements, or do they simply move the bottleneck somewhere else?

---

## 42. Final Research Matrix

| System                 | KV Memory | Dense Compute | Sparse Compute | Custom Kernels | Serving |
| ----------------------- | --------: | -------------: | ---------------: | ---------------: | ------: |
| Hugging Face             |  Baseline |       Baseline |                 - |                 - |   Basic |
| vLLM                     | Optimized |      Optimized |                 - |                 X |       X |
| LatentServe GQA          |     lower |              X |                 - |                 X |       X |
| LatentServe MLA          |  much lower |            X |                 - |                 X |       X |
| LatentServe DSA          |         - |         lower |                 X |                 X |       X |
| LatentServe MLA + DSA    |  much lower |       much lower |             X |                 X |       X |
| LatentServe Adaptive     |  much lower |       much lower |             X |                 X |       X |

The actual values come from experiments.

---

## 43. Final Deliverables

**Software:** Qwen2.5-1.5B-Instruct inference runtime, GQA backend, paged KV cache, latent KV cache, MLA-inspired attention, DSA-inspired sparse attention, sparse attention kernel, prefix cache, continuous batching, scheduler, adaptive attention policy, Triton kernels, CUDA kernels.

**Experimental infrastructure:** automated benchmark harness, vLLM comparison suite, quality evaluation suite, profiling scripts, Kaggle notebooks, reproducible configs, structured result storage.

**Analysis:** KV memory scaling, latency scaling, throughput scaling, compression/quality tradeoffs, sparsity/quality tradeoffs, MLA/DSA interaction, Nsight analysis, kernel optimization analysis, scheduler analysis, 1x vs. 2x T4 scaling.

**Documentation:** architecture document, MLA technical document, DSA technical document, kernel documentation, profiling methodology, benchmark methodology, reproducibility guide, final technical report.

---

## 44. Final Project Thesis

> **Modern LLM inference efficiency is not determined by a single optimization. It emerges from the interaction between attention architecture, KV-cache representation, memory management, sparse computation, GPU kernels, scheduling, and workload characteristics.**

LatentServe takes one real LLM - Qwen2.5-1.5B-Instruct - and progressively changes how that model executes, moving from conventional GQA and paged KV caching through latent KV compression, MLA-style execution, DSA-inspired sparse attention, custom GPU kernels, and adaptive serving.

The strongest conclusion does **not** have to be "LatentServe beats vLLM." It could instead be something like:

> **"Under workload X on T4 hardware, MLA reduced KV-cache memory by X%, while DSA reduced effective attention computation by Y%. Their combination produced Z% end-to-end improvement, but irregular memory access became the dominant bottleneck beyond K% sparsity. Nsight analysis explains why."**

That is a much more interesting result: **model -> attention -> cache -> runtime -> kernel -> GPU -> serving -> empirical research.**
