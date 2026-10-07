"""
Phase 18, experiment A — replicated serving: a router in front of one engine
per GPU.

    python -m benchmarks.runners.phase18_replicas         # ~30 min, Kaggle "GPU T4 x2"

The plan (methodology §25): Scaling Efficiency = Throughput(2 GPU) /
(2 x Throughput(1 GPU)), plus routing, load balancing, aggregate throughput,
latency and GPU utilisation. Each GPU runs its own worker process and engine
(two engines in one Python process would contend for the interpreter lock
during prefill — unlike any real deployment). Workers see only their own GPU
(CUDA_VISIBLE_DEVICES). A `reset` rebuilds a worker's engine — fresh KV and
prefix caches — without reloading the model, so every configuration starts
clean.

Workloads:
  burst   requests submitted at once (1-4K prompts, 128-256 outputs): the
          throughput-bound case, for scaling efficiency
  chat    multi-turn conversations sharing a system prompt, each turn sent
          when the previous reply arrives, prefix caching on: the case where
          routing decides whether turns find their cached history

Per request: time to first token, time per output token, end-to-end time —
p50 / p95 / p99 for each — plus throughput, prefix-cache hit rate, and each
GPU's share of tokens and busy time. The worker backend is pluggable:
"latentserve" or "vllm", through the same router and workloads.

Two measurement details:

* **Graph capture is excluded.** Each configuration rebuilds its engine, so
  LatentServe recaptures CUDA graphs during the timed run — the same seconds
  per GPU whether one GPU runs or two, so on two GPUs (half the makespan) the
  capture share doubles and would bias scaling efficiency down. vLLM captures
  at startup, before any run, so including LatentServe's capture would also
  bias every cross-engine comparison. Workers report capture time;
  throughput is given raw and with capture excluded (the max over the GPUs
  used, since they capture in parallel), and scaling efficiency uses the
  latter.
* **Latencies cross processes:** submit times come from the router, first-
  token and finish times from the workers. `time.perf_counter` on Linux reads
  CLOCK_MONOTONIC, which is system-wide, so the differences are valid.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import statistics
import sys
import time
from pathlib import Path

# --------------------------------------------------------------- worker ---


def _latentserve_engine(opts: dict, state: dict):
    """Build (once) the model, and (each reset) a fresh engine."""
    import torch

    from benchmarks.runners.phase17_workload import pool_blocks
    from config import load_config
    from kernels.gqa.paged_decode import set_decode_backend
    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import QwenReference
    from runtime.engine import ServingEngine

    if "model" not in state:
        cfg = load_config(opts["config"])
        ref = QwenReference(model_name=cfg.model.name, dtype="fp16", device="cuda:0").load()
        set_decode_backend("cuda")
        model = LatentServeQwen.from_reference(ref, max_seq_len_hint=opts["max_seq_len"] + 512,
                                               attn_impl="triton_paged", fuse_projections=True)
        model.set_elementwise(True)
        state["model"] = model
    model = state["model"]
    model.cache = None
    torch.cuda.empty_cache()
    # Sweep options: a fixed sparse budget, or the adaptive policy (a tier
    # and the step table it decides from); neither by default.
    model.set_sparse(opts.get("sparse"))
    policy = None
    if opts.get("tier"):
        from runtime.policy import SparsePolicy

        policy = SparsePolicy.from_json(opts["policy_table"], tier=opts["tier"])
    return ServingEngine(model, max_running=opts["max_running"], max_seq_len=opts["max_seq_len"],
                         block_size=16, num_blocks=pool_blocks(model, opts["headroom_gb"]),
                         use_cuda_graphs=True, kv_dtype=opts["kv_dtype"],
                         prefix_caching=opts["prefix_caching"], policy=policy)


class _VLLMEngine:
    """vLLM behind the interface worker_main drives — add_request(ServedRequest),
    has_work, step(), on_retire(req, engine), prefill_s / decode_s — through
    vLLM's incremental engine (`llm.llm_engine`). Built once per worker; a
    reset clears vLLM's prefix cache rather than reloading the model."""

    def __init__(self, llm, max_running=None):
        self.llm = llm
        self.max_running = max_running          # vLLM's max_num_seqs, fixed at startup
        self.eng = llm.llm_engine
        self.reqs: dict = {}
        self.on_retire = None
        self.prefill_s = 0.0          # vLLM interleaves prefill and decode in one
        self.decode_s = 0.0           # step: busy time accrues here
        self.decoder = None           # graphs are captured at startup, not in runs

    def add_request(self, req) -> None:
        from vllm import SamplingParams

        try:
            from vllm.inputs import TokensPrompt
            prompt = TokensPrompt(prompt_token_ids=list(req.prompt_ids))
        except ImportError:  # pragma: no cover - older vLLM
            prompt = {"prompt_token_ids": list(req.prompt_ids)}
        # detokenize=False: LatentServe never turns tokens into text, so vLLM
        # should not be charged for it either.
        try:
            params = SamplingParams(max_tokens=req.max_new_tokens, temperature=0.0,
                                    ignore_eos=True, detokenize=False)
        except TypeError:  # pragma: no cover - a version without the field
            params = SamplingParams(max_tokens=req.max_new_tokens, temperature=0.0, ignore_eos=True)
        self.eng.add_request(str(req.request_id), prompt, params)
        self.reqs[str(req.request_id)] = req

    @property
    def has_work(self) -> bool:
        return bool(self.reqs) or self.eng.has_unfinished_requests()

    def abort(self, reqs) -> None:
        ids = [str(r.request_id) for r in reqs if str(r.request_id) in self.reqs]
        if ids:
            self.eng.abort_request(ids)
            for i in ids:
                del self.reqs[i]

    @property
    def kv_blocks(self):
        """(KV blocks, block size), if this vLLM version exposes them in the
        main process — from its config, or from its cache_config_info metric.
        None otherwise; the measurement then relies on aborting instead."""
        if getattr(self, "_kv", 0) != 0:
            return self._kv
        self._kv = None
        for path in ("vllm_config.cache_config", "cache_config"):
            obj = self.eng
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            n, bs = getattr(obj, "num_gpu_blocks", None), getattr(obj, "block_size", None)
            if n and bs:
                self._kv = (int(n), int(bs))
                return self._kv
        try:
            for m in self.llm.get_metrics():
                labels = getattr(m, "labels", None) or {}
                if "num_gpu_blocks" in labels and labels.get("block_size"):
                    self._kv = (int(float(labels["num_gpu_blocks"])), int(float(labels["block_size"])))
                    break
        except Exception:  # noqa: BLE001 - metrics are optional
            pass
        return self._kv

    def step(self) -> None:
        t0 = time.perf_counter()
        outs = self.eng.step()
        now = time.perf_counter()
        self.decode_s += now - t0
        for o in outs:
            req = self.reqs.get(o.request_id)
            if req is None:
                continue
            toks = o.outputs[0].token_ids if o.outputs else []
            if toks and req.first_token_time is None:
                req.first_token_time = now
            if o.finished:
                req.output_ids = list(toks)
                req.finish_time = now
                req.prefix_hit_tokens = int(getattr(o, "num_cached_tokens", 0) or 0)
                del self.reqs[o.request_id]
                if self.on_retire is not None:
                    self.on_retire(req, self)


def _vllm_engine(opts: dict, state: dict):
    """Build (once) the vLLM engine; each reset clears its prefix cache."""
    if "llm" not in state:
        from vllm import LLM

        from comparisons.vllm.runner import construct_engine
        from config import load_config

        cfg = load_config(opts["config"])
        kwargs = {"model": cfg.model.name, "dtype": "float16", "seed": 0,
                  "gpu_memory_utilization": opts.get("vllm_memory", 0.85),
                  "max_model_len": opts["max_seq_len"], "tensor_parallel_size": 1,
                  # The same concurrency limit LatentServe runs with: left at
                  # vLLM's default (~256) it was not a like-for-like comparison.
                  "max_num_seqs": opts["max_running"]}
        # Prefix caching is fixed for the engine's life (vLLM cannot toggle it
        # per configuration): on unless opts say otherwise. A reset clears it,
        # so a burst run starts cold and gains nothing from it.
        state["prefix"] = opts.get("vllm_prefix", True)
        state["max_running"] = opts["max_running"]
        state["llm"], _ = construct_engine(LLM, kwargs, state["prefix"], None)
    elif hasattr(state["llm"], "reset_prefix_cache"):
        state["llm"].reset_prefix_cache()
    return _VLLMEngine(state["llm"], state.get("max_running"))


BACKENDS = {"latentserve": _latentserve_engine, "vllm": _vllm_engine}


def worker_main(gpu: int, backend: str, inbox, outbox, opts: dict) -> None:
    """Report any failure with its traceback: a worker that dies silently left
    the parent waiting out a 15-minute timeout (the first vLLM run)."""
    try:
        _worker_loop(gpu, backend, inbox, outbox, opts)
    except Exception:  # noqa: BLE001
        import traceback

        outbox.put(("error", gpu, traceback.format_exc()))


def _worker_loop(gpu: int, backend: str, inbox, outbox, opts: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)        # before torch is imported
    from runtime.request import ServedRequest

    if backend not in BACKENDS:
        outbox.put(("error", gpu, f"backend {backend!r} not available in this worker"))
        return
    state, engine, busy0 = {}, None, 0.0

    def build(o):
        nonlocal engine, busy0
        engine = BACKENDS[backend](o, state)
        busy0 = 0.0

        def on_retire(req, eng):
            outbox.put(("done", {"rid": req.request_id, "gpu": gpu, "n_out": len(req.output_ids),
                                 "output_ids": list(req.output_ids),
                                 "t_first": req.first_token_time, "t_finish": req.finish_time,
                                 "prompt_len": req.prompt_len,
                                 "hit": getattr(req, "prefix_hit_tokens", 0)}))
        engine.on_retire = on_retire

    build(opts)
    outbox.put(("ready", gpu))
    while True:
        msgs = []
        try:
            while True:
                msgs.append(inbox.get_nowait())
        except queue.Empty:
            pass
        if not msgs and not engine.has_work:
            msgs.append(inbox.get())                     # idle: wait for work
        for m in msgs:
            if m[0] == "req":
                _, rid, prompt, max_new = m
                engine.add_request(ServedRequest(request_id=rid, prompt_ids=prompt,
                                                 max_new_tokens=max_new))
            elif m[0] == "reset":
                busy = engine.prefill_s + engine.decode_s
                capture = engine.decoder.capture_s if getattr(engine, "decoder", None) else 0.0
                outbox.put(("stats", gpu, busy, capture))
                build(m[1])
                outbox.put(("reset_ok", gpu))
            elif m[0] == "measure_steps":
                outbox.put(("steps", gpu, _measure_steps(engine, ServedRequest, **m[1])))
            elif m[0] == "stop":
                capture = engine.decoder.capture_s if getattr(engine, "decoder", None) else 0.0
                outbox.put(("stats", gpu, engine.prefill_s + engine.decode_s, capture))
                return
        if engine.has_work:
            engine.step()


def _measure_steps(engine, ServedRequest, batch: int, ctx: int, steps: int, warmup: int,
                   seed: int = 0, vocab: tuple = (1000, 100000)) -> dict:
    """Per-step decode time at (batch, ctx), at the engine level: `batch`
    requests of `ctx` prompt tokens, each generating warmup + steps tokens;
    once *every* request is decoding, each engine step is timed. A batch that
    cannot all run at once is reported as not fitting, never timed short."""
    import random as _r

    limit = getattr(engine, "max_running", None)
    if limit is not None and batch > limit:
        return {"fits": False, "reason": f"batch {batch} exceeds the engine's concurrency limit "
                                         f"of {limit} — a configuration error, not memory"}
    kv = getattr(engine, "kv_blocks", None)                # vLLM: (blocks, block size) if known
    if kv:
        blocks, bs = kv
        need = batch * -(-(ctx + warmup + steps + 16) // bs)
        if need > blocks:
            return {"fits": False, "reason": f"needs {need} KV blocks, the engine has {blocks}"}
    if hasattr(engine, "cache"):                           # LatentServe: check the pool first
        alloc = engine.cache.allocator
        need = batch * alloc.blocks_for_tokens(ctx + warmup + steps + 16)   # the same headroom
        if need > alloc.num_available:
            return {"fits": False, "reason": f"needs {need} blocks, pool has {alloc.num_available}"}
    # Our own completion hook while measuring — restored afterwards, or the
    # next workload on this engine would never report a completion.
    previous = engine.on_retire
    try:
        return _timed_batch(engine, ServedRequest, batch, ctx, steps, warmup, seed, vocab)
    finally:
        engine.on_retire = previous


def _timed_batch(engine, ServedRequest, batch, ctx, steps, warmup, seed, vocab) -> dict:
    import random as _r

    rng = _r.Random(seed)
    reqs = [ServedRequest(request_id=10_000_000 + i,
                          prompt_ids=[rng.randrange(*vocab) for _ in range(ctx)],
                          # Headroom beyond what is timed: engines differ in how many
                          # tokens a request has by the time all are decoding
                          # (LatentServe's step prefills newcomers *and* decodes
                          # them), so exact counting finished requests mid-timing.
                          max_new_tokens=warmup + steps + 16) for i in range(batch)]
    done = []
    engine.on_retire = lambda req, eng: done.append(req)
    for r in reqs:
        engine.add_request(r)
    while engine.has_work and any(r.first_token_time is None for r in reqs):
        if done:                                           # one finished before all started
            break
        engine.step()
    if done or any(r.first_token_time is None for r in reqs):
        # Abort what is left: draining it prefilled every remaining request for
        # nothing — ~25 minutes per non-fitting 32K cell on vLLM.
        rest = [r for r in reqs if r not in done]
        if hasattr(engine, "abort"):
            engine.abort(rest)
        while engine.has_work:
            engine.step()
        return {"fits": False, "reason": "the batch could not all decode at once (memory)"}
    for _ in range(warmup):
        engine.step()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        engine.step()
        times.append((time.perf_counter() - t0) * 1000)
        if done:
            break
    while engine.has_work:
        engine.step()
    return {"fits": len(times) == steps, "step_ms": times}


class Cluster:
    """Worker processes, one per GPU."""

    def __init__(self, gpus: int, backend: str, opts: dict):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.outbox = ctx.Queue()
        self.inboxes = [ctx.Queue() for _ in range(gpus)]
        # Not daemons: vLLM's engine starts a process of its own, and Python
        # forbids daemonic processes from having children — the first vLLM
        # run died on exactly that. stop() tears them down explicitly.
        self.procs = [ctx.Process(target=worker_main, args=(g, backend, self.inboxes[g],
                                                           self.outbox, opts), daemon=False)
                      for g in range(gpus)]
        for p in self.procs:
            p.start()
        ready = 0
        while ready < gpus:
            m = self.recv(timeout=900)
            ready += m[0] == "ready"

    def submit(self, gpu, rid, prompt, max_new):
        self.inboxes[gpu].put(("req", rid, list(prompt), max_new))

    def recv(self, timeout=3600):
        """Next message — failing within seconds, with the worker's own
        traceback, if a worker reports an error or dies."""
        deadline = time.perf_counter() + timeout
        while True:
            # Never wait longer than what remains: polling in fixed 5 s slices
            # let a short timeout overshoot by up to 5 s.
            try:
                m = self.outbox.get(timeout=max(0.001, min(5.0, deadline - time.perf_counter())))
            except queue.Empty:
                dead = [g for g, p in enumerate(self.procs) if not p.is_alive()]
                if dead:
                    codes = {g: self.procs[g].exitcode for g in dead}
                    raise RuntimeError(f"worker(s) died without reporting: exit codes {codes}")
                if time.perf_counter() > deadline:
                    raise
                continue
            if m[0] == "error":
                raise RuntimeError(f"worker {m[1]} failed:\n{m[2]}")
            return m

    def measure_steps(self, gpu: int, **spec) -> dict:
        self.inboxes[gpu].put(("measure_steps", spec))
        while True:
            m = self.recv()
            if m[0] == "steps":
                return m[2]

    def warmup(self, opts, lengths=(1024, 2048, 4096), per_gpu: int = 6) -> None:
        """Run a short workload on every worker, then reset. The first timed
        run otherwise pays one-off compilation — Triton prefill kernels per
        prompt shape, NVRTC kernels, cuBLAS setup — which made the first
        Phase 18 run's single-GPU baseline look 44% slower than it is."""
        import random as _r

        rng = _r.Random(12345)
        n = 0
        for g in range(len(self.inboxes)):
            for i in range(per_gpu):
                plen = lengths[i % len(lengths)]
                self.submit(g, -1 - n, [rng.randrange(1000, 100000) for _ in range(plen)], 32)
                n += 1
        while n:
            if self.recv()[0] == "done":
                n -= 1
        self.reset(opts)

    def reset(self, opts) -> tuple:
        """Rebuild every worker's engine; returns the finished run's busy and
        graph-capture seconds per GPU."""
        for q in self.inboxes:
            q.put(("reset", opts))
        busy, capture, acks = {}, {}, 0
        while acks < len(self.inboxes):
            m = self.recv()
            if m[0] == "stats":
                busy[m[1]], capture[m[1]] = m[2], m[3]
            elif m[0] == "reset_ok":
                acks += 1
        n = len(self.inboxes)
        return [busy.get(g, 0.0) for g in range(n)], [capture.get(g, 0.0) for g in range(n)]

    def stop(self):
        for q in self.inboxes:
            q.put(("stop",))
        for p in self.procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)


# ------------------------------------------------------------- workload ---


def burst(seed: int, n: int = 96) -> dict:
    rng = random.Random(seed)
    reqs = [{"rid": i, "prompt": [rng.randrange(1000, 100000) for _ in range(rng.choice((1024, 2048, 4096)))],
             "max_new": rng.choice((128, 256))} for i in range(n)]
    return {"kind": "burst", "requests": reqs}


def chat(seed: int, convs: int = 16, turns: int = 5) -> dict:
    rng = random.Random(seed)
    tok = lambda k: [rng.randrange(1000, 100000) for _ in range(k)]  # noqa: E731
    system = tok(512)
    msgs = {(c, t): tok(128) for c in range(convs) for t in range(turns)}
    first = [{"rid": c * turns, "prompt": system + msgs[(c, 0)], "max_new": 128, "conv": c, "turn": 0}
             for c in range(convs)]
    return {"kind": "chat", "requests": first, "msgs": msgs, "turns": turns}


def run_workload(cluster, router, wl: dict, gpus_used: int) -> dict:
    """Submit, route, drive conversations closed-loop, collect per-request results."""
    meta, submit_t, results = {}, {}, []

    def send(r):
        g = router.route(r["prompt"])
        meta[r["rid"]] = {**r, "gpu": g}
        submit_t[r["rid"]] = time.perf_counter()
        cluster.submit(g, r["rid"], r["prompt"], r["max_new"])

    # Requests may carry `at` — seconds after the start — for open-loop
    # traffic; without it they are sent at once. A "sequential" workload sends
    # each request when the previous one finishes.
    queue_ = sorted(wl["requests"], key=lambda r: r.get("at", 0.0))
    sequential = wl["kind"] == "sequential"
    t_start = time.perf_counter()
    pending = 0

    def release():
        nonlocal pending
        while queue_ and (sequential and pending == 0 or not sequential
                          and queue_[0].get("at", 0.0) <= time.perf_counter() - t_start):
            send(queue_.pop(0))
            pending += 1
            if sequential:
                break

    release()
    while pending or queue_:
        wait = 3600.0
        if queue_ and not sequential:
            wait = max(0.0, t_start + queue_[0].get("at", 0.0) - time.perf_counter())
        if not pending:
            time.sleep(wait)
            release()
            continue
        try:
            m = cluster.recv(timeout=max(wait, 0.001))
        except queue.Empty:
            release()
            continue
        if m[0] != "done":
            continue
        d = m[1]
        pending -= 1
        router.done(d["gpu"])
        d["t_submit"] = submit_t[d["rid"]]
        results.append(d)
        r = meta[d["rid"]]
        release()
        if wl["kind"] == "chat" and r["turn"] + 1 < wl["turns"]:
            t = r["turn"] + 1
            nxt = {"rid": d["rid"] + 1, "prompt": r["prompt"] + d["output_ids"] + wl["msgs"][(r["conv"], t)],
                   "max_new": 128, "conv": r["conv"], "turn": t}
            send(nxt)
            pending += 1
    return summarise(results, gpus_used, router)


def summarise(results: list, gpus_used: int, router) -> dict:
    from runtime.router import percentiles

    t0 = min(r["t_submit"] for r in results)
    t1 = max(r["t_finish"] for r in results)
    ttft = [(r["t_first"] - r["t_submit"]) * 1000 for r in results]
    e2e = [(r["t_finish"] - r["t_submit"]) * 1000 for r in results]
    tpot = [(r["t_finish"] - r["t_first"]) * 1000 / (r["n_out"] - 1) for r in results if r["n_out"] > 1]
    out_tokens = sum(r["n_out"] for r in results)
    prompt_tokens = sum(r["prompt_len"] for r in results)
    per_gpu = [sum(r["n_out"] for r in results if r["gpu"] == g) for g in range(gpus_used)]
    return {"requests": len(results), "makespan_s": t1 - t0, "out_tok_s": out_tokens / (t1 - t0),
            "out_tokens": out_tokens,
            "ttft_ms": percentiles(ttft), "tpot_ms": percentiles(tpot), "e2e_ms": percentiles(e2e),
            "ttft_mean_ms": statistics.fmean(ttft),
            "hit_rate": sum(r["hit"] for r in results) / prompt_tokens if prompt_tokens else 0.0,
            "tokens_per_gpu": per_gpu, "routed": list(router.routed),
            "affinity_hits": router.affinity_hits}


def aggregate(rs: list) -> dict:
    """Median over rounds of each metric; the throughput spread shows whether
    order or noise could explain a difference."""
    med = statistics.median
    tps = [r["out_tok_s_ex_capture"] for r in rs]
    pct = lambda key: {p: med(r[key][p] for r in rs) for p in rs[0][key]}  # noqa: E731
    return {"rounds": rs, "out_tok_s_ex_capture": med(tps), "out_tok_s": med(r["out_tok_s"] for r in rs),
            "tok_s_spread": (max(tps) - min(tps)) / med(tps) if med(tps) else float("nan"),
            "ttft_ms": pct("ttft_ms"), "tpot_ms": pct("tpot_ms"), "e2e_ms": pct("e2e_ms"),
            "hit_rate": med(r["hit_rate"] for r in rs),
            "busy_frac": [med(r["busy_frac"][g] for r in rs) for g in range(len(rs[0]["busy_frac"]))]}


def ex_capture(res: dict) -> float:
    """Output tokens per second with graph capture excluded. The GPUs capture
    in parallel, so the makespan loses the longest capture, not their sum."""
    span = res["makespan_s"] - max(res.get("capture_s") or [0.0])
    return res["out_tokens"] / span if span > 0 else float("nan")


# ----------------------------------------------------------------- main ---

CONFIGS = [
    # (name, workload, gpus, policy)
    ("burst 1 GPU", "burst", 1, "least_loaded"),
    ("burst 2 GPU least-loaded", "burst", 2, "least_loaded"),
    ("burst 2 GPU round-robin", "burst", 2, "round_robin"),
    ("chat 1 GPU", "chat", 1, "least_loaded"),
    ("chat 2 GPU round-robin", "chat", 2, "round_robin"),
    ("chat 2 GPU least-loaded", "chat", 2, "least_loaded"),
    ("chat 2 GPU prefix-aware", "chat", 2, "prefix_aware"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--backend", default="latentserve", choices=["latentserve", "vllm"])
    p.add_argument("--configs", nargs="+", default=[c[0] for c in CONFIGS])
    p.add_argument("--max-running", type=int, default=16)
    p.add_argument("--headroom-gb", type=float, default=2.5)
    p.add_argument("--kv-dtype", choices=["fp16", "int8"], default="fp16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=2,
                   help="every configuration this many times, odd rounds in reverse order")
    p.add_argument("--config", default="configs/phase6_vllm.yaml")
    p.add_argument("--results-dir", default="results/raw/phase18")
    args = p.parse_args()

    import torch

    from runtime.router import Router

    gpus = torch.cuda.device_count()
    if gpus < 2:
        print(f"[ERROR] {gpus} GPU(s) visible: select the 'GPU T4 x2' accelerator", file=sys.stderr)
        return 1
    base = {"config": args.config, "max_running": args.max_running, "max_seq_len": 4096 + 5 * 256 + 256,
            "headroom_gb": args.headroom_gb, "kv_dtype": args.kv_dtype, "prefix_caching": False}
    print(f"starting {gpus} workers ({args.backend})...", flush=True)
    cluster = Cluster(2, args.backend, base)
    chosen = [c for c in CONFIGS if c[0] in args.configs]
    runs = {c[0]: [] for c in chosen}
    try:
        print("warming up every worker (excluded from every measurement)...", flush=True)
        cluster.warmup(dict(base, prefix_caching=True))
        for rnd in range(args.rounds):
            order = chosen if rnd % 2 == 0 else list(reversed(chosen))
            for name, kind, used, policy in order:
                opts = dict(base, prefix_caching=(kind == "chat"))
                cluster.reset(opts)
                wl = burst(args.seed) if kind == "burst" else chat(args.seed)
                router = Router(used, policy=policy)
                res = run_workload(cluster, router, wl, used)
                busy, capture = cluster.reset(opts)        # this run's busy and capture time
                res["busy_frac"] = [b / res["makespan_s"] for b in busy[:used]]
                res["capture_s"] = capture[:used]
                res["out_tok_s_ex_capture"] = ex_capture(res)
                runs[name].append(res)
                print(f"  round {rnd + 1}  {name:<28} {res['out_tok_s_ex_capture']:7.1f} tok/s  "
                      f"TTFT p50/p95/p99 {res['ttft_ms']['p50']:.0f}/{res['ttft_ms']['p95']:.0f}/"
                      f"{res['ttft_ms']['p99']:.0f} ms  hit {res['hit_rate']:5.1%}  "
                      f"per-GPU tokens {res['tokens_per_gpu']}", flush=True)
    finally:
        cluster.stop()

    out = {name: aggregate(rs) for name, rs in runs.items() if rs}
    print(f"\n{'configuration':<28}{'tok/s*':>8}{'spread':>8}{'TTFT p50':>10}{'p95':>8}{'p99':>8}"
          f"{'TPOT p50':>10}{'p99':>7}{'E2E p99':>10}{'hit':>7}  busy per GPU")
    for name, r in out.items():
        print(f"{name:<28}{r['out_tok_s_ex_capture']:>8.1f}{r['tok_s_spread']:>7.1%} "
              f"{r['ttft_ms']['p50']:>8.0f}ms{r['ttft_ms']['p95']:>6.0f}ms{r['ttft_ms']['p99']:>6.0f}ms"
              f"{r['tpot_ms']['p50']:>8.1f}ms{r['tpot_ms']['p99']:>5.1f}ms"
              f"{r['e2e_ms']['p99'] / 1000:>8.1f}s{r['hit_rate']:>7.1%}  "
              + " ".join(f"{b:.0%}" for b in r["busy_frac"]))
    # Scaling efficiency is a throughput question, so only the saturating
    # burst can answer it. Chat is closed-loop with a fixed number of
    # conversations: a second GPU halves each GPU's batch rather than
    # doubling the work, so chat measures latency, not scalability.
    for two in ("burst 2 GPU least-loaded", "burst 2 GPU round-robin"):
        if "burst 1 GPU" in out and two in out:
            eff = out[two]["out_tok_s_ex_capture"] / (2 * out["burst 1 GPU"]["out_tok_s_ex_capture"])
            print(f"\nscaling efficiency ({two}): {eff:.2f}")
    print("\n* tok/s: median over rounds, graph capture excluded; spread = (max - min) / median.")
    d = Path(args.results_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"replicas_{args.backend}.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())