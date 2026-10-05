"""
Phase 4 — the serving engine.

docs/methodology.md Phase 4: turn the inference backend into a minimal
serving engine with continuous batching.

    Step 1: A B C D
    Step 2: A B   D      <- C finished, its slot is reusable immediately
    Step 3: A B     E    <- E joined without waiting for A, B, D
    Step 4: A

Static batching runs a fixed group to completion, so the whole batch
waits for its slowest member and finished sequences keep their memory.
Continuous batching retires and admits every iteration.

## Why this could not have been built before Phase 3

A ragged decode batch — sequences at positions 512, 9000 and 17 in the
same step — cannot be represented by a contiguous cache at all: it has
one shared fill length. Paging is what makes each sequence's KV
independent, so Phase 3 is a prerequisite for Phase 4, not an
optimization of it. That dependency is worth stating plainly in the
report, because "paged KV" and "continuous batching" are usually
presented as two separate features.

## What one step does

    1. retire finished sequences, freeing their blocks at once
    2. ask the scheduler which waiting requests to admit
    3. prefill each admitted request into its own slot
    4. one ragged decode step across every resident sequence

Prefill runs one request at a time rather than batched. Phase 2 measured
prefill as compute-bound (O(S^2) attention, TTFT 650 ms at 4K against a
31 ms decode step), so batching prefills buys little, while mixing a
prefill into a decode batch requires a combined causal+padding mask that
Phase 3 explicitly deferred. Chunked mixed batching is the upgrade path
and belongs with the adaptive runtime in Phase 17.

The cost of that choice is honest and should be reported: a long prefill
blocks decoding for every resident sequence, so a 32K prompt arriving
mid-flight adds its whole 17 s prefill to everyone else's inter-token
latency. Expect it in the p99 TPOT.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from cache.paged_cache import PagedKVCache
from model.latentserve_qwen import LatentServeQwen
from runtime.request import RequestState, ServedRequest
from runtime.scheduler import Scheduler, build_scheduler


class ServingEngine:
    """Continuous-batching engine over a paged KV cache."""

    def __init__(
        self,
        model: LatentServeQwen,
        max_running: int = 16,
        max_seq_len: int = 8192,
        block_size: int = 16,
        num_blocks: Optional[int] = None,
        scheduler: str | Scheduler = "fifo",
        prefill_chunk_size: Optional[int] = 4096,
        eos_token_id: Optional[int] = None,
        on_retire: Optional[callable] = None,
        use_cuda_graphs: bool = False,
        sample_in_graph: bool = True,
        profile_loop: bool = False,
        kv_dtype: str = "fp16",
        policy=None,
    ):
        from collections import Counter

        self.model = model
        # Phase 17: the sparsity policy (runtime/policy.py), and tokens
        # generated per budget on the eager path.
        self.policy = policy
        self.ratio_tokens: Counter = Counter()
        self.max_running = max_running
        self.prefill_chunk_size = prefill_chunk_size
        self.eos_token_id = eos_token_id
        # Called as each request completes. A long sweep that prints only
        # when a whole configuration finishes is indistinguishable from a
        # hang, and "is it stuck?" is not a question a benchmark should
        # make you ask.
        self.on_retire = on_retire

        cache = model.allocate_cache(
            batch_size=max_running,
            max_seq_len=max_seq_len,
            paged=True,
            block_size=block_size,
            num_blocks=num_blocks,
            kv_dtype=kv_dtype,
        )
        # The INT8 cache mirrors PagedKVCache's interface (allocation, block
        # tables, admission) without inheriting from it.
        from cache.int8_paged_cache import Int8PagedKVCache

        assert isinstance(cache, (PagedKVCache, Int8PagedKVCache))
        if policy is not None and policy.may_sparsify:
            # Bounds kept from the first write, dense steps included: the
            # policy can switch a running batch to sparse at any step, and
            # bounds missing for earlier tokens would scramble its selection.
            if not hasattr(cache, "enable_page_bounds"):
                raise ValueError("a sparsity policy needs the fp16 paged cache")
            cache.enable_page_bounds()
        self.cache: PagedKVCache = cache

        # Phase 13: decode through captured CUDA graphs. Prefill stays
        # eager — it is compute-bound and its shape varies per request.
        #
        # Graphs need the Triton kernel path (the gather path reads a
        # slice whose length changes every step), so enabling them sets
        # it. A configuration a graph cannot capture falls back to eager
        # with a warning rather than failing: a missing graph is a
        # slowdown, not an error.
        #
        # Graphs are keyed by batch size, and continuous batching changes
        # the batch size as requests arrive and finish — so the first
        # step at each new size pays a capture. `warmup_graphs()` moves
        # that cost to start-up instead of onto whichever request
        # happens to trigger it.
        self.decoder = None
        # Phase 14b loop accounting. `profile_loop` adds per-phase timers to
        # every decode step; they cost a few perf_counter calls, which is
        # why they are off unless asked for.
        self.profile_loop = profile_loop
        self.loop_profile: dict = {}
        self.profiled_steps = 0
        # On-device token feedback: (batch key, next-token tensor, positions).
        # Valid only for a step that serves exactly the same requests in the
        # same order as the previous one.
        self._fed = None
        self.device_fed_steps = 0
        self.host_fed_steps = 0
        if use_cuda_graphs:
            import warnings

            from runtime.cuda_graph import GraphedDecoder, GraphUnsupported

            for layer in model.layers:
                layer.attn.attn_impl = "triton_paged"
            try:
                # Phase 14b: greedy token selection inside the graph. The
                # engine only ever samples greedily (argmax), so moving it
                # into the graph changes no output — it removes a launch
                # and a host round trip per step.
                self.decoder = GraphedDecoder(model, greedy=sample_in_graph, policy=policy)
            except GraphUnsupported as e:
                warnings.warn(f"CUDA graphs disabled, decoding eagerly: {e}", stacklevel=2)
                self.decoder = None

        self.scheduler = (
            scheduler if isinstance(scheduler, Scheduler)
            else build_scheduler(scheduler, max_running=max_running)
        )

        self.waiting: list[ServedRequest] = []
        self.running: list[ServedRequest] = []
        self.finished: list[ServedRequest] = []
        self._free_slots = list(range(max_running))

        # Instrumentation. Scheduler overhead is a headline Phase 4
        # metric, so it is measured rather than assumed negligible.
        self.step_count = 0
        self.decode_steps = 0
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.prefill_s = 0.0
        self.decode_s = 0.0
        self.overhead_s = 0.0
        self.batch_occupancy: list[int] = []

    # ------------------------------------------------------------------

    def add_request(self, request: ServedRequest) -> None:
        request.state = RequestState.QUEUED
        self.waiting.append(request)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------------

    def _retire(self, request: ServedRequest) -> None:
        request.finish_time = time.perf_counter()
        request.state = RequestState.FINISHED
        self.cache.free_sequence(request.slot)
        self._free_slots.append(request.slot)
        request.slot = None
        self.finished.append(request)
        if self.on_retire is not None:
            self.on_retire(request, self)

    def _blocks_promised(self) -> int:
        """Blocks running requests have yet to claim for the tokens they may
        still generate."""
        alloc = self.cache.allocator
        return sum(max(0, alloc.blocks_for_tokens(r.prompt_len + r.max_new_tokens)
                       - len(self.cache.tables[r.slot].blocks)) for r in self.running)

    def _fits(self, request: ServedRequest) -> bool:
        """Admit only if the pool can hold this request's prompt *and* every
        token it may generate, after what running requests are still owed.

        Checking the prompt alone over-admits: two requests whose prompts fit
        exhausted a 12-block pool during decode (OutOfBlocks mid-run). Without
        preemption, reserving the worst case is the rule that cannot crash.
        """
        need = self.cache.allocator.blocks_for_tokens(request.prompt_len + request.max_new_tokens)
        return need <= self.cache.allocator.num_free - self._blocks_promised()

    def _admit_and_prefill(self) -> None:
        admitted = self.scheduler.select(
            self.waiting,
            num_running=len(self.running),
            can_admit=self.cache.can_admit,
        )
        for request in admitted:
            if not self._free_slots:
                break
            # Re-check: earlier admissions in this same iteration consumed
            # blocks the scheduler's snapshot did not know about — and the
            # request must fit with its outputs, not just its prompt.
            if not self.cache.can_admit(request.prompt_len) or not self._fits(request):
                break
            slot = self._free_slots.pop(0)
            request.slot = slot
            request.state = RequestState.PREFILL
            request.scheduled_time = time.perf_counter()
            self.waiting.remove(request)

            ids = torch.tensor(
                [request.prompt_ids], dtype=torch.long, device=self.model.device
            )
            t0 = time.perf_counter()
            logits = self.model.prefill_slot(ids, slot, chunk_size=self.prefill_chunk_size)
            if self.model.device.type == "cuda":
                torch.cuda.synchronize()
            self.prefill_s += time.perf_counter() - t0
            self.prefill_tokens += request.prompt_len

            next_id = int(logits[0, -1].argmax().item())
            request.first_token_time = time.perf_counter()
            request.last_token_time = request.first_token_time
            request.output_ids.append(next_id)
            request.state = RequestState.DECODING
            self.running.append(request)
            if self._should_stop(request):
                self._retire(request)

    def _should_stop(self, request: ServedRequest) -> bool:
        if request.generated >= request.max_new_tokens:
            return True
        return self.eos_token_id is not None and request.output_ids[-1] == self.eos_token_id

    def _batch_key(self) -> tuple:
        """Identity of this step's batch: which requests, in which order.

        Keyed on request ids, not slots. A request that finishes frees its
        slot, and a new request admitted into that slot leaves the slot
        list unchanged — so a slot-keyed check would feed the newcomer the
        previous occupant's last token.
        """
        return tuple(r.request_id for r in self.running)

    def _prof(self, name: str, ms: float) -> None:
        if self.profile_loop:
            self.loop_profile[name] = self.loop_profile.get(name, 0.0) + ms

    def _decode(self) -> None:
        if not self.running:
            return
        clock = time.perf_counter
        t_start = clock()
        device = self.model.device
        slots = [r.slot for r in self.running]
        key = self._batch_key()
        greedy_graph = self.decoder is not None and self.decoder.greedy

        if greedy_graph and self._fed is not None and self._fed[0] == key:
            # Same requests, same order: last step's chosen tokens are still
            # on the GPU in the graph's output buffer, and every position is
            # one further on. Nothing to build on the host, nothing to copy up.
            tokens, positions = self._fed[1], self._fed[2] + 1
            self.device_fed_steps += 1
        else:
            tokens = torch.tensor(
                [[r.output_ids[-1]] for r in self.running], dtype=torch.long, device=device
            )
            # Absolute position of the token about to be processed: prompt
            # plus everything generated so far, minus the one being fed in.
            positions = torch.tensor(
                [[r.prompt_len + r.generated - 1] for r in self.running],
                dtype=torch.long, device=device,
            )
            self.host_fed_steps += 1
        t_inputs = clock()

        if greedy_graph:
            next_dev = self.decoder.step_greedy(tokens, positions, slots)
            t_launch = clock()
            # One tiny device-to-host copy, which is also the wait for the GPU.
            next_ids = next_dev.view(-1).tolist()
            t_gpu = clock()
            t_sample = t_gpu
            self._fed = (key, next_dev, positions)
        else:
            if self.decoder is not None:
                # The returned tensor is the graph's static output, overwritten
                # by the next replay. Reading the argmax below (with .tolist(),
                # which copies to host) before the next step is what makes
                # that safe.
                logits = self.decoder.step(tokens, positions, slots)
            else:
                if self.policy is not None:
                    lens = [self.cache.tables[s].length for s in slots]
                    ratio = self.policy.choose(len(slots), sum(lens) / max(1, len(lens)))
                    if ratio != getattr(self.model, "sparse_ratio", None):
                        self.model.set_sparse(ratio)
                    self.ratio_tokens[ratio] += len(slots)
                logits = self.model.decode_step_ragged(tokens, positions, slots)
            t_launch = clock()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_gpu = clock()
            next_ids = logits[:, -1, :].argmax(dim=-1).tolist()
            t_sample = clock()
            self._fed = None

        step_ms = (t_gpu - t_inputs) * 1000
        self.decode_s += step_ms / 1000
        self.decode_steps += 1
        self.batch_occupancy.append(len(self.running))

        now = clock()
        still: list[ServedRequest] = []
        for request, token in zip(self.running, next_ids):
            request.output_ids.append(int(token))
            # Wall gap since this request's previous token. Includes any
            # prefill the engine ran in between, which is exactly the
            # cost the step timer above cannot see.
            request.decode_step_ms.append((now - (request.last_token_time or now)) * 1000)
            request.last_token_time = now
            self.decode_tokens += 1
            if self._should_stop(request):
                self._retire(request)
            else:
                still.append(request)
        self.running = still
        t_end = clock()

        if self.profile_loop:
            host_in_decoder = self.decoder.last_host_ms if self.decoder is not None else 0.0
            self._prof("inputs", (t_inputs - t_start) * 1000)
            self._prof("decoder_host", host_in_decoder)
            self._prof("gpu_wait", (t_gpu - t_inputs) * 1000 - host_in_decoder)
            self._prof("sample", (t_sample - t_gpu) * 1000)
            self._prof("bookkeeping", (t_end - t_sample) * 1000)
            self.profiled_steps += 1

    def step(self) -> None:
        """One engine iteration: admit, prefill, decode, retire."""
        t0 = time.perf_counter()
        gpu_before = self.prefill_s + self.decode_s
        self.step_count += 1
        t_admit = time.perf_counter()
        prefill_before = self.prefill_s
        self._admit_and_prefill()
        if self.profile_loop:
            admit_ms = (time.perf_counter() - t_admit) * 1000
            self._prof("schedule", admit_ms - (self.prefill_s - prefill_before) * 1000)
        self._decode()
        # Everything in the step that was not GPU work: scheduling,
        # block-table updates, Python bookkeeping. Reported as its own
        # number because methodology Phase 4 asks for scheduler overhead
        # explicitly, and "negligible" is a claim, not a given.
        elapsed = time.perf_counter() - t0
        self.overhead_s += max(0.0, elapsed - (self.prefill_s + self.decode_s - gpu_before))

    def run(self, max_steps: int = 1_000_000) -> list[ServedRequest]:
        """Drain the queue. Returns finished requests in completion order."""
        steps = 0
        while self.has_work and steps < max_steps:
            before = (len(self.waiting), len(self.running))
            self.step()
            steps += 1
            if not self.running and self.waiting and before == (len(self.waiting), 0):
                # Nothing admitted and nothing running: the smallest
                # waiting request cannot fit even in an empty pool. Fail
                # loudly rather than spin — a silent hang here would be
                # indistinguishable from a slow workload.
                shortest = min(r.prompt_len for r in self.waiting)
                raise RuntimeError(
                    f"deadlock: {len(self.waiting)} requests waiting, shortest prompt "
                    f"{shortest} tokens, but the block pool cannot seat it "
                    f"({self.cache.allocator.num_free} blocks free)"
                )
        return self.finished

    # ------------------------------------------------------------------

    @torch.no_grad()
    def warmup_graphs(self, batch_sizes, context_length: int) -> int:
        """Capture graphs for the given batch sizes before serving starts.

        Uses throwaway sequences in real cache slots: the capture writes
        their KV, and the cache is reset afterwards. Reset clears contents
        and keeps storage, so the captured addresses stay valid — that is
        the property `test_graph_survives_a_cache_reset` checks.

        Returns the number of graphs captured. Call it before
        `add_request`, since it resets the cache.
        """
        if self.decoder is None:
            return 0
        if self.running or self.waiting:
            raise RuntimeError("warmup_graphs() resets the cache; call it before serving")
        before = self.decoder.captures
        device = self.model.device
        for b in batch_sizes:
            if b > self.max_running:
                continue
            self.cache.reset()
            slots = list(range(b))
            self.cache.advance(context_length - 1, slots=slots)
            tokens = torch.zeros(b, 1, dtype=torch.long, device=device)
            positions = torch.full((b, 1), context_length - 1, dtype=torch.long, device=device)
            self.decoder.step(tokens, positions, slots)
        self.cache.reset()
        self._free_slots = list(range(self.max_running))
        self._fed = None
        return self.decoder.captures - before

    def stats(self) -> dict:
        wall = self.prefill_s + self.decode_s
        mean_batch = (
            sum(self.batch_occupancy) / len(self.batch_occupancy) if self.batch_occupancy else 0.0
        )
        return {
            "engine_steps": self.step_count,
            "decode_steps": self.decode_steps,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "prefill_s": self.prefill_s,
            "decode_s": self.decode_s,
            "gpu_s": wall,
            "decode_tokens_per_s": self.decode_tokens / self.decode_s if self.decode_s else 0.0,
            "mean_batch_occupancy": mean_batch,
            "runtime_overhead_s": self.overhead_s,
            "runtime_overhead_pct": (
                self.overhead_s / (wall + self.overhead_s) * 100 if wall + self.overhead_s else 0.0
            ),
            "runtime_overhead_ms_per_step": (
                self.overhead_s * 1000 / self.step_count if self.step_count else 0.0
            ),
            # Fraction of the theoretical maximum batch actually used.
            # Static batching's loss shows up here as sequences that
            # finished early but still held a slot.
            "batch_efficiency": mean_batch / self.max_running if self.max_running else 0.0,
            "max_running": self.max_running,
            **self.scheduler.stats(),
            "device_fed_steps": self.device_fed_steps,
            "host_fed_steps": self.host_fed_steps,
            **({f"loop_{k}_ms_per_step": v / max(1, self.profiled_steps)
                for k, v in self.loop_profile.items()} if self.profile_loop else {}),
            **({f"graph_{k}": v for k, v in self.decoder.stats().items() if k != "keys"}
               if self.decoder is not None else {"graph_captures": 0}),
            **self.cache.stats(),
        }


def run_static_batching(
    model: LatentServeQwen,
    requests: list[ServedRequest],
    batch_size: int,
    max_seq_len: int,
    block_size: int = 16,
    prefill_chunk_size: Optional[int] = 4096,
    on_retire: Optional[callable] = None,
) -> tuple[list[ServedRequest], dict]:
    """Baseline: fixed groups run to completion.

    The comparison continuous batching has to beat. Two costs are
    structural here, and both are visible in the metrics rather than
    argued for: every request in a group waits for the group's slowest
    member before its slot can be reused, and a request that arrives one
    step after a group starts waits for the whole group.

    Implemented on the same engine primitives so the difference is the
    batching policy alone, not two different code paths.
    """
    finished: list[ServedRequest] = []
    total = {"prefill_s": 0.0, "decode_s": 0.0, "decode_tokens": 0, "occupancy": []}

    for start in range(0, len(requests), batch_size):
        group = requests[start : start + batch_size]
        engine = ServingEngine(
            model,
            max_running=batch_size,
            max_seq_len=max_seq_len,
            block_size=block_size,
            prefill_chunk_size=prefill_chunk_size,
            on_retire=on_retire,
        )
        for r in group:
            engine.add_request(r)
        # Admit the whole group, then decode until all are done — no new
        # request may join mid-flight, which is the definition of static.
        engine.run()
        finished.extend(engine.finished)
        total["prefill_s"] += engine.prefill_s
        total["decode_s"] += engine.decode_s
        total["decode_tokens"] += engine.decode_tokens
        total["occupancy"].extend(engine.batch_occupancy)

    occupancy = total.pop("occupancy")
    total["mean_batch_occupancy"] = sum(occupancy) / len(occupancy) if occupancy else 0.0
    total["batch_efficiency"] = total["mean_batch_occupancy"] / batch_size
    total["gpu_s"] = total["prefill_s"] + total["decode_s"]
    total["decode_tokens_per_s"] = (
        total["decode_tokens"] / total["decode_s"] if total["decode_s"] else 0.0
    )
    return finished, total
