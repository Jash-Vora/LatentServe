"""
Phase 6 — the vLLM baseline.

docs/methodology.md Phase 6. The rule that governs this whole module:

> Do **not** make "Beat vLLM" the objective. Investigate where a
> specialized runtime can approach or exceed a mature serving system and
> where it cannot. If vLLM wins, **explain why**.

## Why this file is mostly about controls

vLLM is not a drop-in peer. Left at its defaults it will, depending on
version, enable automatic prefix caching, chunked prefill, CUDA graphs
and a paged-attention kernel — and stop early on EOS. Every one of those
changes the work done. Comparing against that configuration produces a
number that looks fine and means nothing, which is worse than no number.

The controls below are therefore explicit and recorded in the result row,
so the comparison can be read as "LatentServe vs vLLM **with these
features on/off**" rather than as an unqualified verdict:

* **Same token ids, not same text.** Prompts are passed as
  `prompt_token_ids`, so no tokenizer difference can creep in.
* **`ignore_eos=True`.** Without it vLLM stops early on a random synthetic
  prompt and does less work; the throughput comparison then measures how
  soon each system gave up.
* **Greedy sampling, `temperature=0`,** matching LatentServe's argmax.
* **`enable_prefix_caching=False`.** LatentServe has no prefix cache until
  Phase 13. Leaving vLLM's on would credit it with a feature this project
  has not built yet, and the `mixed` workload shares no prefixes anyway —
  so it would be an unearned win on a workload where it cannot help.
* **fp16, single GPU, same `max_model_len`.**
* **Chunked prefill recorded, not forced.** vLLM's scheduler can mix
  prefill and decode in one step; LatentServe's prefill blocks decoding
  (Phase 4, measured as ~2.3 s inter-token stalls). That is a genuine
  architectural difference and one of the most interesting things to
  report, so it is recorded in `extra` rather than disabled.

## Expect to lose on decode, and know the number

Phase 3 measured LatentServe's paged gather at 2x resident KV per step
(~16 ms at batch 8), and Phase 4 showed it makes batching non-free.
vLLM has the paged-attention kernel Phase 11 is aiming at, which removes
that gather entirely. So **the LatentServe-vs-vLLM gap at large batch is
a direct estimate of what the Phase 11 kernel is worth** — which is a far
more useful framing than a win/loss.
"""

from __future__ import annotations

import time
from typing import Optional

VLLM_UNAVAILABLE_HINT = (
    "vLLM is not installed. Install it in a dedicated session: it pins its own "
    "torch build and will replace the one the rest of LatentServe runs against. "
    "On a T4 (sm75) check that the installed vLLM version still supports Turing; "
    "if not, pin an older release rather than changing the model or dtype."
)


def vllm_available() -> tuple[bool, str]:
    try:
        import vllm  # noqa: F401

        return True, getattr(vllm, "__version__", "unknown")
    except ImportError:
        return False, ""


class VLLMRunner:
    """Offline vLLM engine, configured to match LatentServe's conditions."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        max_model_len: int = 8448,
        max_num_seqs: int = 8,
        gpu_memory_utilization: float = 0.90,
        enable_prefix_caching: bool = False,
        seed: int = 0,
    ):
        ok, version = vllm_available()
        if not ok:
            raise RuntimeError(VLLM_UNAVAILABLE_HINT)
        self.version = version
        self.model_name = model_name
        self.config = {
            "dtype": dtype,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": enable_prefix_caching,
            "seed": seed,
        }

        from vllm import LLM

        kwargs = dict(
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            tensor_parallel_size=1,
        )
        # enable_prefix_caching has moved and been renamed across releases;
        # if this build does not accept it, fall back rather than fail, and
        # record that the control could not be applied.
        # disable_log_stats=False asks V1 to populate per-request metrics.
        # V0 filled RequestOutput.metrics unconditionally; V1 does not, and
        # without this TTFT comes back empty.
        try:
            self.llm = LLM(
                enable_prefix_caching=enable_prefix_caching,
                disable_log_stats=False,
                **kwargs,
            )
            self.config["prefix_caching_control"] = "applied"
        except TypeError:
            self.llm = LLM(**kwargs)
            self.config["prefix_caching_control"] = "unsupported_by_this_version"

    def describe(self) -> dict:
        """Everything that must travel with the numbers."""
        out = {
            "vllm_version": self.version,
            "vllm_detokenize": False,
            **{f"vllm_{k}": v for k, v in self.config.items()},
        }
        try:
            cfg = self.llm.llm_engine.vllm_config
            sched = getattr(cfg, "scheduler_config", None)
            out["vllm_chunked_prefill"] = getattr(sched, "enable_chunked_prefill", None)
            out["vllm_max_num_batched_tokens"] = getattr(sched, "max_num_batched_tokens", None)
            cache = getattr(cfg, "cache_config", None)
            out["vllm_block_size"] = getattr(cache, "block_size", None)
            out["vllm_num_gpu_blocks"] = getattr(cache, "num_gpu_blocks", None)
            model_cfg = getattr(cfg, "model_config", None)
            out["vllm_enforce_eager"] = getattr(model_cfg, "enforce_eager", None)
            comp = getattr(cfg, "compilation_config", None)
            out["vllm_cudagraph_mode"] = str(getattr(comp, "cudagraph_mode", None))
        except Exception:  # noqa: BLE001 - introspection is best-effort across versions
            out["vllm_config_introspection"] = "unavailable"
        return out

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        max_new_tokens: int | list[int],
        warmup: bool = False,
    ) -> dict:
        """Run a batch offline and return timings matched to our schema.

        Offline mode means every request is submitted at once — a burst.
        For an arrival-rate comparison use vLLM's async engine; the burst
        case is the one that maps cleanly onto Phase 4's batching sweep,
        and mixing the two would be its own fairness violation.
        """
        from vllm import SamplingParams

        counts = (
            [max_new_tokens] * len(prompt_token_ids)
            if isinstance(max_new_tokens, int)
            else max_new_tokens
        )
        params = [
            SamplingParams(
                max_tokens=n,
                min_tokens=n,       # forbid early stopping outright
                ignore_eos=True,    # random synthetic prompts hit EOS constantly
                temperature=0.0,    # greedy, matching LatentServe's argmax
                # LatentServe produces token ids and stops; it never
                # detokenizes. vLLM runs incremental detokenization per
                # token per request by default — real CPU work inside the
                # serving loop that the other system does not pay. Leaving
                # it on charges vLLM for output this comparison never
                # reads.
                detokenize=False,
                seed=None,
            )
            for n in counts
        ]

        t0 = time.perf_counter()
        outputs = self.llm.generate(
            prompts=[{"prompt_token_ids": ids} for ids in prompt_token_ids],
            sampling_params=params,
            # vLLM offline mode also defaults to disable_log_stats, so with
            # the bar off a multi-minute run prints nothing at all and is
            # indistinguishable from a hang.
            use_tqdm=not warmup,
        )
        wall = time.perf_counter() - t0
        if warmup:
            return {"wall_s": wall, "warmup": True}

        generated = sum(len(o.outputs[0].token_ids) for o in outputs)
        ttfts, e2es = [], []
        for o in outputs:
            m = getattr(o, "metrics", None)
            if m is None:
                # V1 may not populate per-request metrics at all. Leave the
                # lists empty and let the caller record the metric as
                # unavailable — never as zero. A missing measurement that
                # reads as 0 ms is indistinguishable from an excellent one.
                continue
            arrival = getattr(m, "arrival_time", None)
            first = getattr(m, "first_token_time", None)
            finished = getattr(m, "finished_time", None)
            if arrival is not None and first is not None:
                ttfts.append((first - arrival) * 1000)
            if arrival is not None and finished is not None:
                e2es.append((finished - arrival) * 1000)

        return {
            "wall_s": wall,
            "requests": len(outputs),
            "output_tokens": generated,
            "ttft_measurement": "vllm_request_metrics" if ttfts else "unavailable",
            "output_tokens_per_s": generated / wall if wall else 0.0,
            "requests_per_s": len(outputs) / wall if wall else 0.0,
            "ttft_ms": ttfts,
            "e2e_ms": e2es,
            # vLLM does not expose per-token gaps, so inter-token latency
            # is derived: (e2e - ttft) / (tokens - 1). That is a mean, not
            # a distribution, and it cannot show the stall behaviour
            # Phase 4 measured — so the ITL comparison is p50-only and
            # must be labelled as such rather than compared against
            # LatentServe's p99.
            "mean_itl_ms": [
                (e - t) / max(1, n - 1)
                for e, t, n in zip(
                    e2es, ttfts, [len(o.outputs[0].token_ids) for o in outputs]
                )
            ]
            if e2es and ttfts
            else [],
        }

    # vLLM claims `gpu_memory_utilization` of the card at construction
    # whether or not it needs it (9.5 GiB of KV cache and 42x concurrency
    # headroom on a T4, even with max_num_seqs=4), while LatentServe sizes
    # its pool to the sequences it will actually run. A peak-VRAM column
    # across the two would therefore compare a configuration policy, not
    # efficiency — the same class of asymmetry as the per-request-mean ITL.
    PEAK_VRAM_COMPARABLE = False

    def peak_vram_mb(self) -> Optional[float]:
        """Always None under V1.

        V1 runs the model in a separate worker process ("EngineCore
        pid=..." in the logs), so the parent's
        `torch.cuda.max_memory_allocated()` reports its own allocations —
        which are zero. Rather than publish that, report nothing and let
        `kv_cache_bytes()` carry the memory story, which is the honest
        comparison anyway since vLLM preallocates by policy.
        """
        return None

    def kv_cache_bytes(self) -> Optional[int]:
        """Size of the KV pool vLLM actually reserved, from its config.
        Comparable with LatentServe's `kv_allocated_mb` in a way that
        peak VRAM is not."""
        try:
            cache = self.llm.llm_engine.vllm_config.cache_config
            blocks = getattr(cache, "num_gpu_blocks", None)
            block_size = getattr(cache, "block_size", None)
            if blocks and block_size:
                # Qwen2.5-1.5B native GQA, fp16 — measured in Phase 2.
                return blocks * block_size * 28_672
        except Exception:  # noqa: BLE001
            pass
        return None
