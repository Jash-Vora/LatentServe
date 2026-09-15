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
        try:
            self.llm = LLM(enable_prefix_caching=enable_prefix_caching, **kwargs)
            self.config["prefix_caching_control"] = "applied"
        except TypeError:
            self.llm = LLM(**kwargs)
            self.config["prefix_caching_control"] = "unsupported_by_this_version"

    def describe(self) -> dict:
        """Everything that must travel with the numbers."""
        out = {"vllm_version": self.version, **{f"vllm_{k}": v for k, v in self.config.items()}}
        try:
            cfg = self.llm.llm_engine.vllm_config
            sched = getattr(cfg, "scheduler_config", None)
            out["vllm_chunked_prefill"] = getattr(sched, "enable_chunked_prefill", None)
            out["vllm_max_num_batched_tokens"] = getattr(sched, "max_num_batched_tokens", None)
            cache = getattr(cfg, "cache_config", None)
            out["vllm_block_size"] = getattr(cache, "block_size", None)
            out["vllm_num_gpu_blocks"] = getattr(cache, "num_gpu_blocks", None)
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
                seed=None,
            )
            for n in counts
        ]

        t0 = time.perf_counter()
        outputs = self.llm.generate(
            prompts=[{"prompt_token_ids": ids} for ids in prompt_token_ids],
            sampling_params=params,
            use_tqdm=False,
        )
        wall = time.perf_counter() - t0
        if warmup:
            return {"wall_s": wall, "warmup": True}

        generated = sum(len(o.outputs[0].token_ids) for o in outputs)
        ttfts, e2es = [], []
        for o in outputs:
            m = getattr(o, "metrics", None)
            if m is None:
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

    def peak_vram_mb(self) -> Optional[float]:
        try:
            import torch

            return torch.cuda.max_memory_allocated() / 1024 / 1024
        except Exception:  # noqa: BLE001
            return None
