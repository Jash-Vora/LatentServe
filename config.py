"""
Experiment configuration system.

Every LatentServe experiment is defined by a YAML file matching the
schema below, never by hand-editing code. Load with `load_config(path)`.

Example YAML (see configs/baseline_gqa4.yaml):

    model:
      name: Qwen/Qwen2.5-1.5B-Instruct
      dtype: fp16

    attention:
      type: gqa
      latent_dim: null
      rope_dim: 64

    runtime:
      batch_size: 8
      block_size: 16

    generation:
      input_tokens: 32768
      output_tokens: 256

    hardware:
      devices: [0]

Note: LatentServe uses Qwen2.5-1.5B-Instruct as a *fixed* real model
substrate (see docs/methodology.md, "Model Strategy"). ModelConfig
therefore does not expose architecture hyperparameters (layers, heads,
hidden_dim, ...) for the user to set — there is no scratch model to
configure. Those values are read off the loaded Hugging Face model at
runtime (that discovery is Phase 2's job), not declared here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    # Fixed real model substrate — see docs/methodology.md "Model Strategy".
    # Intentionally does NOT expose layers/hidden_dim/heads/kv_heads: this is
    # a Hugging Face checkpoint identifier, not a from-scratch architecture
    # spec. Actual head/layer/kv-head counts are introspected from the
    # loaded HF config in Phase 2, not declared here.
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: Optional[str] = None
    dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
    trust_remote_code: bool = False
    max_position_embeddings: int = 131072


class AttentionConfig(BaseModel):
    type: Literal["mha", "gqa", "mla", "sparse", "mla_sparse"] = "gqa"
    latent_dim: Optional[int] = None  # required when type in {mla, mla_sparse}
    rope_dim: int = 64
    sparsity: Optional[float] = None  # fraction of tokens KEPT, e.g. 0.25

    # Phase 2. Qwen2.5-1.5B-Instruct's KV head count is fixed by its
    # weights (12 query heads / 2 KV heads), so "vary GQA configuration"
    # has to mean varying what the *cache* stores, not retraining:
    #   native  — 2 KV heads, the real model
    #   mha_sim — 12 KV heads (one per query head); identical math and
    #             identical logits, 6x the KV memory and traffic, so any
    #             latency delta is a pure memory-system effect
    #   mqa_sim — 1 mean-pooled KV head; changes the numerics, so it is a
    #             bandwidth probe only, never a quality claim
    # See cache/kv_cache.py.
    kv_heads_mode: Literal["native", "mha_sim", "mqa_sim"] = "native"

    # Phase 2 reference/kernel-development path. "math" materialises the
    # full score matrix and is the numerical reference Phase 11's Triton
    # and CUDA kernels get checked against; "sdpa" is the baseline used
    # for every benchmark.
    impl: Literal["sdpa", "math"] = "sdpa"

    @model_validator(mode="after")
    def _validate_by_type(self) -> "AttentionConfig":
        if self.type in ("mla", "mla_sparse") and self.latent_dim is None:
            raise ValueError(f"attention.type={self.type} requires latent_dim to be set")
        if self.type in ("sparse", "mla_sparse") and self.sparsity is None:
            raise ValueError(f"attention.type={self.type} requires sparsity to be set")
        return self


class RuntimeConfig(BaseModel):
    batch_size: int = 8
    block_size: int = 16  # paged-cache block size, tokens/block
    # Phase 2: split prefill into blocks of this many tokens. Attention
    # cost is unchanged (each block still attends over the whole prefix);
    # peak activation memory drops from O(context) to O(chunk), which is
    # what makes long prompts reachable on a 16 GB T4. None = one block.
    prefill_chunk_size: Optional[int] = None
    scheduler: Literal["fifo", "length_aware", "fair", "slo_aware"] = "fifo"
    continuous_batching: bool = False
    prefix_caching: bool = False


class GenerationConfig(BaseModel):
    input_tokens: int = 4096
    output_tokens: int = 256
    num_requests: int = 1
    seed: int = 0
    sampling: Literal["greedy", "sample"] = "greedy"


class HardwareConfig(BaseModel):
    devices: list[int] = Field(default_factory=lambda: [0])
    # dtype lives on ModelConfig (it's a property of the weights being
    # loaded, e.g. Qwen2.5-1.5B-Instruct in fp16) — not duplicated here.


class ExperimentConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    # Free-form label used to group/tag results, e.g. "phase2_gqa8"
    tag: str = "unnamed_experiment"


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig(**raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False)


if __name__ == "__main__":
    import sys

    cfg = load_config(sys.argv[1])
    print(cfg.model_dump_json(indent=2))
