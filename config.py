"""
Experiment configuration system.

Every LatentServe experiment is defined by a YAML file matching the
schema below, never by hand-editing code. Load with `load_config(path)`.

Example YAML (see configs/baseline_gqa.yaml):

    model:
      name: research-transformer
      layers: 24
      hidden_dim: 2048
      heads: 16
      kv_heads: 4

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
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    name: str = "research-transformer"
    layers: int = 24
    hidden_dim: int = 2048
    heads: int = 16
    kv_heads: int = 4
    head_dim: Optional[int] = None  # derived from hidden_dim / heads if unset
    vocab_size: int = 32000
    max_position_embeddings: int = 131072

    @model_validator(mode="after")
    def _derive_head_dim(self) -> "ModelConfig":
        if self.head_dim is None:
            if self.hidden_dim % self.heads != 0:
                raise ValueError(
                    f"hidden_dim ({self.hidden_dim}) must be divisible by "
                    f"heads ({self.heads}) when head_dim is not set explicitly"
                )
            self.head_dim = self.hidden_dim // self.heads
        if self.heads % self.kv_heads != 0:
            raise ValueError(
                f"heads ({self.heads}) must be divisible by kv_heads "
                f"({self.kv_heads}) for GQA grouping"
            )
        return self


class AttentionConfig(BaseModel):
    type: Literal["mha", "gqa", "mla", "sparse", "mla_sparse"] = "gqa"
    latent_dim: Optional[int] = None  # required when type in {mla, mla_sparse}
    rope_dim: int = 64
    sparsity: Optional[float] = None  # fraction of tokens KEPT, e.g. 0.25

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
    dtype: Literal["fp16", "bf16", "fp32"] = "fp16"


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
