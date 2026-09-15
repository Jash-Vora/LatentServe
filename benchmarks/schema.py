"""
Structured result schema for every benchmark run.

Rule (see docs/methodology.md, "Reproducibility"): no manually typed
benchmark numbers. Every run goes through `ResultWriter.write(...)`,
which stamps reproducibility metadata (git commit, library versions,
GPU model, seed) automatically and appends one JSON line to
results/raw/<tag>.jsonl.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _lib_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for lib in ("torch", "triton", "vllm"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = None
    return versions


def _cuda_driver_version() -> Optional[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.version.cuda
    except Exception:
        pass
    return None


def _gpu_info() -> list[dict]:
    gpus = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_vram_mb": round(props.total_memory / 1024 / 1024, 1),
                    }
                )
    except Exception:
        pass
    return gpus


@dataclass
class BenchmarkResult:
    # --- experiment identity ---
    system: str  # e.g. "latentserve", "vllm", "pytorch_baseline"
    tag: str  # experiment tag from the config, e.g. "phase2_gqa4"
    attention: str  # "mha" | "gqa" | "mla" | "sparse" | "mla_sparse"
    model: str

    # --- workload ---
    batch_size: int
    context_length: int
    output_length: int
    num_gpus: int

    # --- core metrics (fill in 0.0 / None if not measured this run) ---
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    throughput_tokens_sec: float = 0.0
    requests_per_sec: float = 0.0

    # --- memory ---
    peak_vram_mb: float = 0.0
    kv_cache_mb: float = 0.0

    # --- percentiles (optional, filled by repeated-run aggregation) ---
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    tpot_p50_ms: Optional[float] = None
    tpot_p95_ms: Optional[float] = None
    tpot_p99_ms: Optional[float] = None

    # --- quality (optional) ---
    perplexity: Optional[float] = None
    retrieval_accuracy: Optional[float] = None

    # --- reproducibility (auto-filled, do not set manually) ---
    git_commit: str = field(default_factory=_git_commit)
    seed: int = 0
    gpu_info: list = field(default_factory=_gpu_info)
    cuda_version: Optional[str] = field(default_factory=_cuda_driver_version)
    lib_versions: dict = field(default_factory=_lib_versions)
    hostname: str = field(default_factory=platform.node)
    timestamp_utc: str = ""

    def __post_init__(self):
        if not self.timestamp_utc:
            from datetime import datetime, timezone

            self.timestamp_utc = datetime.now(timezone.utc).isoformat()


class ResultWriter:
    """Appends BenchmarkResult rows as JSON Lines under results/raw/."""

    def __init__(self, results_dir: str | Path = "results/raw"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def write(self, result: BenchmarkResult) -> Path:
        out_path = self.results_dir / f"{result.tag}.jsonl"
        with out_path.open("a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
        return out_path


if __name__ == "__main__":
    # Smoke test: write one dummy row.
    r = BenchmarkResult(
        system="latentserve",
        tag="smoke_test",
        attention="gqa",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        batch_size=1,
        context_length=1024,
        output_length=32,
        num_gpus=1,
    )
    path = ResultWriter().write(r)
    print(f"Wrote smoke-test result to {path}")
