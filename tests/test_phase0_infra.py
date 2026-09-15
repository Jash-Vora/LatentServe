"""
Phase 0 sanity tests. These test the experimental infrastructure itself
(config loading, result schema/writer) — not the model, which doesn't
exist yet. Run with: pytest tests/test_phase0_infra.py -v
"""

import json
from pathlib import Path

import pytest

from config import load_config, save_config, ExperimentConfig
from benchmarks.schema import BenchmarkResult, ResultWriter


def test_load_baseline_gqa_config():
    cfg = load_config("configs/baseline_gqa4.yaml")
    assert cfg.model.heads == 16
    assert cfg.model.kv_heads == 4
    assert cfg.model.head_dim == 128  # derived: 2048 / 16
    assert cfg.attention.type == "gqa"


def test_load_mla_config_requires_latent_dim():
    cfg = load_config("configs/mla_512.yaml")
    assert cfg.attention.type == "mla"
    assert cfg.attention.latent_dim == 512


def test_mla_config_without_latent_dim_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(attention={"type": "mla"})


def test_gqa_grouping_validated():
    # heads=16 not divisible by kv_heads=3 -> invalid grouping
    with pytest.raises(ValueError):
        ExperimentConfig(model={"heads": 16, "kv_heads": 3, "hidden_dim": 2048})


def test_config_roundtrip(tmp_path):
    cfg = load_config("configs/baseline_gqa4.yaml")
    out = tmp_path / "roundtrip.yaml"
    save_config(cfg, out)
    reloaded = load_config(out)
    assert reloaded == cfg


def test_result_writer_produces_valid_jsonl(tmp_path):
    writer = ResultWriter(results_dir=tmp_path)
    result = BenchmarkResult(
        system="latentserve",
        tag="unit_test",
        attention="gqa",
        model="research-1b",
        batch_size=1,
        context_length=1024,
        output_length=32,
        num_gpus=1,
        ttft_ms=12.3,
        tpot_ms=4.5,
    )
    path = writer.write(result)
    assert path.exists()

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["ttft_ms"] == 12.3
    assert "git_commit" in row
    assert "lib_versions" in row
    assert "timestamp_utc" in row
