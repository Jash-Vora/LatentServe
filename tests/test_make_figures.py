"""The figure script parses the saved tables; these pin the numbers the README quotes."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from benchmarks import make_figures as mf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "sweep_stage1_results.md"
PHASE12 = ROOT / "docs" / "phase12_kernel_findings.md"
PHASE13 = ROOT / "docs" / "phase13_prefix.md"


def test_decode_table_parses_including_batches_that_did_not_fit():
    dec = mf.parse_decode(RESULTS.read_text())
    assert dec[(32, 8192, "ls-dense")] == 58.94 and dec[(32, 8192, "vllm")] == 195.35
    assert dec[(32, 16384, "ls-dense")] is None and dec[(32, 16384, "ls-int8")] == 114.69
    assert len({k[:2] for k in dec}) == 20                                  # 5 batches x 4 contexts


def test_headline_ratios_match_the_readme():
    text = RESULTS.read_text()
    dec, ttft = mf.parse_decode(text), mf.parse_ttft(text)
    assert dec[(32, 8192, "vllm")] / dec[(32, 8192, "ls-dense")] == pytest.approx(3.31, abs=0.01)
    assert ttft[(32768, "vllm")] / ttft[(32768, "ls-dense")] == pytest.approx(9.56, abs=0.01)
    assert ttft[(1024, "vllm")] / ttft[(1024, "ls-dense")] == pytest.approx(2.07, abs=0.01)


def test_load_curves_parse_with_their_capacities():
    load = mf.parse_load(RESULTS.read_text())
    assert load["vllm"]["capacity"] == 0.496 and load["ls-dense"]["capacity"] == 1.576
    assert len(load["vllm"]["points"]) == 9 and load["ls-dense"]["points"][0] == (0.158, 315.0, 996.0)


def test_notes_with_numbers_parse():
    assert mf.parse_kernel_bandwidth(PHASE12.read_text()) == {"triton": 69, "loads_only": 243, "cuda": 194}
    assert mf.parse_prefix(PHASE13.read_text())["shared"] == (318.8, 51.1)


def test_every_figure_is_drawn(tmp_path):
    files = mf.make_all(RESULTS, tmp_path, PHASE12, PHASE13)
    assert len(files) == 7 and all(f.exists() and f.stat().st_size > 10_000 for f in files)
