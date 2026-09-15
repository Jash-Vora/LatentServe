"""
Phase 5 harness tests — statistics and fairness guards.

The harness is the instrument every later claim is measured with, so its
own failure modes need tests more than the runners do. Phase 4 lost two
predictions to metric bugs that no unit test would have caught because
no unit test existed; these are the ones that would have.

No GPU, no model, no vLLM.
"""

from __future__ import annotations

import pytest

from benchmarks.harness import (
    UnfairComparison,
    assert_comparable,
    bootstrap_ci,
    check_environment,
    coefficient_of_variation,
    comparison_mismatches,
    percentile,
    repeatability_report,
    speedup,
    summarise_latency,
)


def test_percentile_is_nearest_rank_everywhere():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(xs, 0.0) == 1
    assert percentile(xs, 1.0) == 10
    assert percentile(xs, 0.5) == 5  # nearest-rank: 50% of the sample is <= 5
    assert percentile(xs, 0.55) == 6  # no interpolation; always an observed value
    assert percentile([], 0.5) is None


def test_percentile_has_no_rounding_mode_surprises():
    """The Phase 1-4 formula used round(), which is banker's rounding:
    round(4.5) == 4 but round(5.5) == 6. Nearest-rank has no such tilt."""
    for n in range(2, 40):
        xs = list(range(n))
        assert percentile(xs, 0.5) == xs[max(0, (n + 1) // 2 - 1)]


def test_percentile_ignores_none():
    assert percentile([1.0, None, 3.0], 1.0) == 3.0


def test_bootstrap_ci_refuses_tiny_samples():
    """Two points cannot support an interval, and returning one anyway
    would look more authoritative than the data."""
    assert bootstrap_ci([1.0, 2.0]) == (None, None)
    low, high = bootstrap_ci([10.0, 11.0, 10.5, 10.2, 10.8])
    assert low is not None and low <= 10.5 <= high


def test_coefficient_of_variation_is_the_noise_floor():
    assert coefficient_of_variation([10.0, 10.0, 10.0]) == 0.0
    assert coefficient_of_variation([9.0, 10.0, 11.0]) > 0


def test_summarise_latency_detects_the_stall_population():
    """Inter-token latency under prefill blocking is bimodal. The summary
    must report how often the second mode fires, because p99 lands inside
    or outside it depending on the stall rate."""
    steps = [30.0] * 990 + [2300.0] * 10
    s = summarise_latency(steps)
    assert s.p50 == 30.0
    assert s.max == 2300.0
    assert s.stall_rate == pytest.approx(0.01)
    assert s.stalled_ms_total == pytest.approx(23000.0)


def test_p99_sits_on_the_boundary_when_stalls_are_one_percent():
    """Exactly the Phase 4 situation: with ~1% stalls, p99 is unstable
    while stall_rate is not. This test documents the reason the report
    must not compare policies on p99 inter-token latency."""
    just_under = summarise_latency([30.0] * 992 + [2300.0] * 8)
    just_over = summarise_latency([30.0] * 988 + [2300.0] * 12)
    assert just_under.p99 == 30.0 and just_over.p99 == 2300.0   # flips
    assert just_under.stall_rate < just_over.stall_rate          # moves smoothly


def test_mismatch_detection_reads_both_top_level_and_extra():
    a = {"system": "a", "model": "qwen", "batch_size": 4, "extra": {"dtype": "fp16"}}
    b = {"system": "b", "model": "qwen", "batch_size": 4, "extra": {"dtype": "bf16"}}
    assert comparison_mismatches(a, b) == {"dtype": ("fp16", "bf16")}


def test_key_absent_from_both_is_not_a_mismatch():
    a = {"system": "a", "model": "qwen"}
    b = {"system": "b", "model": "qwen"}
    assert comparison_mismatches(a, b) == {}


def test_key_present_in_only_one_side_is_a_mismatch():
    """Usually means one side quietly defaulted — the exact failure this
    guard exists to catch."""
    a = {"system": "a", "model": "qwen", "extra": {"sampling": "greedy"}}
    b = {"system": "b", "model": "qwen"}
    assert "sampling" in comparison_mismatches(a, b)


def test_assert_comparable_raises_on_different_output_lengths():
    a = {"system": "latentserve", "model": "qwen", "output_length": 256}
    b = {"system": "vllm", "model": "qwen", "output_length": 128}
    with pytest.raises(UnfairComparison, match="output_length"):
        assert_comparable(a, b)


def test_speedup_refuses_to_compute_across_an_unfair_pair():
    """A plausible wrong number is worse than a crash."""
    a = {"system": "a", "model": "qwen", "batch_size": 8, "throughput_tokens_sec": 100.0}
    b = {"system": "b", "model": "qwen", "batch_size": 4, "throughput_tokens_sec": 50.0}
    with pytest.raises(UnfairComparison):
        speedup(a, b, "throughput_tokens_sec")


def test_speedup_computes_on_a_matched_pair():
    a = {"system": "a", "model": "qwen", "batch_size": 8, "throughput_tokens_sec": 100.0}
    b = {"system": "b", "model": "qwen", "batch_size": 8, "throughput_tokens_sec": 50.0}
    assert speedup(a, b, "throughput_tokens_sec") == 2.0


def test_environment_drift_is_surfaced():
    rows = [
        {"git_commit": "aaa", "lib_versions": {"torch": "2.6"}, "gpu_info": [{"name": "Tesla T4"}]},
        {"git_commit": "bbb", "lib_versions": {"torch": "2.6"}, "gpu_info": [{"name": "Tesla T4"}]},
    ]
    assert "git_commit" in check_environment(rows)
    assert "lib_versions" not in check_environment(rows)


def test_repeatability_report_quantifies_run_to_run_noise():
    """Phase 4's scheduler sweep reproduced TTFT within 1.2% across two
    runs; that number is what licenses calling a 3% difference real."""
    trials = [{"throughput_tokens_sec": v} for v in (65.12, 65.32, 65.20, 65.05, 65.40)]
    report = repeatability_report(trials, "throughput_tokens_sec")
    assert report["n"] == 5
    assert report["cv"] < 0.01
