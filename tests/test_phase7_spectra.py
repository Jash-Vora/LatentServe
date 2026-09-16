"""
Phase 7.1 tests — spectral analysis.

The analysis is cheap to run and expensive to misread, so what needs
testing is the arithmetic that turns eigenvalues into a verdict: a
rank-for-energy that is off by one, or a break-even that ignores the
positional dimensions, changes whether six weeks of work looks
worthwhile.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from compression.spectra import (  # noqa: E402
    SpectrumAccumulator,
    SpectrumReport,
    compression_ratio,
    effective_rank,
    energy_curve,
    gqa_numbers_per_token_per_layer,
    mla_break_even,
    rank_for_energy,
    weight_spectrum,
)


def test_gram_accumulation_matches_direct_svd():
    """Streaming A^T A must give the same spectrum as one SVD of A.
    This is what lets the analysis run over 32K tokens in 2 MB."""
    torch.manual_seed(0)
    a = torch.randn(500, 16, dtype=torch.float64)
    acc = SpectrumAccumulator(16)
    for chunk in a.split(97):          # deliberately uneven chunks
        acc.update(chunk)
    direct = torch.linalg.svdvals(a) ** 2
    torch.testing.assert_close(acc.eigenvalues(), direct, rtol=1e-8, atol=1e-8)


def test_accumulator_is_order_independent():
    torch.manual_seed(1)
    a = torch.randn(200, 8, dtype=torch.float64)
    first, second = SpectrumAccumulator(8), SpectrumAccumulator(8)
    first.update(a)
    for row in a.flip(0).split(1):
        second.update(row)
    torch.testing.assert_close(first.eigenvalues(), second.eigenvalues(), rtol=1e-9, atol=1e-9)


def test_rank_for_energy_on_a_known_spectrum():
    eigs = torch.tensor([50.0, 30.0, 15.0, 5.0])  # 50/80/95/100%
    assert rank_for_energy(eigs, 0.50) == 1
    assert rank_for_energy(eigs, 0.80) == 2
    assert rank_for_energy(eigs, 0.95) == 3
    assert rank_for_energy(eigs, 0.999) == 4


def test_rank_for_a_rank_deficient_matrix_is_exact():
    """A matrix built with rank 3 must report rank 3 at 99.9% energy —
    the property the whole analysis rests on."""
    torch.manual_seed(2)
    a = torch.randn(200, 3, dtype=torch.float64) @ torch.randn(3, 20, dtype=torch.float64)
    acc = SpectrumAccumulator(20)
    acc.update(a)
    assert rank_for_energy(acc.eigenvalues(), 0.999) == 3


def test_effective_rank_brackets_the_extremes():
    flat = torch.ones(64, dtype=torch.float64)
    spiky = torch.tensor([1.0] + [1e-12] * 63, dtype=torch.float64)
    assert effective_rank(flat) == pytest.approx(64.0, rel=1e-6)
    assert effective_rank(spiky) < 1.01


def test_energy_curve_is_monotone_and_ends_at_one():
    eigs = torch.tensor([9.0, 3.0, 1.0, 0.5])
    curve = energy_curve(eigs)
    assert torch.all(curve[1:] >= curve[:-1])
    assert curve[-1] == pytest.approx(1.0)


def test_break_even_accounts_for_the_positional_dimensions():
    """A latent cache stores latent_dim + rope_dim. Forgetting rope_dim
    would claim savings that do not exist — and three of the five values
    in the methodology doc's Phase 10 sweep already sit above the real
    break-even."""
    gqa = gqa_numbers_per_token_per_layer(num_kv_heads=2, head_dim=128)
    assert gqa == 512
    assert mla_break_even(rope_dim=64, gqa_numbers=gqa) == 448
    assert compression_ratio(448, 64, gqa) == pytest.approx(1.0)
    assert compression_ratio(128, 64, gqa) == pytest.approx(512 / 192)
    # The doc's sweep values that make the cache bigger, not smaller:
    for latent_dim in (512, 768, 1024):
        assert compression_ratio(latent_dim, 64, gqa) < 1.0


def test_weight_spectrum_is_descending():
    torch.manual_seed(3)
    eigs = weight_spectrum(torch.randn(32, 64))
    assert torch.all(eigs[1:] <= eigs[:-1])


def test_report_summary_has_the_decision_fields():
    eigs = torch.tensor([100.0, 1.0, 0.5, 0.1])
    summary = SpectrumReport("kv_joint", 0, 4, 1000, eigs).summary()
    assert summary["rank_990"] >= 1
    assert 0 < summary["compression_at_99"] <= 1
    assert summary["tokens"] == 1000
