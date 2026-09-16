"""
Phase 7.2 tests — truncation and quantization simulation.

This harness decides whether Phases 7.3-7.4 get built, so its own
correctness matters more than its output. The tests that count are the
ones that would catch a *silent* error: a projection that is not really
lossy, a wrapper that is not really removed, a metric that reports zero
when it should report damage.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from compression.spectra import SpectrumAccumulator  # noqa: E402
from compression.truncation import (  # noqa: E402
    DivergenceMeter,
    fit_basis,
    install_joint_lowrank,
    install_quantization,
    quantize_dequantize,
)

TINY = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(**TINY)).to(torch.float32).eval()


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------


def test_full_rank_projection_is_the_identity():
    """The exactness check the whole method rests on: keep every
    direction and nothing is lost. If this drifts, every KL number the
    harness reports is measuring the implementation, not compression."""
    torch.manual_seed(1)
    a = torch.randn(200, 16, dtype=torch.float64)
    acc = SpectrumAccumulator(16)
    acc.update(a)
    basis = fit_basis(acc.gram, 16)
    torch.testing.assert_close(a @ basis @ basis.T, a, rtol=1e-9, atol=1e-9)


def test_basis_is_orthonormal_and_ordered():
    torch.manual_seed(2)
    acc = SpectrumAccumulator(12)
    acc.update(torch.randn(300, 12, dtype=torch.float64))
    basis = fit_basis(acc.gram, 5)
    assert basis.shape == (12, 5)
    torch.testing.assert_close(
        basis.T @ basis, torch.eye(5, dtype=torch.float64), rtol=1e-8, atol=1e-8
    )


def test_projection_recovers_a_genuinely_low_rank_signal():
    torch.manual_seed(3)
    a = torch.randn(400, 4, dtype=torch.float64) @ torch.randn(4, 24, dtype=torch.float64)
    acc = SpectrumAccumulator(24)
    acc.update(a)
    basis = fit_basis(acc.gram, 4)
    torch.testing.assert_close(a @ basis @ basis.T, a, rtol=1e-8, atol=1e-8)


# ----------------------------------------------------------------------
# Quantization
# ----------------------------------------------------------------------


def test_quantization_none_is_exact():
    x = torch.randn(3, 5, 8)
    torch.testing.assert_close(quantize_dequantize(x, "none", 2, 4), x)


@pytest.mark.parametrize("mode", ["per_tensor", "per_channel", "per_token"])
def test_quantization_is_lossy_but_bounded(mode):
    """Symmetric INT8 error must sit inside half a step of the relevant
    scale. A mode that silently returned its input would make INT8 look
    free, which is exactly the comparison this phase turns on."""
    torch.manual_seed(4)
    x = torch.randn(2, 16, 8)
    q = quantize_dequantize(x, mode, num_heads=2, head_dim=4)
    assert not torch.equal(q, x)
    assert (q - x).abs().max() <= x.abs().max() / 127 * 0.51


def test_per_channel_beats_per_tensor_when_one_channel_dominates():
    """Qwen's pre-RoPE K has effective rank 5.6 — a few channels carrying
    enormous magnitude. A single shared scale spends its resolution on
    those and starves the rest, which is why K wants per-channel."""
    torch.manual_seed(5)
    x = torch.randn(1, 64, 8) * 0.01
    x[..., 0] = 100.0                      # one massive channel
    err = lambda mode: (quantize_dequantize(x, mode, 2, 4) - x).abs().mean()
    assert err("per_channel") < err("per_tensor") / 10


# ----------------------------------------------------------------------
# Install / remove
# ----------------------------------------------------------------------


def test_install_is_exactly_reversible(tiny_model):
    """Every configuration is compared against the same baseline, so a
    leaked wrapper would contaminate every later measurement in the
    sweep with a silent extra loss."""
    ids = torch.randint(0, TINY["vocab_size"], (1, 16))
    with torch.no_grad():
        before = tiny_model(input_ids=ids, use_cache=False).logits
    installed = install_quantization(tiny_model, "per_tensor", "per_token", 2, 16)
    installed.remove()
    with torch.no_grad():
        after = tiny_model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_full_rank_joint_projection_leaves_logits_unchanged(tiny_model):
    """Gate 7's precondition, in output space: at full rank the
    simulated compression must be a no-op on the model's predictions."""
    ids = torch.randint(0, TINY["vocab_size"], (1, 24))
    kv_dim = 2 * TINY["num_key_value_heads"] * (TINY["hidden_size"] // TINY["num_attention_heads"])

    acc = {i: SpectrumAccumulator(kv_dim) for i in range(TINY["num_hidden_layers"])}
    handles = []
    for idx, layer in enumerate(tiny_model.model.layers):
        def hook(_m, _i, out, idx=idx, store={}):
            store.setdefault(idx, []).append(out.detach())
            if len(store[idx]) == 2:
                acc[idx].update(torch.cat(store.pop(idx), dim=-1))
        handles.append(layer.self_attn.k_proj.register_forward_hook(hook))
        handles.append(layer.self_attn.v_proj.register_forward_hook(hook))
    with torch.no_grad():
        baseline = tiny_model(input_ids=ids, use_cache=False).logits
    for h in handles:
        h.remove()

    bases = {i: fit_basis(a.gram, kv_dim) for i, a in acc.items()}
    installed = install_joint_lowrank(tiny_model, bases)
    try:
        with torch.no_grad():
            projected = tiny_model(input_ids=ids, use_cache=False).logits
    finally:
        installed.remove()
    torch.testing.assert_close(projected, baseline, rtol=1e-4, atol=1e-4)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def test_divergence_meter_is_zero_for_identical_logits():
    torch.manual_seed(6)
    logits = torch.randn(2, 8, 50)
    meter = DivergenceMeter()
    meter.update(logits, logits)
    result = meter.result()
    assert result["kl_mean_nats"] == pytest.approx(0.0, abs=1e-6)
    assert result["top1_agreement"] == 1.0
    assert result["tokens"] == 16


def test_divergence_meter_detects_damage():
    torch.manual_seed(7)
    base = torch.randn(1, 32, 40)
    meter = DivergenceMeter()
    meter.update(base, base + torch.randn_like(base) * 3)
    result = meter.result()
    assert result["kl_mean_nats"] > 0.1
    assert result["top1_flip_rate"] > 0.1