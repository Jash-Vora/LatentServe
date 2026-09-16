"""
Phase 7 Experiment B tests — output-aware and learned latent KV.

The claim this code exists to test is "a better-chosen subspace beats
SVD". That claim is only meaningful if the machinery is right, so the
tests pin the two properties everything rests on: the metric fit reduces
to plain SVD when the metric is the identity, and folding the
projections into the adapter reproduces the same arithmetic.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from compression.spectra import SpectrumAccumulator  # noqa: E402
from compression.truncation import fit_basis  # noqa: E402
from compression.learned import (  # noqa: E402
    LatentKVAdapter,
    install_adapters,
    fit_metric_basis,
    joint_metric,
    output_metric_for_v,
)

TINY = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(**TINY)).to(torch.float32).eval()


def test_identity_metric_reduces_to_plain_svd():
    """The generalisation check: with metric = I, the output-aware fit
    must reproduce exactly the subspace 7.2 used. If it does not, any
    difference later is a bug rather than the weighting."""
    torch.manual_seed(1)
    acc = SpectrumAccumulator(16)
    acc.update(torch.randn(400, 16, dtype=torch.float64))
    rank = 5
    down, up = fit_metric_basis(acc.gram, torch.eye(16, dtype=torch.float64), rank)
    plain = fit_basis(acc.gram, rank)

    x = torch.randn(50, 16, dtype=torch.float64)
    torch.testing.assert_close(x @ down @ up, x @ plain @ plain.T, rtol=1e-6, atol=1e-6)


def test_full_rank_metric_fit_is_exact():
    torch.manual_seed(2)
    acc = SpectrumAccumulator(12)
    acc.update(torch.randn(300, 12, dtype=torch.float64))
    metric = torch.randn(12, 12, dtype=torch.float64)
    metric = metric @ metric.T + torch.eye(12, dtype=torch.float64)
    down, up = fit_metric_basis(acc.gram, metric, 12)
    x = torch.randn(40, 12, dtype=torch.float64)
    torch.testing.assert_close(x @ down @ up, x, rtol=1e-6, atol=1e-6)


def test_metric_fit_beats_plain_svd_on_the_metric_it_optimises():
    """The whole point of B1: when some directions matter more, a fit
    that knows which they are should preserve them better — even though
    it is *worse* by plain reconstruction error, which is exactly the
    trade being made."""
    torch.manual_seed(3)
    x = torch.randn(600, 12, dtype=torch.float64)
    x[:, 0] *= 8                       # large but, per the metric below, irrelevant
    acc = SpectrumAccumulator(12)
    acc.update(x)
    metric = torch.eye(12, dtype=torch.float64)
    metric[0, 0] = 1e-4                # the model barely sees direction 0

    down, up = fit_metric_basis(acc.gram, metric, 4)
    plain = fit_basis(acc.gram, 4)
    err = lambda xh: float(((x - xh) @ metric @ (x - xh).T).diagonal().sum())
    assert err(x @ down @ up) < err(x @ plain @ plain.T)


def test_output_metric_for_v_sums_the_group():
    w = torch.randn(16, 4 * 8)
    single = output_metric_for_v(w, range(0, 1), 8)
    pair = output_metric_for_v(w, range(0, 2), 8)
    block = w[:, 8:16].to(torch.float64)
    torch.testing.assert_close(pair - single, block.T @ block, rtol=1e-8, atol=1e-8)


def test_joint_metric_is_block_diagonal():
    a = torch.eye(4, dtype=torch.float64) * 2
    b = torch.eye(4, dtype=torch.float64) * 3
    m = joint_metric([a], [b])
    assert m.shape == (8, 8)
    assert torch.all(m[:4, 4:] == 0) and torch.all(m[4:, :4] == 0)


def test_adapter_folds_projections_and_biases_exactly(tiny_model):
    """Full rank, so the adapter must reproduce the original K and V
    bit-for-bit — including the biases Qwen2 carries on k_proj and
    v_proj, which a fold that ignored them would silently shift."""
    layer = tiny_model.model.layers[0]
    k, v = layer.self_attn.k_proj, layer.self_attn.v_proj
    dim = k.out_features + v.out_features
    eye = torch.eye(dim, dtype=torch.float64)

    adapter = LatentKVAdapter.from_projections(k, v, eye, eye)
    hidden = torch.randn(2, 6, TINY["hidden_size"])
    expected = torch.cat([k(hidden), v(hidden)], dim=-1)
    torch.testing.assert_close(adapter.project(hidden), expected, rtol=1e-4, atol=1e-4)


def test_full_rank_adapter_leaves_logits_unchanged(tiny_model):
    ids = torch.randint(0, TINY["vocab_size"], (1, 16))
    with torch.no_grad():
        baseline = tiny_model(input_ids=ids, use_cache=False).logits

    adapters = {}
    for idx, layer in enumerate(tiny_model.model.layers):
        k, v = layer.self_attn.k_proj, layer.self_attn.v_proj
        dim = k.out_features + v.out_features
        eye = torch.eye(dim, dtype=torch.float64)
        adapters[idx] = LatentKVAdapter.from_projections(k, v, eye, eye)

    installed = install_adapters(tiny_model, adapters)
    try:
        with torch.no_grad():
            out = tiny_model(input_ids=ids, use_cache=False).logits
    finally:
        installed.remove()
    torch.testing.assert_close(out, baseline, rtol=1e-4, atol=1e-4)


def test_install_is_reversible(tiny_model):
    ids = torch.randint(0, TINY["vocab_size"], (1, 8))
    with torch.no_grad():
        before = tiny_model(input_ids=ids, use_cache=False).logits
    k = tiny_model.model.layers[0].self_attn.k_proj
    v = tiny_model.model.layers[0].self_attn.v_proj
    dim = k.out_features + v.out_features
    eye = torch.eye(dim, dtype=torch.float64)
    adapters = {
        i: LatentKVAdapter.from_projections(
            l.self_attn.k_proj, l.self_attn.v_proj, eye, eye
        )
        for i, l in enumerate(tiny_model.model.layers)
    }
    install_adapters(tiny_model, adapters).remove()
    with torch.no_grad():
        after = tiny_model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(before, after, rtol=0, atol=0)