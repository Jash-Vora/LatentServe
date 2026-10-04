"""Phase 14 sparse reference: the math, the hook, and the cache rewind."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from model.attention import sparse as sp  # noqa: E402


def _qkv(b=1, h=2, r=6, t=200, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, r, d, generator=g)
    k = torch.randn(b, h, t, d, generator=g)
    v = torch.randn(b, h, t, d, generator=g)
    return q, k, v


def _dense(q, k, v):
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return torch.softmax(s, -1) @ v


def test_dense_policy_is_plain_attention():
    q, k, v = _qkv()
    out = sp.SparseStudy("dense", 1.0)(q, k, v)
    torch.testing.assert_close(out, _dense(q, k, v), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("policy", sp.POLICIES)
def test_every_policy_at_full_budget_is_dense(policy):
    q, k, v = _qkv(seed=1)
    study = sp.SparseStudy(policy, 1.0)
    torch.testing.assert_close(study(q, k, v), _dense(q, k, v), atol=1e-5, rtol=1e-5)
    assert study.captured[-1] == pytest.approx(1.0) and study.kept[-1] == 1.0


def test_page_mass_sums_to_the_group_size():
    q, k, _ = _qkv(r=6, t=203)
    probs = torch.softmax(q @ k.transpose(-1, -2), -1)
    torch.testing.assert_close(sp.page_mass(probs, 16).sum(-1), torch.full((1, 2), 6.0))


def test_bounds_bound_every_key_in_their_page():
    q, k, _ = _qkv(t=203, seed=2)
    bounds = sp.page_bounds(q, k, 16)                              # [B, H, R, P]
    scores = q @ k.transpose(-1, -2)                               # [B, H, R, T]
    padded = torch.nn.functional.pad(scores, (0, bounds.shape[-1] * 16 - 203), value=-1e9)
    page_max = padded.reshape(*scores.shape[:3], -1, 16).amax(-1)
    assert (bounds >= page_max - 1e-4).all()


def test_bounds_and_window_always_keep_the_sink_and_recent_pages():
    q, k, v = _qkv(t=320, seed=3)                                  # 20 pages
    for policy in ("bounds", "window"):
        study = sp.SparseStudy(policy, 0.25, recent=2)
        mass = sp.page_mass(torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(32), -1), 16)
        keep = sp.select_pages(policy, 0.25, mass=mass, bounds=sp.page_bounds(q, k, 16),
                               pages=20, recent=2)
        assert keep[..., 0].all() and keep[..., 18:].all()
        assert int(keep[0, 0].sum()) == sp.budget(0.25, 20) == 5
        study(q, k, v)


def test_window_is_sink_plus_most_recent():
    q, k, _ = _qkv(t=320, seed=4)
    keep = sp.select_pages("window", 0.25, bounds=sp.page_bounds(q, k, 16), pages=20)
    assert keep[0, 0].nonzero().flatten().tolist() == [0, 16, 17, 18, 19]


def test_the_oracle_captures_at_least_as_much_mass_as_any_policy():
    q, k, v = _qkv(t=640, seed=5)
    captured = {}
    for policy in ("oracle", "bounds", "window"):
        study = sp.SparseStudy(policy, 0.125)
        study(q, k, v)
        captured[policy] = study.captured[-1]
    assert captured["oracle"] >= captured["bounds"] - 1e-6
    assert captured["oracle"] >= captured["window"] - 1e-6


# ---------------------------------------------------------- model level ---


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                      max_position_embeddings=1024)
    hf = Qwen2ForCausalLM(cfg).eval()
    shape = ModelShape(2, 8, 2, 16, 128, 128, 1024, "torch.float32")
    return LatentServeQwen(hf_model=hf, tokenizer=None, shape=shape, device="cpu",
                           attn_impl="sdpa", max_seq_len_hint=512)


def test_dense_study_reproduces_the_models_own_decode():
    """Wiring: query heads folded onto their KV head in the right order, and
    the output reshaped back the same way."""
    ls = _tiny_model()
    prompt = torch.randint(0, 128, (1, 70))
    token = torch.tensor([[5]])

    def run(study):
        sp.install(ls, study)
        ls.allocate_cache(1, 128, paged=True, block_size=16)
        ls.cache.reset()
        with torch.no_grad():
            ls.prefill(prompt)
            return ls.decode_step(token)

    plain = run(None)
    studied = run(sp.SparseStudy("dense", 1.0))
    sp.install(ls, None)
    torch.testing.assert_close(studied, plain, atol=1e-4, rtol=1e-4)


def test_a_sparse_study_changes_the_output():
    ls = _tiny_model()
    prompt = torch.randint(0, 128, (1, 200))
    outs = {}
    for policy, ratio in (("dense", 1.0), ("window", 0.1)):
        sp.install(ls, sp.SparseStudy(policy, ratio))
        ls.allocate_cache(1, 256, paged=True, block_size=16)
        ls.cache.reset()
        with torch.no_grad():
            ls.prefill(prompt)
            outs[policy] = ls.decode_step(torch.tensor([[5]]))
    sp.install(ls, None)
    assert not torch.allclose(outs["dense"], outs["window"])


def test_rewind_reproduces_the_same_decode():
    ls = _tiny_model()
    prompt = torch.randint(0, 128, (1, 75))
    toks = [torch.tensor([[t]]) for t in (3, 9, 27, 81 % 128, 7)]
    ls.allocate_cache(1, 128, paged=True, block_size=16)
    ls.cache.reset()
    with torch.no_grad():
        ls.prefill(prompt)
        first = [ls.decode_step(t) for t in toks]
        ls.cache.rewind(75)
        assert ls.cache.length == 75
        again = [ls.decode_step(t) for t in toks]
    for a, b in zip(first, again):
        torch.testing.assert_close(a, b)


def test_rewind_refuses_to_grow():
    ls = _tiny_model()
    ls.allocate_cache(1, 128, paged=True, block_size=16)
    ls.cache.reset()
    with torch.no_grad():
        ls.prefill(torch.randint(0, 128, (1, 20)))
    with pytest.raises(ValueError):
        ls.cache.rewind(40)


# ------------------------------------------------------------- the runner ---


def test_the_study_runs_end_to_end_on_a_tiny_model(capsys):
    from benchmarks.runners import phase14_oracle as po

    ls = _tiny_model()
    g = torch.Generator().manual_seed(0)
    windows = [torch.randint(0, 128, (180,), generator=g).tolist() for _ in range(2)]
    cfgs = po.configs(["oracle", "bounds", "window"], [0.5, 0.125])
    text = po.eval_text(ls, windows, cfgs, ctx=160, recent=2, device="cpu", log=lambda m: None)
    for c in cfgs:
        assert len(text[c]["nll"]) == 2 * 19
    dense = text[("dense", 1.0)]
    assert all(k == 0.0 for k in dense["kl"])
    assert statistics_mean(dense["captured"]) == pytest.approx(1.0, abs=1e-6)   # float32 sums

    key_ids = [11, 22]
    cases = [po.NeedleCase(context=torch.randint(0, 128, (150,), generator=g).tolist(),
                           question=[1, 2, 3], key="x", key_ids=key_ids, length=4096,
                           depth=0.5)]
    needles = po.eval_needles(ls, cases, cfgs, recent=2, answer_tokens=4, tokenizer=None,
                              device="cpu", log=lambda m: None)
    assert all(len(needles[c]["correct"]) == 1 for c in cfgs)

    rows = po.summarise(text, needles, cfgs, [4096])
    assert rows[0]["policy"] == "dense" and rows[0]["kl"] == 0.0
    for r in rows[1:]:
        assert r["captured"] <= 1.0 + 1e-6 and r["kl"] >= 0.0
    oracle = {r["ratio"]: r["captured"] for r in rows if r["policy"] == "oracle"}
    window = {r["ratio"]: r["captured"] for r in rows if r["policy"] == "window"}
    assert oracle[0.125] >= window[0.125] - 1e-6
    po.print_table(rows, [4096])
    out = capsys.readouterr().out
    assert "needle 4K" in out and "oracle" in out


def statistics_mean(xs):
    return sum(xs) / len(xs)


# ------------------------------------------------------- Phase 15 bake-off ---


@pytest.mark.parametrize("policy", sp.BAKEOFF_POLICIES)
def test_bakeoff_policies_at_full_budget_are_dense(policy):
    q, k, v = _qkv(seed=6)
    torch.testing.assert_close(sp.SparseStudy(policy, 1.0)(q, k, v), _dense(q, k, v),
                               atol=1e-5, rtol=1e-5)


def test_page_means_average_only_written_tokens():
    k = torch.arange(2 * 20, dtype=torch.float32).reshape(1, 1, 40, 1).expand(1, 2, 40, 3)[:, :, :20]
    means = sp.page_means(k.contiguous(), 16)                     # pages: 16 tokens, then 4
    torch.testing.assert_close(means[0, 0, :, 0], torch.tensor([7.5, 17.5]))


def test_estimated_mass_gives_each_head_a_total_of_one():
    logits = torch.randn(1, 2, 6, 9)
    est = sp.estimated_mass(logits, torch.full((9,), 16.0), 0.1)
    torch.testing.assert_close(est.sum(-1), torch.full((1, 2), 6.0))


@pytest.mark.parametrize("policy", sp.BAKEOFF_POLICIES)
def test_bakeoff_policies_keep_the_sink_and_recent_pages(policy):
    q, k, v = _qkv(t=320, seed=7)
    study = sp.SparseStudy(policy, 0.25, recent=3)
    study(q, k, v)
    assert study.kept[-1] == pytest.approx(5 / 20)                # budget honoured
    keep = study._select_bakeoff(policy, q, k, sp.page_mass(
        torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(32), -1), 16), 20, 320, 32)
    assert keep[..., 0].all() and keep[..., 17:].all()


def test_rerank_never_keeps_less_mass_than_bounds():
    """Bounds' top pages are always among rerank's candidates, so choosing
    by exact scores from that superset cannot do worse."""
    for seed in range(5):
        q, k, v = _qkv(t=640, seed=10 + seed)
        got = {}
        for policy in ("bounds", "rerank"):
            study = sp.SparseStudy(policy, 0.125)
            study(q, k, v)
            got[policy] = study.captured[-1]
        assert got["rerank"] >= got["bounds"] - 1e-6


def test_dense_layers_attend_densely():
    q, k, v = _qkv(t=320, seed=8)
    study = sp.SparseStudy("bounds", 0.1, dense_layers=2)
    torch.testing.assert_close(study(q, k, v, layer_idx=1), _dense(q, k, v), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(study(q, k, v, layer_idx=2), _dense(q, k, v), atol=1e-3)
