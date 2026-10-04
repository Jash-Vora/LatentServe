"""Phase 15 harness: case builders, scoring, statistics, verdicts, curves."""

from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from benchmarks.runners import phase15_quality as pq  # noqa: E402


class CharTok:
    """One token per character: lengths and positions are easy to check."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 128 for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _tiny_model():
    """A two-layer Qwen2 on CPU, built here rather than imported from another
    test file: importing test modules only works where the tests folder is
    an importable package, which it is not in every pytest setup."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from model.latentserve_qwen import LatentServeQwen
    from model.qwen import ModelShape

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                      max_position_embeddings=4096)
    hf = Qwen2ForCausalLM(cfg).eval()
    shape = ModelShape(2, 8, 2, 16, 128, 128, 4096, "torch.float32")
    ls = LatentServeQwen(hf_model=hf, tokenizer=None, shape=shape, device="cpu",
                         attn_impl="triton_paged", max_seq_len_hint=1024)
    ls._gpu_mode = True
    return ls


def _filler(n=200_000):
    rng = random.Random(1)
    return pq.Filler([rng.randrange(97, 123) for _ in range(n)])


def test_needles_have_the_requested_length_and_contain_their_key():
    tok = CharTok()
    cases = pq.build_needles(tok, _filler(), [600, 1200], [0.1, 0.9], 2, random.Random(0))
    assert len(cases) == 8
    for c in cases:
        # Room is left for the answer: at 32K this keeps every position inside
        # the range the model was trained on.
        assert len(c.context) + len(c.question) + c.answer_tokens == c.length
        assert c.answers[0] in tok.decode(c.context)


def test_needle_depth_is_where_the_needle_is():
    tok = CharTok()
    for depth in (0.1, 0.9):
        c = pq.build_needles(tok, _filler(), [2000], [depth], 1, random.Random(0))[0]
        at = tok.decode(c.context).index("pass key")
        assert abs(at / len(c.context) - depth) < 0.05


def test_multikey_asks_for_one_key_among_distractors():
    tok = CharTok()
    c = pq.build_multikey(tok, _filler(), [1500], [0.5], 1, random.Random(0))[0]
    text = tok.decode(c.context)
    assert c.answers[0] in text and all(v in text for v in c.meta["distractors"].values())
    assert len(c.meta["distractors"]) == 3


def test_vartrack_chain_runs_in_order_through_the_context():
    tok = CharTok()
    c = pq.build_vartrack(tok, _filler(), [3000], 1, random.Random(0))[0]
    text = tok.decode(c.context)
    chain = c.meta["chain"]
    assert f"VAR {chain[0]} = {c.answers[0]}" in text
    positions = [text.index(f"VAR {chain[i]} = VAR {chain[i - 1]}") for i in range(1, len(chain))]
    assert positions == sorted(positions)


def test_qa_buries_the_gold_paragraph_among_other_articles():
    tok = CharTok()
    squad = [{"title": f"t{i % 5}", "context": f"paragraph {i} " + "x" * 150,
              "question": f"q{i}?", "answers": {"text": [f"paragraph {i}"]}} for i in range(60)]
    c = pq.build_qa(tok, squad, [2500], [0.5], 1, random.Random(0))[0]
    text = tok.decode(c.context)
    assert c.answers[0] in text
    assert len(c.context) + len(c.question) + c.answer_tokens <= 2500


def test_scoring_normalises_case_punctuation_and_articles():
    assert pq.is_correct("The answer is 48213.", ["48213"])
    assert pq.is_correct("It was the Eiffel Tower!", ["eiffel tower"])
    assert not pq.is_correct("unknown", ["48213"])


def test_wilson_interval_brackets_the_rate():
    lo, hi = pq.wilson(45, 50)
    assert lo < 0.9 < hi and 0.75 < lo and hi < 0.97


def test_paired_counts_failures_and_gains_against_dense():
    rows = []
    for case, (d, s) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
        rows.append({"task": "needle", "case": case, "policy": "dense", "ratio": 1.0, "correct": d})
        rows.append({"task": "needle", "case": case, "policy": "gpu", "ratio": 0.25, "correct": s})
    pr = pq.paired(rows, "gpu", 0.25)
    assert (pr["failures"], pr["gains"], pr["dense_correct"]) == (1, 1, 2)


def _rows(failures, n=100):
    rows = []
    for case in range(n):
        rows.append({"task": "needle", "case": case, "policy": "dense", "ratio": 1.0, "correct": True})
        rows.append({"task": "needle", "case": case, "policy": "gpu", "ratio": 0.25,
                     "correct": case >= failures})
    return rows


def test_verdicts_apply_the_preregistered_criteria():
    text = {8192: {("gpu", 0.25): {"kl": 0.009}}, 32768: {("gpu", 0.25): {"kl": 0.008}}}
    assert pq.verdicts(_rows(2), text, [0.25])[0]["pass"]            # 2 <= 2% of 100
    v = pq.verdicts(_rows(3), text, [0.25])[0]
    assert not v["pass"] and "net loss 3" in v["reasons"][0]
    bad_kl = {8192: {("gpu", 0.25): {"kl": 0.02}}}
    assert not pq.verdicts(_rows(0), bad_kl, [0.25])[0]["pass"]
    assert pq.verdicts(_rows(1, n=10), None, [0.25])[0]["pass"]      # at least one allowed


def test_curves_writes_a_summary_with_verdicts(tmp_path, capsys):
    rows = _rows(1, n=60)
    (tmp_path / "needle.json").write_text(json.dumps({"rows": rows}))
    raw = {"8192": {"dense|1.0": {"kl": [0.0], "agree": [1.0], "nll": [2.0]},
                    "gpu|0.25": {"kl": [0.005], "agree": [0.97], "nll": [2.01]}}}
    (tmp_path / "text.json").write_text(json.dumps({"raw": raw}))
    (tmp_path / "latency.json").write_text(json.dumps({"rows": [
        {"batch": 8, "ctx": 32768, "ratio": None, "ms": 48.0},
        {"batch": 8, "ctx": 32768, "ratio": 0.25, "ms": 29.0}]}))
    assert pq.curves(tmp_path, [0.25]) == 0
    summary = (tmp_path / "summary.md").read_text()
    assert "25.0% of pages: PASS" in summary and "1.66x" in summary
    assert "original rule: pass" in summary


def test_cases_run_under_every_policy_on_a_tiny_model():
    ls = _tiny_model()
    tok = CharTok()
    cases = pq.build_needles(tok, _filler(), [400], [0.5], 1, random.Random(0))
    rows = pq.eval_cases(ls, cases, pq.cfg_list([0.5, 0.25]), 2, tok, "cpu",
                         oracle_max_ctx=16384, log=lambda m: None)
    assert {(r["policy"], r["ratio"]) for r in rows} == set(pq.cfg_list([0.5, 0.25]))
    assert all(isinstance(r["correct"], bool) for r in rows)
    skipped = pq.eval_cases(ls, cases, pq.cfg_list([0.25]), 2, tok, "cpu",
                            oracle_max_ctx=100, log=lambda m: None)
    assert [r["correct"] for r in skipped if r["policy"] == "oracle"] == [None]


def test_generation_rows_compare_against_dense():
    ls = _tiny_model()
    rows = pq.eval_gen(ls, [list(range(1, 90))], pq.cfg_list([0.25], oracle=False), 2, 6, "cpu",
                       log=lambda m: None)
    dense = [r for r in rows if r["policy"] == "dense"][0]
    assert dense["identical"] and dense["first_diff"] == 6


def test_every_builder_keeps_the_answer_inside_the_length():
    tok = CharTok()
    rng = random.Random(3)
    cases = (pq.build_needles(tok, _filler(), [3000], [0.5], 2, rng)
             + pq.build_multikey(tok, _filler(), [3000], [0.5], 2, rng)
             + pq.build_vartrack(tok, _filler(), [3000], 2, rng))
    for c in cases:
        assert len(c.context) + len(c.question) + c.answer_tokens <= c.length, c.task


def test_two_hop_vartrack_is_the_default_shape():
    tok = CharTok()
    c = pq.build_vartrack(tok, _filler(), [2000], 1, random.Random(0), hops=2)[0]
    text = tok.decode(c.context)
    a, b = c.meta["chain"]
    assert f"VAR {a} = {c.answers[0]}" in text and f"VAR {b} = VAR {a}" in text


def test_curves_reports_failures_by_length_and_the_gen_control(tmp_path):
    rows = []
    for case, length in enumerate([4096] * 5 + [32768] * 5):
        rows.append({"task": "needle", "case": case, "length": length, "policy": "dense",
                     "ratio": 1.0, "correct": True})
        rows.append({"task": "needle", "case": case, "length": length, "policy": "gpu",
                     "ratio": 0.25, "correct": not (length == 32768 and case % 2)})
    (tmp_path / "needle.json").write_text(json.dumps({"rows": rows}))
    (tmp_path / "gen.json").write_text(json.dumps({"rows": [
        {"policy": "triton", "ratio": 1.0, "prompt": 0, "first_diff": 30, "identical": False},
        {"policy": "gpu", "ratio": 0.25, "prompt": 0, "first_diff": 12, "identical": False}]}))
    pq.curves(tmp_path, [0.25])
    summary = (tmp_path / "summary.md").read_text()
    assert "| 25.0% | 0/5 | 3/5 |" in summary          # cases 5, 7 and 9 fail at 32K
    assert "dense, Triton kernel (control) | 0/1 | 30" in summary


def test_gen_control_runs_dense_on_the_triton_kernel():
    from kernels.gqa import paged_decode as pd

    ls = _tiny_model()
    before = pd.decode_backend()
    try:
        rows = pq.eval_gen(ls, [list(range(1, 60))], [("dense", 1.0), ("triton", 1.0)], 2, 4,
                           "cpu", log=lambda m: None)
    finally:
        pd.set_decode_backend(before)
    assert {r["policy"] for r in rows} == {"dense", "triton"}


# ------------------------------------------------------------- bake-off ---

from benchmarks.runners import phase15_bakeoff as bo  # noqa: E402


def _row(name, kl, cost):
    return {"variant": name, "kl": kl, "cost": cost}


def test_rule_picks_the_cheapest_within_ten_percent_of_the_best():
    rows = [_row("rerank", 0.0060, 3), _row("mass", 0.0065, 1), _row("bounds", 0.0090, 0),
            _row("oracle", 0.0010, None)]
    assert bo.choose(rows)["variant"] == "mass"          # 0.0065 <= 1.1 x 0.0060, cheaper


def test_rule_takes_the_best_when_nothing_cheap_is_close():
    rows = [_row("rerank", 0.0060, 3), _row("mass", 0.0080, 1)]
    assert bo.choose(rows)["variant"] == "rerank"


def test_rule_never_selects_the_oracle():
    assert bo.choose([_row("oracle", 0.0001, None), _row("bounds", 0.01, 0)])["variant"] == "bounds"


def test_seeds_change_the_contexts_and_seed_zero_keeps_the_published_ones():
    stream = list(range(1000))
    assert pq.Filler(stream).take(5) == [0, 1, 2, 3, 4]
    assert pq.Filler(stream, offset=1 * 50_021).take(5) != pq.Filler(stream).take(5)
    assert pq.circular(stream, 998, 4) == [998, 999, 0, 1]


def test_each_variant_installs_what_it_names():
    from model.attention import sparse as sp

    ls = _tiny_model()
    for name, v in bo.VARIANTS.items():
        study = bo.apply_variant(ls, (name, 0.25), 2)
        assert (study.policy, study.recent, study.dense_layers) == \
               (v["policy"], v["recent"], v["dense_layers"])
        assert ls.layers[0].attn.sparse_study is study
    sp.install(ls, None)


def test_bakeoff_loop_runs_on_a_tiny_model():
    from benchmarks.runners.phase14_oracle import eval_text

    ls = _tiny_model()
    cfgs = [("dense", 1.0)] + [(v, 0.25) for v in ("bounds", "mass", "mean", "rerank+dense2+window8")]
    g = torch.Generator().manual_seed(0)
    windows = [torch.randint(0, 128, (300,), generator=g).tolist()]
    out = eval_text(ls, windows, cfgs, 280, 2, "cpu", log=lambda m: None,
                    apply_fn=bo.apply_variant)
    assert all(len(out[c]["kl"]) == 19 for c in cfgs)
    assert statistics_mean(out[("dense", 1.0)]["captured"]) == pytest.approx(1.0, abs=1e-6)


def statistics_mean(xs):
    return sum(xs) / len(xs)



def test_symmetric_churn_passes_the_revised_rule_but_not_the_original():
    """Seed 3's 50% replication: 2 failures, 3 gains. Counting failures only
    fails a configuration that lost nothing on net."""
    rows = []
    outcomes = [(True, False)] * 2 + [(False, True)] * 3 + [(True, True)] * 80
    for case, (d, s) in enumerate(outcomes):
        rows.append({"task": "qa", "case": case, "policy": "dense", "ratio": 1.0, "correct": d})
        rows.append({"task": "qa", "case": case, "policy": "gpu", "ratio": 0.5, "correct": s})
    v = pq.verdicts(rows, {8192: {("gpu", 0.5): {"kl": 0.003}}}, [0.5])[0]
    assert v["pass"] and not v["original_pass"] and v["net_loss"] == -1


def test_the_revised_rule_still_fails_a_net_loss():
    rows = []
    outcomes = [(True, False)] * 4 + [(False, True)] * 1 + [(True, True)] * 80
    for case, (d, s) in enumerate(outcomes):
        rows.append({"task": "qa", "case": case, "policy": "dense", "ratio": 1.0, "correct": d})
        rows.append({"task": "qa", "case": case, "policy": "gpu", "ratio": 0.25, "correct": s})
    v = pq.verdicts(rows, None, [0.25])[0]
    assert not v["pass"] and v["net_loss"] == 3


def test_the_dense_control_runs_on_case_tasks_and_is_reported_as_the_noise_floor(tmp_path):
    from kernels.gqa import paged_decode as pd

    ls = _tiny_model()
    tok = CharTok()
    cases = pq.build_needles(tok, _filler(), [400], [0.5], 1, random.Random(0))
    before = pd.decode_backend()
    try:
        rows = pq.eval_cases(ls, cases, [("dense", 1.0), ("triton", 1.0), ("gpu", 0.25)], 2, tok,
                             "cpu", oracle_max_ctx=16384, log=lambda m: None)
    finally:
        pd.set_decode_backend(before)
    assert {r["policy"] for r in rows} == {"dense", "triton", "gpu"}
    (tmp_path / "needle.json").write_text(json.dumps({"rows": rows}))
    pq.curves(tmp_path, [0.25])
    summary = (tmp_path / "summary.md").read_text()
    assert "Noise floor" in summary and "| triton | 100.0% |" in summary