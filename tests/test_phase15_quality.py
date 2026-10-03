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
    assert not v["pass"] and "3 paired failures" in v["reasons"][0]
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