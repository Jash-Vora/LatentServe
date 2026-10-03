"""The quality benchmark's own arithmetic, and its LatentServe arm end to end.

The vLLM arm cannot run here; its scoring is the same helpers applied to
vLLM's log-probabilities, so the helpers are what is tested, plus the full
LatentServe arm and the comparison on CPU with a tiny model.
"""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from benchmarks.runners import quality_vs_vllm as q  # noqa: E402


def test_nll_top1_kl_against_hand_computation():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.0]])
    ids = torch.tensor([0, 1, 2])
    nll, top1, kl = q.nll_top1_kl(logits, ids, torch.log_softmax(logits, -1)[:-1])
    # position 0 predicts ids[1]=1 under logits [2,0,0]
    assert nll[0] == pytest.approx(-math.log(1 / (math.e**2 + 2)))
    assert top1 == [0, 1]
    assert kl == pytest.approx([0.0, 0.0], abs=1e-6)        # identical to itself


def test_slicing_changes_no_score():
    """Scoring in slices must give exactly what scoring the whole sequence
    gives — that is what makes the memory fix free."""
    torch.manual_seed(4)
    logits, ids = torch.randn(9, 7), torch.randint(0, 7, (9,))
    ref = torch.log_softmax(torch.randn(8, 7), -1)
    whole = q.nll_top1_kl(logits, ids, ref)
    parts = [[], [], []]
    for a in range(0, 8, 3):
        b = min(a + 3, 8)
        n, t1, kl, _ = q.score_positions(logits[a:b], ids[a + 1 : b + 1], ref[a:b])
        for acc, piece in zip(parts, (n, t1, kl)):
            acc += piece
    for got, want in zip(parts, whole):
        assert got == pytest.approx(want)


def test_continuation_logprob_scores_only_the_continuation():
    logits = torch.zeros(4, 5)
    logits[1, 3] = 10.0                                     # predicts token 3 at pos 2
    ids = torch.tensor([0, 1, 3, 4])
    lp = q.continuation_logprob(logits, ids, start=2)
    expect = torch.log_softmax(logits[1], -1)[3] + torch.log_softmax(logits[2], -1)[4]
    assert lp == pytest.approx(float(expect))


def test_acc_norm_can_disagree_with_acc():
    """A long option loses on total log-probability but wins per character."""
    assert q.mc_predict([-10.0, -6.0], [40, 5]) == (1, 0)


def test_first_divergence():
    assert q.first_divergence([1, 2, 3], [1, 2, 3]) == 3
    assert q.first_divergence([1, 2, 3], [1, 9, 3]) == 1


def test_verdict_categories():
    assert q.verdict(1.0, 1.0).startswith("on par")
    assert q.verdict(2.0, 1.0).startswith("degraded")
    assert q.verdict(0.5, 1.0).startswith("better")
    assert q.verdict(None, 1.0) == "n/a"


def test_latentserve_arm_and_compare_end_to_end(tmp_path, capsys):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from model.qwen import ModelShape

    cfg = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
               num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=2048)

    class Ref:
        def __init__(self):
            torch.manual_seed(0)
            self.model = Qwen2ForCausalLM(Qwen2Config(**cfg)).to(torch.float32).eval()
            self.tokenizer = None
            self.device = "cpu"
            self.shape = ModelShape(2, 8, 2, 16, 128, 128, 2048, "torch.float32")

    class Args:
        chunks, chunk_len, mc_items, score_slice = 2, 48, 4, 10
        gen_prompts, gen_prompt_len, gen_new_tokens, gen_batch = 3, 10, 12, 2
        synthetic, results_dir = True, str(tmp_path)

    ls_side = q.run_latentserve_arm(Args, lambda dtype: Ref())
    for s in ("hf_fp32", "hf_fp16", "latentserve_unfused", "latentserve_fused"):
        assert ls_side["metrics"][s]["tokens"] == 2 * 47
        assert len(ls_side["generations"].get(s, [[]] * 3)) == 3
    # On CPU every system computes the same numbers: perfect agreement.
    assert ls_side["metrics"]["latentserve_fused"]["top1_agree"] == pytest.approx(1.0)
    assert all(len(g) == 12 for g in ls_side["generations"]["latentserve_fused"])

    fake_vllm = {"mc_preds": ls_side["mc_preds"]["latentserve_fused"],
                 "generations": ls_side["generations"]["latentserve_fused"],
                 "metrics": dict(ls_side["metrics"]["latentserve_fused"], kl_mean=None, kl_max=None)}
    (tmp_path / q.RESULTS[2]).write_text(json.dumps(fake_vllm))
    capsys.readouterr()
    q.compare(Args)
    out = capsys.readouterr().out
    assert "synthetic data" in out
    for section in ("perplexity", "ARC-Easy", "Generation", "fused vs vllm"):
        assert section in out

