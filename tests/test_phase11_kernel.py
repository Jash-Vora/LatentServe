"""
Phase 11 — paged decode attention.

The kernel replaces a gather plus SDPA with a single in-place read, so
the only thing that makes it trustworthy is equivalence to what it
replaces. These tests check the *algorithm* (block walk, online softmax,
split-and-combine) against dense attention on CPU, so the logic is
verified independently of whether Triton lowers it correctly.

That separation is the point: given a wrong output tensor, "the
algorithm is wrong" and "the Triton lowering is wrong" are otherwise
indistinguishable. The GPU-gated tests at the bottom compare the Triton
kernel against this same reference, so a failure there localises to the
lowering.

    pytest tests/test_phase11_kernel.py -v            # CPU, reference only
    pytest tests/test_phase11_kernel.py -v -m gpu     # adds the Triton path
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from kernels.gqa.paged_decode import (  # noqa: E402
    HAS_TRITON,
    combine_splits,
    paged_decode_attention,
    paged_decode_reference,
)

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON),
    reason="needs a CUDA device with Triton",
)


def build_case(batch=2, h_kv=2, n_rep=3, d=32, page=16, lengths=None, seed=0):
    """A paged cache with deliberately *scattered* block tables.

    Contiguous block ids would let an indexing bug pass: page p landing
    at block p is exactly what a broken block-table lookup produces.
    """
    torch.manual_seed(seed)
    lengths = lengths or [40, 17]
    max_pages = max((l + page - 1) // page for l in lengths)
    num_blocks = batch * max_pages + 5

    k_pool = torch.randn(num_blocks, page, h_kv, d)
    v_pool = torch.randn(num_blocks, page, h_kv, d)
    perm = torch.randperm(num_blocks)[: batch * max_pages].reshape(batch, max_pages)
    q = torch.randn(batch, h_kv, n_rep, d)
    return q, k_pool, v_pool, perm.to(torch.int32), torch.tensor(lengths, dtype=torch.int32)


def dense_attention(q, k_pool, v_pool, block_tables, seq_lens):
    """Ground truth: gather every live token, then one softmax. This is
    what the current runtime does — gather then SDPA — so agreeing with
    it is precisely the property Phase 11 must preserve."""
    b, h_kv, n_rep, d = q.shape
    page = k_pool.shape[1]
    out = torch.zeros_like(q)
    for i in range(b):
        length = int(seq_lens[i])
        idx = []
        for p in range((length + page - 1) // page):
            blk = int(block_tables[i, p])
            for off in range(page):
                if p * page + off < length:
                    idx.append((blk, off))
        for h in range(h_kv):
            k = torch.stack([k_pool[blk, off, h] for blk, off in idx])
            v = torch.stack([v_pool[blk, off, h] for blk, off in idx])
            logits = (q[i, h] @ k.T) / math.sqrt(d)
            out[i, h] = torch.softmax(logits, dim=-1) @ v
    return out


# ----------------------------------------------------------------------
# The algorithm
# ----------------------------------------------------------------------


def test_reference_matches_dense_attention():
    q, k_pool, v_pool, tables, lens = build_case()
    got = paged_decode_reference(q, k_pool, v_pool, tables, lens)
    want = dense_attention(q, k_pool, v_pool, tables, lens)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("num_splits", [1, 2, 3, 4, 8])
def test_split_k_is_exact(num_splits):
    """Splitting the KV length changes how the softmax is accumulated but
    must not change the result. If it does, the partial-softmax combine
    is wrong — and at batch 1 this kernel *needs* splitting to fill the
    GPU, so an approximate combine would be silently wrong everywhere."""
    q, k_pool, v_pool, tables, lens = build_case(lengths=[63, 40])
    want = dense_attention(q, k_pool, v_pool, tables, lens)
    got = paged_decode_reference(q, k_pool, v_pool, tables, lens, num_splits=num_splits)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_more_splits_than_pages_is_harmless():
    """The default split count is chosen to fill the SMs, so it can
    exceed the number of pages on a short sequence. Empty splits must
    contribute nothing rather than NaN — exp(-inf - -inf)."""
    q, k_pool, v_pool, tables, lens = build_case(lengths=[10, 5], page=16)
    want = dense_attention(q, k_pool, v_pool, tables, lens)
    got = paged_decode_reference(q, k_pool, v_pool, tables, lens, num_splits=8)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33])
def test_ragged_final_page(length):
    """Sequence lengths are not multiples of the page size, and tokens
    past the end must be masked rather than attended to."""
    q, k_pool, v_pool, tables, lens = build_case(batch=1, lengths=[length])
    want = dense_attention(q, k_pool, v_pool, tables, lens)
    got = paged_decode_reference(q, k_pool, v_pool, tables, lens, num_splits=2)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_sequences_do_not_leak_into_each_other():
    """Two sequences of very different lengths in one batch: the short
    one must not see the long one's blocks."""
    q, k_pool, v_pool, tables, lens = build_case(batch=2, lengths=[64, 3])
    got = paged_decode_reference(q, k_pool, v_pool, tables, lens)
    solo = paged_decode_reference(
        q[1:2], k_pool, v_pool, tables[1:2], lens[1:2]
    )
    torch.testing.assert_close(got[1:2], solo, rtol=1e-5, atol=1e-6)


def test_int8_path_matches_its_own_dequantized_tensors():
    """The INT8 read must equal dense attention over the values it
    decodes to — separating quantization error (expected) from a wrong
    scale axis (not). K's scale is per channel, V's per token; swapping
    them produces plausible output and wrong numbers."""
    torch.manual_seed(3)
    b, h_kv, n_rep, d, page = 1, 2, 4, 32, 16
    num_blocks, length = 6, 40
    k_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8)
    v_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8)
    k_scale = torch.rand(num_blocks, h_kv, d) * 0.01 + 0.001
    v_scale = torch.rand(num_blocks, page, h_kv) * 0.01 + 0.001
    tables = torch.randperm(num_blocks)[:3].reshape(1, 3).to(torch.int32)
    lens = torch.tensor([length], dtype=torch.int32)
    q = torch.randn(b, h_kv, n_rep, d)

    k_deq = k_q.float() * k_scale[:, None, :, :]
    v_deq = v_q.float() * v_scale[:, :, :, None]
    want = dense_attention(q, k_deq, v_deq, tables, lens)
    got = paged_decode_reference(
        q, k_q, v_q, tables, lens, k_scale=k_scale, v_scale=v_scale, num_splits=2
    )
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_asymmetric_zero_points_are_applied_on_the_right_axis():
    torch.manual_seed(4)
    num_blocks, page, h_kv, d, length = 5, 16, 2, 32, 30
    k_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8)
    v_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8)
    k_scale = torch.rand(num_blocks, h_kv, d) * 0.01 + 0.001
    v_scale = torch.rand(num_blocks, page, h_kv) * 0.01 + 0.001
    k_zero = torch.randn(num_blocks, h_kv, d)
    v_zero = torch.randn(num_blocks, page, h_kv)
    tables = torch.randperm(num_blocks)[:2].reshape(1, 2).to(torch.int32)
    lens = torch.tensor([length], dtype=torch.int32)
    q = torch.randn(1, h_kv, 3, d)

    k_deq = k_q.float() * k_scale[:, None, :, :] + k_zero[:, None, :, :]
    v_deq = v_q.float() * v_scale[:, :, :, None] + v_zero[:, :, :, None]
    want = dense_attention(q, k_deq, v_deq, tables, lens)
    got = paged_decode_reference(
        q, k_q, v_q, tables, lens, k_scale=k_scale, v_scale=v_scale,
        k_zero=k_zero, v_zero=v_zero, num_splits=2,
    )
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_combine_splits_handles_an_all_empty_split():
    acc = torch.zeros(1, 1, 3, 4, 8)
    m = torch.full((1, 1, 3, 4), float("-inf"))
    l = torch.zeros(1, 1, 3, 4)
    m[:, :, 0] = 0.0
    l[:, :, 0] = 1.0
    acc[:, :, 0] = 1.0
    out = combine_splits(acc, m, l, torch.float32)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.ones(1, 1, 4, 8))


# ----------------------------------------------------------------------
# The Triton lowering (GPU only)
# ----------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("num_splits", [1, 4])
def test_triton_matches_reference_fp16(num_splits):
    q, k_pool, v_pool, tables, lens = build_case(n_rep=6, d=128, page=16, lengths=[130, 48])
    args = [t.cuda() for t in (q.half(), k_pool.half(), v_pool.half(), tables, lens)]
    want = paged_decode_reference(*args, num_splits=num_splits)
    got = paged_decode_attention(*args, num_splits=num_splits)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-3)


@requires_gpu
def test_triton_matches_reference_int8():
    torch.manual_seed(5)
    num_blocks, page, h_kv, d = 12, 16, 2, 128
    k_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8).cuda()
    v_q = torch.randint(-127, 127, (num_blocks, page, h_kv, d), dtype=torch.int8).cuda()
    k_scale = (torch.rand(num_blocks, h_kv, d) * 0.01 + 0.001).cuda()
    v_scale = (torch.rand(num_blocks, page, h_kv) * 0.01 + 0.001).cuda()
    tables = torch.randperm(num_blocks)[:8].reshape(1, 8).to(torch.int32).cuda()
    lens = torch.tensor([120], dtype=torch.int32).cuda()
    q = torch.randn(1, h_kv, 6, d).half().cuda()

    want = paged_decode_reference(q, k_q, v_q, tables, lens,
                                  k_scale=k_scale, v_scale=v_scale, num_splits=4)
    got = paged_decode_attention(q, k_q, v_q, tables, lens,
                                 k_scale=k_scale, v_scale=v_scale, num_splits=4)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-3)