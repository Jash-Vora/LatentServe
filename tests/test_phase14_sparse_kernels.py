"""Phase 14 sparse kernels (kernels/cuda/paged_sparse*): each kernel against a
plain reference, then the whole pipeline against the dense kernel."""

from __future__ import annotations

import math
import re

import pytest

torch = pytest.importorskip("torch")

from kernels.cuda import paged_sparse as ps  # noqa: E402

try:
    ps.compile_cubin("sm_75")
    CAN_COMPILE = True
except Exception:  # noqa: BLE001 - environmental
    CAN_COMPILE = False

requires_nvrtc = pytest.mark.skipif(not CAN_COMPILE, reason="needs CuPy + NVRTC")
requires_gpu = pytest.mark.skipif(not (CAN_COMPILE and torch.cuda.is_available()),
                                  reason="needs a GPU and CuPy")


@requires_nvrtc
def test_kernels_compile_without_spills():
    _, log = ps.compile_cubin("sm_75")
    if "registers" in (log or ""):
        kernels = log.count("Compiling entry function")
        assert kernels >= 4 and log.count("0 bytes spill stores") == kernels, log
        sparse = re.search(r"paged_sparse_fp16.*?Used (\d+) registers", log, re.S)
        assert sparse and int(sparse.group(1)) <= 200, log


def _setup(lens, h=2, n_rep=6, d=128, page=16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = (max(lens) + page - 1) // page
    nb = len(lens) * pages + 3
    k = torch.randn(nb, page, h, d, dtype=torch.float16, device="cuda", generator=g)
    v = torch.randn(nb, page, h, d, dtype=torch.float16, device="cuda", generator=g)
    tables = torch.randperm(nb, device="cuda", generator=g)[: len(lens) * pages]
    tables = tables.reshape(len(lens), pages).to(torch.int32)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q = torch.randn(len(lens), h, n_rep, d, dtype=torch.float16, device="cuda", generator=g)
    kmin = torch.empty(nb, h, d, dtype=torch.float16, device="cuda")
    kmax = torch.empty_like(kmin)
    ps.rebuild_bounds(k, kmin, kmax)
    return q, k, v, tables, seq, kmin, kmax


def _gather(pool, tables, b, length, page=16):
    rows = [pool[int(tables[b, p // page]), p % page] for p in range(length)]   # [H, D] each
    return torch.stack(rows, dim=1).float()                                     # [H, T, D]


def _reference(q, k, v, tables, seq, keep_pages, page=16):
    """Attention for each sequence over the pages in keep_pages[b][h] only."""
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for b in range(q.shape[0]):
        L = int(seq[b])
        K, V = _gather(k, tables, b, L), _gather(v, tables, b, L)
        for h in range(q.shape[1]):
            keep = torch.zeros(L, dtype=torch.bool, device=q.device)
            for p in keep_pages[b][h]:
                keep[p * page:min(L, (p + 1) * page)] = True
            s = (q[b, h].float() @ K[h].T) / math.sqrt(q.shape[-1])
            s = s.masked_fill(~keep[None, :], float("-inf"))
            out[b, h] = torch.softmax(s, -1) @ V[h]
    return out


@requires_gpu
def test_bounds_updated_token_by_token_equal_the_true_min_and_max():
    page, h, d, nb = 16, 2, 128, 8
    g = torch.Generator(device="cuda").manual_seed(1)
    kmin = torch.full((nb, h, d), 99.0, dtype=torch.float16, device="cuda")
    kmax = torch.full_like(kmin, -99.0)
    written = {}
    blocks = [5, 2]                                   # one sequence across two pages
    for pos in range(27):
        slot = blocks[pos // page] * page + pos % page
        k_new = torch.randn(1, h, d, dtype=torch.float16, device="cuda", generator=g)
        ps.update_bounds(k_new, torch.tensor([slot], device="cuda"), kmin, kmax)
        written.setdefault(blocks[pos // page], []).append(k_new[0])
    torch.cuda.synchronize()
    for blk, rows in written.items():
        stack = torch.stack(rows)                     # [n, H, D]
        assert torch.equal(kmin[blk], stack.amin(0)) and torch.equal(kmax[blk], stack.amax(0))


@requires_gpu
def test_indexer_scores_are_the_reference_bounds():
    from model.attention.sparse import page_bounds

    lens = [700, 333]
    q, k, v, tables, seq, kmin, kmax = _setup(lens)
    scores = ps.page_scores(q, kmin, kmax, tables, seq, recent=2).clone()
    for b, L in enumerate(lens):
        np_ = -(-L // 16)
        K = _gather(k, tables, b, np_ * 16)[None]                    # whole pages, as rebuilt
        ref = page_bounds(q[b:b + 1].float(), K, 16).amax(dim=2)[0]  # [H, P]
        mid = slice(1, np_ - 2)
        torch.testing.assert_close(scores[b, :, mid], ref[:, mid], rtol=1e-4, atol=1e-3)
        assert torch.isinf(scores[b, :, 0]).all() and (scores[b, :, 0] > 0).all()
        assert (scores[b, :, np_ - 2:np_] == float("inf")).all()
        assert (scores[b, :, np_:] == float("-inf")).all()


@requires_gpu
@pytest.mark.parametrize("num_splits", [None, 1, 3])
def test_sparse_kernel_attends_exactly_the_given_pages(num_splits):
    """A fixed selection isolates the kernel from top-k tie-breaking. The
    shorter sequence's selection includes pages past its end, which a ragged
    batch produces and the kernel must skip."""
    lens = [700, 150]
    q, k, v, tables, seq, _, _ = _setup(lens, seed=2)
    picks = [[0, 3, 7, 12, 20, 43], [0, 2, 5, 9, 30, 40]]          # seq 1 has 10 pages
    sel = torch.tensor([[p, p[::-1]] for p in picks], dtype=torch.int32, device="cuda")
    got = ps.sparse_decode(q, k, v, tables, seq, sel, num_splits=num_splits)
    keep = [[[p for p in picks[b] if p < -(-lens[b] // 16)]] * 2 for b in range(2)]
    want = _reference(q, k, v, tables, seq, keep)
    torch.testing.assert_close(got.float(), want, atol=5e-3, rtol=1e-2)


@requires_gpu
def test_full_budget_reproduces_the_dense_kernel():
    from kernels.cuda import paged_decode_cuda as pdc

    lens = [1024, 300]
    q, k, v, tables, seq, kmin, kmax = _setup(lens, seed=3)
    dense = pdc.paged_decode_cuda(q, k, v, tables, seq, max(lens)).clone()
    sparse = ps.sparse_attention(q, k, v, tables, seq, kmin, kmax, ratio=1.0)
    torch.testing.assert_close(sparse.float(), dense.float(), atol=2e-3, rtol=1e-2)


@requires_gpu
def test_quarter_budget_matches_reference_attention_on_its_own_selection():
    lens = [2048]
    q, k, v, tables, seq, kmin, kmax = _setup(lens, seed=4)
    kk = ps.budget(0.25, tables.shape[1])
    sel = ps.select(ps.page_scores(q, kmin, kmax, tables, seq), kk)
    got = ps.sparse_decode(q, k, v, tables, seq, sel)
    keep = [[sorted(set(sel[0, h].tolist())) for h in range(2)]]
    assert all(0 in pages and 127 in pages and 126 in pages for pages in keep[0])
    torch.testing.assert_close(got.float(), _reference(q, k, v, tables, seq, keep),
                               atol=5e-3, rtol=1e-2)


@requires_gpu
def test_the_pipeline_survives_cuda_graph_capture():
    lens = [1500, 900]
    q, k, v, tables, seq, kmin, kmax = _setup(lens, seed=5)
    ps.sparse_attention(q, k, v, tables, seq, kmin, kmax, ratio=0.25)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ps.sparse_attention(q, k, v, tables, seq, kmin, kmax, ratio=0.25)
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.cuda.synchronize()
    want = ps.sparse_attention(q, k, v, tables, seq, kmin, kmax, ratio=0.25)
    torch.testing.assert_close(out.float(), want.float(), atol=1e-3, rtol=1e-3)


@requires_gpu
def test_kernels_load_without_spills_or_shared_memory():
    for name, r in ps.kernel_resources().items():
        assert r["local_bytes"] == 0 and r["shared_bytes"] == 0, (name, r)


# ------------------------------------------- Phase 15 follow-up: mass ---


@requires_gpu
def test_mass_scores_match_the_reference_estimate():
    from model.attention.sparse import estimated_mass, page_bounds, page_counts

    lens = [700, 333]
    q, k, v, tables, seq, kmin, kmax = _setup(lens, seed=11)
    got = ps.page_scores_mass(q, kmin, kmax, tables, seq, recent=2).clone()
    scale = 1.0 / math.sqrt(q.shape[-1])
    for b, L in enumerate(lens):
        np_ = -(-L // 16)
        K = _gather(k, tables, b, np_ * 16)[None]                 # whole pages, as rebuilt
        ref = estimated_mass(page_bounds(q[b:b + 1].float(), K, 16), page_counts(L, 16, "cuda"),
                             scale)[0]                             # [H, P]
        mid = slice(1, np_ - 2)
        torch.testing.assert_close(got[b, :, mid], ref[:, mid], rtol=1e-3, atol=1e-6)
        assert (got[b, :, 0] == float("inf")).all()
        assert (got[b, :, np_ - 2:np_] == float("inf")).all()
        assert (got[b, :, np_:] == float("-inf")).all()


@requires_gpu
def test_mass_scoring_at_full_budget_reproduces_dense():
    from kernels.cuda import paged_decode_cuda as pdc

    q, k, v, tables, seq, kmin, kmax = _setup([1024, 300], seed=12)
    dense = pdc.paged_decode_cuda(q, k, v, tables, seq, 1024).clone()
    got = ps.sparse_attention(q, k, v, tables, seq, kmin, kmax, ratio=1.0, scoring="mass")
    torch.testing.assert_close(got.float(), dense.float(), atol=2e-3, rtol=1e-2)


@requires_gpu
def test_mass_scoring_survives_cuda_graph_capture():
    q, k, v, tables, seq, kmin, kmax = _setup([1500, 900], seed=13)
    ps.sparse_attention(q, k, v, tables, seq, kmin, kmax, ratio=0.25, scoring="mass")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ps.sparse_attention(q, k, v, tables, seq, kmin, kmax, ratio=0.25, scoring="mass")
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.cuda.synchronize()
    want = ps.sparse_attention(q, k, v, tables, seq, kmin, kmax, ratio=0.25, scoring="mass")
    torch.testing.assert_close(out.float(), want.float(), atol=1e-3, rtol=1e-3)
