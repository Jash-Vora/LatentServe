# DSA-inspired Sparse Attention

DeepSeek's sparse attention trains a small indexer to pick which tokens each query attends to. LatentServe uses a training-free, page-level stand-in (per-page minimum and maximum key bounds) and measures what that costs. The full story is in two notes:

- `phase14_sparse.md`: the indexer, the sparse decode kernels and the speedups (up to 1.66× per decode step at 37.5% of pages, batch 8 / 32K).
- `phase15_quality.md`: the quality study. How many answers each page budget loses, and why page selection, not sparsity itself, is the limit.

A learned indexer, as DeepSeek trains, was out of scope and is the main open follow-up. Source: `model/attention/sparse.py` and `kernels/cuda/paged_sparse_fp16.cu`.
