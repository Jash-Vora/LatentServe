# Multi-head Latent Attention (MLA)

TODO — write once Phase 7 (MLA) lands. Should cover:
- Mathematical derivation: c_t^KV = W^DKV h_t, and why we cache the latent instead of full K/V.
- Decoupled RoPE (Phase 8).
- Matrix absorption (Phase 9): MLA-1 (materialize K/V) vs MLA-2 (reconstruct from latent)
  vs MLA-3 (absorbed projections), with the benchmark numbers showing where each wins.
- Compression study results (Phase 10): latent_dim in {256, 384, 512, 768, 1024} vs
  memory/quality/latency.
