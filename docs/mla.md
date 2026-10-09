# Multi-head Latent Attention (MLA)

The MLA-inspired track, a compressed latent KV cache (Phases 7–10), was closed after Phase 7's compressibility study, and INT8 caching took its place as the memory-saving technique.

The reason is in the study's numbers (`phase7.md`). The 2.24× smaller cache that an energy threshold suggested was an illusion: values carry 3–5× the reconstruction error of keys, so a latent at 10% value error compresses only about 1.6×, or 1.49× with separate K and V latents. INT8 gives exactly 2× with no reconstruction compute. The experiments are in `compression/` (spectra, truncation and learned projections). The final go/no-go wasn't written down at the time, and no MLA kernels were built; `kernels/mla/` is an empty package.
