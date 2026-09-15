# Custom GPU Kernels

TODO — write once Phase 11 lands. For each kernel (GQA decode, MLA decode,
latent-cache attention, RoPE, fused variants), record:
- Triton vs CUDA C++ implementation notes
- occupancy, memory bandwidth achieved vs peak, register pressure
- before/after Nsight Compute comparison (this is the whole point — don't skip it)
