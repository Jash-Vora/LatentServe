# Custom GPU Kernels

Where each kernel lives and where it is documented.

| Kernel | Source | Documented in |
| --- | --- | --- |
| Decode attention, fp16 (CUDA-core) | `kernels/cuda/paged_decode_fp16.cu` | `phase12_kernel_findings.md` |
| Decode attention, INT8 | `kernels/cuda/paged_decode_int8.cu` | `phase14.md`, `phase16_int8_sparse.md` |
| Sparse decode: page bounds, indexer, attention | `kernels/cuda/paged_sparse_fp16.cu` | `phase14_sparse.md` |
| INT8 decode-step write, fused | `kernels/cuda/int8_write.cu` | `phase16_int8_sparse.md` |
| Decode attention, Triton (the first version) | `kernels/gqa/paged_decode.py` | `phase12_kernel_findings.md` |

The CUDA kernels compile at runtime with NVRTC through CuPy. Nsight Compute isn't permitted in the container these runs used, so kernels were analysed from compiler resource counts, ablation kernels and machine-code censuses instead (see `profiling.md`).
