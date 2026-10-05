// Phase 16 — INT8's decode-step write, fused: one launch per layer.
//
// The torch path does about a dozen small ops per layer for one token per
// sequence — V to fp32, its per-token absolute max (or min and max), a
// scale, divide, round, clamp, cast, two scatters, and K's scatter into the
// fp16 residual — ~280 extra launches per decode step, the bulk of INT8's
// end-to-end cost since Phase 14c. Here: one block per (token, head), one
// thread per channel.
//
// Byte-identical to the torch path (cache/int8_paged_cache.py, write()):
// fp32 throughout, the same epsilon floor, round-half-to-even (rintf, as
// torch.round), and every multiply/add/divide that the torch path rounds
// separately is rounded separately here too (__fmul_rn etc.) — otherwise
// the compiler may contract a multiply-add into one FMA that rounds once.

#include <cuda_fp16.h>

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif
#define WARPS (HEAD_DIM / 32)
#define FULL 0xffffffffu

static __device__ __forceinline__ float block_max(float x, float* buf) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) x = fmaxf(x, __shfl_xor_sync(FULL, x, off));
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) buf[w] = x;
    __syncthreads();
    float r = buf[0];
#pragma unroll
    for (int i = 1; i < WARPS; ++i) r = fmaxf(r, buf[i]);
    __syncthreads();
    return r;
}

extern "C" __global__ void __launch_bounds__(HEAD_DIM)
int8_decode_write(const __half* __restrict__ k, const __half* __restrict__ v,
                  const long long* __restrict__ slots, const long long* __restrict__ res_idx,
                  signed char* __restrict__ flat_v, float* __restrict__ flat_v_scale,
                  float* __restrict__ flat_v_zero, __half* __restrict__ k_res,
                  long long sk_b, long long sk_h, long long sv_b, long long sv_h,
                  int heads, int asym, float eps, float qmax, float levels, float offset) {
    __shared__ float buf[WARPS];
    const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float x = __half2float(v[b * sv_b + h * sv_h + d]);
    const long long slot = slots[b];
    const long long row = slot * heads + h;

    if (!asym) {
        const float amax = block_max(fabsf(x), buf);
        const float step = __fdiv_rn(fmaxf(amax, eps), qmax);
        const float q = fminf(fmaxf(rintf(__fdiv_rn(x, step)), -qmax), qmax);
        flat_v[row * HEAD_DIM + d] = (signed char)(int)q;
        if (d == 0) flat_v_scale[row] = step;
    } else {
        const float hi = block_max(x, buf);
        const float lo = -block_max(-x, buf);
        const float step = fmaxf(__fdiv_rn(__fsub_rn(hi, lo), levels), eps);
        const float u = fminf(fmaxf(rintf(__fdiv_rn(__fsub_rn(x, lo), step)), 0.f), levels);
        flat_v[row * HEAD_DIM + d] = (signed char)(int)__fsub_rn(u, offset);
        if (d == 0) {
            flat_v_scale[row] = step;
            flat_v_zero[row] = __fadd_rn(lo, __fmul_rn(offset, step));
        }
    }
    // K into the fp16 residual: the page is quantized when it completes.
    k_res[(res_idx[b] * heads + h) * HEAD_DIM + d] = k[b * sk_b + h * sk_h + d];
}
