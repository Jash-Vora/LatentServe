// Paged decode attention over an INT8 KV cache, on CUDA cores, for Turing.
//
// The fp16 kernel (paged_decode_fp16.cu) reaches ~80% of the memory ceiling
// at batch 16 / 8K: it is limited by bytes. INT8 halves the bytes, so this
// is where it finally pays — on the Triton kernel, which was limited by
// arithmetic, INT8 was 7-30% *slower* than fp16. Same layout, same partial
// format, same merge; the differences are all in how a row is read:
//
//   K  int8, scale (and zero point) per (block, head, channel). A lane
//      loads its 8 channels' scales once per page and applies them as each
//      token is converted: 8 multiplies per token against the 48 of the dot
//      products. Pre-scaling the query per page instead would hold 48 more
//      values in registers, and registers are what Phase 12 was about.
//   V  int8, scale (and zero point) per (block, token, head). The scale
//      folds into the softmax weight — acc += (w * scale) * v + w * zero —
//      so V is never dequantized.
//   The last page's K comes from the fp16 residual (Phase 14c): it is not
//      quantized until the page is complete. One branch per page, taken
//      identically by every lane.
//
// Dequantization follows _dequant_k/_dequant_v in kernels/gqa/paged_decode.py:
// value = int8 * scale (+ zero), in fp32.

#include <cuda_fp16.h>

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif
#ifndef NREP
#define NREP 6
#endif
#ifndef PAGE
#define PAGE 16
#endif
#ifndef TOKG
#define TOKG 4
#endif
#ifndef ASYM
#define ASYM 0                      // zero points present
#endif
#ifndef HAS_RES
#define HAS_RES 0                   // last page's K in the fp16 residual
#endif

#define HALF_LANES 16
#define DPL (HEAD_DIM / HALF_LANES)  // dimensions per lane: 8
#define GROUPS (PAGE / (2 * TOKG))
#define FULL 0xffffffffu
#define NEG_INF __int_as_float(0xff800000)

static __device__ __forceinline__ void load8h(const __half* p, float* f) {
    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(p));
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(h2[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

// Eight int8 values: one 8-byte load.
static __device__ __forceinline__ void load8i(const signed char* p, float* f) {
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(p));
    const char4 a = *reinterpret_cast<const char4*>(&raw.x);
    const char4 b = *reinterpret_cast<const char4*>(&raw.y);
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
    f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}

static __device__ __forceinline__ void load8f(const float* p, float* f) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p));
    const float4 b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
    f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}

// Minimum resident one-warp blocks per SM the compiler must fit in the
// register file (65,536 / (MIN_BLOCKS x 32) registers per thread). Left
// alone, the INT8 variants use 199-233 registers against the fp16 kernel's
// 168 — 8-10 resident warps instead of 12. --maxrregcount cannot fix that:
// it is ignored for kernels that declare launch bounds.
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif

extern "C" __global__ void __launch_bounds__(32, MIN_BLOCKS)
paged_decode_int8(const __half* __restrict__ q, const signed char* __restrict__ k_pool,
                  const signed char* __restrict__ v_pool, const int* __restrict__ block_tables,
                  const int* __restrict__ seq_lens, float* __restrict__ part_acc,
                  float* __restrict__ part_m, float* __restrict__ part_l,
                  const float* __restrict__ k_scale, const float* __restrict__ v_scale,
                  const float* __restrict__ k_zero, const float* __restrict__ v_zero,
                  const __half* __restrict__ k_res, const int* __restrict__ res_rows,
                  int max_pages, int num_splits,
                  long long sq_b, long long sq_h, long long sq_m,
                  long long sk_b, long long sk_p, long long sk_h,
                  long long ks_b, long long ks_h,
                  long long vs_b, long long vs_p, long long vs_h,
                  long long sr_r, long long sr_p, long long sr_h,
                  long long so_b, long long so_h, long long so_s, long long so_m,
                  long long sm_b, long long sm_h, long long sm_s, float scale) {
    const int b = blockIdx.x, split = blockIdx.y, h = blockIdx.z;
    const int lane = threadIdx.x;
    const int half_id = lane >> 4;
    const int d0 = (lane & 15) * DPL;

    float qf[NREP][DPL];
#pragma unroll
    for (int r = 0; r < NREP; ++r) {
        load8h(q + b * sq_b + h * sq_h + r * sq_m + d0, qf[r]);
#pragma unroll
        for (int j = 0; j < DPL; ++j) qf[r][j] *= scale;
    }

    const int L = seq_lens[b];
    const int num_pages = (L + PAGE - 1) / PAGE;
    const int pps = (num_pages + num_splits - 1) / num_splits;
    const int lo = split * pps;
    const int hi = min(lo + pps, num_pages);
#if HAS_RES
    const __half* kr = k_res + (long long)res_rows[b] * sr_r + h * sr_h + d0;
#endif

    float m[NREP], l[NREP], acc[NREP][DPL];
#pragma unroll
    for (int r = 0; r < NREP; ++r) {
        m[r] = NEG_INF;
        l[r] = 0.f;
#pragma unroll
        for (int j = 0; j < DPL; ++j) acc[r][j] = 0.f;
    }

    for (int p = lo; p < hi; ++p) {
        const int blk = block_tables[(long long)b * max_pages + p];
        const signed char* kb = k_pool + blk * sk_b + h * sk_h + d0;
        const signed char* vb = v_pool + blk * sk_b + h * sk_h + d0;
        const float* vsb = v_scale + blk * vs_b + h * vs_h;
#if ASYM
        const float* vzb = v_zero + blk * vs_b + h * vs_h;
#endif
#if HAS_RES
        const bool from_res = (p == num_pages - 1);
#else
        const bool from_res = false;
#endif
        // This lane's 8 channel scales for the page's K (unused, and not
        // loaded, when K comes from the residual).
        float ksf[DPL];
#if ASYM
        float kzf[DPL];
#endif
        if (!from_res) {
            load8f(k_scale + blk * ks_b + h * ks_h + d0, ksf);
#if ASYM
            load8f(k_zero + blk * ks_b + h * ks_h + d0, kzf);
#endif
        }
#pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            float s[TOKG][NREP];
#pragma unroll
            for (int i = 0; i < TOKG; ++i) {
                const int t = 2 * (g * TOKG + i) + half_id;
                float kf[DPL];
#if HAS_RES
                if (from_res) {
                    load8h(kr + t * sr_p, kf);
                } else
#endif
                {
                    load8i(kb + t * sk_p, kf);
#pragma unroll
                    for (int j = 0; j < DPL; ++j)
#if ASYM
                        kf[j] = fmaf(kf[j], ksf[j], kzf[j]);
#else
                        kf[j] *= ksf[j];
#endif
                }
#pragma unroll
                for (int r = 0; r < NREP; ++r) {
                    float dot = 0.f;
#pragma unroll
                    for (int j = 0; j < DPL; ++j) dot = fmaf(qf[r][j], kf[j], dot);
                    s[i][r] = dot;
                }
            }
#pragma unroll
            for (int i = 0; i < TOKG; ++i)
#pragma unroll
                for (int r = 0; r < NREP; ++r)
#pragma unroll
                    for (int off = 8; off > 0; off >>= 1)
                        s[i][r] += __shfl_xor_sync(FULL, s[i][r], off);

            bool valid[TOKG];
#pragma unroll
            for (int i = 0; i < TOKG; ++i) {
                valid[i] = p * PAGE + 2 * (g * TOKG + i) + half_id < L;
                if (!valid[i])
#pragma unroll
                    for (int r = 0; r < NREP; ++r) s[i][r] = NEG_INF;
            }

#pragma unroll
            for (int r = 0; r < NREP; ++r) {
                float mx = m[r];
#pragma unroll
                for (int i = 0; i < TOKG; ++i) mx = fmaxf(mx, s[i][r]);
                const float a = (mx == NEG_INF) ? 0.f : __expf(m[r] - mx);
                l[r] *= a;
#pragma unroll
                for (int j = 0; j < DPL; ++j) acc[r][j] *= a;
                m[r] = mx;
#pragma unroll
                for (int i = 0; i < TOKG; ++i) s[i][r] = valid[i] ? __expf(s[i][r] - mx) : 0.f;
            }
#pragma unroll
            for (int i = 0; i < TOKG; ++i) {
                if (!valid[i]) continue;
                const int t = 2 * (g * TOKG + i) + half_id;
                float vf[DPL];
                load8i(vb + t * sk_p, vf);
                const float vs_t = vsb[t * vs_p];
#if ASYM
                const float vz_t = vzb[t * vs_p];
#endif
#pragma unroll
                for (int r = 0; r < NREP; ++r) {
                    l[r] += s[i][r];
                    // V's per-token scale rides on the weight; the zero
                    // point adds w * zero to every channel.
                    const float w = s[i][r] * vs_t;
#pragma unroll
                    for (int j = 0; j < DPL; ++j) acc[r][j] = fmaf(w, vf[j], acc[r][j]);
#if ASYM
                    const float wz = s[i][r] * vz_t;
#pragma unroll
                    for (int j = 0; j < DPL; ++j) acc[r][j] += wz;
#endif
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < NREP; ++r) {
        const float mo = __shfl_xor_sync(FULL, m[r], 16);
        const float lo_ = __shfl_xor_sync(FULL, l[r], 16);
        const float M = fmaxf(m[r], mo);
        const float ea = (m[r] == NEG_INF) ? 0.f : __expf(m[r] - M);
        const float eb = (mo == NEG_INF) ? 0.f : __expf(mo - M);
        l[r] = l[r] * ea + lo_ * eb;
#pragma unroll
        for (int j = 0; j < DPL; ++j) {
            const float ao = __shfl_xor_sync(FULL, acc[r][j], 16);
            acc[r][j] = acc[r][j] * ea + ao * eb;
        }
        m[r] = M;
    }

    if (half_id == 0) {
        float* out = part_acc + b * so_b + h * so_h + split * so_s + d0;
#pragma unroll
        for (int r = 0; r < NREP; ++r) {
            float4* o = reinterpret_cast<float4*>(out + r * so_m);
            o[0] = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
            o[1] = make_float4(acc[r][4], acc[r][5], acc[r][6], acc[r][7]);
        }
        if (lane == 0) {
            const long long base = b * sm_b + h * sm_h + split * sm_s;
#pragma unroll
            for (int r = 0; r < NREP; ++r) {
                part_m[base + r] = m[r];
                part_l[base + r] = l[r];
            }
        }
    }
}
