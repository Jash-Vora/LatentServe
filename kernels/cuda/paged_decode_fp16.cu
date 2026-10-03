// Paged decode attention on CUDA cores, for Turing (sm_75).
//
// Phase 12 found that Triton compiles the production kernel's matrix
// multiplies to scalar fp32 FMA on this GPU (HMMA = 0, no `mma` in its
// PTX), staged through shared memory: 40 KB per program, one program per
// SM, 12% occupancy, ~69 GB/s. Decode attention is a matrix-*vector*
// problem here — 6 query rows per KV head — so this kernel does it with
// CUDA-core arithmetic directly, and pads nothing.
//
// Layout: one warp per (sequence, split, KV head). The two half-warps take
// alternate tokens; within a half, each of the 16 lanes owns DPL = 8 of the
// 128 dimensions, so a K or V row is one 16-byte vector load per lane, and
// the 16 lanes together read 256 contiguous bytes. Each lane forms partial
// dot products for all NREP query rows over its 8 dimensions; four shuffle
// steps reduce them across the half-warp.
//
// The softmax is updated once per group of TOKG tokens per half, not per
// token: scores for the group, one rescale of the running sums, then V.
// Partials are written in the Triton kernel's layout ([B, H, S, 16, D] and
// [B, H, S, 16]), so its combine kernel merges them unchanged.

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
#define TOKG 4                      // tokens per half-warp per softmax group
#endif

#define HALF_LANES 16
#define DPL (HEAD_DIM / HALF_LANES)  // dimensions per lane: 8
#define GROUPS (PAGE / (2 * TOKG))   // softmax groups per page
#define FULL 0xffffffffu
// NVRTC has no <math.h>, so no INFINITY: build -inf from its bit pattern.
#define NEG_INF __int_as_float(0xff800000)

static __device__ __forceinline__ void load8(const __half* p, float* f) {
    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(p));
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(h2[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

extern "C" __global__ void __launch_bounds__(32)
paged_decode_fp16(const __half* __restrict__ q, const __half* __restrict__ k_pool,
                  const __half* __restrict__ v_pool, const int* __restrict__ block_tables,
                  const int* __restrict__ seq_lens, float* __restrict__ part_acc,
                  float* __restrict__ part_m, float* __restrict__ part_l,
                  int max_pages, int num_splits,
                  long long sq_b, long long sq_h, long long sq_m,
                  long long sk_b, long long sk_p, long long sk_h,
                  long long so_b, long long so_h, long long so_s, long long so_m,
                  long long sm_b, long long sm_h, long long sm_s, float scale) {
    const int b = blockIdx.x, split = blockIdx.y, h = blockIdx.z;
    const int lane = threadIdx.x;
    const int half_id = lane >> 4;
    const int d0 = (lane & 15) * DPL;

    // This lane's slice of every query row, pre-scaled, in fp32.
    float qf[NREP][DPL];
#pragma unroll
    for (int r = 0; r < NREP; ++r) {
        load8(q + b * sq_b + h * sq_h + r * sq_m + d0, qf[r]);
#pragma unroll
        for (int j = 0; j < DPL; ++j) qf[r][j] *= scale;
    }

    const int L = seq_lens[b];
    const int num_pages = (L + PAGE - 1) / PAGE;
    const int pps = (num_pages + num_splits - 1) / num_splits;
    const int lo = split * pps;
    const int hi = min(lo + pps, num_pages);

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
        const __half* kb = k_pool + blk * sk_b + h * sk_h + d0;
        const __half* vb = v_pool + blk * sk_b + h * sk_h + d0;
#pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            // Pass 1: this half's TOKG scores, reduced across its 16 lanes.
            float s[TOKG][NREP];
#pragma unroll
            for (int i = 0; i < TOKG; ++i) {
                const int t = 2 * (g * TOKG + i) + half_id;
                float kf[DPL];
                load8(kb + t * sk_p, kf);
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

            // Tokens past the sequence end: score -inf, and their V is never
            // touched below — a zero weight on stale memory is still NaN if
            // that memory holds NaN bits.
            bool valid[TOKG];
#pragma unroll
            for (int i = 0; i < TOKG; ++i) {
                valid[i] = p * PAGE + 2 * (g * TOKG + i) + half_id < L;
                if (!valid[i])
#pragma unroll
                    for (int r = 0; r < NREP; ++r) s[i][r] = NEG_INF;
            }

            // Pass 2: one rescale per group, then the V accumulation.
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
                load8(vb + t * sk_p, vf);
#pragma unroll
                for (int r = 0; r < NREP; ++r) {
                    l[r] += s[i][r];
#pragma unroll
                    for (int j = 0; j < DPL; ++j) acc[r][j] = fmaf(s[i][r], vf[j], acc[r][j]);
                }
            }
        }
    }

    // Merge the two half-warps' states: lane x and lane x^16 own the same
    // dimensions of different tokens.
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
