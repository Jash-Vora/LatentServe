// Phase 14 — sparse decode attention on CUDA cores, for Turing (sm_75).
//
// Three kernels:
//
//   page_bounds_update  keeps each page's per-channel min and max of K as
//                       tokens are written: one launch per layer per step,
//                       one thread per channel. (INT8's write side cost 280
//                       extra kernels per step; this costs 28.)
//   page_index          the indexer: for every page, the Quest-style upper
//                       bound sum_d max(q_d min_d, q_d max_d), maximised over
//                       the six query heads that share the KV head. The first
//                       page and the most recent ones score +inf (always
//                       kept); pages past the sequence score -inf. Reads two
//                       128-value vectors per page instead of 16 K and 16 V
//                       rows.
//   paged_sparse_fp16   attention over a list of selected pages: the fp16
//                       kernel (paged_decode_fp16.cu) with its page loop
//                       running over `sel` instead of a contiguous range. A
//                       selected page is read exactly as any page is — 256
//                       contiguous bytes per row — so selection adds no
//                       irregular gather. Kept as a copy so the proven dense
//                       kernel is untouched.
//
// Online softmax is order-independent, so the selection need not be sorted.

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
#ifndef PAGES_PER_WARP
#define PAGES_PER_WARP 16           // indexer: pages scored per warp, q loaded once
#endif

#define HALF_LANES 16
#define DPL (HEAD_DIM / HALF_LANES)
#define GROUPS (PAGE / (2 * TOKG))
#define FULL 0xffffffffu
#define NEG_INF __int_as_float(0xff800000)
#define POS_INF __int_as_float(0x7f800000)

// ------------------------------------------------------------------------
// 1. Page bounds, maintained on write.
// grid (B, H_kv), block HEAD_DIM threads.
extern "C" __global__ void
page_bounds_update(const __half* __restrict__ k_new, const long long* __restrict__ slots,
                   __half* __restrict__ kmin, __half* __restrict__ kmax,
                   long long sk_b, long long sk_h, long long sb_b, long long sb_h) {
    const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const long long slot = slots[b];
    const long long blk = slot / PAGE;
    const int off = (int)(slot % PAGE);
    const __half v = k_new[b * sk_b + h * sk_h + d];
    const long long i = blk * sb_b + h * sb_h + d;
    if (off == 0) {                 // first token of a fresh page: reset
        kmin[i] = v;
        kmax[i] = v;
    } else {                        // fp16 min/max through fp32 is exact
        const float f = __half2float(v);
        kmin[i] = __float2half(fminf(__half2float(kmin[i]), f));
        kmax[i] = __float2half(fmaxf(__half2float(kmax[i]), f));
    }
}

// ------------------------------------------------------------------------
// 2. The indexer.
// grid (B, H_kv, ceil(max_pages / (4 * PAGES_PER_WARP))), block 128 threads.
// Each lane owns 4 of the 128 channels.
extern "C" __global__ void __launch_bounds__(128)
page_index(const __half* __restrict__ q, const __half* __restrict__ kmin,
           const __half* __restrict__ kmax, const int* __restrict__ block_tables,
           const int* __restrict__ seq_lens, float* __restrict__ scores,
           int max_pages, int recent,
           long long sq_b, long long sq_h, long long sq_m,
           long long sb_b, long long sb_h, long long ss_b, long long ss_h) {
    const int b = blockIdx.x, h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int first = (blockIdx.z * 4 + warp) * PAGES_PER_WARP;
    if (first >= max_pages) return;
    const int d0 = lane * 4;

    float qf[NREP][4];
#pragma unroll
    for (int r = 0; r < NREP; ++r) {
        const __half2* p = reinterpret_cast<const __half2*>(q + b * sq_b + h * sq_h + r * sq_m + d0);
        const float2 a = __half22float2(p[0]), c = __half22float2(p[1]);
        qf[r][0] = a.x; qf[r][1] = a.y; qf[r][2] = c.x; qf[r][3] = c.y;
    }
    const int L = seq_lens[b];
    const int np = (L + PAGE - 1) / PAGE;

    for (int k = 0; k < PAGES_PER_WARP; ++k) {
        const int p = first + k;
        if (p >= max_pages) break;
        float out;
        if (p >= np) {
            out = NEG_INF;                              // past the end: never chosen
        } else if (p == 0 || p >= np - recent) {
            out = POS_INF;                              // sink and recent: always kept
        } else {
            const int blk = block_tables[(long long)b * max_pages + p];
            const __half2* mn = reinterpret_cast<const __half2*>(kmin + blk * sb_b + h * sb_h + d0);
            const __half2* mx = reinterpret_cast<const __half2*>(kmax + blk * sb_b + h * sb_h + d0);
            const float2 n0 = __half22float2(mn[0]), n1 = __half22float2(mn[1]);
            const float2 x0 = __half22float2(mx[0]), x1 = __half22float2(mx[1]);
            const float lo[4] = {n0.x, n0.y, n1.x, n1.y};
            const float hi[4] = {x0.x, x0.y, x1.x, x1.y};
            out = NEG_INF;
#pragma unroll
            for (int r = 0; r < NREP; ++r) {
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < 4; ++j) s += fmaxf(qf[r][j] * lo[j], qf[r][j] * hi[j]);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(FULL, s, off);
                out = fmaxf(out, s);
            }
        }
        if (lane == 0) scores[b * ss_b + h * ss_h + p] = out;
    }
}

// ------------------------------------------------------------------------
// 2b. The indexer, per head: each query head's bound for every page, not
// their maximum — for summed-mass scoring (Phase 15), which turns each head's
// bounds into estimated attention weights and sums them over the group, as
// the oracle ranks. Pages past the end get 0, a *finite* value: the scratch
// buffer may hold stale bit patterns that read as NaN, and NaN + -inf is
// still NaN, which would poison the head's softmax. Their token count of 0
// (log 0 = -inf) removes them downstream.
// grid (B, H_kv, ceil(max_pages / (4 * PAGES_PER_WARP))), block 128 threads.
extern "C" __global__ void __launch_bounds__(128)
page_index_heads(const __half* __restrict__ q, const __half* __restrict__ kmin,
                 const __half* __restrict__ kmax, const int* __restrict__ block_tables,
                 const int* __restrict__ seq_lens, float* __restrict__ heads,
                 int max_pages,
                 long long sq_b, long long sq_h, long long sq_m,
                 long long sb_b, long long sb_h,
                 long long sh_b, long long sh_h, long long sh_r) {
    const int b = blockIdx.x, h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int first = (blockIdx.z * 4 + warp) * PAGES_PER_WARP;
    if (first >= max_pages) return;
    const int d0 = lane * 4;

    float qf[NREP][4];
#pragma unroll
    for (int r = 0; r < NREP; ++r) {
        const __half2* p = reinterpret_cast<const __half2*>(q + b * sq_b + h * sq_h + r * sq_m + d0);
        const float2 a = __half22float2(p[0]), c = __half22float2(p[1]);
        qf[r][0] = a.x; qf[r][1] = a.y; qf[r][2] = c.x; qf[r][3] = c.y;
    }
    const int L = seq_lens[b];
    const int np = (L + PAGE - 1) / PAGE;
    float* out = heads + b * sh_b + h * sh_h;

    for (int k = 0; k < PAGES_PER_WARP; ++k) {
        const int p = first + k;
        if (p >= max_pages) break;
        if (p >= np) {
            if (lane < NREP) out[lane * sh_r + p] = 0.f;
            continue;
        }
        const int blk = block_tables[(long long)b * max_pages + p];
        const __half2* mn = reinterpret_cast<const __half2*>(kmin + blk * sb_b + h * sb_h + d0);
        const __half2* mx = reinterpret_cast<const __half2*>(kmax + blk * sb_b + h * sb_h + d0);
        const float2 n0 = __half22float2(mn[0]), n1 = __half22float2(mn[1]);
        const float2 x0 = __half22float2(mx[0]), x1 = __half22float2(mx[1]);
        const float lo[4] = {n0.x, n0.y, n1.x, n1.y};
        const float hi[4] = {x0.x, x0.y, x1.x, x1.y};
#pragma unroll
        for (int r = 0; r < NREP; ++r) {
            float s = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) s += fmaxf(qf[r][j] * lo[j], qf[r][j] * hi[j]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(FULL, s, off);
            if (lane == 0) out[r * sh_r + p] = s;
        }
    }
}

// ------------------------------------------------------------------------
// 3. Attention over the selected pages.
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
paged_sparse_fp16(const __half* __restrict__ q, const __half* __restrict__ k_pool,
                  const __half* __restrict__ v_pool, const int* __restrict__ block_tables,
                  const int* __restrict__ seq_lens, const int* __restrict__ sel,
                  float ratio, int recent,
                  float* __restrict__ part_acc, float* __restrict__ part_m,
                  float* __restrict__ part_l,
                  int max_pages, int num_sel, int num_splits,
                  long long sq_b, long long sq_h, long long sq_m,
                  long long sk_b, long long sk_p, long long sk_h,
                  long long sl_b, long long sl_h,
                  long long so_b, long long so_h, long long so_s, long long so_m,
                  long long sm_b, long long sm_h, long long sm_s, float scale) {
    const int b = blockIdx.x, split = blockIdx.y, h = blockIdx.z;
    const int lane = threadIdx.x;
    const int half_id = lane >> 4;
    const int d0 = (lane & 15) * DPL;

    float qf[NREP][DPL];
#pragma unroll
    for (int r = 0; r < NREP; ++r) {
        load8(q + b * sq_b + h * sq_h + r * sq_m + d0, qf[r]);
#pragma unroll
        for (int j = 0; j < DPL; ++j) qf[r][j] *= scale;
    }

    const int L = seq_lens[b];
    const int num_pages = (L + PAGE - 1) / PAGE;
    // How many of this sequence's (score-sorted) selections to attend: its
    // own budget for its *current* length — max(recent + 1, ceil(ratio x
    // pages)), capped at its pages — computed here, every step. A captured
    // graph's selection is sized for the top of its context bucket; without
    // this, a budget fixed at capture decayed toward half its nominal ratio
    // as a sequence grew through a doubling bucket. Computed in the kernel,
    // not as separate torch ops: those cost ~200 launches per step and made
    // small-batch sparse up to 10% slower than dense. ratio <= 0: attend to
    // every entry of `sel`.
    int n = num_sel;
    if (ratio > 0.f) {
        int want = ratio >= 1.f ? num_pages
                                : max(recent + 1, (int)ceilf(ratio * (float)num_pages));
        n = min(min(want, num_pages), num_sel);
    }
    const int per = (n + num_splits - 1) / num_splits;
    const int lo = split * per;
    const int hi = min(lo + per, n);
    const int* my_sel = sel + b * sl_b + h * sl_h;

    float m[NREP], l[NREP], acc[NREP][DPL];
#pragma unroll
    for (int r = 0; r < NREP; ++r) {
        m[r] = NEG_INF;
        l[r] = 0.f;
#pragma unroll
        for (int j = 0; j < DPL; ++j) acc[r][j] = 0.f;
    }

    for (int i = lo; i < hi; ++i) {
        const int p = my_sel[i];
        // A budget larger than this sequence's pages is padded with pages
        // past its end (they scored -inf): nothing to attend.
        if (p < 0 || p >= num_pages) continue;
        const int blk = block_tables[(long long)b * max_pages + p];
        const __half* kb = k_pool + blk * sk_b + h * sk_h + d0;
        const __half* vb = v_pool + blk * sk_b + h * sk_h + d0;
#pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            float s[TOKG][NREP];
#pragma unroll
            for (int t_ = 0; t_ < TOKG; ++t_) {
                const int t = 2 * (g * TOKG + t_) + half_id;
                float kf[DPL];
                load8(kb + t * sk_p, kf);
#pragma unroll
                for (int r = 0; r < NREP; ++r) {
                    float dot = 0.f;
#pragma unroll
                    for (int j = 0; j < DPL; ++j) dot = fmaf(qf[r][j], kf[j], dot);
                    s[t_][r] = dot;
                }
            }
#pragma unroll
            for (int t_ = 0; t_ < TOKG; ++t_)
#pragma unroll
                for (int r = 0; r < NREP; ++r)
#pragma unroll
                    for (int off = 8; off > 0; off >>= 1)
                        s[t_][r] += __shfl_xor_sync(FULL, s[t_][r], off);

            bool valid[TOKG];
#pragma unroll
            for (int t_ = 0; t_ < TOKG; ++t_) {
                valid[t_] = p * PAGE + 2 * (g * TOKG + t_) + half_id < L;
                if (!valid[t_])
#pragma unroll
                    for (int r = 0; r < NREP; ++r) s[t_][r] = NEG_INF;
            }
#pragma unroll
            for (int r = 0; r < NREP; ++r) {
                float mx = m[r];
#pragma unroll
                for (int t_ = 0; t_ < TOKG; ++t_) mx = fmaxf(mx, s[t_][r]);
                const float a = (mx == NEG_INF) ? 0.f : __expf(m[r] - mx);
                l[r] *= a;
#pragma unroll
                for (int j = 0; j < DPL; ++j) acc[r][j] *= a;
                m[r] = mx;
#pragma unroll
                for (int t_ = 0; t_ < TOKG; ++t_) s[t_][r] = valid[t_] ? __expf(s[t_][r] - mx) : 0.f;
            }
#pragma unroll
            for (int t_ = 0; t_ < TOKG; ++t_) {
                if (!valid[t_]) continue;
                const int t = 2 * (g * TOKG + t_) + half_id;
                float vf[DPL];
                load8(vb + t * sk_p, vf);
#pragma unroll
                for (int r = 0; r < NREP; ++r) {
                    l[r] += s[t_][r];
#pragma unroll
                    for (int j = 0; j < DPL; ++j) acc[r][j] = fmaf(s[t_][r], vf[j], acc[r][j]);
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
