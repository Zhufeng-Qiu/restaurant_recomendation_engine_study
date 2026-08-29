// Shared CUDA kernels for the Pearson contract.
//
// Mapping (phase-3 initial design from the project brief): one WARP per
// candidate pair. Lanes stride over the shorter row slice and binary-search
// the longer slice; the six statistics are combined with warp shuffles. The
// kernel accumulates sufficient statistics only — finalization is a separate
// trivial kernel — so the same code serves the single-GPU backend
// (full dim range) and the NCCL backends (dim subrange per GPU + AllReduce).
//
// The per-lane algorithm (pair_lane_stats) and the finalization math
// (finalize_six) are plain host-compilable functions so that
// tests/cuda_emulation_test.cpp can execute the exact device code path on
// the CPU and compare it against golden similarities before any GPU exists.
// Only the __global__ wrappers require nvcc.
#pragma once

#include <cstdint>
#include <cmath>

#if defined(__CUDACC__)
#define EC_HD __host__ __device__
#else
#define EC_HD inline
#endif

namespace engine_cuda {

constexpr int kWarp = 32;
constexpr int kBlock = 128;  // 4 warps per block

EC_HD int64_t lower_bound_i32(const int32_t* a, int64_t lo, int64_t hi,
                              int32_t v) {
  while (lo < hi) {
    const int64_t mid = (lo + hi) >> 1;
    if (a[mid] < v) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// Accumulates lane `lane`'s share (elements lane, lane+stride, ...) of the
// six sufficient statistics for pair (ei, ej), restricted to dims in
// [dim_lo, dim_hi). `s` is n, sx, sy, sxx, syy, sxy; the caller zeroes it.
// x is always entity ei's rating and y entity ej's, regardless of which row
// slice is shorter.
EC_HD void pair_lane_stats(const int64_t* offsets, const int32_t* dims,
                           const double* vals, int32_t ei, int32_t ej,
                           int32_t dim_lo, int32_t dim_hi, int lane,
                           int stride, double s[6]) {
  int64_t a0 = lower_bound_i32(dims, offsets[ei], offsets[ei + 1], dim_lo);
  int64_t a1 = lower_bound_i32(dims, a0, offsets[ei + 1], dim_hi);
  int64_t b0 = lower_bound_i32(dims, offsets[ej], offsets[ej + 1], dim_lo);
  int64_t b1 = lower_bound_i32(dims, b0, offsets[ej + 1], dim_hi);

  // Iterate the shorter slice, binary-search the longer one.
  const bool swapped = (a1 - a0) > (b1 - b0);
  const int64_t s0 = swapped ? b0 : a0, s1 = swapped ? b1 : a1;
  const int64_t l0 = swapped ? a0 : b0, l1 = swapped ? a1 : b1;

  for (int64_t t = s0 + lane; t < s1; t += stride) {
    const int32_t d = dims[t];
    const int64_t pos = lower_bound_i32(dims, l0, l1, d);
    if (pos < l1 && dims[pos] == d) {
      const double xs = vals[t];    // value from the short side
      const double xl = vals[pos];  // value from the long side
      const double x = swapped ? xl : xs;
      const double y = swapped ? xs : xl;
      s[0] += 1.0;
      s[1] += x; s[2] += y;
      s[3] += x * x; s[4] += y * y; s[5] += x * y;
    }
  }
}

// Contract finalization (docs/pearson_contract.md §3) from a six-stat vector.
EC_HD double finalize_six(const double* s) {
  const double num = s[0] * s[5] - s[1] * s[2];
  const double vx = s[0] * s[3] - s[1] * s[1];
  const double vy = s[0] * s[4] - s[2] * s[2];
  return (num == 0.0 || vx <= 0.0 || vy <= 0.0) ? 0.0 : num / std::sqrt(vx * vy);
}

// ---------------------------------------------------------------------------
// Compressed collective payloads (docs/pearson_contract.md §9).
//
// The contract's ratings are small non-negative integers, so every one of the
// six sufficient statistics is an exact non-negative integer bounded by the
// longest rating row N:  n <= N,  Sx,Sy <= vmax*N,  Sxx,Syy,Sxy <= vmax^2*N.
// The AllReduce payload is therefore compressible with no loss at all, which
// is what lets the bit-exact contract survive compression:
//
//   f64     6 x double    48 B/pair    baseline
//   i32     6 x int32     24 B/pair    2.00x
//   packed  2 x uint64    16 B/pair    3.00x, and reducible in place
//
// `packed` gives each field its own kPackBits with no shared carry space.
// Partitioning is over the rating dimension, so a field's cross-GPU sum is
// exactly the global total, which the domain check bounds below 2^kPackBits.
// No field can therefore carry into its neighbour, and the packed words can
// be handed straight to ncclSum without being decompressed first --
// homomorphic reduction, in the sense that sum(pack(a), pack(b)) ==
// pack(a + b).
//
// 21 bits x 3 fields = 63 of the 64 available bits per word.
constexpr int kPackBits = 21;
constexpr uint64_t kPackMask = (1ull << kPackBits) - 1;
constexpr int64_t kPackMaxField = (1ll << kPackBits) - 1;

EC_HD void pack_six(const double* s, uint64_t* w) {
  w[0] = static_cast<uint64_t>(static_cast<int64_t>(s[0]))
       | (static_cast<uint64_t>(static_cast<int64_t>(s[1])) << kPackBits)
       | (static_cast<uint64_t>(static_cast<int64_t>(s[2])) << (2 * kPackBits));
  w[1] = static_cast<uint64_t>(static_cast<int64_t>(s[3]))
       | (static_cast<uint64_t>(static_cast<int64_t>(s[4])) << kPackBits)
       | (static_cast<uint64_t>(static_cast<int64_t>(s[5])) << (2 * kPackBits));
}

EC_HD void unpack_six(const uint64_t* w, double* s) {
  s[0] = static_cast<double>(w[0] & kPackMask);
  s[1] = static_cast<double>((w[0] >> kPackBits) & kPackMask);
  s[2] = static_cast<double>((w[0] >> (2 * kPackBits)) & kPackMask);
  s[3] = static_cast<double>(w[1] & kPackMask);
  s[4] = static_cast<double>((w[1] >> kPackBits) & kPackMask);
  s[5] = static_cast<double>((w[1] >> (2 * kPackBits)) & kPackMask);
}

// Convenience for the emulation test: the round trip a packed AllReduce
// performs on one pair, including the in-compressed-form summation.
EC_HD void pack_add(const uint64_t* a, const uint64_t* b, uint64_t* out) {
  out[0] = a[0] + b[0];
  out[1] = a[1] + b[1];
}

#if defined(__CUDACC__)

// Accumulate + warp-reduce one pair's six statistics. Shared by every payload
// variant so the arithmetic that produces the numbers is identical across
// them; the variants differ only in how lane 0 writes them out. The
// accumulation stays in double: the values are exact integers far inside
// double's 53-bit exact range, so narrowing at emit is lossless.
__device__ inline bool warp_pair_stats(const int64_t* __restrict__ offsets,
                                       const int32_t* __restrict__ dims,
                                       const double* __restrict__ vals,
                                       const int32_t* __restrict__ pairs,
                                       int64_t n_pairs, int32_t dim_lo,
                                       int32_t dim_hi, int64_t* warp_id_out,
                                       double* s) {
  const int64_t warp_id =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarp;
  const int lane = threadIdx.x % kWarp;
  *warp_id_out = warp_id;
  if (warp_id >= n_pairs) return false;

#pragma unroll
  for (int i = 0; i < 6; ++i) s[i] = 0.0;
  pair_lane_stats(offsets, dims, vals, pairs[2 * warp_id],
                  pairs[2 * warp_id + 1], dim_lo, dim_hi, lane, kWarp, s);
#pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1) {
#pragma unroll
    for (int i = 0; i < 6; ++i)
      s[i] += __shfl_down_sync(0xffffffff, s[i], off);
  }
  return lane == 0;
}

// stats layout: [n_pairs][6] doubles = n, sx, sy, sxx, syy, sxy.
__global__ void pair_stats_kernel(const int64_t* __restrict__ offsets,
                                  const int32_t* __restrict__ dims,
                                  const double* __restrict__ vals,
                                  const int32_t* __restrict__ pairs,
                                  int64_t n_pairs, int32_t dim_lo,
                                  int32_t dim_hi, double* __restrict__ stats) {
  int64_t k;
  double s[6];
  if (!warp_pair_stats(offsets, dims, vals, pairs, n_pairs, dim_lo, dim_hi,
                       &k, s))
    return;
  double* dst = stats + 6 * k;
#pragma unroll
  for (int i = 0; i < 6; ++i) dst[i] = s[i];
}

// stats layout: [n_pairs][6] int32 — 2x smaller collective payload.
__global__ void pair_stats_kernel_i32(const int64_t* __restrict__ offsets,
                                      const int32_t* __restrict__ dims,
                                      const double* __restrict__ vals,
                                      const int32_t* __restrict__ pairs,
                                      int64_t n_pairs, int32_t dim_lo,
                                      int32_t dim_hi,
                                      int32_t* __restrict__ stats) {
  int64_t k;
  double s[6];
  if (!warp_pair_stats(offsets, dims, vals, pairs, n_pairs, dim_lo, dim_hi,
                       &k, s))
    return;
  int32_t* dst = stats + 6 * k;
#pragma unroll
  for (int i = 0; i < 6; ++i) dst[i] = static_cast<int32_t>(s[i]);
}

// stats layout: [n_pairs][2] uint64 — 3x smaller, and summable as-is.
__global__ void pair_stats_kernel_packed(const int64_t* __restrict__ offsets,
                                         const int32_t* __restrict__ dims,
                                         const double* __restrict__ vals,
                                         const int32_t* __restrict__ pairs,
                                         int64_t n_pairs, int32_t dim_lo,
                                         int32_t dim_hi,
                                         uint64_t* __restrict__ stats) {
  int64_t k;
  double s[6];
  if (!warp_pair_stats(offsets, dims, vals, pairs, n_pairs, dim_lo, dim_hi,
                       &k, s))
    return;
  pack_six(s, stats + 2 * k);
}

// One thread per pair.
__global__ void finalize_kernel(const double* __restrict__ stats,
                                int64_t n_pairs, double* __restrict__ sims) {
  const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= n_pairs) return;
  sims[k] = finalize_six(stats + 6 * k);
}

__global__ void finalize_kernel_i32(const int32_t* __restrict__ stats,
                                    int64_t n_pairs,
                                    double* __restrict__ sims) {
  const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= n_pairs) return;
  const int32_t* src = stats + 6 * k;
  double s[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) s[i] = static_cast<double>(src[i]);
  sims[k] = finalize_six(s);
}

// Unpacking happens here, after the reduction — the collective itself never
// sees the uncompressed form.
__global__ void finalize_kernel_packed(const uint64_t* __restrict__ stats,
                                       int64_t n_pairs,
                                       double* __restrict__ sims) {
  const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= n_pairs) return;
  double s[6];
  unpack_six(stats + 2 * k, s);
  sims[k] = finalize_six(s);
}

#endif  // __CUDACC__

}  // namespace engine_cuda
