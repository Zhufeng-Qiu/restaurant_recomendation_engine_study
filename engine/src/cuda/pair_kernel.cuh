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

#if defined(__CUDACC__)

// stats layout: [n_pairs][6] doubles = n, sx, sy, sxx, syy, sxy.
__global__ void pair_stats_kernel(const int64_t* __restrict__ offsets,
                                  const int32_t* __restrict__ dims,
                                  const double* __restrict__ vals,
                                  const int32_t* __restrict__ pairs,
                                  int64_t n_pairs, int32_t dim_lo,
                                  int32_t dim_hi, double* __restrict__ stats) {
  const int64_t warp_id =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (warp_id >= n_pairs) return;

  double s[6] = {0, 0, 0, 0, 0, 0};
  pair_lane_stats(offsets, dims, vals, pairs[2 * warp_id],
                  pairs[2 * warp_id + 1], dim_lo, dim_hi, lane, kWarp, s);
#pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1) {
#pragma unroll
    for (int i = 0; i < 6; ++i)
      s[i] += __shfl_down_sync(0xffffffff, s[i], off);
  }
  if (lane == 0) {
    double* dst = stats + 6 * warp_id;
#pragma unroll
    for (int i = 0; i < 6; ++i) dst[i] = s[i];
  }
}

// One thread per pair.
__global__ void finalize_kernel(const double* __restrict__ stats,
                                int64_t n_pairs, double* __restrict__ sims) {
  const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= n_pairs) return;
  sims[k] = finalize_six(stats + 6 * k);
}

#endif  // __CUDACC__

}  // namespace engine_cuda
