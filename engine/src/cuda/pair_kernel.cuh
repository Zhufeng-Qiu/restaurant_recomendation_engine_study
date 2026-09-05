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
#include <stdexcept>

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
// The slice bounds for one pair: where each row's ratings intersect
// [dim_lo, dim_hi), and which of the two is shorter. Four binary searches, and
// they do not depend on the lane -- which is the point of separating them.
struct PairSlices {
  int64_t s0, s1;   // shorter slice
  int64_t l0, l1;   // longer slice, the one that gets binary-searched
  bool swapped;     // true when the shorter slice belongs to ej
};

EC_HD PairSlices pair_slice_bounds(const int64_t* offsets, const int32_t* dims,
                                   int32_t ei, int32_t ej, int32_t dim_lo,
                                   int32_t dim_hi) {
  const int64_t a0 = lower_bound_i32(dims, offsets[ei], offsets[ei + 1], dim_lo);
  const int64_t a1 = lower_bound_i32(dims, a0, offsets[ei + 1], dim_hi);
  const int64_t b0 = lower_bound_i32(dims, offsets[ej], offsets[ej + 1], dim_lo);
  const int64_t b1 = lower_bound_i32(dims, b0, offsets[ej + 1], dim_hi);
  PairSlices p;
  p.swapped = (a1 - a0) > (b1 - b0);   // iterate the shorter side
  p.s0 = p.swapped ? b0 : a0;
  p.s1 = p.swapped ? b1 : a1;
  p.l0 = p.swapped ? a0 : b0;
  p.l1 = p.swapped ? a1 : b1;
  return p;
}

// Accumulates lane `lane`'s share (elements lane, lane+stride, ...) of the six
// sufficient statistics, given bounds already computed. `s` is
// n, sx, sy, sxx, syy, sxy; the caller zeroes it. x is always entity ei's
// rating and y entity ej's, regardless of which slice is shorter.
EC_HD void pair_lane_scan(const int32_t* dims, const double* vals,
                          const PairSlices& p, int lane, int stride,
                          double s[6]) {
  for (int64_t t = p.s0 + lane; t < p.s1; t += stride) {
    const int32_t d = dims[t];
    const int64_t pos = lower_bound_i32(dims, p.l0, p.l1, d);
    if (pos < p.l1 && dims[pos] == d) {
      const double xs = vals[t];    // value from the short side
      const double xl = vals[pos];  // value from the long side
      const double x = p.swapped ? xl : xs;
      const double y = p.swapped ? xs : xl;
      s[0] += 1.0;
      s[1] += x; s[2] += y;
      s[3] += x * x; s[4] += y * y; s[5] += x * y;
    }
  }
}

// Bounds + scan, the way every lane did it before hoisting existed. Kept as
// the reference composition: the host emulation and the unhoisted device path
// both go through it.
EC_HD void pair_lane_stats(const int64_t* offsets, const int32_t* dims,
                           const double* vals, int32_t ei, int32_t ej,
                           int32_t dim_lo, int32_t dim_hi, int lane,
                           int stride, double s[6]) {
  pair_lane_scan(dims, vals,
                 pair_slice_bounds(offsets, dims, ei, ej, dim_lo, dim_hi),
                 lane, stride, s);
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

// ---------------------------------------------------------------------------
// Warp packing. See common/pair_order.hpp for why the ordering matters.
//
// The phase-3 mapping was one warp per pair: 32 lanes stride the shorter
// restricted slice, so a pair costs 32 * ceil(L / 32) lane slots and the
// partly-filled last round is waste -- 14.42% of the slots at one GPU, 25.32%
// at two.
//
// Packing gives a pair a SUB-WARP of G lanes, so a warp carries 32/G pairs and
// the tail wastes at most G-1 slots. The groups in a warp run in lockstep, so
// the warp costs 32 * max_g ceil(L_g / G): one long pair among short ones pays
// for all of them. `order` therefore arrives sorted by shorter-slice length.
// Unsorted, the model says G=1 costs 63% MORE than the baseline.
//
// Indexing. `order` is slot -> pair, and the caller may pass a slice of it
// (the async path launches per chunk). Statistics are written at the SLOT
// index, so the collective payload stays contiguous and chunkable exactly as
// before; the permutation is undone in finalize, which scatters to
// sims[order[slot]]. Every GPU must be given the same `order`, or the
// AllReduce would sum slots that stand for different pairs -- which is why the
// plan is built from the full dimension range rather than each GPU's slice.
//
// G == 32 with an identity order is the phase-3 mapping plus one broadcast
// load, so the baseline and every packed arm run this same code.
//
// No lane returns early: threads past the end keep zeroed accumulators and
// still take part in the shuffles, so the full-warp mask stays honest.
// HOIST: compute the four slice-bound searches ONCE per group and broadcast
// them, instead of repeating them in all G lanes. The measured cost of the
// unhoisted kernel scales with the thread count, and these searches are the
// largest identifiable per-thread term -- this is the A/B that decides whether
// they are actually it. Kept as a switch rather than a replacement so packing
// and hoisting can be attributed separately.
//
// The broadcast sits OUTSIDE the active test: a group is G consecutive lanes
// sharing one slot, so a group is uniformly active or inactive, but the whole
// warp must still reach the shuffle for the full mask to be honest. An
// inactive group broadcasts zeroes and scans nothing.
template <int G, bool HOIST>
__device__ inline bool subwarp_pair_stats(const int64_t* __restrict__ offsets,
                                          const int32_t* __restrict__ dims,
                                          const double* __restrict__ vals,
                                          const int32_t* __restrict__ pairs,
                                          const int32_t* __restrict__ order,
                                          int64_t n_slots, int32_t dim_lo,
                                          int32_t dim_hi, int64_t* slot_out,
                                          double* s) {
  static_assert(G >= 1 && G <= kWarp && (G & (G - 1)) == 0,
                "group size must be a power of two in [1, 32]");
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t slot = tid / G;
  const int sublane = static_cast<int>(threadIdx.x) % G;
  const bool active = slot < n_slots;
  *slot_out = slot;

#pragma unroll
  for (int i = 0; i < 6; ++i) s[i] = 0.0;

  PairSlices sl{0, 0, 0, 0, false};
  const int64_t k = active ? static_cast<int64_t>(order[slot]) : 0;
  if (HOIST && G > 1) {
    if (active && sublane == 0)
      sl = pair_slice_bounds(offsets, dims, pairs[2 * k], pairs[2 * k + 1],
                             dim_lo, dim_hi);
    __syncwarp();
    sl.s0 = __shfl_sync(0xffffffff, static_cast<long long>(sl.s0), 0, G);
    sl.s1 = __shfl_sync(0xffffffff, static_cast<long long>(sl.s1), 0, G);
    sl.l0 = __shfl_sync(0xffffffff, static_cast<long long>(sl.l0), 0, G);
    sl.l1 = __shfl_sync(0xffffffff, static_cast<long long>(sl.l1), 0, G);
    sl.swapped = __shfl_sync(0xffffffff, static_cast<int>(sl.swapped), 0, G) != 0;
  } else if (active) {
    sl = pair_slice_bounds(offsets, dims, pairs[2 * k], pairs[2 * k + 1],
                           dim_lo, dim_hi);
  }
  if (active) pair_lane_scan(dims, vals, sl, sublane, G, s);

  __syncwarp();
#pragma unroll
  for (int off = G / 2; off > 0; off >>= 1) {
#pragma unroll
    for (int i = 0; i < 6; ++i)
      s[i] += __shfl_down_sync(0xffffffff, s[i], off, G);
  }
  return active && sublane == 0;
}

// stats layout: [n_slots][6] doubles = n, sx, sy, sxx, syy, sxy.
template <int G, bool HOIST>
__global__ void pair_stats_kernel(const int64_t* __restrict__ offsets,
                                  const int32_t* __restrict__ dims,
                                  const double* __restrict__ vals,
                                  const int32_t* __restrict__ pairs,
                                  const int32_t* __restrict__ order,
                                  int64_t n_slots, int32_t dim_lo,
                                  int32_t dim_hi, double* __restrict__ stats) {
  int64_t slot;
  double s[6];
  if (!subwarp_pair_stats<G, HOIST>(offsets, dims, vals, pairs, order,
                                    n_slots, dim_lo, dim_hi, &slot, s))
    return;
  double* dst = stats + 6 * slot;
#pragma unroll
  for (int i = 0; i < 6; ++i) dst[i] = s[i];
}

// stats layout: [n_slots][6] int32 -- 2x smaller collective payload.
template <int G, bool HOIST>
__global__ void pair_stats_kernel_i32(const int64_t* __restrict__ offsets,
                                      const int32_t* __restrict__ dims,
                                      const double* __restrict__ vals,
                                      const int32_t* __restrict__ pairs,
                                      const int32_t* __restrict__ order,
                                      int64_t n_slots, int32_t dim_lo,
                                      int32_t dim_hi,
                                      int32_t* __restrict__ stats) {
  int64_t slot;
  double s[6];
  if (!subwarp_pair_stats<G, HOIST>(offsets, dims, vals, pairs, order,
                                    n_slots, dim_lo, dim_hi, &slot, s))
    return;
  int32_t* dst = stats + 6 * slot;
#pragma unroll
  for (int i = 0; i < 6; ++i) dst[i] = static_cast<int32_t>(s[i]);
}

// stats layout: [n_slots][2] uint64 -- 3x smaller, and summable as-is.
template <int G, bool HOIST>
__global__ void pair_stats_kernel_packed(const int64_t* __restrict__ offsets,
                                         const int32_t* __restrict__ dims,
                                         const double* __restrict__ vals,
                                         const int32_t* __restrict__ pairs,
                                         const int32_t* __restrict__ order,
                                         int64_t n_slots, int32_t dim_lo,
                                         int32_t dim_hi,
                                         uint64_t* __restrict__ stats) {
  int64_t slot;
  double s[6];
  if (!subwarp_pair_stats<G, HOIST>(offsets, dims, vals, pairs, order,
                                    n_slots, dim_lo, dim_hi, &slot, s))
    return;
  pack_six(s, stats + 2 * slot);
}

// Runtime G -> template instantiation. Anything else is a caller bug, not a
// fallback: quietly running a mapping other than the one on the command line
// would corrupt the comparison.
#define ENGINE_DISPATCH_G(group, H, KERNEL, GRID, BLOCK, STREAM, ...)          \
  switch (group) {                                                            \
    case 1:  KERNEL<1, H><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break;     \
    case 2:  KERNEL<2, H><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break;     \
    case 4:  KERNEL<4, H><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break;     \
    case 8:  KERNEL<8, H><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break;     \
    case 16: KERNEL<16, H><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break;    \
    case 32: KERNEL<32, H><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break;    \
    default: throw std::runtime_error("group must be 1, 2, 4, 8, 16 or 32");   \
  }

#define ENGINE_DISPATCH_GROUP(group, hoist, KERNEL, GRID, BLOCK, STREAM, ...)  \
  do {                                                                        \
    if (hoist) {                                                              \
      ENGINE_DISPATCH_G(group, true, KERNEL, GRID, BLOCK, STREAM, __VA_ARGS__) \
    } else {                                                                  \
      ENGINE_DISPATCH_G(group, false, KERNEL, GRID, BLOCK, STREAM, __VA_ARGS__)\
    }                                                                         \
  } while (0)

// One thread per slot. Undoes the permutation: slot s holds pair order[s].
__global__ void finalize_kernel(const double* __restrict__ stats,
                                const int32_t* __restrict__ order,
                                int64_t n_slots, double* __restrict__ sims) {
  const int64_t s = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (s >= n_slots) return;
  sims[order[s]] = finalize_six(stats + 6 * s);
}

__global__ void finalize_kernel_i32(const int32_t* __restrict__ stats,
                                    const int32_t* __restrict__ order,
                                    int64_t n_slots, double* __restrict__ sims) {
  const int64_t s = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (s >= n_slots) return;
  const int32_t* src = stats + 6 * s;
  double v[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) v[i] = static_cast<double>(src[i]);
  sims[order[s]] = finalize_six(v);
}

// Unpacking happens here, after the reduction -- the collective itself never
// sees the uncompressed form.
__global__ void finalize_kernel_packed(const uint64_t* __restrict__ stats,
                                       const int32_t* __restrict__ order,
                                       int64_t n_slots,
                                       double* __restrict__ sims) {
  const int64_t s = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (s >= n_slots) return;
  double v[6];
  unpack_six(stats + 2 * s, v);
  sims[order[s]] = finalize_six(v);
}

#endif  // __CUDACC__

}  // namespace engine_cuda
