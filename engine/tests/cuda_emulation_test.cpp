// Host-side emulation of the CUDA warp-per-pair kernel.
//
// The GPU cannot be tested on this machine, but its algorithm can: this test
// executes pair_lane_stats / finalize_six (the exact functions the __global__
// wrappers call) for all 32 lanes of the emulated warp, reduces them with the
// same shuffle-tree order as the device code, and compares every pair against
// the fixture's golden similarities. It also emulates the NCCL decomposition:
// dim ranges split across 2 and 3 virtual GPUs, partial six-stat tensors
// summed in rank order (the AllReduce), then finalized.
//
// This catches algorithmic bugs (slice bounds, x/y orientation, binary
// search, range splits) before the code ever sees an nvcc. It cannot catch
// launch-config or memory bugs; those wait for the GPU host (gates A-D in
// docs/gpu_runbook.md).
//
// Usage: cuda_emulation_test <fixture_dir> [<fixture_dir> ...]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "common/fixture.hpp"
#include "common/pearson.hpp"
#include "cuda/pair_kernel.cuh"

namespace {

using engine::Fixture;
using engine_cuda::finalize_six;
using engine_cuda::kWarp;
using engine_cuda::pair_lane_stats;

// Emulates pair_stats_kernel for one pair over [dim_lo, dim_hi): 32 lane
// accumulators reduced with the same tree order as __shfl_down_sync.
void warp_stats(const Fixture& fx, int32_t ei, int32_t ej, int32_t dim_lo,
                int32_t dim_hi, double out[6]) {
  double lanes[kWarp][6] = {};
  for (int lane = 0; lane < kWarp; ++lane)
    pair_lane_stats(fx.offsets.data(), fx.dims.data(), fx.vals.data(), ei, ej,
                    dim_lo, dim_hi, lane, kWarp, lanes[lane]);
  for (int off = kWarp / 2; off > 0; off >>= 1)
    for (int l = 0; l < off; ++l)
      for (int i = 0; i < 6; ++i) lanes[l][i] += lanes[l + off][i];
  for (int i = 0; i < 6; ++i) out[i] = lanes[0][i];
}

struct Result {
  double max_diff = 0.0;
  int64_t failures = 0;
};

Result check(const Fixture& fx, int n_ranges) {
  int32_t n_dims = 0;
  for (int32_t d : fx.dims) n_dims = std::max(n_dims, d + 1);

  Result r;
  const int64_t n_pairs = fx.n_pairs();
  for (int64_t k = 0; k < n_pairs; ++k) {
    double total[6] = {};
    for (int g = 0; g < n_ranges; ++g) {  // rank-order sum == AllReduce result
      const int32_t lo = static_cast<int32_t>(int64_t(n_dims) * g / n_ranges);
      const int32_t hi = static_cast<int32_t>(int64_t(n_dims) * (g + 1) / n_ranges);
      double part[6];
      warp_stats(fx, fx.pairs[2 * k], fx.pairs[2 * k + 1], lo, hi, part);
      for (int i = 0; i < 6; ++i) total[i] += part[i];
    }
    const double sim = finalize_six(total);
    const double diff = std::fabs(sim - fx.golden[k]);
    if (diff > r.max_diff) r.max_diff = diff;
    if (diff > engine::kTol) ++r.failures;
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <fixture_dir> [...]\n", argv[0]);
    return 2;
  }
  int rc = 0;
  for (int a = 1; a < argc; ++a) {
    const Fixture fx = Fixture::load(argv[a]);
    for (int n_ranges : {1, 2, 3}) {
      const Result r = check(fx, n_ranges);
      std::printf("%-40s ranges=%d pairs=%lld max_abs_diff=%.3e failures=%lld\n",
                  argv[a], n_ranges, static_cast<long long>(fx.n_pairs()),
                  r.max_diff, static_cast<long long>(r.failures));
      if (r.failures > 0) rc = 1;
    }
  }
  std::puts(rc == 0 ? "CUDA emulation: ALL PASS" : "CUDA emulation: FAILURES");
  return rc;
}
