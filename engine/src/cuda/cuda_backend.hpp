#pragma once

#include <vector>

#include "common/fixture.hpp"
#include "common/pair_order.hpp"

namespace engine {

// How pairs are mapped onto lanes.
//
// The default is four lanes per pair in the fixture's own order, because that
// configuration is better than the phase-3 mapping on BOTH timing bases and on
// both real fixtures -- it is not a trade:
//
//   device_total   -17.59% [-19.75, -15.38] on item_full, -42.23% on user_full
//   cold path       -3.08% [ -9.75,  +4.08], i.e. no penalty
//   vs the pre-packing binary  -11.43% [-11.84, -11.02]
//
// Sorting by shorter-slice length (--pair-order bylen) buys a further ~25% of
// steady state and is NOT the default, because building that order costs ~100 ms
// against a ~3 ms kernel: it pays back after roughly 57 queries, so it belongs
// to a resident engine and not to a one-shot invocation of this binary.
struct CudaMapOptions {
  int group = 4;                         // lanes per pair; power of two <= 32
  PairOrder order = PairOrder::kSource;  // kByShortLen enables packing
  // Compute each pair's slice bounds once per group and broadcast, instead of
  // repeating the four searches in every lane. Orthogonal to packing on
  // purpose, so the two effects can be attributed separately.
  bool hoist = false;
  // Plan diagnostics (lengths, lane slots, utilisation) are OFF by default.
  // Excluding them from a total while still executing them makes the total
  // a reclassification rather than a saving; the production path does not
  // compute them at all.
  bool plan_metrics = false;
  bool is_default() const {
    return group == 4 && order == PairOrder::kSource && !hoist && !plan_metrics;
  }
};

// Process-wide, set from the command line before the backend runs. The backend
// registry hands backends only (fixture, out), so this is the seam rather than
// threading an options struct through every backend signature.
CudaMapOptions& cuda_map_options();

// Stage timings from the last compute_cuda call, so main can assemble the
// cold-path total (it owns t_load; the backend owns everything after it).
//
// Two bases, and they answer different questions:
//   device_total    steady state -- what a resident engine pays per query.
//                   Excludes the plan, which such an engine builds once.
//   cold_data_path  entry to results in host memory, plan INCLUDED, because a
//                   caller that runs the binary once pays to build it.
// Neither is a CLI one-shot: process launch and the CUDA driver's first touch
// happen before any code here could time them.
struct CudaTimings {
  double plan = 0, h2d = 0, stats = 0, finalize = 0, d2h = 0;
  // Describing the plan (lengths, lane slots, utilisation) is reporting, not
  // execution. It is timed so it can be shown, and excluded from both totals
  // so it cannot inflate either. On item_full it is ~50 ms against a ~3 ms
  // kernel, so counting it would have dominated the cold path it was added to
  // make honest.
  double plan_metrics = 0;

  double device_total() const { return stats + finalize; }
  double cold_data_path(double t_load) const {
    return t_load + plan + h2d + device_total() + d2h;
  }
};
CudaTimings& cuda_last_timings();

// Single-GPU CUDA backend: sub-warp-per-pair sufficient statistics + device
// finalization. Prints a cuda_detail JSON line with kernel/transfer timings and
// the mapping actually used.
void compute_cuda(const Fixture& fx, std::vector<double>& out);

}  // namespace engine
