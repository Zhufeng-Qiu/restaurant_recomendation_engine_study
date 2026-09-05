#pragma once

#include <vector>

#include "common/fixture.hpp"
#include "common/pair_order.hpp"

namespace engine {

// How pairs are mapped onto lanes. The defaults reproduce the phase-3 mapping
// (one warp per pair, fixture order), which is the baseline every warp-packing
// number in the README is measured against.
struct CudaMapOptions {
  int group = 32;                          // lanes per pair; power of two <= 32
  PairOrder order = PairOrder::kSource;    // kByShortLen enables packing
};

// Process-wide, set from the command line before the backend runs. The backend
// registry hands backends only (fixture, out), so this is the seam rather than
// threading an options struct through every backend signature.
CudaMapOptions& cuda_map_options();

// Single-GPU CUDA backend: sub-warp-per-pair sufficient statistics + device
// finalization. Prints a cuda_detail JSON line with kernel/transfer timings and
// the mapping actually used.
void compute_cuda(const Fixture& fx, std::vector<double>& out);

}  // namespace engine
