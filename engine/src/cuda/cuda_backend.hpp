#pragma once

#include <vector>

#include "common/fixture.hpp"

namespace engine {

// Single-GPU CUDA backend: warp-per-pair sufficient statistics + device
// finalization. Prints a cuda_detail JSON line with kernel/transfer timings.
void compute_cuda(const Fixture& fx, std::vector<double>& out);

}  // namespace engine
