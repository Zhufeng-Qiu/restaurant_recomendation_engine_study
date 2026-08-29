#pragma once

#include <vector>

#include "common/fixture.hpp"

namespace engine {

// Pair-parallel OpenMP backend. Each candidate pair is independent, and each
// thread writes only its own out[k] slot, so no synchronization is needed.
// Loop scheduling is schedule(runtime): control it with OMP_SCHEDULE
// (e.g. "static" vs "dynamic,1024") to study skew from uneven overlap sizes.
void compute_openmp(const Fixture& fx, std::vector<double>& out);

}  // namespace engine
