#pragma once

#include <vector>

#include "common/fixture.hpp"
#include "common/pearson.hpp"

namespace engine {

// Evaluate one candidate pair: sorted two-pointer intersection of the two CSR
// rows, six-stat accumulation, contract finalization. Shared by serial and
// (pair-parallel) OpenMP backends.
inline double evaluate_pair(const Fixture& fx, int32_t i, int32_t j) {
  int64_t a = fx.offsets[i], a_end = fx.offsets[i + 1];
  int64_t b = fx.offsets[j], b_end = fx.offsets[j + 1];
  PairStats s;
  while (a < a_end && b < b_end) {
    const int32_t da = fx.dims[a], db = fx.dims[b];
    if (da < db) {
      ++a;
    } else if (db < da) {
      ++b;
    } else {
      s.add(fx.vals[a], fx.vals[b]);
      ++a;
      ++b;
    }
  }
  return pearson_finalize(s);
}

// Serial oracle: evaluates every candidate pair in order.
void compute_serial(const Fixture& fx, std::vector<double>& out);

}  // namespace engine
