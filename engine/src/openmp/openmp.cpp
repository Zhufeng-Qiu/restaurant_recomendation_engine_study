#include "openmp/openmp.hpp"

#include "serial/serial.hpp"

namespace engine {

void compute_openmp(const Fixture& fx, std::vector<double>& out) {
  const int64_t n = fx.n_pairs();
  out.resize(n);
  const int32_t* pairs = fx.pairs.data();
  double* o = out.data();
#pragma omp parallel for schedule(runtime)
  for (int64_t k = 0; k < n; ++k) {
    o[k] = evaluate_pair(fx, pairs[2 * k], pairs[2 * k + 1]);
  }
}

}  // namespace engine
