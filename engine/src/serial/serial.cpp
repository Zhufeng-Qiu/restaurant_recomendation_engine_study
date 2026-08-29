#include "serial/serial.hpp"

namespace engine {

void compute_serial(const Fixture& fx, std::vector<double>& out) {
  const int64_t n = fx.n_pairs();
  out.resize(n);
  for (int64_t k = 0; k < n; ++k) {
    out[k] = evaluate_pair(fx, fx.pairs[2 * k], fx.pairs[2 * k + 1]);
  }
}

}  // namespace engine
