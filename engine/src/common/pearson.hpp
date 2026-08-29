// Frozen Pearson contract, native form. Normative text: docs/pearson_contract.md.
#pragma once

#include <cmath>
#include <cstdint>

namespace engine {

constexpr int64_t kMinOverlap = 3;
constexpr double kEps = 1e-14;  // emission filter: sim > kEps
constexpr double kTol = 1e-12;  // cross-backend agreement tolerance

// Six additive sufficient statistics for one candidate pair (contract §3).
// All six slots are doubles so a batch of PairStats is directly usable as an
// AllReduce buffer (MPI_SUM / ncclSum) without repacking.
struct PairStats {
  double n = 0.0;
  double sx = 0.0, sy = 0.0;
  double sxx = 0.0, syy = 0.0, sxy = 0.0;

  void add(double x, double y) {
    n += 1.0;
    sx += x;
    sy += y;
    sxx += x * x;
    syy += y * y;
    sxy += x * y;
  }
};

inline double pearson_finalize(const PairStats& s) {
  const double num = s.n * s.sxy - s.sx * s.sy;
  const double vx = s.n * s.sxx - s.sx * s.sx;
  const double vy = s.n * s.syy - s.sy * s.sy;
  if (num == 0.0 || vx <= 0.0 || vy <= 0.0) return 0.0;
  return num / std::sqrt(vx * vy);
}

}  // namespace engine
