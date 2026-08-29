// Unit test for the frozen Pearson contract microcases.
// Normative source: data/fixtures/microcases.json (hand-copied here; the two
// must be kept in sync — the JSON is the reference).

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "common/pearson.hpp"

using engine::kEps;
using engine::kMinOverlap;
using engine::kTol;
using engine::PairStats;
using engine::pearson_finalize;

namespace {

using Vec = std::vector<std::pair<int, double>>;  // (dim, rating), dim ascending

struct Case {
  std::string name;
  Vec x, y;
  bool candidate;       // n >= kMinOverlap
  double sim;           // meaningful iff candidate
  bool emit;            // meaningful iff candidate
  long long expect_n;
};

// Sorted two-pointer intersection, same as the engine kernel.
PairStats stats_of(const Vec& x, const Vec& y) {
  PairStats s;
  size_t a = 0, b = 0;
  while (a < x.size() && b < y.size()) {
    if (x[a].first < y[b].first) ++a;
    else if (y[b].first < x[a].first) ++b;
    else { s.add(x[a].second, y[b].second); ++a; ++b; }
  }
  return s;
}

const double kFrac = 0.7559289460184544;  // 2/sqrt(7)

const std::vector<Case> kCases = {
    {"identical", {{1,1},{2,2},{3,3}}, {{1,1},{2,2},{3,3}}, true, 1.0, true, 3},
    {"inverse", {{1,1},{2,2},{3,3}}, {{1,3},{2,2},{3,1}}, true, -1.0, false, 3},
    {"one_overlap", {{1,5},{2,1}}, {{1,4},{3,2}}, false, 0, false, 1},
    {"no_overlap", {{1,1},{2,2}}, {{3,5},{4,4}}, false, 0, false, 0},
    {"two_overlap", {{1,1},{2,2}}, {{1,2},{2,4}}, false, 0, false, 2},
    {"zero_variance", {{1,4},{2,4},{3,4}}, {{1,1},{2,3},{3,5}}, true, 0.0, false, 3},
    {"orthogonal", {{1,1},{2,2},{3,3}}, {{1,2},{2,4},{3,2}}, true, 0.0, false, 3},
    {"fractional", {{1,1},{2,2},{3,4}}, {{1,1},{2,3},{3,3}}, true, kFrac, true, 3},
    {"intersection_only", {{1,1},{2,2},{3,4},{4,5}}, {{1,1},{2,3},{3,3},{5,2}},
     true, kFrac, true, 3},
};

}  // namespace

int main() {
  int failures = 0;
  for (const auto& c : kCases) {
    const PairStats s = stats_of(c.x, c.y);
    const long long n = static_cast<long long>(s.n);
    if (n != c.expect_n) {
      std::printf("FAIL %s: n=%lld expected %lld\n", c.name.c_str(), n, c.expect_n);
      ++failures;
      continue;
    }
    const bool candidate = n >= kMinOverlap;
    if (candidate != c.candidate) {
      std::printf("FAIL %s: candidate=%d expected %d\n", c.name.c_str(), candidate,
                  c.candidate);
      ++failures;
      continue;
    }
    if (!candidate) continue;
    const double sim = pearson_finalize(s);
    if (std::fabs(sim - c.sim) > kTol) {
      std::printf("FAIL %s: sim=%.17g expected %.17g\n", c.name.c_str(), sim, c.sim);
      ++failures;
    }
    if ((sim > kEps) != c.emit) {
      std::printf("FAIL %s: emit=%d expected %d\n", c.name.c_str(), sim > kEps, c.emit);
      ++failures;
    }
  }
  if (failures) {
    std::printf("FAIL: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("OK: all %zu microcases pass\n", kCases.size());
  return 0;
}
