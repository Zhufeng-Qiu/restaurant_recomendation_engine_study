// Warp packing: choosing which pair each lane group works on.
//
// The baseline mapping is one warp per pair: 32 lanes stride over the shorter
// restricted rating slice and binary-search the longer one. Its cost is not the
// intersection size but the LANE SLOTS it occupies, 32 * ceil(L / 32) for a
// shorter slice of length L -- a model this project measured directly (see the
// lane-band experiment in docs/measurement_audit_20260905.md). The waste is the
// partly-filled last round, and it is 14.42% of the slots at one GPU and 25.32%
// at two, because splitting the dimension range halves L and leaves shorter
// tails.
//
// Packing recovers that by giving a pair a SUB-WARP of G lanes instead of all
// 32, so a warp carries 32/G pairs at once and a pair's tail wastes at most
// G-1 slots. That only works if the pairs sharing a warp need the same number
// of rounds: the groups run in lockstep, so the warp costs
// 32 * max_g ceil(L_g / G), and one long pair among short ones pays for all of
// them. Hence the ordering below. Sorting by L is not an optimisation on top of
// packing -- it is what makes packing profitable at all. Unsorted, the model
// says G=1 costs 63% MORE than the baseline; sorted, 14.39% less.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <vector>

#include "common/fixture.hpp"

namespace engine {

// Length of entity e's rating slice restricted to dims [lo, hi). Mirrors the
// two lower_bounds pair_lane_stats performs on the device.
inline int64_t restricted_row_len(const Fixture& fx, int32_t e, int32_t lo,
                                  int32_t hi) {
  const int32_t* b = fx.dims.data() + fx.offsets[e];
  const int32_t* e_ = fx.dims.data() + fx.offsets[e + 1];
  const int32_t* p = std::lower_bound(b, e_, lo);
  return std::lower_bound(p, e_, hi) - p;
}

// Elements the lanes actually stride over for pair k: the SHORTER slice.
inline int64_t pair_short_len(const Fixture& fx, int64_t k, int32_t lo,
                              int32_t hi) {
  return std::min(restricted_row_len(fx, fx.pairs[2 * k], lo, hi),
                  restricted_row_len(fx, fx.pairs[2 * k + 1], lo, hi));
}

enum class PairOrder { kSource, kByShortLen };

struct PairPlan {
  std::vector<int32_t> order;   // slot -> pair index
  int group = 32;               // lanes per pair
  int64_t effective_elements = 0;
  int64_t lane_slots = 0;       // what the model says this plan occupies
  double build_seconds = 0;

  double utilisation() const {
    return lane_slots ? static_cast<double>(effective_elements) / lane_slots : 0;
  }
};

// Builds the slot -> pair permutation and the model's slot count for it.
// `group` must be a power of two in [1, 32]. kSource leaves the fixture's own
// order (identity), so the baseline goes through the same indirection as every
// packed arm and the comparison isolates G and the sort.
inline PairPlan plan_pairs(const Fixture& fx, int32_t lo, int32_t hi,
                           PairOrder how, int group) {
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t n = fx.n_pairs();
  PairPlan p;
  p.group = group;
  p.order.resize(static_cast<size_t>(n));

  std::vector<int32_t> len(static_cast<size_t>(n));
  for (int64_t k = 0; k < n; ++k)
    len[k] = static_cast<int32_t>(pair_short_len(fx, k, lo, hi));

  std::iota(p.order.begin(), p.order.end(), 0);
  if (how == PairOrder::kByShortLen) {
    // Ascending, ties by pair index, so the plan is a deterministic function of
    // (fixture, dim range, group) and two runs are comparable.
    std::stable_sort(p.order.begin(), p.order.end(),
                     [&](int32_t a, int32_t b) { return len[a] < len[b]; });
  }

  const int per_warp = 32 / group;
  for (int64_t s = 0; s < n; s += per_warp) {
    int64_t rounds = 0;
    for (int64_t t = s; t < std::min(s + per_warp, n); ++t) {
      const int64_t L = len[p.order[t]];
      p.effective_elements += L;
      rounds = std::max(rounds, (L + group - 1) / group);
    }
    p.lane_slots += 32 * rounds;
  }
  p.build_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return p;
}

}  // namespace engine
