// Warp packing: choosing which pair each lane group works on.
//
// The baseline mapping is one warp per pair: 32 lanes stride over the shorter
// restricted rating slice and binary-search the longer one. Its cost is not the
// intersection size but the LANE SLOTS it occupies, 32 * ceil(L / 32) for a
// shorter slice of length L -- a model this project measured directly (see the
// lane-band experiment in docs/measurement_audit_20260905.md). The waste is the
// partly-filled last round.
//
// Packing recovers that by giving a pair a SUB-WARP of G lanes instead of all
// 32, so a warp carries 32/G pairs and a pair's tail wastes at most G-1 slots.
// That only works if the pairs sharing a warp need the same number of rounds:
// the groups run in lockstep, so the warp costs 32 * max_g ceil(L_g / G), and
// one long pair among short ones pays for all of them. Hence the ordering.
// Sorting by L is not an optimisation on top of packing -- it is what makes
// packing profitable at all.
//
// WHAT THE MODEL IS AND IS NOT. It counts scan-loop lane slots under a
// zero-overhead assumption. It says nothing about register pressure,
// occupancy, memory coalescing, the binary search into the longer row, the
// finalize scatter, or the collective. It is a mechanism hypothesis to be
// tested against measurement, never a substitute for one, and a modelled
// percentage is not an achieved speedup.
//
// MULTI-GPU. Every device must be handed the SAME order: slot s must stand for
// the same pair everywhere or the AllReduce would sum statistics belonging to
// different pairs. The shared order is therefore built over the full dimension
// range, which is nobody's own optimum -- each device then executes that
// order against its own [dim_lo, dim_hi) slice, where the row lengths differ.
// Reporting the full-dimension plan's utilisation as if it described what a
// GPU executed would overstate it. evaluate_order re-evaluates a fixed order
// against a given slice, which is what the devices actually run.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "common/fixture.hpp"

namespace engine {

constexpr int kLaneWidth = 32;

inline bool valid_group(int g) {
  return g >= 1 && g <= kLaneWidth && (g & (g - 1)) == 0;
}

// Length of entity e's rating slice restricted to dims [lo, hi). Mirrors the
// two lower_bounds pair_lane_stats performs on the device.
inline int64_t restricted_row_len(const Fixture& fx, int32_t e, int32_t lo,
                                  int32_t hi) {
  const int32_t* b = fx.dims.data() + fx.offsets[e];
  const int32_t* end = fx.dims.data() + fx.offsets[e + 1];
  const int32_t* p = std::lower_bound(b, end, lo);
  return std::lower_bound(p, end, hi) - p;
}

// Elements the lanes actually stride over for pair k: the SHORTER slice.
inline int64_t pair_short_len(const Fixture& fx, int64_t k, int32_t lo,
                              int32_t hi) {
  return std::min(restricted_row_len(fx, fx.pairs[2 * k], lo, hi),
                  restricted_row_len(fx, fx.pairs[2 * k + 1], lo, hi));
}

inline std::vector<int32_t> short_lens(const Fixture& fx, int32_t lo,
                                       int32_t hi) {
  std::vector<int32_t> len(static_cast<size_t>(fx.n_pairs()));
  for (int64_t k = 0; k < fx.n_pairs(); ++k)
    len[k] = static_cast<int32_t>(pair_short_len(fx, k, lo, hi));
  return len;
}

enum class PairOrder { kSource, kByShortLen };

inline const char* pair_order_name(PairOrder o) {
  return o == PairOrder::kByShortLen ? "bylen" : "source";
}

// What a given (order, group) occupies on a given dimension slice.
struct PlanMetrics {
  int64_t effective_elements = 0;  // sum of shorter-slice lengths
  int64_t lane_slots = 0;          // 32 * sum over warps of the warp's rounds
  double utilisation() const {
    return lane_slots ? static_cast<double>(effective_elements) / lane_slots
                      : 0.0;
  }
};

// Cost of executing `order` at group size `group`, given each pair's shorter
// slice length on the slice of interest.
//
// Warps are formed from consecutive slots -- 32 threads cover 32/group slots,
// and 32 divides the block size, so warp boundaries land on slot boundaries.
// A warp costs 32 * (its longest member's round count): the groups are in
// lockstep, and a trailing partial warp still occupies a whole warp.
inline PlanMetrics evaluate_order(const std::vector<int32_t>& order,
                                  const std::vector<int32_t>& len, int group) {
  if (!valid_group(group))
    throw std::runtime_error("group must be 1, 2, 4, 8, 16 or 32");
  PlanMetrics m;
  const int per_warp = kLaneWidth / group;
  const int64_t n = static_cast<int64_t>(order.size());
  for (int64_t s = 0; s < n; s += per_warp) {
    int64_t rounds = 0;
    for (int64_t t = s; t < std::min(s + per_warp, n); ++t) {
      const int64_t L = len[order[t]];
      m.effective_elements += L;
      rounds = std::max(rounds, (L + group - 1) / group);
    }
    m.lane_slots += kLaneWidth * rounds;
  }
  return m;
}

// Convenience: derives the lengths for [lo, hi) first. Prefer the overload
// above when evaluating several orders or groups against one slice -- the
// length pass is the expensive half.
inline PlanMetrics evaluate_order(const Fixture& fx,
                                  const std::vector<int32_t>& order, int group,
                                  int32_t lo, int32_t hi) {
  return evaluate_order(order, short_lens(fx, lo, hi), group);
}

struct PairPlan {
  std::vector<int32_t> order;   // slot -> pair index
  int group = 32;               // lanes per pair
  PairOrder basis = PairOrder::kSource;
  int32_t basis_dim_lo = 0, basis_dim_hi = 0;  // slice the order was sorted on
  PlanMetrics on_basis;         // cost on that same slice
  double build_seconds = 0;
};

// Builds the slot -> pair permutation over dims [lo, hi).
//
// kSource leaves the fixture's own order (identity), so the baseline goes
// through the same indirection as every packed arm and a comparison between
// them isolates the group size and the sort rather than the plumbing.
inline PairPlan plan_pairs(const Fixture& fx, int32_t lo, int32_t hi,
                           PairOrder how, int group) {
  if (!valid_group(group))
    throw std::runtime_error("group must be 1, 2, 4, 8, 16 or 32");
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t n = fx.n_pairs();
  PairPlan p;
  p.group = group;
  p.basis = how;
  p.basis_dim_lo = lo;
  p.basis_dim_hi = hi;
  p.order.resize(static_cast<size_t>(n));
  std::iota(p.order.begin(), p.order.end(), 0);

  const std::vector<int32_t> len = short_lens(fx, lo, hi);
  if (how == PairOrder::kByShortLen) {
    // Ascending, ties by pair index, so the plan is a deterministic function
    // of (fixture, dim range, group) and two runs are comparable.
    std::stable_sort(p.order.begin(), p.order.end(),
                     [&](int32_t a, int32_t b) { return len[a] < len[b]; });
  }
  p.on_basis = evaluate_order(p.order, len, group);
  p.build_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  return p;
}

// COUNTERFACTUAL ONLY. What this slice would cost if it could sort its own
// pairs, which a multi-GPU run cannot do: the collective is elementwise, so
// all devices are locked to one shared order. Useful as an upper bound on what
// a smarter mapping could reach; never report it as achieved or achievable.
inline PlanMetrics counterfactual_per_device_ideal(const Fixture& fx, int group,
                                                   int32_t lo, int32_t hi) {
  const std::vector<int32_t> len = short_lens(fx, lo, hi);
  std::vector<int32_t> own(len.size());
  std::iota(own.begin(), own.end(), 0);
  std::stable_sort(own.begin(), own.end(),
                   [&](int32_t a, int32_t b) { return len[a] < len[b]; });
  return evaluate_order(own, len, group);
}

}  // namespace engine
