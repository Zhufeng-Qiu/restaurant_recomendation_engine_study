// Compressed-payload domain predicate (docs/pearson_contract.md section 9.4).
//
// `--payload i32` / `packed` are lossless only while the fixture stays inside
// the domain they assume: non-negative integer ratings, and a longest rating
// row small enough that no statistic can overflow its field. Outside it, a
// silently overflowed field would corrupt the AllReduce sum while every
// backend still agreed with itself -- the worst failure mode this project has,
// because the bit-exact gates would all still pass.
//
// The predicate lives here, host-compilable and free of CUDA/NCCL, so
// tests/payload_domain_test.cpp can exercise BOTH branches on any machine.
// The field capacity comes from engine_cuda::kPackMaxField rather than a
// second copy of the 21, so the test cannot drift from the packing it guards.
// src/nccl/nccl_main.cu calls this header; its wrapper only turns the verdict
// into an exception.
//
// Two things changed on 2026-09-17:
//
//   * The capacity test was a single bound, `vmax^2 * max_row <= L`, which
//     covers Sxx/Syy/Sxy and -- for vmax >= 1 -- happens to dominate
//     `n <= max_row` as well. At vmax = 0 it collapses to `0 <= L` and a row
//     of any length whatsoever passes, with n unbounded. The three bounds are
//     now stated separately, one per group of statistics, instead of relying
//     on one of them to imply the others.
//
//   * Non-finite ratings are now refused on EVERY path, f64 included. A NaN
//     rating is outside the contract for all backends, and finiteness of the
//     OUTPUT is not evidence of it: pearson_finalize's zero-variance branch
//     can turn an abnormal intermediate into a clean 0.0.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>

#include "common/fixture.hpp"
#include "cuda/pair_kernel.cuh"

namespace engine {

enum class PayloadKind { F64, I32, Packed };

inline const char* payload_kind_name(PayloadKind p) {
  switch (p) {
    case PayloadKind::I32: return "i32";
    case PayloadKind::Packed: return "packed";
    default: return "f64";
  }
}

// Field capacity per statistic, matching the wire layouts in contract 9.2.
inline double payload_field_limit(PayloadKind p) {
  switch (p) {
    case PayloadKind::Packed:
      return static_cast<double>(engine_cuda::kPackMaxField);  // 2^21 - 1
    case PayloadKind::I32:
      return 2147483647.0;  // 2^31 - 1
    default:
      return 0.0;  // f64 carries no precondition
  }
}

struct DomainVerdict {
  bool ok = true;
  std::string reason;  // empty iff ok
};

// The capacity decision, separated from the scan that feeds it. Kept pure so
// the boundary can be tested at L-1, L, L+1 and vmax = 0 without constructing
// a fixture with two billion elements -- the reason the vmax = 0 hole above
// survived as long as it did.
//
// The three bounds cover the six statistics the AllReduce sums:
//
//     n                 <= max_row              (n <= the shorter row)
//     Sx,  Sy           <= vmax * max_row
//     Sxx, Syy, Sxy     <= vmax^2 * max_row
//
// They constrain the statistics AFTER the cross-GPU sum, not each device's
// partial: max_row is the global row length, and the rating dimension is
// partitioned, so the partials sum to at most these values. Checking that
// each device's own partial fits would not be enough.
//
// Double arithmetic is the overflow-safe choice here, not a shortcut.
// Integers below 2^53 are exact, so every accepted product (at most 2^31-1)
// is represented exactly and no rounding can move a value across the limit;
// above that the products only lose precision upward, and an overflow to
// +inf still compares greater than the limit. There is no wraparound to
// worry about, which is exactly what an int64 multiply would have.
inline DomainVerdict check_payload_bounds(int64_t max_row, double vmax,
                                          PayloadKind p) {
  if (p == PayloadKind::F64) return {};  // no field, no capacity

  const double limit = payload_field_limit(p);
  const double rows = static_cast<double>(max_row);
  const struct {
    double value;
    const char* covers;
  } bounds[] = {
      {rows, "n"},
      {vmax * rows, "sum(x), sum(y)"},
      {vmax * vmax * rows, "sum(x^2), sum(y^2), sum(xy)"},
  };

  for (const auto& b : bounds) {
    if (b.value > limit) {
      return {false, std::string("--payload ") + payload_kind_name(p) +
                         ": bound on " + b.covers + " is " +
                         std::to_string(b.value) + ", exceeding field capacity " +
                         std::to_string(limit) +
                         " (max_row=" + std::to_string(max_row) +
                         ", max_rating=" + std::to_string(vmax) +
                         "); use --payload f64"};
    }
  }
  return {};
}

struct RatingExtent {
  int64_t max_row = 0;
  double vmax = 0.0;
};

inline DomainVerdict check_payload_domain(const Fixture& fx, PayloadKind p,
                                          RatingExtent* out_extent = nullptr) {
  RatingExtent ext;
  for (size_t i = 0; i + 1 < fx.offsets.size(); ++i)
    ext.max_row = std::max(ext.max_row, fx.offsets[i + 1] - fx.offsets[i]);

  const bool needs_integers = (p != PayloadKind::F64);
  for (size_t i = 0; i < fx.vals.size(); ++i) {
    const double v = fx.vals[i];
    // Every path, every payload: a rating that is not a number is outside the
    // contract, and no downstream check can recover the fact afterwards.
    if (!std::isfinite(v)) {
      return {false, "rating[" + std::to_string(i) +
                         "] is not finite (" + std::to_string(v) +
                         "); every backend requires finite ratings"};
    }
    if (needs_integers && (v < 0.0 || v != std::floor(v))) {
      return {false, std::string("--payload ") + payload_kind_name(p) +
                         " requires non-negative integer ratings; found " +
                         std::to_string(v)};
    }
    ext.vmax = std::max(ext.vmax, v);
  }

  if (out_extent) *out_extent = ext;
  return check_payload_bounds(ext.max_row, ext.vmax, p);
}

}  // namespace engine
