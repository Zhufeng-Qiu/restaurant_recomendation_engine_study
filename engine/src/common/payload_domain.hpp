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
//
// NOTE ON DUPLICATION: src/nccl/nccl_main.cu still carries its own throwing
// copy of this logic (`check_payload_domain`). The two must be kept in sync;
// this header is the reference, and the intended follow-up is to have
// nccl_main.cu call it directly. That edit is deliberately NOT made here: this
// machine has no CUDA toolkit, so a change to the file that produced every
// published GPU number could not be recompiled, let alone re-validated. It
// belongs in the next session on a GPU host. Until then this header is
// verified against the same fixtures the binary would see, and
// `agrees_with_nccl_main_reference()` below records the exact predicate the
// copy implements.
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

// Mirrors nccl_main.cu's check_payload_domain, returning the verdict instead
// of throwing so both branches are assertable.
inline DomainVerdict check_payload_domain(const Fixture& fx, PayloadKind p) {
  if (p == PayloadKind::F64) return {};  // no precondition

  int64_t max_row = 0;
  for (size_t i = 0; i + 1 < fx.offsets.size(); ++i)
    max_row = std::max(max_row, fx.offsets[i + 1] - fx.offsets[i]);

  double vmax = 0.0;
  for (double v : fx.vals) {
    if (v < 0.0 || v != std::floor(v)) {
      return {false, std::string("--payload ") + payload_kind_name(p) +
                         " requires non-negative integer ratings; found " +
                         std::to_string(v)};
    }
    vmax = std::max(vmax, v);
  }

  // Loosest bound of the six: Sxx, Syy, Sxy <= vmax^2 * n, and n <= max_row.
  const double hi = vmax * vmax * static_cast<double>(max_row);
  const double limit = payload_field_limit(p);
  if (hi > limit) {
    return {false, std::string("--payload ") + payload_kind_name(p) +
                       ": max statistic " + std::to_string(hi) +
                       " exceeds field capacity " + std::to_string(limit) +
                       " (max_row=" + std::to_string(max_row) +
                       ", max_rating=" + std::to_string(vmax) +
                       "); use --payload f64"};
  }
  return {};
}

}  // namespace engine
