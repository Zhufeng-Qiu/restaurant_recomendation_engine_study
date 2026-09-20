// Shared output verification for every backend entry point.
//
// What this replaces, copied verbatim into main.cpp, mpi_main.cpp and
// nccl_main.cu:
//
//     const double d = std::fabs(sims[k] - golden[k]);
//     if (d > max_diff) max_diff = d;
//     if (d > engine::kTol) ++failures;
//
// Those three lines cannot see a NaN. Every comparison against a NaN is
// false, so `d > max_diff` and `d > kTol` are both false and a run that
// produced nothing but NaN reported max_abs_diff = 0.000e+00,
// tol_failures = 0, and exited 0. A silent pass, in triplicate. That is the
// reason this is one header and not a fourth copy.
//
// This is a validation hole, not evidence that any past run produced NaN --
// no published number is being withdrawn here. It means the gate could not
// have told us either way, which is the same reason the domain gate in
// payload_domain.hpp got a test: a check that has only ever returned "pass"
// is not evidence of anything.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

#include "common/pearson.hpp"  // kTol, kEps

namespace engine {

struct ResultVerdict {
  // "Was it asked for" and "did it succeed" are different questions, and the
  // old JSON answered only the first: `validated: true` meant --validate was
  // on the command line. A consumer could not distinguish a passing run from
  // one that ran the check and hated the answer.
  bool ran = false;
  bool passed = false;

  int64_t n_expected = 0;
  int64_t n_output = 0;
  int64_t n_golden = 0;

  int64_t nonfinite_output = 0;
  int64_t nonfinite_golden = 0;
  int64_t finite_compared = 0;  // pairs where BOTH sides are finite
  int64_t tol_failures = 0;
  double max_abs_diff = 0.0;  // over finite_compared only; 0 when that is 0

  int64_t emitted = 0;         // |{k : out[k] > kEps}|
  int64_t emitted_golden = 0;  // |{k : golden[k] > kEps}|
  int64_t emitted_mismatches = 0;

  std::string reason;  // first thing found wrong; empty iff passed
};

// Comparing only the COUNT of retained pairs cannot distinguish "the right
// similarities in the wrong places" from a correct run: any permutation of
// the output preserves the count exactly. Membership is compared per index.
//
// Failing on a membership disagreement is safe here rather than merely
// strict. Over all ten shipped golden.bin files the smallest non-zero |sim|
// is 4.986e-04 -- ten orders of magnitude above kEps = 1e-14 -- and every
// other value is exactly 0. Nothing straddles the threshold, so a
// disagreement cannot be a value wobbling across it within tolerance. It is a
// misplacement.
inline ResultVerdict check_result(const double* out, int64_t n_out,
                                  const double* golden, int64_t n_golden,
                                  int64_t n_expected) {
  ResultVerdict v;
  v.ran = true;
  v.n_expected = n_expected;
  v.n_output = n_out;
  v.n_golden = n_golden;

  // 1. Lengths, before any element is touched. A short array must not reach
  //    the comparison loop -- that is a read past the end, not a tolerance
  //    failure.
  if (n_out != n_expected || n_golden != n_expected) {
    char buf[192];
    std::snprintf(buf, sizeof buf,
                  "length mismatch: output=%lld golden=%lld expected=%lld",
                  static_cast<long long>(n_out),
                  static_cast<long long>(n_golden),
                  static_cast<long long>(n_expected));
    v.reason = buf;
    return v;
  }

  // 2. Finiteness of each side separately, so "the kernel produced garbage"
  //    and "the fixture shipped garbage" are distinguishable.
  for (int64_t k = 0; k < n_expected; ++k) {
    const bool fo = std::isfinite(out[k]);
    const bool fg = std::isfinite(golden[k]);
    if (!fo) ++v.nonfinite_output;
    if (!fg) ++v.nonfinite_golden;

    // 3. Absolute error, but only where both sides are a number. kTol is
    //    unchanged at 1e-12; what changes is that a NaN now lands in the
    //    count above instead of vanishing.
    if (fo && fg) {
      ++v.finite_compared;
      const double d = std::fabs(out[k] - golden[k]);
      if (d > v.max_abs_diff) v.max_abs_diff = d;
      if (d > kTol) ++v.tol_failures;
    }

    // 4. Retained-set membership, index by index.
    const bool eo = fo && out[k] > kEps;
    const bool eg = fg && golden[k] > kEps;
    if (eo) ++v.emitted;
    if (eg) ++v.emitted_golden;
    if (eo != eg) ++v.emitted_mismatches;
  }

  // max_abs_diff is a maximum over finite differences, so it is finite by
  // construction and never reaches the JSON as nan or inf. When
  // finite_compared is 0 there is no sample at all, and 0.0 would read as a
  // perfect run -- finite_compared and the non-finite counts are what say
  // otherwise, which is why they are emitted rather than folded into a flag.
  char buf[256];
  if (v.nonfinite_output || v.nonfinite_golden) {
    std::snprintf(buf, sizeof buf,
                  "non-finite similarities: output=%lld golden=%lld of %lld",
                  static_cast<long long>(v.nonfinite_output),
                  static_cast<long long>(v.nonfinite_golden),
                  static_cast<long long>(n_expected));
    v.reason = buf;
  } else if (v.tol_failures) {
    std::snprintf(buf, sizeof buf,
                  "%lld of %lld pairs exceed tolerance %.1e (max_abs_diff %.6e)",
                  static_cast<long long>(v.tol_failures),
                  static_cast<long long>(n_expected), kTol, v.max_abs_diff);
    v.reason = buf;
  } else if (v.emitted_mismatches) {
    std::snprintf(buf, sizeof buf,
                  "retained set differs at %lld of %lld indices "
                  "(emitted=%lld golden=%lld)",
                  static_cast<long long>(v.emitted_mismatches),
                  static_cast<long long>(n_expected),
                  static_cast<long long>(v.emitted),
                  static_cast<long long>(v.emitted_golden));
    v.reason = buf;
  } else {
    v.passed = true;
  }
  return v;
}

// One formatter for all three entry points, so the JSON cannot drift between
// them the way the check itself did. The old `validated` field is gone rather
// than aliased: it meant "--validate was on the command line", nothing read
// it, and keeping it beside validation_ran would have shipped two identical
// booleans with different names. Returned as a string and printed with a
// single %s: this project has already shipped two JSON fields as "(null)" by
// miscounting printf arguments, and a fragment with no arguments cannot.
inline std::string result_json_fields(const ResultVerdict& v) {
  char buf[512];
  std::snprintf(
      buf, sizeof buf,
      "\"validation_ran\":%s,\"validation_passed\":%s,"
      "\"max_abs_diff\":%.3e,\"tol_failures\":%lld,\"emitted\":%lld,"
      "\"emitted_golden\":%lld,\"emitted_mismatches\":%lld,"
      "\"nonfinite_output\":%lld,\"nonfinite_golden\":%lld,"
      "\"finite_compared\":%lld",
      v.ran ? "true" : "false", v.passed ? "true" : "false", v.max_abs_diff,
      static_cast<long long>(v.tol_failures),
      static_cast<long long>(v.emitted),
      static_cast<long long>(v.emitted_golden),
      static_cast<long long>(v.emitted_mismatches),
      static_cast<long long>(v.nonfinite_output),
      static_cast<long long>(v.nonfinite_golden),
      static_cast<long long>(v.finite_compared));
  return buf;
}

}  // namespace engine
