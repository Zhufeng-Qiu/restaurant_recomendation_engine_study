// Fault injection for the shared output validator (common/result_check.hpp).
//
// The check this replaced could not see a NaN: `fabs(NaN - g) > kTol` is
// false, so an all-NaN run reported max_abs_diff = 0, tol_failures = 0 and
// exited 0. Case "the defect itself" below reconstructs those exact three
// lines and asserts that they DO pass on all-NaN output while check_result
// rejects it -- so the hole is demonstrated rather than asserted, and it
// cannot quietly come back.
//
// Every rejection case is paired with the in-domain control at the top: a
// validator that refuses everything would pass all the injections and is not
// a validator. No fixture files, no GPU.
//
// Usage: result_check_test

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "common/result_check.hpp"

namespace {

int checks = 0;
int failures = 0;

void expect(bool ok, const std::string& what) {
  ++checks;
  if (ok) {
    std::printf("ok   %s\n", what.c_str());
  } else {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
  }
}

engine::ResultVerdict run(const std::vector<double>& out,
                          const std::vector<double>& golden,
                          int64_t n_expected) {
  return engine::check_result(out.data(), static_cast<int64_t>(out.size()),
                              golden.data(),
                              static_cast<int64_t>(golden.size()), n_expected);
}

// A small, realistic similarity vector: some retained, some exactly zero,
// matching the shape of every shipped golden.bin.
std::vector<double> base() {
  return {0.9134, 0.0, 0.4471, 0.0, -0.2250, 0.7788, 0.0, 0.0501};
}

const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();

// The three lines that shipped in main.cpp, mpi_main.cpp and nccl_main.cu.
// Returns true if the OLD check would have called this output good.
bool old_check_passes(const std::vector<double>& out,
                      const std::vector<double>& golden) {
  int fails = 0;
  double max_diff = 0.0;
  for (size_t k = 0; k < out.size(); ++k) {
    const double d = std::fabs(out[k] - golden[k]);
    if (d > max_diff) max_diff = d;
    if (d > engine::kTol) ++fails;
  }
  return fails == 0;
}

void injection(const char* what, std::vector<double> out,
               std::vector<double> golden, int64_t n,
               bool want_nonfinite_out, bool want_nonfinite_golden) {
  const engine::ResultVerdict v = run(out, golden, n);
  expect(!v.passed, std::string("reject: ") + what);
  expect(!v.reason.empty(), std::string("reject explains itself: ") + what);
  if (want_nonfinite_out)
    expect(v.nonfinite_output > 0,
           std::string("counts non-finite output: ") + what);
  if (want_nonfinite_golden)
    expect(v.nonfinite_golden > 0,
           std::string("counts non-finite golden: ") + what);
  // Whatever went wrong, the number that reaches the JSON must still be a
  // number. %.3e on a NaN prints "nan", which is not valid JSON and takes the
  // whole benchmark record down with it.
  expect(std::isfinite(v.max_abs_diff),
         std::string("max_abs_diff stays finite: ") + what);
  const std::string j = engine::result_json_fields(v);
  expect(j.find("nan") == std::string::npos && j.find("inf") == std::string::npos,
         std::string("JSON carries no nan/inf: ") + what);
}

}  // namespace

int main() {
  const std::vector<double> g = base();
  const int64_t n = static_cast<int64_t>(g.size());

  // ---- positive controls: the gate must discriminate, not just refuse ----
  {
    const engine::ResultVerdict v = run(g, g, n);
    expect(v.passed, "accept: output identical to golden");
    expect(v.ran, "accept: validation_ran is set");
    expect(v.tol_failures == 0, "accept: no tolerance failures");
    expect(v.nonfinite_output == 0 && v.nonfinite_golden == 0,
           "accept: no non-finite values");
    expect(v.finite_compared == n, "accept: every pair compared");
    // 4 of 8: the filter is `sim > kEps`, not `|sim| > kEps`, so the
    // negative correlation at index 4 is dropped along with the three zeros.
    expect(v.emitted == 4 && v.emitted_golden == 4,
           "accept: retained counts agree (4 of 8)");
    expect(v.emitted_mismatches == 0, "accept: retained sets agree");
    expect(v.reason.empty(), "accept: passing verdict gives no reason");
  }
  {
    std::vector<double> out = g;
    out[2] += 9e-13;  // inside kTol = 1e-12
    const engine::ResultVerdict v = run(out, g, n);
    expect(v.passed, "accept: difference just inside tolerance");
    expect(v.max_abs_diff > 0.0, "accept: the difference is still reported");
  }
  {
    std::vector<double> out = g;
    out[2] += 2e-12;  // outside kTol
    const engine::ResultVerdict v = run(out, g, n);
    expect(!v.passed, "reject: difference just outside tolerance");
    expect(v.tol_failures == 1, "reject: exactly one pair over tolerance");
  }

  // ---- the defect itself ----
  {
    std::vector<double> out(g.size(), kNaN);
    expect(old_check_passes(out, g),
           "the old three-line check PASSES all-NaN output (the defect)");
    const engine::ResultVerdict v = run(out, g, n);
    expect(!v.passed, "reject: all-NaN output");
    expect(v.nonfinite_output == n, "reject: all 8 counted as non-finite");
    expect(v.finite_compared == 0, "reject: no pair was comparable");
    expect(v.max_abs_diff == 0.0 && std::isfinite(v.max_abs_diff),
           "reject: max_abs_diff has no sample and stays 0, not NaN");
  }

  // ---- non-finite injection, each of three values, into each of two sides --
  for (auto kv : {std::pair<const char*, double>{"NaN", kNaN},
                  {"+Inf", kInf},
                  {"-Inf", -kInf}}) {
    std::vector<double> out = g;
    out[3] = kv.second;
    injection((std::string("output[3] = ") + kv.first).c_str(), out, g, n,
              true, false);

    std::vector<double> bad_golden = g;
    bad_golden[5] = kv.second;
    injection((std::string("golden[5] = ") + kv.first).c_str(), g, bad_golden, n,
              false, true);
  }

  // ---- length ----
  {
    std::vector<double> shortened(g.begin(), g.end() - 1);
    const engine::ResultVerdict v = run(shortened, g, n);
    expect(!v.passed, "reject: output one element short");
    expect(v.reason.find("length mismatch") != std::string::npos,
           "reject: short output names the length, not a tolerance failure");
    // The loop must not have run at all -- reading golden[7] against a
    // 7-element output is the bug this check exists to prevent.
    expect(v.finite_compared == 0, "reject: short output compares nothing");
  }
  {
    std::vector<double> shortened(g.begin(), g.end() - 1);
    const engine::ResultVerdict v = run(g, shortened, n);
    expect(!v.passed, "reject: golden one element short");
  }

  // ---- misplacement: same values, wrong positions ----
  {
    std::vector<double> out = g;
    std::swap(out[0], out[2]);  // 0.9134 <-> 0.4471
    const engine::ResultVerdict v = run(out, g, n);
    expect(!v.passed, "reject: two similarities swapped");
    expect(v.emitted == v.emitted_golden,
           "reject: the swap leaves the retained COUNT identical");
  }

  // ---- what the count alone cannot see, and the set can ----
  {
    // Membership differs, but by less than kTol: the tolerance check is
    // blind to it and the retained-set check is not. This is the only
    // situation where the two disagree, and it is why both are kept.
    std::vector<double> golden = g;
    std::vector<double> out = g;
    golden[1] = 5e-15;   // below kEps = 1e-14 -> not retained
    out[1] = 2e-14;      // above kEps        -> retained
    const engine::ResultVerdict v = run(out, golden, n);
    expect(old_check_passes(out, golden),
           "tolerance alone accepts a retained-set change (difference 1.5e-14)");
    expect(!v.passed, "reject: retained set differs within tolerance");
    expect(v.emitted_mismatches == 1, "reject: exactly one membership flip");
    expect(v.tol_failures == 0,
           "reject: and it is NOT a tolerance failure -- separate check");
  }

  // ---- JSON shape on the happy path ----
  {
    const std::string j = engine::result_json_fields(run(g, g, n));
    expect(j.find("\"validation_ran\":true") != std::string::npos,
           "json: validation_ran true");
    expect(j.find("\"validation_passed\":true") != std::string::npos,
           "json: validation_passed true");
    const engine::ResultVerdict never;  // --validate not given
    const std::string jn = engine::result_json_fields(never);
    expect(jn.find("\"validation_ran\":false") != std::string::npos &&
               jn.find("\"validation_passed\":false") != std::string::npos,
           "json: a run that never validated is distinguishable from a pass");
  }

  std::printf("\n%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
