// The timing contract, as a test rather than as a comment.
//
// Two bases are reported for every backend and they mean different things:
//
//   device_total    steady state. What a resident engine pays per query, once
//                   the fixture is staged and the lane plan is built.
//   cold_data_path  entry to results in host memory: load, plan, runtime and
//                   communicator init, staging, compute, readback. NOT a CLI
//                   one-shot -- process launch and the CUDA driver's first
//                   touch happen before any code here could time them.
//
// Warp packing added a stage (plan construction) and it was initially left out
// of the cold path in both GPU backends, which made a cold caller look cheaper
// than it is. A comment saying "remember the plan" would not have caught that.
// This does: it perturbs each component in turn and requires the totals to
// move by exactly that amount, so a term that stops being summed fails here
// instead of quietly biasing a benchmark.

#include <cmath>
#include <cstdio>
#include <string>

#include "cuda/cuda_backend.hpp"

namespace {

int failures = 0, checks = 0;

void expect_near(double got, double want, const std::string& what) {
  ++checks;
  if (std::fabs(got - want) > 1e-12) {
    ++failures;
    std::printf("  FAIL  %s: got %.12f want %.12f\n", what.c_str(), got, want);
  }
}

engine::CudaTimings base() {
  engine::CudaTimings t;
  t.plan = 0.011;      // distinct primes-ish values, so a swapped pair of
  t.h2d = 0.022;       // fields cannot cancel out and pass by accident
  t.stats = 0.044;
  t.finalize = 0.088;
  t.d2h = 0.176;
  return t;
}

const double kLoad = 0.352;

}  // namespace

int main() {
  const engine::CudaTimings b = base();

  expect_near(b.device_total(), b.stats + b.finalize, "device_total = stats + finalize");
  expect_near(b.cold_data_path(kLoad),
              kLoad + b.plan + b.h2d + b.stats + b.finalize + b.d2h,
              "cold_data_path sums every stage");

  // Each stage must move the cold path by its own delta -- a dropped term
  // shows up as a zero here.
  const double d = 0.5;
  struct Field { const char* name; double engine::CudaTimings::*p; };
  for (const Field& f : {Field{"plan", &engine::CudaTimings::plan},
                         Field{"h2d", &engine::CudaTimings::h2d},
                         Field{"stats", &engine::CudaTimings::stats},
                         Field{"finalize", &engine::CudaTimings::finalize},
                         Field{"d2h", &engine::CudaTimings::d2h}}) {
    engine::CudaTimings t = b;
    t.*(f.p) += d;
    expect_near(t.cold_data_path(kLoad) - b.cold_data_path(kLoad), d,
                std::string("cold_data_path tracks ") + f.name);
  }
  {
    engine::CudaTimings t = b;
    t.plan += d;
    expect_near(t.cold_data_path(0.0) - b.cold_data_path(0.0), d,
                "the plan is inside the cold path");
    expect_near(t.device_total(), b.device_total(),
                "the plan is outside device_total");
  }
  // Fixture load is the caller's contribution and must pass through unscaled.
  expect_near(b.cold_data_path(kLoad + d) - b.cold_data_path(kLoad), d,
              "cold_data_path tracks t_load");

  // Staging and readback are cold-path only; steady state excludes them.
  for (auto p : {&engine::CudaTimings::h2d, &engine::CudaTimings::d2h}) {
    engine::CudaTimings t = b;
    t.*p += d;
    expect_near(t.device_total(), b.device_total(),
                "transfers stay outside device_total");
  }

  std::printf("timing contract: %d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
