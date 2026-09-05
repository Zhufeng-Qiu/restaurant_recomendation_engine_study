// pearson_engine: evaluate a fixture's candidate pairs with a selected backend.
//
// Usage:
//   pearson_engine <fixture_dir> [--backend serial] [--validate] [--out sims.bin]
//
// --validate compares computed similarities against the fixture's golden.bin
// (value comparison per contract §5) and exits non-zero on failure.
// Timing is reported as a single JSON line on stdout for the bench harness.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "common/fixture.hpp"
#include "common/pearson.hpp"
#include "serial/serial.hpp"
#ifdef ENGINE_HAVE_OPENMP
#include "openmp/openmp.hpp"
#endif
#ifdef ENGINE_HAVE_CUDA
#include "cuda/cuda_backend.hpp"
#endif

namespace {

using Backend = std::function<void(const engine::Fixture&, std::vector<double>&)>;

std::map<std::string, Backend> backends() {
  std::map<std::string, Backend> m;
  m["serial"] = engine::compute_serial;
#ifdef ENGINE_HAVE_OPENMP
  m["openmp"] = engine::compute_openmp;
#endif
#ifdef ENGINE_HAVE_CUDA
  m["cuda"] = engine::compute_cuda;
#endif
  return m;
}

double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Body of main; main() below is a thin exception boundary. Fixture::load and
// the CUDA backend both throw, and an escaping exception terminates on
// SIGABRT with the message buried under "terminate called after throwing...".
int run(int argc, char** argv);

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}

namespace {

int run(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <fixture_dir> [--backend serial] [--validate] [--out f]\n"
                 "       [--group 1|2|4|8|16|32] [--pair-order source|bylen]\n",
                 argv[0]);
    return 2;
  }
  std::string dir = argv[1];
  std::string backend_name = "serial";
  std::string out_path;
  bool validate = false;
  for (int a = 2; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--backend") && a + 1 < argc) backend_name = argv[++a];
    else if (!std::strcmp(argv[a], "--out") && a + 1 < argc) out_path = argv[++a];
    else if (!std::strcmp(argv[a], "--validate")) validate = true;
#if defined(ENGINE_HAVE_CUDA)
    // Lane mapping (warp packing). Accepted only where a GPU backend exists,
    // so a CPU-only build rejects them instead of silently ignoring them and
    // reporting a mapping it never ran.
    else if (!std::strcmp(argv[a], "--group") && a + 1 < argc) {
      const int g = std::atoi(argv[++a]);
      if (g < 1 || g > 32 || (g & (g - 1)) != 0) {
        std::fprintf(stderr, "--group must be 1, 2, 4, 8, 16 or 32\n");
        return 2;
      }
      engine::cuda_map_options().group = g;
    }
    else if (!std::strcmp(argv[a], "--pair-order") && a + 1 < argc) {
      const std::string v = argv[++a];
      if (v == "source") engine::cuda_map_options().order = engine::PairOrder::kSource;
      else if (v == "bylen") engine::cuda_map_options().order = engine::PairOrder::kByShortLen;
      else { std::fprintf(stderr, "--pair-order must be source or bylen\n"); return 2; }
    }
#endif
    else { std::fprintf(stderr, "unknown arg %s\n", argv[a]); return 2; }
  }

  auto reg = backends();
  auto it = reg.find(backend_name);
  if (it == reg.end()) {
    std::fprintf(stderr, "unknown backend '%s'\n", backend_name.c_str());
    return 2;
  }

  const double t0 = now_s();
  engine::Fixture fx = engine::Fixture::load(dir);
  const double t_load = now_s() - t0;

  std::vector<double> sims;
  const double t1 = now_s();
  it->second(fx, sims);
  const double t_compute = now_s() - t1;

  int failures = 0;
  double max_diff = 0.0;
  int64_t emitted = 0;
  if (validate) {
    for (int64_t k = 0; k < fx.n_pairs(); ++k) {
      const double d = std::fabs(sims[k] - fx.golden[k]);
      if (d > max_diff) max_diff = d;
      if (d > engine::kTol) ++failures;
      if (sims[k] > engine::kEps) ++emitted;
    }
  }

  if (!out_path.empty()) {
    std::ofstream f(out_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(sims.data()),
            static_cast<std::streamsize>(sims.size() * sizeof(double)));
  }

  // Unified timing schema (README, 2026-09-05). For serial/openmp the whole
  // kernel is one fused pass -- finalization happens inside evaluate_pair --
  // so stats carries it and finalize is 0 rather than unmeasured. The cuda
  // backend prints its own stage breakdown and its own device/one-shot totals
  // on the cuda_detail line, so they are omitted here to avoid overwriting it.
  const bool cuda_owns_totals = (backend_name == "cuda");
  if (cuda_owns_totals) {
    std::printf(
        "{\"fixture\":\"%s\",\"backend\":\"%s\",\"n_entities\":%lld,\"nnz\":%lld,"
        "\"n_pairs\":%lld,\"t_load_s\":%.6f,\"t_compute_s\":%.6f,"
        "\"pairs_per_s\":%.0f,\"validated\":%s,\"max_abs_diff\":%.3e,"
        "\"tol_failures\":%d,\"emitted\":%lld}\n",
        dir.c_str(), backend_name.c_str(),
        static_cast<long long>(fx.n_entities()), static_cast<long long>(fx.nnz()),
        static_cast<long long>(fx.n_pairs()), t_load, t_compute,
        fx.n_pairs() / (t_compute > 0 ? t_compute : 1e-9),
        validate ? "true" : "false", max_diff, failures,
        static_cast<long long>(emitted));
  } else {
    std::printf(
        "{\"fixture\":\"%s\",\"backend\":\"%s\",\"n_entities\":%lld,\"nnz\":%lld,"
        "\"n_pairs\":%lld,\"timing_basis\":\"device_total\","
        "\"t_load_s\":%.6f,\"t_setup_s\":0.000000,"
        "\"t_stats_s\":%.6f,\"t_allreduce_s\":0.000000,\"t_finalize_s\":0.000000,"
        "\"t_d2h_s\":0.000000,"
        "\"device_total_s\":%.6f,\"cold_data_path_s\":%.6f,"
        "\"t_compute_s\":%.6f,"
        "\"pairs_per_s\":%.0f,\"validated\":%s,\"max_abs_diff\":%.3e,"
        "\"tol_failures\":%d,\"emitted\":%lld}\n",
        dir.c_str(), backend_name.c_str(),
        static_cast<long long>(fx.n_entities()), static_cast<long long>(fx.nnz()),
        static_cast<long long>(fx.n_pairs()), t_load, t_compute,
        t_compute, t_load + t_compute, t_compute,
        fx.n_pairs() / (t_compute > 0 ? t_compute : 1e-9),
        validate ? "true" : "false", max_diff, failures,
        static_cast<long long>(emitted));
  }

  return (validate && failures > 0) ? 1 : 0;
}

}  // namespace
