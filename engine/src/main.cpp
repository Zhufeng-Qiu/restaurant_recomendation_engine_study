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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <fixture_dir> [--backend serial] [--validate] [--out f]\n",
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

  return (validate && failures > 0) ? 1 : 0;
}
