// MPI backend: distributed Pearson via partial sufficient statistics.
//
// Decomposition (contract §6 / project brief): ranks partition the RATING
// DIMENSION axis (users in item mode) into contiguous index ranges. Every
// rank evaluates EVERY candidate pair, but only over the dims it owns,
// producing a local (n_pairs x 6) statistics tensor. A single
// MPI_Allreduce(MPI_SUM) combines the tensors; finalization then runs on
// rank 0. Pairs are never sharded — this is deliberately the same
// communication pattern the two-GPU NCCL backend will use.
//
// Usage:
//   mpirun -np R pearson_engine_mpi <fixture_dir> [--validate] [--out sims.bin]

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common/fixture.hpp"
#include "common/pearson.hpp"

namespace {

// Accumulate the six statistics for every pair, restricted to dim indices in
// [dim_lo, dim_hi). Rows are dim-ascending, so the owned slice of each row is
// located with binary search and then intersected with a two-pointer walk.
void partial_stats(const engine::Fixture& fx, int32_t dim_lo, int32_t dim_hi,
                   std::vector<double>& stats /* 6 * n_pairs */) {
  const int64_t n = fx.n_pairs();
  stats.assign(6 * static_cast<size_t>(n), 0.0);
  const int32_t* dims = fx.dims.data();
  const double* vals = fx.vals.data();

  for (int64_t k = 0; k < n; ++k) {
    const int32_t i = fx.pairs[2 * k], j = fx.pairs[2 * k + 1];
    const int32_t* ai = dims + fx.offsets[i];
    const int32_t* ai_end = dims + fx.offsets[i + 1];
    const int32_t* bj = dims + fx.offsets[j];
    const int32_t* bj_end = dims + fx.offsets[j + 1];
    const int32_t* a = std::lower_bound(ai, ai_end, dim_lo);
    const int32_t* a_end = std::lower_bound(a, ai_end, dim_hi);
    const int32_t* b = std::lower_bound(bj, bj_end, dim_lo);
    const int32_t* b_end = std::lower_bound(b, bj_end, dim_hi);

    double* s = stats.data() + 6 * k;  // n, sx, sy, sxx, syy, sxy
    while (a < a_end && b < b_end) {
      if (*a < *b) {
        ++a;
      } else if (*b < *a) {
        ++b;
      } else {
        const double x = vals[(a - dims)];
        const double y = vals[(b - dims)];
        s[0] += 1.0;
        s[1] += x;
        s[2] += y;
        s[3] += x * x;
        s[4] += y * y;
        s[5] += x * y;
        ++a;
        ++b;
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  // Process entry, before MPI_Init: MPI_Wtime is not callable until the
  // runtime is up, so the cold path has to start on a plain clock. What this
  // measures is `cold_data_path_s` -- entry to data-ready, including MPI_Init
  // -- NOT a CLI one-shot, which would also have to include mpirun's own
  // process launch and is only measurable from outside this binary.
  const auto proc_entry = std::chrono::steady_clock::now();
  MPI_Init(&argc, &argv);
  int rank = 0, world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  if (argc < 2) {
    if (rank == 0) std::fprintf(stderr, "usage: %s <fixture_dir> [--validate] [--out f]\n", argv[0]);
    MPI_Finalize();
    return 2;
  }
  std::string dir = argv[1];
  std::string out_path;
  bool validate = false;
  for (int a = 2; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--validate")) validate = true;
    else if (!std::strcmp(argv[a], "--out") && a + 1 < argc) out_path = argv[++a];
  }

  const double t0 = MPI_Wtime();
  engine::Fixture fx = engine::Fixture::load(dir);
  const double t_load = MPI_Wtime() - t0;

  // Slowest rank's entry->ready, reduced as ONE quantity. Taking
  // max(init) + max(load) separately would invent a critical path that no
  // single rank walked, since the slowest init and the slowest load need not
  // be the same rank.
  MPI_Barrier(MPI_COMM_WORLD);
  double ready = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - proc_entry).count();
  double cold_data_path = 0.0;
  MPI_Reduce(&ready, &cold_data_path, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  // t_load_s is rank 0's own load and is kept for continuity; the quantity
  // that bounds the collective is the SLOWEST rank's, since every rank waits
  // for it. Reported separately rather than folded into cold_data_path, which
  // is already a single reduced entry->ready measurement.
  double t_load_max = 0.0;
  MPI_Reduce(&t_load, &t_load_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // Dimension range owned by this rank. n_dims = max dim index + 1 is not
  // stored in the bins; derive it from the data (dims are dense 0..n_dims-1).
  int32_t n_dims = 0;
  for (int32_t d : fx.dims) n_dims = std::max(n_dims, d + 1);
  MPI_Allreduce(MPI_IN_PLACE, &n_dims, 1, MPI_INT32_T, MPI_MAX, MPI_COMM_WORLD);
  const int32_t lo = static_cast<int32_t>(static_cast<int64_t>(n_dims) * rank / world);
  const int32_t hi = static_cast<int32_t>(static_cast<int64_t>(n_dims) * (rank + 1) / world);

  std::vector<double> stats;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t1 = MPI_Wtime();
  partial_stats(fx, lo, hi, stats);
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_local = MPI_Wtime() - t1;

  const double t2 = MPI_Wtime();
  MPI_Allreduce(MPI_IN_PLACE, stats.data(), static_cast<int>(stats.size()),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double t_allreduce = MPI_Wtime() - t2;

  double t_finalize = 0.0;
  int failures = 0;
  double max_diff = 0.0;
  int64_t emitted = 0;
  if (rank == 0) {
    const double t3 = MPI_Wtime();
    std::vector<double> sims(fx.n_pairs());
    for (int64_t k = 0; k < fx.n_pairs(); ++k) {
      engine::PairStats s;
      const double* p = stats.data() + 6 * k;
      s.n = p[0]; s.sx = p[1]; s.sy = p[2]; s.sxx = p[3]; s.syy = p[4]; s.sxy = p[5];
      sims[k] = engine::pearson_finalize(s);
    }
    t_finalize = MPI_Wtime() - t3;

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

    const double bytes = 6.0 * fx.n_pairs() * sizeof(double);
    const double t_total = t_local + t_allreduce + t_finalize;
    std::printf(
        "{\"fixture\":\"%s\",\"backend\":\"mpi\",\"ranks\":%d,"
        "\"n_pairs\":%lld,\"timing_basis\":\"device_total\","
        "\"t_load_s\":%.6f,\"t_load_max_s\":%.6f,"
        "\"t_setup_s\":0.000000,\"t_local_s\":%.6f,"
        "\"t_stats_s\":%.6f,"
        "\"t_allreduce_s\":%.6f,\"t_finalize_s\":%.6f,\"t_d2h_s\":0.000000,"
        "\"device_total_s\":%.6f,\"cold_data_path_s\":%.6f,"
        "\"comm_fraction\":%.4f,\"allreduce_bytes\":%.0f,"
        "\"validated\":%s,\"max_abs_diff\":%.3e,\"tol_failures\":%d,"
        "\"emitted\":%lld}\n",
        dir.c_str(), world, static_cast<long long>(fx.n_pairs()), t_load,
        t_load_max, t_local, t_local, t_allreduce, t_finalize,
        t_total, cold_data_path + t_total,
        t_total > 0 ? t_allreduce / t_total : 0.0, bytes,
        validate ? "true" : "false", max_diff, failures,
        static_cast<long long>(emitted));
  }

  MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return (validate && failures > 0) ? 1 : 0;
}
