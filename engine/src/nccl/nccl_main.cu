// Multi-GPU NCCL backends (phases 5-6). One host process drives N GPUs
// (ncclCommInitAll). Decomposition mirrors the MPI backend: each GPU owns a
// contiguous dim range, computes a partial (n_pairs x 6) statistics tensor
// for ALL pairs, and the tensors are summed with ncclAllReduce. Pairs are
// never sharded across GPUs.
//
//   sync  mode: one kernel per GPU over all pairs, one AllReduce, finalize.
//   async mode: candidate pairs are processed in chunks with double
//               buffering; chunk c+1's stats kernel (compute stream) runs
//               while chunk c's AllReduce (comm stream) is in flight.
//               Event-based cross-stream dependencies keep it race-free.
//
// Usage:
//   pearson_engine_nccl <fixture_dir> [--gpus N] [--mode sync|async]
//                       [--chunk PAIRS] [--validate] [--out sims.bin]

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/fixture.hpp"
#include "common/pearson.hpp"
#include "cuda/pair_kernel.cuh"

#define CUDA_CHECK(x)                                                   \
  do {                                                                  \
    cudaError_t e__ = (x);                                              \
    if (e__ != cudaSuccess)                                             \
      throw std::runtime_error(std::string("CUDA: ") +                  \
                               cudaGetErrorString(e__));                \
  } while (0)
#define NCCL_CHECK(x)                                                   \
  do {                                                                  \
    ncclResult_t r__ = (x);                                             \
    if (r__ != ncclSuccess)                                             \
      throw std::runtime_error(std::string("NCCL: ") +                  \
                               ncclGetErrorString(r__));                \
  } while (0)

namespace {

struct Device {
  int id = 0;
  int32_t dim_lo = 0, dim_hi = 0;
  int64_t* offsets = nullptr;
  int32_t* dims = nullptr;
  double* vals = nullptr;
  int32_t* pairs = nullptr;
  double* stats = nullptr;   // 6 * n_pairs (sync) or 6 * 2*chunk (async)
  double* sims = nullptr;    // n_pairs (device 0 finalizes)
  cudaStream_t compute = nullptr, comm = nullptr;
  cudaEvent_t chunk_ready[2] = {nullptr, nullptr};   // stats written (per buffer)
  cudaEvent_t chunk_reduced[2] = {nullptr, nullptr}; // AllReduce done (per buffer)
};

double wall() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <fixture_dir> [--gpus N] [--mode sync|async] "
                 "[--chunk PAIRS] [--validate] [--out f]\n", argv[0]);
    return 2;
  }
  std::string dir = argv[1], out_path, mode = "sync";
  int n_gpus = 2;
  int64_t chunk = 1 << 18;  // 262144 pairs per chunk (async mode)
  bool validate = false;
  for (int a = 2; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--gpus") && a + 1 < argc) n_gpus = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--mode") && a + 1 < argc) mode = argv[++a];
    else if (!std::strcmp(argv[a], "--chunk") && a + 1 < argc) chunk = std::atoll(argv[++a]);
    else if (!std::strcmp(argv[a], "--validate")) validate = true;
    else if (!std::strcmp(argv[a], "--out") && a + 1 < argc) out_path = argv[++a];
  }

  const double t0 = wall();
  engine::Fixture fx = engine::Fixture::load(dir);
  const double t_load = wall() - t0;
  const int64_t n_pairs = fx.n_pairs();

  int32_t n_dims = 0;
  for (int32_t d : fx.dims) n_dims = std::max(n_dims, d + 1);

  std::vector<Device> devs(n_gpus);
  std::vector<ncclComm_t> comms(n_gpus);
  std::vector<int> ids(n_gpus);
  for (int g = 0; g < n_gpus; ++g) ids[g] = g;
  NCCL_CHECK(ncclCommInitAll(comms.data(), n_gpus, ids.data()));

  const int64_t stats_len = (mode == "sync") ? 6 * n_pairs : 6 * 2 * chunk;
  const double t1 = wall();
  for (int g = 0; g < n_gpus; ++g) {
    Device& d = devs[g];
    d.id = g;
    d.dim_lo = static_cast<int32_t>(static_cast<int64_t>(n_dims) * g / n_gpus);
    d.dim_hi = static_cast<int32_t>(static_cast<int64_t>(n_dims) * (g + 1) / n_gpus);
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamCreate(&d.compute));
    CUDA_CHECK(cudaStreamCreate(&d.comm));
    for (int b = 0; b < 2; ++b) {
      CUDA_CHECK(cudaEventCreateWithFlags(&d.chunk_ready[b], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&d.chunk_reduced[b], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaMalloc(&d.offsets, fx.offsets.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d.dims, fx.dims.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.vals, fx.vals.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.pairs, fx.pairs.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.stats, stats_len * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.sims, n_pairs * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d.offsets, fx.offsets.data(),
                          fx.offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.dims, fx.dims.data(),
                          fx.dims.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vals, fx.vals.data(),
                          fx.vals.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.pairs, fx.pairs.data(),
                          fx.pairs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  }
  for (int g = 0; g < n_gpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  // Untimed warm-up collective: NCCL initializes its kernels/buffers lazily
  // on the first call, which would otherwise land inside the timed region.
  NCCL_CHECK(ncclGroupStart());
  for (int g = 0; g < n_gpus; ++g) {
    NCCL_CHECK(ncclAllReduce(devs[g].stats, devs[g].stats, 6 * 1024,
                             ncclDouble, ncclSum, comms[g], devs[g].compute));
  }
  NCCL_CHECK(ncclGroupEnd());
  for (int g = 0; g < n_gpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamSynchronize(devs[g].compute));
    // The warm-up dirtied the stats buffer; stats kernels overwrite (not
    // accumulate) every slot they own, so no reset is needed in sync mode,
    // but async chunk tails only write 6*len — zero to be safe.
    CUDA_CHECK(cudaMemset(devs[g].stats, 0, stats_len * sizeof(double)));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  const double t_setup = wall() - t1;

  using namespace engine_cuda;
  const int wpb = kBlock / kWarp;
  double t_kernel = 0, t_allreduce = 0, t_pipeline = 0;

  if (mode == "sync") {
    // 1. Partial statistics on every GPU over its dim range (all pairs).
    const double k0 = wall();
    for (int g = 0; g < n_gpus; ++g) {
      Device& d = devs[g];
      CUDA_CHECK(cudaSetDevice(g));
      const int64_t blocks = (n_pairs + wpb - 1) / wpb;
      pair_stats_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, d.compute>>>(
          d.offsets, d.dims, d.vals, d.pairs, n_pairs, d.dim_lo, d.dim_hi, d.stats);
    }
    for (int g = 0; g < n_gpus; ++g) {
      CUDA_CHECK(cudaSetDevice(g));
      CUDA_CHECK(cudaStreamSynchronize(devs[g].compute));
    }
    t_kernel = wall() - k0;

    // 2. Sum the six-stat tensors across GPUs.
    const double a0 = wall();
    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < n_gpus; ++g) {
      NCCL_CHECK(ncclAllReduce(devs[g].stats, devs[g].stats, 6 * n_pairs,
                               ncclDouble, ncclSum, comms[g], devs[g].compute));
    }
    NCCL_CHECK(ncclGroupEnd());
    for (int g = 0; g < n_gpus; ++g) {
      CUDA_CHECK(cudaSetDevice(g));
      CUDA_CHECK(cudaStreamSynchronize(devs[g].compute));
    }
    t_allreduce = wall() - a0;

    // 3. Finalize on GPU 0.
    CUDA_CHECK(cudaSetDevice(0));
    finalize_kernel<<<static_cast<unsigned>((n_pairs + 255) / 256), 256, 0,
                      devs[0].compute>>>(devs[0].stats, n_pairs, devs[0].sims);
    CUDA_CHECK(cudaStreamSynchronize(devs[0].compute));
  } else {
    // Async: chunked, double-buffered pipeline. Buffer b of GPU g holds the
    // stats of the chunk currently using slot b.
    const double p0 = wall();
    const int64_t n_chunks = (n_pairs + chunk - 1) / chunk;
    for (int64_t c = 0; c < n_chunks; ++c) {
      const int b = static_cast<int>(c & 1);
      const int64_t base = c * chunk;
      const int64_t len = std::min(chunk, n_pairs - base);
      NCCL_CHECK(ncclGroupStart());
      for (int g = 0; g < n_gpus; ++g) {
        Device& d = devs[g];
        CUDA_CHECK(cudaSetDevice(g));
        // Reuse of slot b must wait until its previous AllReduce finished.
        if (c >= 2) CUDA_CHECK(cudaStreamWaitEvent(d.compute, d.chunk_reduced[b], 0));
        const int64_t blocks = (len + wpb - 1) / wpb;
        pair_stats_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, d.compute>>>(
            d.offsets, d.dims, d.vals, d.pairs + 2 * base, len, d.dim_lo,
            d.dim_hi, d.stats + 6 * chunk * b);
        CUDA_CHECK(cudaEventRecord(d.chunk_ready[b], d.compute));
        // Communication stream reduces this chunk while the compute stream
        // moves on to the next one.
        CUDA_CHECK(cudaStreamWaitEvent(d.comm, d.chunk_ready[b], 0));
        NCCL_CHECK(ncclAllReduce(d.stats + 6 * chunk * b, d.stats + 6 * chunk * b,
                                 6 * len, ncclDouble, ncclSum, comms[g], d.comm));
      }
      NCCL_CHECK(ncclGroupEnd());
      // GPU 0 finalizes the reduced chunk on its comm stream (in order) and
      // only THEN records the slot-free event: recording before finalize let
      // chunk c+2's stats kernel overwrite slot b while finalize still read
      // it (caught by the Gate D chunk sweep at chunk < 262144).
      CUDA_CHECK(cudaSetDevice(0));
      finalize_kernel<<<static_cast<unsigned>((len + 255) / 256), 256, 0,
                        devs[0].comm>>>(devs[0].stats + 6 * chunk * b, len,
                                        devs[0].sims + base);
      for (int g = 0; g < n_gpus; ++g) {
        Device& d = devs[g];
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventRecord(d.chunk_reduced[b], d.comm));
      }
    }
    for (int g = 0; g < n_gpus; ++g) {
      CUDA_CHECK(cudaSetDevice(g));
      CUDA_CHECK(cudaStreamSynchronize(devs[g].compute));
      CUDA_CHECK(cudaStreamSynchronize(devs[g].comm));
    }
    t_pipeline = wall() - p0;
  }

  const double d0 = wall();
  std::vector<double> sims(n_pairs);
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMemcpy(sims.data(), devs[0].sims, n_pairs * sizeof(double),
                        cudaMemcpyDeviceToHost));
  const double t_d2h = wall() - d0;

  int failures = 0;
  double max_diff = 0.0;
  int64_t emitted = 0;
  if (validate) {
    for (int64_t k = 0; k < n_pairs; ++k) {
      const double df = std::fabs(sims[k] - fx.golden[k]);
      if (df > max_diff) max_diff = df;
      if (df > engine::kTol) ++failures;
      if (sims[k] > engine::kEps) ++emitted;
    }
  }
  if (!out_path.empty()) {
    std::ofstream f(out_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(sims.data()),
            static_cast<std::streamsize>(sims.size() * sizeof(double)));
  }

  std::printf(
      "{\"fixture\":\"%s\",\"backend\":\"nccl\",\"mode\":\"%s\",\"gpus\":%d,"
      "\"chunk\":%lld,\"n_pairs\":%lld,\"t_load_s\":%.6f,\"t_setup_s\":%.6f,"
      "\"t_kernel_s\":%.6f,\"t_allreduce_s\":%.6f,\"t_pipeline_s\":%.6f,"
      "\"t_d2h_s\":%.6f,\"validated\":%s,\"max_abs_diff\":%.3e,"
      "\"tol_failures\":%d,\"emitted\":%lld}\n",
      dir.c_str(), mode.c_str(), n_gpus, static_cast<long long>(chunk),
      static_cast<long long>(n_pairs), t_load, t_setup, t_kernel, t_allreduce,
      t_pipeline, t_d2h, validate ? "true" : "false", max_diff, failures,
      static_cast<long long>(emitted));

  for (int g = 0; g < n_gpus; ++g) ncclCommDestroy(comms[g]);
  return (validate && failures > 0) ? 1 : 0;
}
