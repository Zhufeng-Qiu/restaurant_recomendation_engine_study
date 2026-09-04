// Multi-GPU NCCL backends (phases 5-6). One host process drives N GPUs
// (ncclCommInitAll). Decomposition mirrors the MPI backend: each GPU owns a
// contiguous dim range, computes a partial (n_pairs x 6) statistics tensor
// for ALL pairs, and the tensors are summed with ncclAllReduce. Pairs are
// never sharded across GPUs.
//
// The AllReduce payload representation is selectable (--payload, contract §9):
// the six statistics are exact small integers on integer-rated fixtures, so
// they compress losslessly from 48 B/pair to 24 (i32) or 16 (packed uint64).
// The packed form is summed by NCCL in compressed form -- fields never carry
// into one another -- so nothing is decompressed until finalize.
//
//   sync  mode: one kernel per GPU over all pairs, one AllReduce, finalize.
//   async mode: candidate pairs are processed in chunks with double
//               buffering; chunk c+1's stats kernel (compute stream) runs
//               while chunk c's AllReduce (comm stream) is in flight.
//               Event-based cross-stream dependencies keep it race-free.
//
// Usage:
//   pearson_engine_nccl <fixture_dir> [--gpus N] [--mode sync|async]
//                       [--chunk PAIRS] [--payload f64|i32|packed]
//                       [--validate] [--out sims.bin]

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
#include "common/payload_domain.hpp"
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
  void* stats = nullptr;     // elems_per_pair(payload) entries per pair
  double* sims = nullptr;    // n_pairs (device 0 finalizes)
  cudaStream_t compute = nullptr, comm = nullptr;
  cudaEvent_t chunk_ready[2] = {nullptr, nullptr};   // stats written (per buffer)
  cudaEvent_t chunk_reduced[2] = {nullptr, nullptr}; // AllReduce done (per buffer)
};

double wall() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// --- Collective payload representation (docs/pearson_contract.md §9) -------

enum class Payload { F64, I32, Packed };

Payload parse_payload(const std::string& s) {
  if (s == "f64") return Payload::F64;
  if (s == "i32") return Payload::I32;
  if (s == "packed") return Payload::Packed;
  throw std::runtime_error("--payload must be f64, i32, or packed");
}

const char* payload_name(Payload p) {
  switch (p) {
    case Payload::I32: return "i32";
    case Payload::Packed: return "packed";
    default: return "f64";
  }
}

int64_t elems_per_pair(Payload p) { return p == Payload::Packed ? 2 : 6; }

size_t elem_bytes(Payload p) {
  switch (p) {
    case Payload::I32: return sizeof(int32_t);
    case Payload::Packed: return sizeof(uint64_t);
    default: return sizeof(double);
  }
}

ncclDataType_t nccl_dtype(Payload p) {
  switch (p) {
    case Payload::I32: return ncclInt32;
    case Payload::Packed: return ncclUint64;
    default: return ncclDouble;
  }
}

int64_t payload_bytes_per_pair(Payload p) {
  return elems_per_pair(p) * static_cast<int64_t>(elem_bytes(p));
}

// Byte offset of the slot holding `pair_offset` pairs into the buffer.
void* stats_at(void* base, Payload p, int64_t pair_offset) {
  return static_cast<char*>(base) + payload_bytes_per_pair(p) * pair_offset;
}

// The compressed payloads are lossless only while the fixture stays inside
// the domain they assume: non-negative integer ratings, and a largest rating
// row small enough that no statistic can overflow its field. Checked up front
// and refused loudly -- a silently overflowed field would corrupt the sum
// while every backend still agreed with itself.
//
// The predicate itself lives in common/payload_domain.hpp so that it is
// testable without a GPU (engine/tests/payload_domain_test.cpp); this wrapper
// only turns its verdict into the exception main() reports.
engine::PayloadKind domain_kind(Payload p) {
  switch (p) {
    case Payload::I32: return engine::PayloadKind::I32;
    case Payload::Packed: return engine::PayloadKind::Packed;
    default: return engine::PayloadKind::F64;
  }
}

void check_payload_domain(const engine::Fixture& fx, Payload p) {
  const engine::DomainVerdict v =
      engine::check_payload_domain(fx, domain_kind(p));
  if (!v.ok) throw std::runtime_error(v.reason);
}

// Payload-dispatching launchers. The three variants share their accumulation
// path (engine_cuda::warp_pair_stats) and differ only in the width they emit.
void launch_stats(Payload p, unsigned blocks, cudaStream_t s, const Device& d,
                  const int32_t* pairs, int64_t len, void* dst) {
  using namespace engine_cuda;
  switch (p) {
    case Payload::I32:
      pair_stats_kernel_i32<<<blocks, kBlock, 0, s>>>(
          d.offsets, d.dims, d.vals, pairs, len, d.dim_lo, d.dim_hi,
          static_cast<int32_t*>(dst));
      break;
    case Payload::Packed:
      pair_stats_kernel_packed<<<blocks, kBlock, 0, s>>>(
          d.offsets, d.dims, d.vals, pairs, len, d.dim_lo, d.dim_hi,
          static_cast<uint64_t*>(dst));
      break;
    default:
      pair_stats_kernel<<<blocks, kBlock, 0, s>>>(
          d.offsets, d.dims, d.vals, pairs, len, d.dim_lo, d.dim_hi,
          static_cast<double*>(dst));
  }
}

void launch_finalize(Payload p, int64_t len, cudaStream_t s, const void* stats,
                     double* sims) {
  using namespace engine_cuda;
  const unsigned blocks = static_cast<unsigned>((len + 255) / 256);
  switch (p) {
    case Payload::I32:
      finalize_kernel_i32<<<blocks, 256, 0, s>>>(
          static_cast<const int32_t*>(stats), len, sims);
      break;
    case Payload::Packed:
      finalize_kernel_packed<<<blocks, 256, 0, s>>>(
          static_cast<const uint64_t*>(stats), len, sims);
      break;
    default:
      finalize_kernel<<<blocks, 256, 0, s>>>(
          static_cast<const double*>(stats), len, sims);
  }
}

// Body of main. Kept separate so main() can be a thin exception boundary:
// a refused payload is a diagnosable user error, not a crash, and letting the
// exception escape main() terminates on SIGABRT (exit 134, core dumped) with
// the message buried under "terminate called after throwing...".
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
    std::fprintf(stderr,
                 "usage: %s <fixture_dir> [--gpus N] [--mode sync|async] "
                 "[--chunk PAIRS] [--payload f64|i32|packed] [--validate] "
                 "[--out f]\n", argv[0]);
    return 2;
  }
  std::string dir = argv[1], out_path, mode = "sync", payload_arg = "f64";
  int n_gpus = 2;
  int64_t chunk = 1 << 18;  // 262144 pairs per chunk (async mode)
  bool validate = false;
  for (int a = 2; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--gpus") && a + 1 < argc) n_gpus = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--mode") && a + 1 < argc) mode = argv[++a];
    else if (!std::strcmp(argv[a], "--chunk") && a + 1 < argc) chunk = std::atoll(argv[++a]);
    else if (!std::strcmp(argv[a], "--payload") && a + 1 < argc) payload_arg = argv[++a];
    else if (!std::strcmp(argv[a], "--validate")) validate = true;
    else if (!std::strcmp(argv[a], "--out") && a + 1 < argc) out_path = argv[++a];
  }
  const Payload payload = parse_payload(payload_arg);

  const double t0 = wall();
  engine::Fixture fx = engine::Fixture::load(dir);
  const double t_load = wall() - t0;
  const int64_t n_pairs = fx.n_pairs();
  check_payload_domain(fx, payload);

  int32_t n_dims = 0;
  for (int32_t d : fx.dims) n_dims = std::max(n_dims, d + 1);

  std::vector<Device> devs(n_gpus);
  std::vector<ncclComm_t> comms(n_gpus);
  std::vector<int> ids(n_gpus);
  for (int g = 0; g < n_gpus; ++g) ids[g] = g;
  NCCL_CHECK(ncclCommInitAll(comms.data(), n_gpus, ids.data()));

  const int64_t epp = elems_per_pair(payload);
  const int64_t stats_pairs = (mode == "sync") ? n_pairs : 2 * chunk;
  const int64_t stats_len = epp * stats_pairs;
  const size_t stats_bytes = static_cast<size_t>(stats_len) * elem_bytes(payload);
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
    CUDA_CHECK(cudaMalloc(&d.stats, stats_bytes));
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
    NCCL_CHECK(ncclAllReduce(devs[g].stats, devs[g].stats, epp * 1024,
                             nccl_dtype(payload), ncclSum, comms[g],
                             devs[g].compute));
  }
  NCCL_CHECK(ncclGroupEnd());
  for (int g = 0; g < n_gpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamSynchronize(devs[g].compute));
    // The warm-up dirtied the stats buffer; stats kernels overwrite (not
    // accumulate) every slot they own, so no reset is needed in sync mode,
    // but async chunk tails only write the live prefix — zero to be safe.
    CUDA_CHECK(cudaMemset(devs[g].stats, 0, stats_bytes));
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
      launch_stats(payload, static_cast<unsigned>(blocks), d.compute, d,
                   d.pairs, n_pairs, d.stats);
    }
    for (int g = 0; g < n_gpus; ++g) {
      CUDA_CHECK(cudaSetDevice(g));
      CUDA_CHECK(cudaStreamSynchronize(devs[g].compute));
    }
    t_kernel = wall() - k0;

    // 2. Sum the six-stat tensors across GPUs. Under --payload packed this
    //    reduction runs on the compressed words directly: the fields never
    //    carry into one another, so the sum of packed words is the packing of
    //    the summed statistics. Nothing is decompressed until finalize.
    const double a0 = wall();
    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < n_gpus; ++g) {
      NCCL_CHECK(ncclAllReduce(devs[g].stats, devs[g].stats, epp * n_pairs,
                               nccl_dtype(payload), ncclSum, comms[g],
                               devs[g].compute));
    }
    NCCL_CHECK(ncclGroupEnd());
    for (int g = 0; g < n_gpus; ++g) {
      CUDA_CHECK(cudaSetDevice(g));
      CUDA_CHECK(cudaStreamSynchronize(devs[g].compute));
    }
    t_allreduce = wall() - a0;

    // 3. Finalize on GPU 0.
    CUDA_CHECK(cudaSetDevice(0));
    launch_finalize(payload, n_pairs, devs[0].compute, devs[0].stats,
                    devs[0].sims);
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
        void* slot = stats_at(d.stats, payload, chunk * b);
        launch_stats(payload, static_cast<unsigned>(blocks), d.compute, d,
                     d.pairs + 2 * base, len, slot);
        CUDA_CHECK(cudaEventRecord(d.chunk_ready[b], d.compute));
        // Communication stream reduces this chunk while the compute stream
        // moves on to the next one.
        CUDA_CHECK(cudaStreamWaitEvent(d.comm, d.chunk_ready[b], 0));
        NCCL_CHECK(ncclAllReduce(slot, slot, epp * len, nccl_dtype(payload),
                                 ncclSum, comms[g], d.comm));
      }
      NCCL_CHECK(ncclGroupEnd());
      // GPU 0 finalizes the reduced chunk on its comm stream (in order) and
      // only THEN records the slot-free event: recording before finalize let
      // chunk c+2's stats kernel overwrite slot b while finalize still read
      // it (caught by the async chunk-size sweep at chunk < 262144).
      CUDA_CHECK(cudaSetDevice(0));
      launch_finalize(payload, len, devs[0].comm,
                      stats_at(devs[0].stats, payload, chunk * b),
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
      "\"chunk\":%lld,\"payload\":\"%s\",\"payload_bytes_per_pair\":%lld,"
      "\"allreduce_bytes\":%lld,\"n_pairs\":%lld,"
      "\"t_load_s\":%.6f,\"t_setup_s\":%.6f,"
      "\"t_kernel_s\":%.6f,\"t_allreduce_s\":%.6f,\"t_pipeline_s\":%.6f,"
      "\"t_d2h_s\":%.6f,\"validated\":%s,\"max_abs_diff\":%.3e,"
      "\"tol_failures\":%d,\"emitted\":%lld}\n",
      dir.c_str(), mode.c_str(), n_gpus, static_cast<long long>(chunk),
      payload_name(payload),
      static_cast<long long>(payload_bytes_per_pair(payload)),
      static_cast<long long>(payload_bytes_per_pair(payload) * n_pairs),
      static_cast<long long>(n_pairs), t_load, t_setup, t_kernel, t_allreduce,
      t_pipeline, t_d2h, validate ? "true" : "false", max_diff, failures,
      static_cast<long long>(emitted));

  for (int g = 0; g < n_gpus; ++g) ncclCommDestroy(comms[g]);
  return (validate && failures > 0) ? 1 : 0;
}

}  // namespace
