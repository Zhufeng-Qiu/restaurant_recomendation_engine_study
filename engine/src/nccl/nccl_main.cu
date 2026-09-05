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
// --finalize-stream is an A/B on where GPU0's finalize kernel runs in async
// mode. `comm` (default, the original behaviour) puts it on the communication
// stream; `separate` gives it its own stream chained to the collective by an
// event. The PCIe traces showed GPU0 achieving almost no overlap while GPU1
// hid 55% of its compute, and GPU0 is the one also running finalize on `comm`
// -- this flag is what tests whether that is the cause.
//
// Usage:
//   pearson_engine_nccl <fixture_dir> [--gpus N] [--mode sync|async]
//                       [--chunk PAIRS] [--payload f64|i32|packed]
//                       [--finalize-stream comm|separate]
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
#include "common/pair_order.hpp"
#include "cuda/occupancy.cuh"
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
  int32_t* order = nullptr;  // slot -> pair; identical on every device
  void* stats = nullptr;     // elems_per_pair(payload) entries per pair
  double* sims = nullptr;    // n_pairs (device 0 finalizes)
  cudaStream_t compute = nullptr, comm = nullptr;
  // Device 0 only, and only under --finalize-stream separate: finalize runs
  // here instead of on `comm`. See the note at the async loop.
  cudaStream_t fin = nullptr;
  cudaEvent_t chunk_ready[2] = {nullptr, nullptr};   // stats written (per buffer)
  cudaEvent_t chunk_reduced[2] = {nullptr, nullptr}; // AllReduce done (per buffer)
  cudaEvent_t chunk_allreduced[2] = {nullptr, nullptr}; // collective done (per buffer)
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
void launch_stats(Payload p, int group, unsigned blocks, cudaStream_t s,
                  const Device& d, const int32_t* order, int64_t len,
                  void* dst) {
  using namespace engine_cuda;
  switch (p) {
    case Payload::I32:
      ENGINE_DISPATCH_GROUP(group, pair_stats_kernel_i32, blocks, kBlock, s,
                            d.offsets, d.dims, d.vals, d.pairs, order, len,
                            d.dim_lo, d.dim_hi, static_cast<int32_t*>(dst));
      break;
    case Payload::Packed:
      ENGINE_DISPATCH_GROUP(group, pair_stats_kernel_packed, blocks, kBlock, s,
                            d.offsets, d.dims, d.vals, d.pairs, order, len,
                            d.dim_lo, d.dim_hi, static_cast<uint64_t*>(dst));
      break;
    default:
      ENGINE_DISPATCH_GROUP(group, pair_stats_kernel, blocks, kBlock, s,
                            d.offsets, d.dims, d.vals, d.pairs, order, len,
                            d.dim_lo, d.dim_hi, static_cast<double*>(dst));
  }
}

void launch_finalize(Payload p, int64_t len, cudaStream_t s, const void* stats,
                     const int32_t* order, double* sims) {
  using namespace engine_cuda;
  const unsigned blocks = static_cast<unsigned>((len + 255) / 256);
  switch (p) {
    case Payload::I32:
      finalize_kernel_i32<<<blocks, 256, 0, s>>>(
          static_cast<const int32_t*>(stats), order, len, sims);
      break;
    case Payload::Packed:
      finalize_kernel_packed<<<blocks, 256, 0, s>>>(
          static_cast<const uint64_t*>(stats), order, len, sims);
      break;
    default:
      finalize_kernel<<<blocks, 256, 0, s>>>(
          static_cast<const double*>(stats), order, len, sims);
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
                 "[--chunk PAIRS] [--payload f64|i32|packed] "
                 "[--finalize-stream comm|separate] [--validate] "
                 "[--group 1|2|4|8|16|32] [--pair-order source|bylen] "
                 "[--out f]\n", argv[0]);
    return 2;
  }
  std::string dir = argv[1], out_path, mode = "sync", payload_arg = "f64";
  std::string finalize_stream = "comm";  // async only; "comm" | "separate"
  int n_gpus = 2;
  int64_t chunk = 1 << 18;  // 262144 pairs per chunk (async mode)
  int group = 32;                     // lanes per pair; 32 == phase-3 mapping
  std::string order_arg = "source";   // "bylen" enables warp packing
  bool validate = false;
  // Argument parsing refuses what it cannot honour. Silently ignoring an
  // unknown flag, or a flag whose value went missing at the end of the line,
  // would run a configuration nobody asked for and label the results with the
  // one they did -- the kind of thing that only surfaces after a benchmark
  // campaign is already written up.
  auto need_value = [&](int a, const char* flag) {
    if (a + 1 >= argc)
      throw std::runtime_error(std::string(flag) + " needs a value");
    return argv[a + 1];
  };
  for (int a = 2; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--gpus")) n_gpus = std::atoi(need_value(a++, "--gpus"));
    else if (!std::strcmp(argv[a], "--mode")) mode = need_value(a++, "--mode");
    else if (!std::strcmp(argv[a], "--chunk")) chunk = std::atoll(need_value(a++, "--chunk"));
    else if (!std::strcmp(argv[a], "--payload")) payload_arg = need_value(a++, "--payload");
    else if (!std::strcmp(argv[a], "--finalize-stream"))
      finalize_stream = need_value(a++, "--finalize-stream");
    else if (!std::strcmp(argv[a], "--out")) out_path = need_value(a++, "--out");
    else if (!std::strcmp(argv[a], "--group")) group = std::atoi(need_value(a++, "--group"));
    else if (!std::strcmp(argv[a], "--pair-order")) order_arg = need_value(a++, "--pair-order");
    else if (!std::strcmp(argv[a], "--validate")) validate = true;
    else throw std::runtime_error(std::string("unknown argument ") + argv[a]);
  }
  if (mode != "sync" && mode != "async")
    throw std::runtime_error("--mode must be sync or async");
  if (chunk <= 0) throw std::runtime_error("--chunk must be positive");
  if (n_gpus <= 0) throw std::runtime_error("--gpus must be positive");
  {
    int visible = 0;
    CUDA_CHECK(cudaGetDeviceCount(&visible));
    if (n_gpus > visible)
      throw std::runtime_error("--gpus " + std::to_string(n_gpus) +
                               " exceeds the " + std::to_string(visible) +
                               " visible device(s)");
  }
  if (!engine::valid_group(group))
    throw std::runtime_error("--group must be 1, 2, 4, 8, 16 or 32");
  if (order_arg != "source" && order_arg != "bylen")
    throw std::runtime_error("--pair-order must be source or bylen");
  const Payload payload = parse_payload(payload_arg);
  if (finalize_stream != "comm" && finalize_stream != "separate")
    throw std::runtime_error("--finalize-stream must be comm or separate");
  if (mode == "sync" && finalize_stream != "comm")
    throw std::runtime_error(
        "--finalize-stream applies to --mode async only; a sync run has a "
        "single finalize outside the pipeline. Refusing rather than recording "
        "a sync result labelled with a setting that did nothing.");
  const bool finalize_separate = (finalize_stream == "separate");

  const double t0 = wall();
  engine::Fixture fx = engine::Fixture::load(dir);
  const double t_load = wall() - t0;
  const int64_t n_pairs = fx.n_pairs();
  check_payload_domain(fx, payload);

  int32_t n_dims = 0;
  for (int32_t d : fx.dims) n_dims = std::max(n_dims, d + 1);

  // One plan for every device. It is built over the FULL dimension range, not
  // each GPU's slice: slot s must mean the same pair on every GPU or the
  // AllReduce would sum statistics belonging to different pairs. Sorting by
  // the global shorter-slice length is therefore nobody's own optimum -- each
  // device executes this shared order against its own [dim_lo, dim_hi), where
  // the rows are shorter and the tails fall differently.
  const engine::PairPlan plan =
      engine::plan_pairs(fx, 0, n_dims,
                         order_arg == "bylen" ? engine::PairOrder::kByShortLen
                                              : engine::PairOrder::kSource,
                         group);

  std::vector<Device> devs(n_gpus);
  std::vector<ncclComm_t> comms(n_gpus);
  std::vector<int> ids(n_gpus);
  for (int g = 0; g < n_gpus; ++g) ids[g] = g;
  // Timed: ncclCommInitAll is a real cold-start cost (bootstrap, topology
  // detection, buffer allocation) and sat outside the timed region until
  // 2026-09-05, so the reported cold path understated itself. Process launch
  // is still excluded -- hence cold_data_path_s rather than a CLI one-shot.
  const double c0 = wall();
  NCCL_CHECK(ncclCommInitAll(comms.data(), n_gpus, ids.data()));
  const double t_comm_init = wall() - c0;

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
    if (g == 0) CUDA_CHECK(cudaStreamCreate(&d.fin));
    for (int b = 0; b < 2; ++b) {
      CUDA_CHECK(cudaEventCreateWithFlags(&d.chunk_ready[b], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&d.chunk_reduced[b], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&d.chunk_allreduced[b], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaMalloc(&d.offsets, fx.offsets.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d.dims, fx.dims.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.vals, fx.vals.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.pairs, fx.pairs.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.order, plan.order.size() * sizeof(int32_t)));
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
    CUDA_CHECK(cudaMemcpy(d.order, plan.order.data(),
                          plan.order.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  }
  // What the shared order actually costs on each device's own slice.
  //
  // The plan's own metrics describe the full dimension range, which no device
  // executes; reporting those as the GPUs' utilisation would overstate it,
  // because splitting the dimensions shortens every row and leaves shorter
  // tails. So the fixed order is re-evaluated against each [dim_lo, dim_hi).
  // The counterfactual is what that slice would cost if it could sort its own
  // pairs -- unreachable while the collective is elementwise, and recorded
  // only as an upper bound on what a better mapping could reach.
  //
  // This is diagnostics, not execution, so it is timed separately and kept out
  // of cold_data_path_s. Building the order is required to run; measuring it
  // is not.
  //
  // The full-dimension figure is diagnostic too, and used to hide inside
  // t_plan_s: plan_pairs computed it unconditionally, so even --pair-order
  // source paid a ~50 ms length pass it never needed. Both live here now.
  const double pm0 = wall();
  const engine::PlanMetrics on_basis = engine::measure_plan(fx, plan, 0, n_dims);
  std::vector<engine::PlanMetrics> per_dev(n_gpus), per_dev_ideal(n_gpus);
  engine::PlanMetrics agg;
  int64_t critical_slots = 0;
  for (int g = 0; g < n_gpus; ++g) {
    const std::vector<int32_t> len =
        engine::short_lens(fx, devs[g].dim_lo, devs[g].dim_hi);
    per_dev[g] = engine::evaluate_order(plan.order, len, group);
    per_dev_ideal[g] = engine::counterfactual_per_device_ideal(
        fx, group, devs[g].dim_lo, devs[g].dim_hi);
    agg.effective_elements += per_dev[g].effective_elements;
    agg.lane_slots += per_dev[g].lane_slots;
    critical_slots = std::max(critical_slots, per_dev[g].lane_slots);
  }
  const double t_plan_metrics = wall() - pm0;

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
  // Timing contract (see README, unified 2026-09-05):
  //   device_total = stats + allreduce + finalize   -- comparable across
  //                                                    backends, headline uses it
  //   cold_data_path = load + plan + comm_init + setup + device_total + d2h
  //                                                 -- entry to results in
  //                                                    host memory
  // Async cannot sum its stages (that is the point of overlapping them), so it
  // reports pipeline_total as its comparable number and the three stage SUMS
  // separately, for mechanism only.
  double t_kernel = 0, t_allreduce = 0, t_finalize = 0, t_pipeline = 0;
  double t_stats_sum = 0, t_allreduce_sum = 0, t_finalize_sum = 0;

  if (mode == "sync") {
    // 1. Partial statistics on every GPU over its dim range (all pairs).
    const double k0 = wall();
    for (int g = 0; g < n_gpus; ++g) {
      Device& d = devs[g];
      CUDA_CHECK(cudaSetDevice(g));
      const int64_t blocks = (n_pairs * group + kBlock - 1) / kBlock;
      launch_stats(payload, group, static_cast<unsigned>(blocks), d.compute, d,
                   d.order, n_pairs, d.stats);
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

    // 3. Finalize on GPU 0. Timed: cuda and mpi both count their finalize in
    //    their totals, and until 2026-09-05 this one ran outside the timed
    //    region, which made every NCCL sync number a stats+AllReduce subtotal
    //    and quietly favoured it in cross-backend comparisons by 0.5-1%.
    const double f0 = wall();
    CUDA_CHECK(cudaSetDevice(0));
    launch_finalize(payload, n_pairs, devs[0].compute, devs[0].stats,
                    devs[0].order, devs[0].sims);
    CUDA_CHECK(cudaStreamSynchronize(devs[0].compute));
    t_finalize = wall() - f0;
  } else {
    // ORDER AND ASYNC DO NOT MIX CLEANLY YET. Chunks are contiguous runs of
    // slots, so an ascending-by-length order hands the early chunks the short
    // pairs and the late chunks the long ones. That changes chunk cost,
    // pipeline balance and the tail -- effects that have nothing to do with
    // lane packing but land in the same number. So async is deliberately NOT
    // part of the first-stage decision on whether packing is worth defaulting
    // to; it is measured here for correctness only. A chunk-balanced order is
    // a separate piece of work: pack first, cost each pack on every device's
    // slice, then distribute packs across chunks so the critical lane slots
    // per chunk are even, keeping one shared order and near-length neighbours
    // inside each chunk.
    // Async: chunked, double-buffered pipeline. Buffer b of GPU g holds the
    // stats of the chunk currently using slot b.
    const int64_t n_chunks = (n_pairs + chunk - 1) / chunk;

    // Per-chunk stage timing on GPU0 (the binding side). Events are stream
    // markers, so recording them does not reorder anything; all elapsed times
    // are read after the final synchronize so the pipeline is never stalled to
    // measure it. These sums are for MECHANISM ONLY -- they overlap, so they
    // do not add up to pipeline_total, which is the comparable number.
    std::vector<cudaEvent_t> es0(n_chunks), es1(n_chunks), ea0(n_chunks),
        ea1(n_chunks), ef0(n_chunks), ef1(n_chunks);
    CUDA_CHECK(cudaSetDevice(0));
    for (int64_t c = 0; c < n_chunks; ++c) {
      CUDA_CHECK(cudaEventCreate(&es0[c])); CUDA_CHECK(cudaEventCreate(&es1[c]));
      CUDA_CHECK(cudaEventCreate(&ea0[c])); CUDA_CHECK(cudaEventCreate(&ea1[c]));
      CUDA_CHECK(cudaEventCreate(&ef0[c])); CUDA_CHECK(cudaEventCreate(&ef1[c]));
    }

    const double p0 = wall();
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
        const int64_t blocks = (len * group + kBlock - 1) / kBlock;
        void* slot = stats_at(d.stats, payload, chunk * b);
        if (g == 0) CUDA_CHECK(cudaEventRecord(es0[c], d.compute));
        launch_stats(payload, group, static_cast<unsigned>(blocks), d.compute,
                     d, d.order + base, len, slot);
        if (g == 0) CUDA_CHECK(cudaEventRecord(es1[c], d.compute));
        CUDA_CHECK(cudaEventRecord(d.chunk_ready[b], d.compute));
        // Communication stream reduces this chunk while the compute stream
        // moves on to the next one.
        CUDA_CHECK(cudaStreamWaitEvent(d.comm, d.chunk_ready[b], 0));
        if (g == 0) CUDA_CHECK(cudaEventRecord(ea0[c], d.comm));
        NCCL_CHECK(ncclAllReduce(slot, slot, epp * len, nccl_dtype(payload),
                                 ncclSum, comms[g], d.comm));
      }
      NCCL_CHECK(ncclGroupEnd());
      // The closing event must come AFTER ncclGroupEnd: grouped collectives are
      // only enqueued to the stream when the group closes, so an event recorded
      // between the ncclAllReduce call and the group end brackets nothing and
      // reports ~0.02 ms of pure event overhead instead of the collective.
      CUDA_CHECK(cudaSetDevice(0));
      CUDA_CHECK(cudaEventRecord(ea1[c], devs[0].comm));
      // GPU 0 finalizes the reduced chunk, and only THEN is slot b declared
      // free: recording the slot-free event before finalize let chunk c+2's
      // stats kernel overwrite slot b while finalize still read it (caught by
      // the async chunk-size sweep at chunk < 262144).
      //
      // Which stream finalize runs on is an A/B (--finalize-stream). Putting
      // it on `comm` keeps the ordering implicit but leaves GPU0's
      // communication stream doing compute, which the PCIe traces suggested
      // was why GPU0 achieved almost no overlap there. `separate` gives it its
      // own stream, chained to the collective by an explicit event so the
      // ordering is unchanged; the slot-free event then comes off that stream.
      CUDA_CHECK(cudaSetDevice(0));
      {
        cudaStream_t fs = finalize_separate ? devs[0].fin : devs[0].comm;
        if (finalize_separate) {
          CUDA_CHECK(cudaEventRecord(devs[0].chunk_allreduced[b], devs[0].comm));
          CUDA_CHECK(cudaStreamWaitEvent(fs, devs[0].chunk_allreduced[b], 0));
        }
        CUDA_CHECK(cudaEventRecord(ef0[c], fs));
        // sims is not offset here: finalize scatters through order, which
        // holds global pair indices, so the chunk writes straight to its
        // pairs' slots in the full output.
        launch_finalize(payload, len, fs,
                        stats_at(devs[0].stats, payload, chunk * b),
                        devs[0].order + base, devs[0].sims);
        CUDA_CHECK(cudaEventRecord(ef1[c], fs));
      }
      for (int g = 0; g < n_gpus; ++g) {
        Device& d = devs[g];
        CUDA_CHECK(cudaSetDevice(g));
        // Slot b is reusable once everything that reads it has finished. On
        // GPU0 under `separate` that is the finalize stream, not `comm`.
        CUDA_CHECK(cudaEventRecord(d.chunk_reduced[b],
                                   (g == 0 && finalize_separate) ? d.fin : d.comm));
      }
    }
    for (int g = 0; g < n_gpus; ++g) {
      CUDA_CHECK(cudaSetDevice(g));
      CUDA_CHECK(cudaStreamSynchronize(devs[g].compute));
      CUDA_CHECK(cudaStreamSynchronize(devs[g].comm));
      if (g == 0 && finalize_separate)
        CUDA_CHECK(cudaStreamSynchronize(devs[0].fin));
    }
    t_pipeline = wall() - p0;

    CUDA_CHECK(cudaSetDevice(0));
    for (int64_t c = 0; c < n_chunks; ++c) {
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, es0[c], es1[c])); t_stats_sum += ms / 1e3;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ea0[c], ea1[c])); t_allreduce_sum += ms / 1e3;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ef0[c], ef1[c])); t_finalize_sum += ms / 1e3;
      cudaEventDestroy(es0[c]); cudaEventDestroy(es1[c]);
      cudaEventDestroy(ea0[c]); cudaEventDestroy(ea1[c]);
      cudaEventDestroy(ef0[c]); cudaEventDestroy(ef1[c]);
    }
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

  // Unified timing schema. device_total is the cross-backend comparable number
  // for sync (stats + allreduce + finalize); async cannot sum overlapping
  // stages, so it reports pipeline_total and keeps the stage SUMS for
  // mechanism only. cold_data_path adds everything from entry to results in
  // host memory, the plan included.
  const bool is_async = (mode != "sync");
  const double device_total = is_async ? t_pipeline
                                       : (t_kernel + t_allreduce + t_finalize);
  // Plan construction is in here and NOT in device_total: a resident engine
  // builds the order once and pays device_total per query, while a caller that
  // runs this binary once pays for the order too. Still not a CLI one-shot --
  // process launch and the CUDA driver's first touch precede any code that
  // could time them.
  const double cold_data_path = t_load + plan.build_seconds + t_comm_init +
                                t_setup + device_total + t_d2h;
  // Held in a named string: taking .c_str() off a temporary inside the printf
  // argument list would dangle before printf reads it.
  const std::string fs_field =
      is_async ? ("\"" + finalize_stream + "\"") : std::string("\"not_applicable\"");

  // Per-device lane accounting for the shared order. Named so it cannot be
  // confused with the full-dimension plan below it: these are the slices the
  // GPUs ran, that one is the range the order was sorted on and nobody
  // executed.
  engine_cuda::OccupancyInfo occ{};
  switch (payload) {
    case Payload::I32:
      ENGINE_OCCUPANCY_FOR_GROUP(group, engine_cuda::pair_stats_kernel_i32,
                                 engine_cuda::kBlock, occ);
      break;
    case Payload::Packed:
      ENGINE_OCCUPANCY_FOR_GROUP(group, engine_cuda::pair_stats_kernel_packed,
                                 engine_cuda::kBlock, occ);
      break;
    default:
      ENGINE_OCCUPANCY_FOR_GROUP(group, engine_cuda::pair_stats_kernel,
                                 engine_cuda::kBlock, occ);
  }
  char occ_json[512];
  engine_cuda::occupancy_json(occ_json, sizeof occ_json, occ);

  std::string per_device = "[";
  for (int g = 0; g < n_gpus; ++g) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "%s{\"device\":%d,\"dim_lo\":%d,\"dim_hi\":%d,"
                  "\"effective_elements\":%lld,\"lane_slots\":%lld,"
                  "\"lane_utilisation\":%.5f,"
                  "\"counterfactual_per_device_sorted_lane_slots\":%lld,"
                  "\"counterfactual_per_device_sorted_utilisation\":%.5f}",
                  g ? "," : "", g, devs[g].dim_lo, devs[g].dim_hi,
                  static_cast<long long>(per_dev[g].effective_elements),
                  static_cast<long long>(per_dev[g].lane_slots),
                  per_dev[g].utilisation(),
                  static_cast<long long>(per_dev_ideal[g].lane_slots),
                  per_dev_ideal[g].utilisation());
    per_device += buf;
  }
  per_device += "]";
  std::printf(
      "{\"fixture\":\"%s\",\"backend\":\"nccl\",\"mode\":\"%s\",\"gpus\":%d,"
      "\"chunk\":%lld,\"finalize_stream\":%s,"
      "\"payload\":\"%s\",\"payload_bytes_per_pair\":%lld,"
      "\"allreduce_bytes\":%lld,\"n_pairs\":%lld,"
      "\"timing_basis\":\"%s\","
      "\"t_load_s\":%.6f,\"t_setup_s\":%.6f,"
      "\"t_stats_s\":%.6f,\"t_allreduce_s\":%.6f,\"t_finalize_s\":%.6f,"
      "\"t_kernel_s\":%.6f,\"t_pipeline_s\":%.6f,"
      "\"t_stats_sum_s\":%.6f,\"t_allreduce_sum_s\":%.6f,"
      "\"t_finalize_sum_s\":%.6f,"
      "\"device_total_s\":%.6f,\"t_comm_init_s\":%.6f,"
      "\"cold_data_path_s\":%.6f,"
      "\"group\":%d,\"pair_order\":\"%s\","
      "\"t_plan_s\":%.6f,\"t_plan_metrics_s\":%.6f,"
      "\"plan_order_basis\":\"global_full_dims\","
      "\"plan_per_device\":%s,"
      "\"plan_critical_lane_slots\":%lld,"
      "\"plan_aggregate_effective_elements\":%lld,"
      "\"plan_aggregate_lane_slots\":%lld,"
      "\"plan_aggregate_lane_utilisation\":%.5f,"
      "\"global_full_dimension_effective_elements\":%lld,"
      "\"global_full_dimension_lane_slots\":%lld,"
      "\"global_full_dimension_lane_utilisation\":%.5f,"
      "\"occupancy\":%s,"
      "\"t_d2h_s\":%.6f,\"validated\":%s,\"max_abs_diff\":%.3e,"
      "\"tol_failures\":%d,\"emitted\":%lld}\n",
      dir.c_str(), mode.c_str(), n_gpus, static_cast<long long>(chunk),
      fs_field.c_str(),
      payload_name(payload),
      static_cast<long long>(payload_bytes_per_pair(payload)),
      static_cast<long long>(payload_bytes_per_pair(payload) * n_pairs),
      static_cast<long long>(n_pairs),
      is_async ? "pipeline_total" : "device_total",
      t_load, t_setup,
      is_async ? t_stats_sum : t_kernel, is_async ? t_allreduce_sum : t_allreduce,
      is_async ? t_finalize_sum : t_finalize,
      t_kernel, t_pipeline,
      t_stats_sum, t_allreduce_sum, t_finalize_sum,
      device_total, t_comm_init, cold_data_path,
      group, order_arg.c_str(), plan.build_seconds, t_plan_metrics,
      per_device.c_str(),
      static_cast<long long>(critical_slots),
      static_cast<long long>(agg.effective_elements),
      static_cast<long long>(agg.lane_slots), agg.utilisation(),
      static_cast<long long>(on_basis.effective_elements),
      static_cast<long long>(on_basis.lane_slots), on_basis.utilisation(),
      occ_json,
      t_d2h, validate ? "true" : "false", max_diff, failures,
      static_cast<long long>(emitted));

  for (int g = 0; g < n_gpus; ++g) ncclCommDestroy(comms[g]);
  return (validate && failures > 0) ? 1 : 0;
}

}  // namespace
