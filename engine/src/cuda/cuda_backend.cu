// Single-GPU CUDA backend (phase 3). Full dim range on one device.
// Prints a "cuda_detail" JSON line with device-side timings (CUDA events for
// kernels, wall clock for transfers) before main prints its summary line.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/fixture.hpp"
#include "common/pair_order.hpp"
#include "cuda/cuda_backend.hpp"
#include "cuda/gpu_state.cuh"
#include "cuda/occupancy.cuh"
#include "cuda/pair_kernel.cuh"

#define CUDA_CHECK(x)                                                    \
  do {                                                                   \
    cudaError_t err__ = (x);                                             \
    if (err__ != cudaSuccess)                                            \
      throw std::runtime_error(std::string("CUDA: ") +                   \
                               cudaGetErrorString(err__));               \
  } while (0)

namespace engine {

CudaMapOptions& cuda_map_options() {
  static CudaMapOptions opts;
  return opts;
}

CudaTimings& cuda_last_timings() {
  static CudaTimings t;
  return t;
}

void compute_cuda(const Fixture& fx, std::vector<double>& out) {
  using namespace engine_cuda;
  const int64_t n_pairs = fx.n_pairs();
  out.resize(n_pairs);

  int32_t n_dims = 0;
  for (int32_t d : fx.dims) n_dims = std::max(n_dims, d + 1);

  // The lane mapping. Building the plan is host work over the whole pair list,
  // so it is setup, not steady state: it stays out of device_total, which a
  // resident engine pays per query after building the plan once. It IS inside
  // cold_data_path_s, which a caller running the binary once pays in full.
  const CudaMapOptions map = cuda_map_options();
  const PairPlan plan = plan_pairs(fx, 0, n_dims, map.order, map.group);
  // Diagnostics, timed apart and excluded from every total.
  const auto pm0 = std::chrono::steady_clock::now();
  const PlanMetrics plan_metrics =
      map.plan_metrics ? measure_plan(fx, plan, 0, n_dims) : PlanMetrics{};
  const double t_plan_metrics =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - pm0).count();
  if (map.pre_timing_delay_ms > 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(map.pre_timing_delay_ms));
  char gpu_state[256];
  gpu_state_json(gpu_state, sizeof gpu_state, read_gpu_state(0));

  int64_t *d_offsets;
  int32_t *d_dims, *d_pairs, *d_order;
  double *d_vals, *d_stats, *d_sims;

  cudaEvent_t ev0, ev1, ev2;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventCreate(&ev2));

  const auto b0 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaMalloc(&d_offsets, fx.offsets.size() * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_dims, fx.dims.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_vals, fx.vals.size() * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_pairs, fx.pairs.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_order, plan.order.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_stats, 6 * n_pairs * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sims, n_pairs * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_offsets, fx.offsets.data(),
                        fx.offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dims, fx.dims.data(), fx.dims.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vals, fx.vals.data(), fx.vals.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pairs, fx.pairs.data(), fx.pairs.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_order, plan.order.data(),
                        plan.order.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  const double t_h2d = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - b0).count();

  // One thread per (pair, lane-in-group): n_pairs * group threads.
  const int64_t threads = n_pairs * map.group;
  const int64_t blocks = (threads + kBlock - 1) / kBlock;
  CUDA_CHECK(cudaEventRecord(ev0));
  ENGINE_DISPATCH_GROUP(map.group, map.hoist, pair_stats_kernel,
                        static_cast<unsigned>(blocks), kBlock, 0, d_offsets,
                        d_dims, d_vals, d_pairs, d_order, n_pairs, 0, n_dims,
                        d_stats);
  CUDA_CHECK(cudaEventRecord(ev1));
  finalize_kernel<<<static_cast<unsigned>((n_pairs + 255) / 256), 256>>>(
      d_stats, d_order, n_pairs, d_sims);
  CUDA_CHECK(cudaEventRecord(ev2));
  CUDA_CHECK(cudaEventSynchronize(ev2));
  CUDA_CHECK(cudaGetLastError());

  float ms_stats = 0, ms_final = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_stats, ev0, ev1));
  CUDA_CHECK(cudaEventElapsedTime(&ms_final, ev1, ev2));

  const auto b1 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaMemcpy(out.data(), d_sims, n_pairs * sizeof(double),
                        cudaMemcpyDeviceToHost));
  const double t_d2h = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - b1).count();

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  // Unified timing schema: device_total is stats + finalize (no collective on
  // one GPU). t_load lives on main's line, so main assembles cold_data_path_s
  // from these stage timings -- see CudaTimings.
  const double t_stats = ms_stats / 1e3, t_final = ms_final / 1e3;
  OccupancyInfo occ{};
  ENGINE_OCCUPANCY_FOR_GROUP(map.group, map.hoist, pair_stats_kernel, kBlock, occ);
  char occ_json[512];
  occupancy_json(occ_json, sizeof occ_json, occ);

  CudaTimings& tm = cuda_last_timings();
  tm.plan = plan.build_seconds;
  tm.plan_metrics = t_plan_metrics;
  tm.h2d = t_h2d;
  tm.stats = t_stats;
  tm.finalize = t_final;
  tm.d2h = t_d2h;
  const double device_total = tm.device_total();
  std::printf(
      "{\"cuda_detail\":{\"device\":\"%s\",\"timing_basis\":\"device_total\","
      "\"t_setup_s\":%.6f,\"t_h2d_s\":%.6f,"
      "\"t_stats_s\":%.6f,\"t_allreduce_s\":0.000000,\"t_finalize_s\":%.6f,"
      "\"t_kernel_stats_s\":%.6f,\"t_kernel_finalize_s\":%.6f,"
      "\"device_total_s\":%.6f,\"t_d2h_s\":%.6f,"
      "\"group\":%d,\"pair_order\":\"%s\",\"hoist\":%s,"
      "\"t_plan_s\":%.6f,\"t_plan_metrics_s\":%.6f,"
      "\"plan_order_basis\":\"global_full_dims\","
      "\"plan_metrics_computed\":%s,"
      "\"pre_timing_delay_ms\":%d,\"gpu_state_at_timing_start\":%s,"
      "\"plan_effective_elements\":%lld,\"plan_lane_slots\":%lld,"
      "\"plan_lane_utilisation\":%.5f,\"occupancy\":%s}}\n",
      prop.name, t_h2d, t_h2d, t_stats, t_final, t_stats, t_final,
      device_total, t_d2h, map.group, pair_order_name(map.order),
      map.hoist ? "true" : "false", plan.build_seconds, t_plan_metrics,
      map.plan_metrics ? "true" : "false",
      map.pre_timing_delay_ms, gpu_state,
      static_cast<long long>(plan_metrics.effective_elements),
      static_cast<long long>(plan_metrics.lane_slots),
      plan_metrics.utilisation(), occ_json);

  cudaFree(d_offsets); cudaFree(d_dims); cudaFree(d_vals);
  cudaFree(d_pairs); cudaFree(d_order); cudaFree(d_stats); cudaFree(d_sims);
}

}  // namespace engine
