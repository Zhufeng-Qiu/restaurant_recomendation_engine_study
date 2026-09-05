// Single-GPU CUDA backend (phase 3). Full dim range on one device.
// Prints a "cuda_detail" JSON line with device-side timings (CUDA events for
// kernels, wall clock for transfers) before main prints its summary line.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/fixture.hpp"
#include "cuda/pair_kernel.cuh"

#define CUDA_CHECK(x)                                                    \
  do {                                                                   \
    cudaError_t err__ = (x);                                             \
    if (err__ != cudaSuccess)                                            \
      throw std::runtime_error(std::string("CUDA: ") +                   \
                               cudaGetErrorString(err__));               \
  } while (0)

namespace engine {

void compute_cuda(const Fixture& fx, std::vector<double>& out) {
  using namespace engine_cuda;
  const int64_t n_pairs = fx.n_pairs();
  out.resize(n_pairs);

  int32_t n_dims = 0;
  for (int32_t d : fx.dims) n_dims = std::max(n_dims, d + 1);

  int64_t *d_offsets;
  int32_t *d_dims, *d_pairs;
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
  CUDA_CHECK(cudaDeviceSynchronize());
  const double t_h2d = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - b0).count();

  const int warps_per_block = kBlock / kWarp;
  const int64_t blocks = (n_pairs + warps_per_block - 1) / warps_per_block;
  CUDA_CHECK(cudaEventRecord(ev0));
  pair_stats_kernel<<<static_cast<unsigned>(blocks), kBlock>>>(
      d_offsets, d_dims, d_vals, d_pairs, n_pairs, 0, n_dims, d_stats);
  CUDA_CHECK(cudaEventRecord(ev1));
  finalize_kernel<<<static_cast<unsigned>((n_pairs + 255) / 256), 256>>>(
      d_stats, n_pairs, d_sims);
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
  // one GPU); one_shot adds the H2D staging and D2H readback a cold call pays.
  // t_load lives on main's line and is added by the bench harness.
  const double d_stats = ms_stats / 1e3, d_final = ms_final / 1e3;
  const double device_total = d_stats + d_final;
  std::printf(
      "{\"cuda_detail\":{\"device\":\"%s\",\"timing_basis\":\"device_total\","
      "\"t_setup_s\":%.6f,\"t_h2d_s\":%.6f,"
      "\"t_stats_s\":%.6f,\"t_allreduce_s\":0.000000,\"t_finalize_s\":%.6f,"
      "\"t_kernel_stats_s\":%.6f,\"t_kernel_finalize_s\":%.6f,"
      "\"device_total_s\":%.6f,\"t_d2h_s\":%.6f}}\n",
      prop.name, t_h2d, t_h2d, d_stats, d_final, d_stats, d_final,
      device_total, t_d2h);

  cudaFree(d_offsets); cudaFree(d_dims); cudaFree(d_vals);
  cudaFree(d_pairs); cudaFree(d_stats); cudaFree(d_sims);
}

}  // namespace engine
