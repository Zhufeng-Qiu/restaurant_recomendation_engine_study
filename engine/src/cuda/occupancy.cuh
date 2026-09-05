// Static resource use and THEORETICAL occupancy for a launched kernel.
//
// ncu is unavailable on every host this project has run on
// (ERR_NVGPUCTRPERM), so achieved occupancy cannot be measured. Theoretical
// occupancy can, from the driver, with no counters and no profiling: what the
// register and shared-memory allocation permit at this block size. It is an
// upper bound on the achieved figure, not a substitute for it.
#pragma once

#include <cuda_runtime.h>
#include <string>

namespace engine_cuda {

struct OccupancyInfo {
  int registers_per_thread = 0;
  int max_threads_per_block = 0;
  size_t local_bytes_per_thread = 0;   // nonzero means spilling
  size_t shared_bytes_static = 0;
  size_t const_bytes = 0;
  int block_size = 0;
  int active_blocks_per_sm = 0;
  int active_warps_per_sm = 0;
  int max_warps_per_sm = 0;
  int multiprocessors = 0;
  double theoretical_occupancy = 0.0;  // active warps / hardware maximum
  int blocks_if_register_limited = 0;
  const char* limiter = "unknown";
};

// `kernel` is a __global__ function pointer; templates are fine.
template <typename K>
inline OccupancyInfo occupancy_of(K kernel, int block_size, int device = 0) {
  OccupancyInfo o;
  o.block_size = block_size;
  cudaFuncAttributes fa{};
  if (cudaFuncGetAttributes(&fa, kernel) != cudaSuccess) return o;
  o.registers_per_thread = fa.numRegs;
  o.max_threads_per_block = fa.maxThreadsPerBlock;
  o.local_bytes_per_thread = fa.localSizeBytes;
  o.shared_bytes_static = fa.sharedSizeBytes;
  o.const_bytes = fa.constSizeBytes;

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return o;
  o.multiprocessors = prop.multiProcessorCount;
  o.max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;

  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &o.active_blocks_per_sm, kernel, block_size, 0) != cudaSuccess)
    return o;
  o.active_warps_per_sm = o.active_blocks_per_sm * (block_size / prop.warpSize);
  o.theoretical_occupancy =
      o.max_warps_per_sm ? double(o.active_warps_per_sm) / o.max_warps_per_sm : 0.0;

  // Which resource ran out first. Registers are allocated per warp in fixed
  // units (256 on sm_80), not per thread, so a naive regsPerSM /
  // (numRegs * block_size) overestimates how many blocks fit and mislabels a
  // register-limited kernel as warp-limited -- which is exactly what the first
  // version of this function did with 37 registers at 128 threads.
  const int warps_per_block = block_size / prop.warpSize;
  const int kGranularity = 256;
  const int regs_per_warp =
      ((fa.numRegs * prop.warpSize + kGranularity - 1) / kGranularity) * kGranularity;
  const int by_regs = (fa.numRegs > 0 && warps_per_block > 0)
      ? prop.regsPerMultiprocessor / (regs_per_warp * warps_per_block)
      : 1 << 20;
  const int by_blocks = prop.maxBlocksPerMultiProcessor;
  const int by_warps = (prop.maxThreadsPerMultiProcessor / prop.warpSize) / warps_per_block;
  o.blocks_if_register_limited = by_regs;
  if (o.shared_bytes_static > 0) o.limiter = "shared_memory";
  else if (by_regs <= by_blocks && by_regs <= by_warps) o.limiter = "registers";
  else if (by_blocks <= by_warps) o.limiter = "blocks_per_sm";
  else o.limiter = "warps_per_sm";
  // The driver is the authority; if the estimate disagrees, say so instead of
  // asserting a cause.
  if (by_regs != o.active_blocks_per_sm && o.limiter == std::string("registers")
      && by_blocks != o.active_blocks_per_sm && by_warps != o.active_blocks_per_sm)
    o.limiter = "not_determined";
  return o;
}

// JSON body WITHOUT the enclosing braces, so callers can embed it.
inline int occupancy_json(char* buf, size_t n, const OccupancyInfo& o) {
  return snprintf(buf, n,
      "{\"registers_per_thread\":%d,\"local_bytes_per_thread\":%zu,"
      "\"shared_bytes_static\":%zu,\"block_size\":%d,"
      "\"active_blocks_per_sm\":%d,\"active_warps_per_sm\":%d,"
      "\"max_warps_per_sm\":%d,\"multiprocessors\":%d,"
      "\"theoretical_occupancy\":%.4f,"
      "\"blocks_if_register_limited\":%d,\"limiter\":\"%s\","
      "\"measured\":\"theoretical_only_ncu_unavailable\"}",
      o.registers_per_thread, o.local_bytes_per_thread, o.shared_bytes_static,
      o.block_size, o.active_blocks_per_sm, o.active_warps_per_sm,
      o.max_warps_per_sm, o.multiprocessors, o.theoretical_occupancy,
      o.blocks_if_register_limited, o.limiter);
}

#define ENGINE_OCC_G(group, H, KERNEL, BLOCK, OUT)                     \
  switch (group) {                                                     \
    case 1:  OUT = occupancy_of(KERNEL<1, H>, BLOCK); break;           \
    case 2:  OUT = occupancy_of(KERNEL<2, H>, BLOCK); break;           \
    case 4:  OUT = occupancy_of(KERNEL<4, H>, BLOCK); break;           \
    case 8:  OUT = occupancy_of(KERNEL<8, H>, BLOCK); break;           \
    case 16: OUT = occupancy_of(KERNEL<16, H>, BLOCK); break;          \
    default: OUT = occupancy_of(KERNEL<32, H>, BLOCK); break;          \
  }

#define ENGINE_OCCUPANCY_FOR_GROUP(group, hoist, KERNEL, BLOCK, OUT)   \
  do {                                                                 \
    if (hoist) { ENGINE_OCC_G(group, true, KERNEL, BLOCK, OUT) }       \
    else       { ENGINE_OCC_G(group, false, KERNEL, BLOCK, OUT) }      \
  } while (0)

}  // namespace engine_cuda
