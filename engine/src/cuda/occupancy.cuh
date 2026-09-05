// Static resource use and THEORETICAL occupancy for a launched kernel.
//
// ncu is unavailable on every host this project has run on
// (ERR_NVGPUCTRPERM), so achieved occupancy cannot be measured. Theoretical
// occupancy can, from the driver, with no counters and no profiling: what the
// register and shared-memory allocation permit at this block size. It is an
// upper bound on the achieved figure, not a substitute for it.
#pragma once

#include <cuda_runtime.h>

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

  // Which resource ran out first. Registers and blocks-per-SM are the only
  // candidates here: these kernels allocate no static shared memory.
  const int by_regs = fa.numRegs > 0
      ? prop.regsPerMultiprocessor / (fa.numRegs * block_size) : 1 << 20;
  if (o.shared_bytes_static > 0) o.limiter = "shared_memory";
  else if (by_regs <= o.active_blocks_per_sm) o.limiter = "registers";
  else if (o.active_blocks_per_sm >= prop.maxBlocksPerMultiProcessor)
    o.limiter = "blocks_per_sm";
  else o.limiter = "warps_per_sm";
  return o;
}

// JSON body WITHOUT the enclosing braces, so callers can embed it.
inline int occupancy_json(char* buf, size_t n, const OccupancyInfo& o) {
  return snprintf(buf, n,
      "{\"registers_per_thread\":%d,\"local_bytes_per_thread\":%zu,"
      "\"shared_bytes_static\":%zu,\"block_size\":%d,"
      "\"active_blocks_per_sm\":%d,\"active_warps_per_sm\":%d,"
      "\"max_warps_per_sm\":%d,\"multiprocessors\":%d,"
      "\"theoretical_occupancy\":%.4f,\"limiter\":\"%s\","
      "\"measured\":\"theoretical_only_ncu_unavailable\"}",
      o.registers_per_thread, o.local_bytes_per_thread, o.shared_bytes_static,
      o.block_size, o.active_blocks_per_sm, o.active_warps_per_sm,
      o.max_warps_per_sm, o.multiprocessors, o.theoretical_occupancy,
      o.limiter);
}

#define ENGINE_OCCUPANCY_FOR_GROUP(group, KERNEL, BLOCK, OUT)          \
  do {                                                                 \
    switch (group) {                                                   \
      case 1:  OUT = occupancy_of(KERNEL<1>, BLOCK); break;            \
      case 2:  OUT = occupancy_of(KERNEL<2>, BLOCK); break;            \
      case 4:  OUT = occupancy_of(KERNEL<4>, BLOCK); break;            \
      case 8:  OUT = occupancy_of(KERNEL<8>, BLOCK); break;            \
      case 16: OUT = occupancy_of(KERNEL<16>, BLOCK); break;           \
      default: OUT = occupancy_of(KERNEL<32>, BLOCK); break;           \
    }                                                                  \
  } while (0)

}  // namespace engine_cuda
