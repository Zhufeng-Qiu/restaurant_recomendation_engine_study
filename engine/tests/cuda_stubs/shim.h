// Device-side shims so the __CUDACC__-guarded kernels type-check on the host.
#pragma once
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
struct engine_dim3_ { unsigned x, y, z; };
extern engine_dim3_ threadIdx, blockIdx, blockDim, gridDim;
template <class T> T __shfl_sync(unsigned, T v, int, int = 32) { return v; }
template <class T> T __shfl_down_sync(unsigned, T v, unsigned, int = 32) { return v; }
inline void __syncthreads() {}
inline void __syncwarp(unsigned = 0xffffffffu) {}
