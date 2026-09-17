// Syntax-check stubs. Not a CUDA implementation -- just enough declarations
// to let a host compiler type-check a .cu file's host code.
#pragma once
#include <cstddef>
#include <cstdint>
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorPeerAccessAlreadyEnabled = 704 };
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                      cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
enum { cudaEventDisableTiming = 2, cudaEventDefault = 0 };
struct cudaDeviceProp { char name[256]; int major, minor, multiProcessorCount,
                            maxThreadsPerMultiProcessor, warpSize,
                            regsPerMultiprocessor, sharedMemPerMultiprocessor,
                            maxThreadsPerBlock, maxBlocksPerMultiProcessor; };
struct cudaFuncAttributes { int numRegs, maxThreadsPerBlock;
                            size_t sharedSizeBytes, constSizeBytes,
                            localSizeBytes; };
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
cudaError_t cudaDeviceGetAttribute(int*, int, int);
cudaError_t cudaStreamCreate(cudaStream_t*);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
cudaError_t cudaEventCreate(cudaEvent_t*);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
cudaError_t cudaMalloc(void**, size_t);
template <class T> cudaError_t cudaMalloc(T** p, size_t n) {
  return cudaMalloc(reinterpret_cast<void**>(p), n);
}
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t cudaMemset(void*, int, size_t);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaDeviceEnablePeerAccess(int, unsigned);
cudaError_t cudaGetLastError();
template <class F>
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, F) {
  a->numRegs = 32; a->maxThreadsPerBlock = 1024; a->sharedSizeBytes = 0;
  a->constSizeBytes = 0; a->localSizeBytes = 0; return cudaSuccess;
}
template <class F>
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, F, int, size_t) {
  *n = 1; return cudaSuccess;
}
