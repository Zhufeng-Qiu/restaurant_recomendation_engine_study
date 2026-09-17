#pragma once
#include <cstdint>
#include <cstddef>
typedef int ncclResult_t;
enum { ncclSuccess = 0 };
typedef struct ncclComm* ncclComm_t;
typedef enum { ncclInt32, ncclUint64, ncclDouble, ncclFloat } ncclDataType_t;
typedef enum { ncclSum, ncclProd, ncclMax, ncclMin } ncclRedOp_t;
const char* ncclGetErrorString(ncclResult_t);
ncclResult_t ncclCommInitAll(ncclComm_t*, int, const int*);
ncclResult_t ncclCommDestroy(ncclComm_t);
ncclResult_t ncclGroupStart();
ncclResult_t ncclGroupEnd();
ncclResult_t ncclAllReduce(const void*, void*, size_t, ncclDataType_t,
                           ncclRedOp_t, ncclComm_t, cudaStream_t);
