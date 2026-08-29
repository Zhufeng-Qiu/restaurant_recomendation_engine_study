cmake_minimum_required(VERSION 3.20)
project(pearson_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Backend toggles. Serial is always built (it is the correctness oracle).
# Each later phase turns exactly one of these on as the backend lands.
option(ENGINE_OPENMP "Build the OpenMP backend" OFF)
option(ENGINE_CUDA   "Build the CUDA backend"   OFF)
option(ENGINE_MPI    "Build the MPI backend"    OFF)
option(ENGINE_NCCL   "Build the NCCL backends"  OFF)

add_library(engine_common INTERFACE)
target_include_directories(engine_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(engine_serial src/serial/serial.cpp)
target_link_libraries(engine_serial PUBLIC engine_common)

add_executable(pearson_engine src/main.cpp)
target_link_libraries(pearson_engine PRIVATE engine_serial)

if(ENGINE_OPENMP)
  find_package(OpenMP REQUIRED)
  add_library(engine_openmp src/openmp/openmp.cpp)
  target_link_libraries(engine_openmp PUBLIC engine_common OpenMP::OpenMP_CXX)
  target_link_libraries(pearson_engine PRIVATE engine_openmp)
  target_compile_definitions(pearson_engine PRIVATE ENGINE_HAVE_OPENMP)
endif()

if(ENGINE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_executable(pearson_engine_mpi src/mpi/mpi_main.cpp)
  target_link_libraries(pearson_engine_mpi PRIVATE engine_common MPI::MPI_CXX)
endif()

if(ENGINE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  add_library(engine_cuda src/cuda/cuda_backend.cu)
  target_include_directories(engine_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(pearson_engine PRIVATE engine_cuda)
  target_compile_definitions(pearson_engine PRIVATE ENGINE_HAVE_CUDA)
endif()

if(ENGINE_NCCL)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_library(NCCL_LIB nccl REQUIRED)
  find_path(NCCL_INC nccl.h REQUIRED)
  add_executable(pearson_engine_nccl src/nccl/nccl_main.cu)
  target_include_directories(pearson_engine_nccl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${NCCL_INC})
  target_link_libraries(pearson_engine_nccl PRIVATE ${NCCL_LIB})
endif()

enable_testing()
add_executable(microcases_test tests/microcases_test.cpp)
target_link_libraries(microcases_test PRIVATE engine_serial)
add_test(NAME microcases COMMAND microcases_test)

# Host-side emulation of the CUDA warp-per-pair algorithm (no GPU needed):
# runs the exact device code path against golden similarities.
add_executable(cuda_emulation_test tests/cuda_emulation_test.cpp)
target_link_libraries(cuda_emulation_test PRIVATE engine_common)
add_test(NAME cuda_emulation
         COMMAND cuda_emulation_test
                 ${CMAKE_CURRENT_SOURCE_DIR}/../data/fixtures/item_tiny)
