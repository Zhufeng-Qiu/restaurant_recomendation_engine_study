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

# printf format/argument mismatches are how two JSON fields shipped as
# "(null)" and broke a benchmark configuration mid-run. Make them fatal rather
# than a warning nobody reads in a 400-line build log.
set(ENGINE_FORMAT_FLAGS -Wall -Wformat=2 -Werror=format -Werror=format-extra-args)

if(ENGINE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  add_library(engine_cuda src/cuda/cuda_backend.cu)
  target_compile_options(engine_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wformat=2,-Werror=format>)
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
  target_compile_options(pearson_engine_nccl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wformat=2,-Werror=format>)
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

# The two timing bases, checked by perturbation rather than by inspection.
# Header-only: no GPU and no CUDA toolkit needed.
add_executable(timing_contract_test tests/timing_contract_test.cpp)
target_link_libraries(timing_contract_test PRIVATE engine_common)
add_test(NAME timing_contract COMMAND timing_contract_test)

# Warp-packing pair planning: boundary cases the shipped fixtures do not
# contain (empty restricted slices, trailing partial warps at every group size,
# tie structure), on synthetic in-memory fixtures. No arguments, no data files.
add_executable(pair_plan_test tests/pair_plan_test.cpp)
target_link_libraries(pair_plan_test PRIVATE engine_common)
add_test(NAME pair_plan COMMAND pair_plan_test)

# The bench harness's own timing check, both branches including the failure
# message -- which once raised NameError only on the failure path, after a
# 30-round matrix had finished measuring and before anything was written.
find_program(PYTHON3 python3)
if(PYTHON3)
  add_test(NAME bench_timing_selftest
           COMMAND ${PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/bench/run_bench.py --selftest)
endif()

# Compressed-payload domain gate (contract 9.4). The shipped fixtures are all
# inside the domain, so the gate's rejection branches only fire on the
# synthetic fixtures in data/fixtures/domain/ (tools/make_domain_fixtures.py).
add_executable(payload_domain_test tests/payload_domain_test.cpp)
target_link_libraries(payload_domain_test PRIVATE engine_common)
add_test(NAME payload_domain
         COMMAND payload_domain_test
                 ${CMAKE_CURRENT_SOURCE_DIR}/../data/fixtures/domain)
