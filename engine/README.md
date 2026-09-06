# pearson_engine — native similarity backends

Native implementations of the frozen Pearson kernel
(`docs/pearson_contract.md`). All backends consume the fixtures in
`data/fixtures/` (exported by `spark_pipeline/export_fixture.py`) and stop at
similarities; candidate generation and rating prediction stay in Python.

## Layout

```
src/common/   fixture loader, PairStats, contract finalization (shared)
src/serial/   serial C++ oracle (always built; correctness reference)
src/openmp/   pair-parallel CPU backend            (phase 2)
src/cuda/     single-GPU kernel                    (phase 3)
src/mpi/      rating-partitioned CPU AllReduce     (phase 4)
src/nccl/     two-GPU sync + async collectives     (phases 5-6)
tests/        contract microcases (CTest)
bench/        benchmark harness and results schema (phase 7)
```

## Build and run

```bash
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j
ctest --test-dir engine/build --output-on-failure

# Validate a backend against golden similarities:
engine/build/pearson_engine data/fixtures/item_full --backend serial --validate

# OpenMP (build with -DENGINE_OPENMP=ON -DOpenMP_ROOT=/opt/homebrew/opt/libomp):
OMP_NUM_THREADS=8 OMP_SCHEDULE=static \
  engine/build/pearson_engine data/fixtures/item_full --backend openmp --validate

# MPI (build with -DENGINE_MPI=ON): dimension-partitioned partial statistics
# + MPI_Allreduce; the communication pattern the NCCL backend will reuse.
mpirun -np 4 engine/build/pearson_engine_mpi data/fixtures/item_full --validate
```

Output is one JSON line with load/compute timings, pairs/s, max abs diff vs
golden, tolerance failures, and emitted count. Exit code is non-zero if any
pair differs from golden by more than the contract tolerance (1e-12).

Backend toggles: `-DENGINE_OPENMP=ON`, `-DENGINE_CUDA=ON`, `-DENGINE_MPI=ON`,
`-DENGINE_NCCL=ON` (each lands in its own phase).

On macOS, Apple Clang ships no OpenMP runtime, so `-DENGINE_OPENMP=ON` fails to
configure from a clean tree unless Homebrew's libomp is pointed at explicitly:

```
cmake -S engine -B engine/build -DENGINE_OPENMP=ON \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
```

## Status

**Correctness gates** (unchanged in kind since 2026-08-26; the counts below are
the 2026-09-06 ladder at commit `cfb8993`, which also sweeps the lane mapping):

| Backend | Gate | Result |
| ------- | ---- | ------ |
| serial  | microcases + 4 fixtures vs golden | max abs diff 0.0 |
| openmp  | bit-identical at 1/2/4/8/16 threads, static+dynamic | pass |
| mpi     | invariant at 1/2/4/8 ranks vs golden and each other | pass, bit-identical |
| cuda    | 5 fixtures x 6 group sizes x 2 orderings x 2 hoist settings | 120 gates, all `max_abs_diff = 0.0` |
| cuda    | output byte-compared against the baseline mapping | 24 gates, 24 identical |
| nccl    | 1 and 2 GPU x f64/i32/packed x every mapping | 88 gates, all bit-exact |
| nccl    | async at chunk 16384 / 262144 / 100003 x `comm`/`separate` | 12 gates, all bit-exact |
| cli     | invalid group/order/hoist/mode/gpus/chunk, unknown flag, missing value | 13 rejected, 0 accepted |
| sanitizer | `compute-sanitizer memcheck`, CUDA and NCCL | no memory errors |

**Performance** is not duplicated here, because two copies of a benchmark table
drift apart: the current same-machine matrix, its provenance and its figures
live in the top-level [README](../README.md#backend-comparison), and the
warp-packing analysis in
[docs/warp_packing_experiment_20260905.md](../docs/warp_packing_experiment_20260905.md).
The table that used to sit here was measured on a 2026-08-26 laptop against the
pre-warp-packing default and is superseded on both counts.

**Defaults** are `--group 4 --pair-order source` for both GPU binaries since
`1abc477`; `--pair-order bylen` is the resident-engine opt-in, and
`--plan-metrics on` enables the lane-plan diagnostics, which are off by default
because they cost far more than the kernel they describe.

MPI note: contiguous dimension ranges summed in rank order reproduce the
serial ascending-dim summation order exactly, which is why partition
invariance is bit-exact rather than merely within tolerance.

## End-to-end RMSE check (2026-08-26)

Serial backend similarities were converted to a cf_predict model
(`tools/sims_to_model.py`) and evaluated on the 58,480 test pairs with the
HW3 protocol (`tools/rmse.py`, business-average fallback):

| Model                        | Coverage | RMSE     |
| ---------------------------- | -------- | -------- |
| native serial (sim > 1e-14)  | 100%     | 0.865193 |
| archived Spark (sim > 0)     | 100%     | 0.865726 |

Both beat the assignment's 0.9 target. 873 predictions (1.5%) differ, all
caused by the 206 noise-level (~1e-16) similarities the archived filter kept:
when one lands in a top-3 neighborhood the prediction divides by a ~1e-16
denominator. The contract's EPS filter removes them and slightly improves
RMSE — evidence the sim > EPS policy is correct, not just convenient.
