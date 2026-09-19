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

`-DENGINE_PROFILING=ON` additionally builds the **measurement apparatus**: the
`--pre-timing-delay-ms` flag and NVML device-state sampling, used to
investigate why host work before a timed region changes the result. It is OFF
by default and compiled out entirely, because leaving an NVML call on the
default path would perturb the very thing it exists to study — and the flags
are refused, not ignored, in a normal build. Ordinary benchmarking does not
need it.

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
the 2026-09-06 ladder at commit `cfb8993`, which also sweeps the lane mapping).
The 2026-09-17 work-division ladder is separate and covers the pair-split path:
48 pair-count boundary gates, 8 full-workload gates, 4 reuse gates, 4 refused
combinations, and compute-sanitizer on all four configurations — see
[docs/architecture_ab_20260917.md](../docs/architecture_ab_20260917.md):

| Backend | Gate | Result |
| ------- | ---- | ------ |
| serial  | microcases + 4 fixtures vs golden | max abs diff 0.0 |
| openmp  | invariant at 1/2/4/8/16 threads, static+dynamic | pass; pairs are independent, so the thread count cannot change any summation order |
| mpi     | invariant at 1/2/4/8 ranks vs golden and each other | pass at `max_abs_diff = 0.0`; bitwise identity is expected for the reason in the note below, but was not separately checked |
| cuda    | 5 fixtures x 6 group sizes x 2 orderings x 2 hoist settings | 120 gates, all `max_abs_diff = 0.0` |
| cuda    | output byte-compared against the baseline mapping | 24 gates, 24 identical |
| nccl    | 1 and 2 GPU x f64/i32/packed x every mapping | 88 gates, all within the 1e-12 tolerance |
| nccl    | async at chunk 16384 / 262144 / 100003 x `comm`/`separate` | 12 gates, all within the 1e-12 tolerance |
| cli     | invalid group/order/hoist/mode/gpus/chunk, unknown flag, missing value | 13 rejected, 0 accepted |
| sanitizer | `compute-sanitizer memcheck`, CUDA and NCCL | no memory errors |

**Performance** is not duplicated here, because two copies of a benchmark table
drift apart: the current same-machine matrix, its provenance and its figures
live in the top-level [README](../README.md#results), and the
warp-packing analysis in
[docs/warp_packing_experiment_20260905.md](../docs/warp_packing_experiment_20260905.md).
The table that used to sit here was measured on a 2026-08-26 laptop against the
pre-warp-packing default and is superseded on both counts.

**Defaults** are `--group 4 --pair-order source` for both GPU binaries since
`1abc477`, and `--partition dim` for `pearson_engine_nccl` since 2026-09-17.
`--partition` and `--resident-bench` exist **only** on `pearson_engine_nccl`;
the `pearson_engine --backend cuda` entry point rejects them with
`unknown arg`. `--pair-order bylen` is the resident-engine opt-in, and
`--plan-metrics on` enables the lane-plan diagnostics, which are off by default
because they cost far more than the kernel they describe.

`--partition pair` gives each device a disjoint run of pairs and runs no
collective. It requires `--mode sync --payload f64 --pair-order source` and
refuses anything else: each device's results come back as a contiguous window,
which is only valid while the order is the identity.
`--resident-bench --warmup N --repeat N` sets up once and times N complete
batches in-process, reporting `resident_host_complete_ms` — a different timing
basis from `device_total`, and not comparable to it.

MPI note: the gates measure `max_abs_diff = 0.0` across 1/2/4/8 ranks. That
is an observation on this data, not a general floating-point guarantee, and an
earlier version of this note got the reason wrong. Splitting a sum into
contiguous ranges and reducing the partials does **not** reproduce the serial
addition order — it regroups it, and floating-point addition is not
associative; on random fractional values of wide magnitude the regrouped sum
differs from the serial one in about two thirds of trials.

Zero difference holds here because the six statistics are **exact integers**
under the shipped fixtures' 1-5 star ratings: the worst is 34,075 against
2^53, so every partial sum is exactly representable and no rounding happens
for the grouping to affect. Expect the invariance to survive only while that
is true. `max_abs_diff = 0.0` is also numeric equality rather than a bitwise
comparison; for doubles it permits exactly one distinct pair of bit patterns,
+0.0 against -0.0.

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
