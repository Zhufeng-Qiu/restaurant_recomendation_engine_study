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

## Status (2026-08-26, Apple M-series 10-core laptop)

| Backend | Gate                                                | Result |
| ------- | --------------------------------------------------- | ------ |
| serial  | microcases + 4 fixtures vs golden                    | max abs diff 0.0; 1.44M pairs/s on item_full |
| openmp  | bit-identical at 1/2/4/8/16 threads, static+dynamic  | pass; 0.794s -> 0.108s (7.4x at 16 threads) |
| mpi     | invariant at 1/2/4/8 ranks vs golden and each other   | pass, bit-identical; comm fraction 2.6% (2r) -> 18.7% (8r), 56 MB AllReduce |
| cuda    | validated on A100: bit-exact on all 4 fixtures         | stats kernel 5.0 ms on item_full (~262x serial) |
| nccl    | validated on 2x A100 NVLink: sync + async bit-exact    | sync 4.4 ms; async 4.8 ms, 31-34% of comm overlapped (Nsight) |
| nccl    | `--payload f64/i32/packed` x sync/async x 1/2 GPU: 18 gates | pass, all bit-exact; packed 4.2 ms (3x smaller collective, 1.9x faster AllReduce, 4.3% end to end) |
| nccl    | same 18 gates re-run on 2x A100 PCIe (PHB, no P2P)     | pass, all bit-exact; AllReduce 78x costlier, so packed cuts the iteration 55% and async turns positive — best config 15.9 ms `async packed` |

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
