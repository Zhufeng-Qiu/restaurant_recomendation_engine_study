# Multi-Backend Pearson Similarity Engine

An archived INF553 Yelp recommender (PySpark, MinHash/LSH + collaborative
filtering) evolved into a systems study: the Pearson-similarity kernel is
extracted behind a frozen numerical contract, exported as backend-neutral
binary workloads, and re-implemented across serial C++, OpenMP, MPI,
single-GPU CUDA, and two-GPU NCCL (synchronous and async/overlapped)
backends — every one validated against the same golden similarities.
Prediction and RMSE evaluation stay in Python; native backends stop at
similarities.

## Layout

| Path | Contents |
| --- | --- |
| `original_code/` | Frozen original homework code + data (SHA-256 manifest, read-only) |
| `spark_pipeline/` | Maintained PySpark baseline: `cf_train.py`, `cf_predict.py`, fixture exporter, pinned env |
| `docs/pearson_contract.md` | The frozen numerical contract every backend implements |
| `docs/gpu_runbook.md` | Scripted GPU-host session: build, correctness gates, bench, Nsight |
| `tools/` | Spark-free validator, reference impl, archived-model cross-check, RMSE loop |
| `data/fixtures/` | 4 exported workloads (CSR ratings + candidate pairs + golden sims) |
| `engine/` | Native engine: CMake, serial/OpenMP/MPI/CUDA/NCCL backends, tests, bench |
| `results/` | Benchmark JSON/CSV, figures, prediction outputs |

## The contract (docs/pearson_contract.md)

All backends compute, for the same candidate pairs, the six sufficient
statistics (n, Σx, Σy, Σx², Σy², Σxy) over the intersection of two rating
rows, with pair-local means, minimum overlap 3, zero-variance ⇒ sim 0, an
output filter `sim > 1e-14`, and absolute tolerance ≤ 1e-12 against the
serial reference. Distributed backends (MPI, NCCL) partition the RATING
DIMENSION and AllReduce partial six-stat tensors — pairs are never sharded.

## Results

Workload: `item_full` — 1,171,857 candidate pairs, 488,560 ratings. Every
benchmark trial ran with `--validate`; **every backend matches the golden
similarities bit-exactly (max |diff| = 0.0)**. CPU study on Apple M5; GPU
study on a RunPod pod with 2x A100-SXM4-80GB (NV12 NVLink), CUDA 12.8,
NCCL 2.25.1 (environment recorded in `results/gpu_env.txt`).

### Backend comparison

Median of 5 trials after 2 warm-ups; GPU times are device-side (kernels +
collectives), speedup vs same-machine serial.

| Backend | Hardware | Median | Speedup |
| --- | --- | --- | --- |
| Serial C++ (oracle) | EPYC (pod) | 1323.1 ms | 1.0x |
| OpenMP 16t dynamic | EPYC (pod) | 86.1 ms | 15.4x |
| MPI 8 ranks | Apple M5 | 256.3 ms | 3.1x |
| CUDA warp-per-pair | 1x A100 | 4.66 ms | 284x |
| NCCL sync | 2x A100 | 4.03 ms | 328x |
| NCCL async double-buffered | 2x A100 | 5.03 ms | 263x |

![CPU vs GPU backend latency on item_full](results/figures/gpu_comparison.png)

Key takeaways:

- **GPUs vs best CPU config**: ~17-21x over OpenMP 16t (the 284-328x
  headline is vs one CPU core).
- **Two GPUs gain only 1.16x** — each GPU still touches every pair, so
  per-pair fixed costs don't halve; the AllReduce itself is nearly free on
  NVLink (0.45 ms for the 56 MB six-stat tensor).
- **Async overlap is real but doesn't pay here**: 31-34% of comm is hidden
  (Nsight-verified), yet sync wins because comm is only ~11% of kernel time.
- **End-to-end RMSE 0.8652** (archived Spark model: 0.8657; target 0.9).

OpenMP scaling (M5; the 4→8t knee is the P-core/E-core boundary):

![OpenMP speedup and efficiency](results/figures/openmp_scaling.png)

MPI scaling (M5; the AllReduce share grows to ~13% at 8 ranks — the same
tensor NVLink moves in 0.45 ms):

![MPI strong scaling and communication fraction](results/figures/mpi_scaling.png)

Raw benchmark logs live in `results/bench/` (regenerate figures with
`engine/bench/make_figures.py <bench.json>`); Nsight traces in
`results/profiles/`.

## Lossless compression of the collective payload

The AllReduce carries `(n, Sx, Sy, Sxx, Syy, Sxy)` as six float64 per pair --
56.25 MB for `item_full`. But the fixture's ratings are Yelp stars, taking
exactly five values `{1,2,3,4,5}`, so **all six statistics are exact
non-negative integers**, bounded by the longest rating row `N = 1363`:
`n <= N`, `Sx,Sy <= 5N`, `Sxx,Syy,Sxy <= 25N = 34,075`. Sixteen bits of
content in 384 bits of wire format.

`--payload` selects the representation (`docs/pearson_contract.md` §9):

| `--payload` | Layout | Bytes/pair | AllReduce | Ratio |
| --- | --- | --- | --- | --- |
| `f64` (default) | 6 x float64 | 48 | 56.25 MB | 1.00x |
| `i32` | 6 x int32 | 24 | 28.12 MB | 2.00x |
| `packed` | 2 x uint64 | 16 | 18.75 MB | 3.00x |

`packed` gives each field 21 bits with no shared carry space
(`word0 = n | Sx<<21 | Sy<<42`). Because the partition is over the rating
dimension, a field's cross-GPU sum *is* the global total, which the domain
gate bounds below 2^21 -- so no field can carry into its neighbour and
`pack(a) + pack(b) == pack(a + b)`. The packed words therefore go **straight
to `ncclSum` over `ncclUint64`**: the collective never sees the uncompressed
form, and unpacking happens once, in the finalize kernel.

This is lossless by construction rather than lossy within a tolerance, which
is what lets the frozen bit-exact contract survive compression untouched.
`check_payload_domain()` refuses `i32`/`packed` up front on any fixture
outside the domain -- a silently overflowed field would corrupt the sum while
every backend still agreed with itself.

**Verified on CPU, not yet on GPU.** `cuda_emulation_test` runs all three
representations against golden similarities at 1-, 2-, and 3-way rank splits,
packing each rank's partial and summing the *packed words*. All pass at
`max_abs_diff = 0.0` over 1,171,857 pairs (`item_full`) and 1,411,864
(`user_full`). The device path compiles from the same header but has not been
run on a GPU host; `--payload` timings are not yet measured, so the table
above reports payload size, which is exact, and no speedup, which is not.

## Findings / limitations

- The Gate D chunk sweep caught a real double-buffering race in the async
  pipeline (slot-free event recorded before the finalize kernel that reads
  the slot); fixed by reordering, after which every chunk size validates
  bit-exact. See the comment in `engine/src/nccl/nccl_main.cu`.
- On NVLink this workload's AllReduce (56 MB six-stat tensor, ~0.5 ms) is
  ~11% of kernel time, so async overlap gains do not beat per-chunk launch
  overhead: sync wins the A/B. The async design targets slower interconnects
  (PCIe) or larger stat tensors; the Nsight traces show the overlap is real
  (31-34% of comm hidden).
- The original homework's `user_based` predict branch writes `user_id` and
  `business_id` swapped in its output JSON (inherited defect, kept for
  faithfulness); swap keys when evaluating (native user model: RMSE 0.9461).
- The dataset is the course-provided California subset of Yelp; candidate
  generation (item-item Cartesian over qualified pairs, user-user LSH)
  follows the original assignment and is part of the frozen contract.

## Reproduction

### 0. Environment (macOS arm64 used for CPU phases)

```bash
brew install openjdk@17 cmake libomp open-mpi
uv venv .venv && uv pip install -r spark_pipeline/requirements.txt matplotlib
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export PYSPARK_PYTHON=$PWD/.venv/bin/python PYSPARK_DRIVER_PYTHON=$PWD/.venv/bin/python
```

### 1. Spark baseline and fixtures

```bash
DATA=original_code/python/data
# Reproduce the archived golden model (item-based CF):
.venv/bin/python spark_pipeline/cf_train.py $DATA/train_review.json /tmp/model item_based
# Export + validate a backend-neutral fixture:
.venv/bin/python spark_pipeline/export_fixture.py $DATA/train_review.json data/fixtures/item_full item
python3 tools/validate_fixture.py data/fixtures/item_full
```

### 2. Native engine (CPU)

```bash
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release -DENGINE_OPENMP=ON -DENGINE_MPI=ON
cmake --build engine/build -j
ctest --test-dir engine/build --output-on-failure   # microcases + CUDA emulation
engine/build/pearson_engine data/fixtures/item_full --backend serial --validate
OMP_NUM_THREADS=8 engine/build/pearson_engine data/fixtures/item_full --backend openmp --validate
mpirun -np 4 engine/build/pearson_engine_mpi data/fixtures/item_full --validate
```

### 3. Benchmarks and figures

```bash
python3 engine/bench/run_bench.py                    # add --gpu on a GPU host
.venv/bin/python engine/bench/make_figures.py results/bench/bench_<stamp>.json
```

### 4. End-to-end RMSE (native similarities → Python prediction)

```bash
engine/build/pearson_engine data/fixtures/item_full --backend serial --out /tmp/sims.bin
python3 tools/sims_to_model.py data/fixtures/item_full /tmp/sims.bin /tmp/native.model
.venv/bin/python spark_pipeline/cf_predict.py $DATA/train_review.json \
    $DATA/test_review.json /tmp/native.model /tmp/pred.json item_based
python3 tools/rmse.py /tmp/pred.json $DATA/test_review_ratings.json \
    $DATA/business_avg.json item
```

### 5. GPU phases

Follow `docs/gpu_runbook.md` on a dual-GPU Linux host (correctness gates
before benchmarks, then Nsight profiles).
