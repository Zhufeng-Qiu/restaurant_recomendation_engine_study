# Reproduction

Everything the README's results rest on, from the raw corpus to the figures.
Split out of the README so that document can stay short; nothing here is
summarised there.

## Scaling detail: OpenMP and MPI

OpenMP scaling (M5, 30 trials; the 4→8t knee is the P-core/E-core boundary —
and the measurement audit below shows that knee, not workload skew, is most of what
dynamic scheduling buys):

![OpenMP speedup and efficiency](../results/figures/openmp_scaling.png)

MPI scaling (EPYC 7742 pod, 30 trials, ranks to 48; peak 5.04x at 16, and the
AllReduce share climbs to 51% by 48 ranks — the CPU-time quota and the growing
collective are confounded there, see the [measurement audit](measurement_audit_20260905.md). On the smaller
fixtures the collective dominates far earlier: at 9,054 pairs it is 83% and MPI
turns net slower than serial):

![MPI strong scaling and communication fraction](../results/figures/mpi_scaling.png)

Collective payload compression (A100 NVLink pod, 30 trials; right panel is the
one that matters — the collective shrinks 1.9x, but it was ~9-11% of the iteration. In the left panel the `async 2 GPU` group is the only one whose
IQR whiskers are wide enough to see):

![AllReduce payload comparison](../results/figures/payload_comparison.png)

Raw benchmark logs live in `results/bench/` (regenerate figures with
`engine/bench/make_figures.py <bench.json>`). The Nsight traces behind
[docs/analysis.md](analysis.md) are not shipped: `nsys` records the
profiled process's environment into the trace, and these were captured on
hosts with credentials in the environment. Everything derived from them is
written up there.


## Reproduction

### 0. Environment (macOS arm64 used for CPU phases)

```bash
brew install openjdk@17 cmake libomp open-mpi
uv venv .venv && uv pip install -r spark_pipeline/requirements.txt matplotlib
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export PYSPARK_PYTHON=$PWD/.venv/bin/python PYSPARK_DRIVER_PYTHON=$PWD/.venv/bin/python
```

### 1. Spark baseline and fixtures

The exported fixtures are committed, so **a fresh clone can skip straight to
step 2**. This step re-derives them and needs the raw Yelp corpus
(`original_code/python/data`, 492 MB), which is too large to ship.

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
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release \
      -DENGINE_OPENMP=ON -DENGINE_MPI=ON \
      -DOpenMP_ROOT=/opt/homebrew/opt/libomp   # macOS only: Apple clang needs the hint
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

On a dual-GPU Linux host, build with `-DENGINE_CUDA=ON -DENGINE_NCCL=ON` and
clear the correctness ladder before timing anything:

```bash
# single-GPU CUDA vs golden, smallest fixture first
for f in item_tiny item_medium item_full user_full; do
  engine/build/pearson_engine data/fixtures/$f --backend cuda --validate
done
# NCCL: one GPU, then two, then the async pipeline across chunk sizes
engine/build/pearson_engine_nccl data/fixtures/item_full --gpus 1 --mode sync --validate
engine/build/pearson_engine_nccl data/fixtures/item_full --gpus 2 --mode sync --validate
for c in 32768 65536 131072 262144 524288; do
  engine/build/pearson_engine_nccl data/fixtures/item_full \
      --gpus 2 --mode async --chunk $c --validate
done
```

Every run must report `max_abs_diff = 0` and `tol_failures = 0`. The async
chunk sweep is not optional: the double-buffering race below only surfaced
below 262144 pairs per chunk. Benchmarks come after.

