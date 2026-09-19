# Reproduction

Everything the README's results rest on, from the raw corpus to the figures.
Split out of the README so that document can stay short; nothing here is
summarised there.

## Scaling detail: OpenMP and MPI

All three figures below are regenerated from
`results/bench/bench_20260906_044118.json` — EPYC 7742 + 2x A100-SXM4-80GB,
30 trials after 3 warm-ups, ranks swept to 32, at the current default
(`--group 4 --pair-order source`). Regenerate with
`engine/bench/make_figures.py <bench JSON>`.

![OpenMP speedup and efficiency](../results/figures/openmp_scaling.png)

OpenMP on the EPYC's 64 physical cores behind a ~27-CPU-equivalent cgroup
quota. Efficiency falls off well before 16 threads, and the quota — not a
core-type boundary — is the ceiling: this host has no P-core/E-core split. An
earlier version of this caption described an Apple M-series laptop and no
longer matches the figure.

![MPI strong scaling and communication fraction](../results/figures/mpi_scaling.png)

MPI to 32 ranks. The AllReduce share climbs as ranks grow, and the CPU-time
quota and the growing collective are confounded — see the
[measurement audit](measurement_audit_20260905.md). On smaller fixtures the
collective dominates far earlier: at 9,054 pairs it is most of the iteration
and MPI turns net slower than serial. `item_full` OOMs beyond 32 ranks, each
rank holding its own copy of the fixture.

![AllReduce payload comparison](../results/figures/payload_comparison.png)

Payload comparison at the current default. The right panel is the one that
matters: the collective shrinks ~1.9x, but it is a single-digit share of the
iteration on NVLink, so the end-to-end gain is bounded by that share.

Raw benchmark logs live in `results/bench/` (regenerate figures with
`engine/bench/make_figures.py <bench.json>`). The Nsight traces behind
[docs/analysis.md](analysis.md) are not shipped: `nsys` records the
profiled process's environment into the trace, and these were captured on
hosts with credentials in the environment. Everything derived from them is
written up there.


## Steps

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

`headline_table.py` regenerates the README's backend table, but with no
argument it reads the **newest** bench JSON. Name the file to reproduce a
published table:

```bash
python3 engine/bench/headline_table.py results/bench/bench_20260906_044118.json
```

The work-division matrix is a separate harness, on a different timing basis
(`resident_host_complete`, which includes the copy back to host):

```bash
for fx in item_full user_full; do
  python3 engine/bench/architecture_ab.py run \
      --binary engine/build/pearson_engine_nccl --fixture data/fixtures/$fx \
      --out results/bench/architecture_formal_<stamp>/formal_$fx.json \
      --warmup 50 --repeat 500 --segment-pause 180 --seed 20260917
  python3 engine/bench/architecture_ab.py analyze <that file>
done
.venv/bin/python engine/bench/make_architecture_figures.py
```

Run `--pilot 3` first: it writes to its own file and is labelled not formal
data. See [architecture_ab_20260917.md](architecture_ab_20260917.md).

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
# the pair split: disjoint pairs per device, no collective
python3 tools/make_prefix_fixtures.py data/fixtures/item_tiny
for n in 1 2 3 63 64 65 127 128 129 511 512 513; do
  engine/build/pearson_engine_nccl data/fixtures/item_tiny_n$n \
      --gpus 2 --partition pair --payload f64 --validate
done
engine/build/pearson_engine_nccl data/fixtures/item_full \
    --gpus 2 --partition pair --payload f64 --validate
```

Every run must report `validation_passed: true`, `tol_failures: 0`,
`nonfinite_output: 0`, `nonfinite_golden: 0` and `emitted_mismatches: 0`, and
exit 0. `max_abs_diff` should be `0.000e+00`, but note that a zero there is
numeric equality, not a bitwise comparison. The async chunk sweep is not
optional: the double-buffering race below only surfaced under 262144 pairs
per chunk. The pair-count prefixes exist because the shipped fixtures all
have six-figure pair counts and contain none of the boundaries the pair split
can get wrong. Benchmarks come after.

