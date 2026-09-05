# Multi-Backend Pearson Similarity Engine

An archived Yelp recommender — originally USC INF553-Data-Mining-2020 project: PySpark, MinHash/LSH, item- and user-based collaborative filtering — rebuilt as a systems study.

Collaborative filtering scores a candidate pair of items by how alike their co-raters rated them, and that score is a Pearson correlation. This project extracts that one kernel behind a frozen numerical contract, exports it as backend-neutral binary workloads, and re-implements it across serial C++, OpenMP, MPI, single-GPU CUDA, and two-GPU NCCL (synchronous and async/overlapped) — every backend validated against the same golden similarities. Prediction and RMSE evaluation stay in Python; the native backends stop at similarities.

Hope this project can start my research journey of MLSys, AI infra, and parallel computing.

## Start here

This is a study of one question: **when is it worth compressing what a
collective sends?**

The workload is a recommender kernel, but its shape will be familiar from
data-parallel training. Each GPU computes a partial result over its shard of
the data, then an **AllReduce** sums the partials. In DDP those partials are
gradients; here they are six statistics per item pair — 1,171,857 pairs x 6
float64 = **56 MB moved every iteration**.

The finding: those six numbers are secretly small integers, because the
ratings behind them are 1-5 stars. So the 56 MB packs into **18.7 MB with no
loss at all** — and the packed words can be summed *while still packed*, so
NCCL never sees the uncompressed form.

Whether that 3x is worth anything depends entirely on the link:

| | NVLink | PCIe |
| --- | --- | --- |
| Communication is... | 11% of the iteration | 91% of the iteration |
| ...so compressing 3x buys | **4%** | **55%** |

Same binary, same data, same GPU model. Only the interconnect changed. That
is the point: **compressing a collective pays when the collective is the
bottleneck, and not otherwise** — and the second half of that sentence is the
part usually left out.

Three terms used throughout, if the vocabulary is unfamiliar:

- **six-stat tensor** — the AllReduce payload: `(n, Sx, Sy, Sxx, Syy, Sxy)`
  per pair. Enough to finish the correlation after summing, the way gradient
  buffers are enough to finish an optimizer step.
- **sync vs async** — sync does one big AllReduce after all the compute.
  Async splits the pairs into chunks and overlaps chunk *c*'s AllReduce with
  chunk *c+1*'s compute, the way DDP overlaps gradient buckets with backward.
- **f64 / i32 / packed** — the three payload encodings, 48 / 24 / 16 bytes
  per pair. All three produce bit-identical results.


## Layout

| Path | Contents |
| --- | --- |
| `original_code/` | Frozen original homework code + data (SHA-256 manifest, read-only) |
| `spark_pipeline/` | Maintained PySpark baseline: `cf_train.py`, `cf_predict.py`, fixture exporter, pinned env |
| `docs/pearson_contract.md` | The frozen numerical contract every backend implements |
| `docs/analysis.md` | Mechanism: roofline of the overlap, Nsight evidence, per-link regimes |
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
study on RunPod pods with 2x A100-SXM4-80GB (NV12 NVLink) and, for the
interconnect A/B, 2x A100 80GB PCIe (PHB, no P2P) — CUDA 12.8, NCCL 2.25.1
throughout (environments recorded in `results/gpu_env*.txt`).

### Backend comparison

Median of 5 trials after 2 warm-ups; GPU times are device-side (kernels +
collectives), speedup vs same-machine serial.

| Backend | Hardware | Median | Speedup |
| --- | --- | --- | --- |
| Serial C++ (oracle) | EPYC (pod) | 1312.0 ms | 1.0x |
| OpenMP 16t dynamic | EPYC (pod) | 85.8 ms | 15.3x |
| MPI 8 ranks | Apple M5 | 256.3 ms | 3.1x |
| CUDA warp-per-pair | 1x A100 | 5.00 ms | 262x |
| NCCL sync (`packed`) | 2x A100 | 4.21 ms | 312x |
| NCCL async double-buffered | 2x A100 | 4.80 ms | 273x |

![CPU vs GPU backend latency on item_full](results/figures/gpu_comparison.png)

Key takeaways:

- **262-312x over one CPU core**, ~17-20x over the best 16-thread OpenMP
  configuration.
- **A second GPU adds only 1.14-1.20x.** Only the rating dimension is split;
  every GPU still touches every pair, so the per-pair work does not halve.
- **On NVLink, neither optimisation is worth it.** Compressing the collective
  3x buys 4.3%; async overlap *costs* 9.3%. Communication is 11% of the
  iteration, so there is very little there to win.
- **On PCIe, both reverse.** Compression buys 55%, async turns positive, and
  together they cut the iteration 60% — same binary, same data.
- **End-to-end RMSE 0.8652** (archived Spark model 0.8657; target 0.9).

<details>
<summary>Why the GPU numbers differ slightly from the earlier session</summary>

The GPU rows come from the 2026-08-29 run, which re-measured everything after
`pair_stats_kernel` and `finalize_kernel` were refactored to share
`warp_pair_stats`. Two effects separate cleanly, and both are measured rather
than assumed: a same-GPU A/B against the pre-refactor header puts the
refactor's cost at **+2.4%** (4.854 → 4.972 ms), and the identical
pre-refactor code runs **4.2%** slower on this pod than on the one used in the
earlier session (4.854 vs 4.660 ms) — host-to-host variation, not code.
Serial and OpenMP, which do not include the CUDA header, reproduced their
archived numbers to within 0.8%, confirming the drift is specific to the GPU
path.

</details>

OpenMP scaling (M5; the 4→8t knee is the P-core/E-core boundary):

![OpenMP speedup and efficiency](results/figures/openmp_scaling.png)

MPI scaling (M5; the AllReduce share grows to ~13% at 8 ranks — the same
tensor NVLink moves in 0.47 ms):

![MPI strong scaling and communication fraction](results/figures/mpi_scaling.png)

Collective payload compression (A100 pod; right panel is the one that matters
— the collective shrinks 1.9x, but it was only 10.7% of the iteration):

![AllReduce payload comparison](results/figures/payload_comparison.png)

Raw benchmark logs live in `results/bench/` (regenerate figures with
`engine/bench/make_figures.py <bench.json>`). The Nsight traces behind
[docs/analysis.md](docs/analysis.md) are not shipped: `nsys` records the
profiled process's environment into the trace, and these were captured on
hosts with credentials in the environment. Everything derived from them is
written up there.

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

Correctness first: all 18 device gates pass bit-exact — 3 payloads x
{sync, async} x {1, 2} GPUs on `item_full`, plus 3 payloads x 2 modes on
`user_full`, every one at `max_abs_diff = 0.0` with identical emitted counts
(557,478 / 634,993). The CPU-side `cuda_emulation_test` agrees over the same
2.58 M pairs at 1-, 2-, and 3-way rank splits.

On NVLink the 3x payload cut made the collective **1.9x faster**
(0.471 -> 0.246 ms) but the iteration only **4.3%** faster, because the
collective was just 10.7% of it to begin with. The single-GPU rows are the
control: with no peer to reduce against, totals sit within 0.6% of each other
at every payload, so the packing itself costs nothing measurable and the whole
effect lives in the collective, exactly where the design put it.
[Full breakdown](docs/analysis.md#where-the-nvlink-gain-went).

Read together with the async result this is one coherent negative finding
rather than two disappointments: **on a fast interconnect this workload is not
communication-bound, so neither hiding communication nor shrinking it moves
the needle.** Both techniques target the same regime, and NVLink is not it.

### The same techniques on a slow interconnect

To test the converse, the identical binary was run on 2x A100 80GB **PCIe**
(`PHB` topology, P2P reported `NS` — so NCCL stages through host memory
rather than moving GPU-to-GPU). Same GPU generation, same memory class, same
CUDA 12.8 and NCCL 2.25.1, same SHA-verified sources: the interconnect is
close to the only variable. All 18 correctness gates pass bit-exact here too,
which is what the contract promised — the partition is over the rating
dimension, so the answer cannot depend on the link.

![NVLink vs PCIe payload comparison](results/figures/interconnect_comparison.png)

| | NVLink (NV12) | PCIe (PHB, no P2P) |
| --- | --- | --- |
| AllReduce, `f64` sync 2 GPU | 0.47 ms | 36.74 ms (78x) |
| Communication share of iteration | 10.7% | 91.1% |
| Compression, `f64` → `packed` | **-4.3%** | **-55.5%** |
| Async overlap, sync → async at `f64` | **+9.3%** (worse) | **-4.3%** (better) |
| Both together | +18.0% (worse) | **-60.5%** (40.3 → 15.9 ms) |
| Best configuration | `sync packed`, 4.21 ms | `async packed`, 15.92 ms |

**Every verdict reverses.** Compression goes from a 4.3% curiosity to a 55%
win, async overlap changes sign, and the best configuration moves from sync
to async. Nothing about the code changed; only the ratio of communication to
computation did.

The deciding factor is not the interconnect's name but whether the
collective is **bandwidth-bound**. Under payload reduction, effective
bandwidth stays flat on PCIe (~1.5 GB/s, the link is genuinely saturated) and
*falls* on NVLink (119 -> 76 GB/s, the collective is latency- and launch-bound
and smaller messages cannot fill it). Only a saturated link converts saved
bytes into saved time roughly one-for-one.
[Full analysis](docs/analysis.md#which-regime-the-collective-is-in).

## Findings / limitations

- The async chunk-size sweep caught a real double-buffering race in the async
  pipeline (slot-free event recorded before the finalize kernel that reads
  the slot); fixed by reordering, after which every chunk size validates
  bit-exact. See the comment in `engine/src/nccl/nccl_main.cu`.
- Async overlap loses its A/B on NVLink even though the pipeline works,
  because overlap's ceiling is `min(compute, comm)/total` — it pays when the
  two are comparable, not when communication is large. That corrects this
  project's original "async targets slow interconnects" hypothesis. The cost
  is paid differently on each link and both were measured; see
  [the mechanism](docs/analysis.md#why-overlap-is-capped-on-both-links).
- Lossless 3x payload compression hits the same ceiling from the other side:
  the collective really does get 1.9x faster, but it was 10.7% of the
  iteration, so end-to-end gain is 4.3%.
- Running the identical binary on 2x A100 PCIe reverses both verdicts
  (compression -55.5%, async overlap positive, best config moves from sync to
  async). The deciding factor is not the interconnect's name but whether the
  collective is bandwidth-bound: effective bandwidth stays flat under payload
  reduction on PCIe (~1.5 GB/s) and falls on NVLink (119 → 76 GB/s), so only
  the former converts saved bytes into saved time. Both optimisations are
  therefore worth their complexity on commodity multi-GPU hosts and not on
  NVLink ones — a conclusion neither measurement could have reached alone.
- `--payload i32`/`packed` are only valid inside their integer domain, and
  `check_payload_domain()` enforces it. Neither branch fires on the shipped
  fixtures (all {1,2,3,4,5}-rated, worst statistic 34,075 against a 2^21-1
  field), so the gate is **not exercised** — coverable with a synthetic
  fixture, not untestable.
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

## Sep. 4th 2026 - Update

Two corrections, one new test, and a re-measurement on fresh hardware that
adds a third interconnect. Everything above is unchanged.

### `Sixteen bits of content` is wrong — it is 85

16 bits is the widest single field, not the total. For `item_full`
(`N = 1363`, `vmax = 5`):

| field | bound | bits |
| --- | --- | --- |
| `n` | 1,363 | 11 |
| `Sx`, `Sy` | 6,815 | 13 each |
| `Sxx`, `Syy`, `Sxy` | 34,075 | 16 each |
| total | | **85** |

85 bits of content in 384 bits of wire format; 82 for `user_full`. Audit with
`python3 tools/payload_bit_audit.py --observed data/fixtures/item_full`.

Nothing downstream moves — the gate bounds each field separately, and 16 ≤ 21.
Measured rather than bounded, over all 1,171,857 pairs, the tightest field
(`Syy`, max 4,437) keeps 473x headroom in its 21-bit slot.

### `Only the interconnect changed` overstates the control

[docs/analysis.md](docs/analysis.md) already said so under *Caveats on scope*.
The pods differ by more than the link:

| | NVLink pod | PCIe pod |
| --- | --- | --- |
| CPU | EPYC 7742 | EPYC 7763 |
| `cuda_1gpu` (no collective) | 5.002 ms | 4.311 ms — 13.8% faster |
| `serial` | 1311.96 ms | 1326.39 ms — 1.1% slower |
| `openmp_t16_dynamic` | 85.82 ms | 88.85 ms — 3.5% slower |

The PCIe pod's GPU path is faster while its CPU path is slower. That is not
the interconnect. Read the comparison as hosts representing communication
regimes, not a controlled A/B — which is what the three-regime table below
now makes explicit.

### Domain gate: the rejection branches now run

`check_payload_domain()` had only ever accepted, since every shipped fixture
is Yelp stars far inside the domain. Five synthetic fixtures make it refuse.

| fixture | violates | `f64` | `i32` | `packed` |
| --- | --- | --- | --- | --- |
| `ok` | — (control) | accept | accept | accept |
| `fractional` | 3.5 not an integer | accept | reject | reject |
| `negative` | rating −2 | accept | reject | reject |
| `packed_overflow` | 5.0e6 > 2²¹−1 | accept | accept | reject |
| `i32_overflow` | 3.0e10 > 2³¹−1 | accept | reject | reject |

```bash
python3 tools/make_domain_fixtures.py
ctest --test-dir engine/build -R payload_domain
```

15 checks, 7 of them rejections. All five are valid workloads too — the serial
oracle reproduces their goldens at `max_abs_diff = 0.0`. The predicate now
sits in [`engine/src/common/payload_domain.hpp`](engine/src/common/payload_domain.hpp),
host-compilable, taking its field width from `engine_cuda::kPackMaxField`.

The A100 run that first exercised this found the gate refusing correctly but
doing it by letting the exception escape `main`, so the process died on
`SIGABRT` (exit 134, core dumped) with the message buried under
`terminate called after throwing...`. Both `nccl_main.cu` and `main.cpp` now
wrap their body in a thin exception boundary, and `nccl_main.cu` calls the
shared predicate instead of carrying a second copy:

| | before | after |
| --- | --- | --- |
| refused payload | exit 134, core dumped | **exit 2**, `error: <reason>` |
| valid workload | exit 0 | exit 0 |

Re-verified on 2x A100-SXM4-80GB after the change: **all 18 NCCL gates still
bit-exact** at unchanged emitted counts (557,478 / 634,993), single-GPU CUDA
unchanged on all three fixtures, and no timing regression — `sync f64`
4.137 ms and `sync packed` 3.933 ms against the 4.122 / 3.928 ms measured
before it.

### Three interconnect regimes, re-measured at 30 trials

Re-run on fresh pods at `--repeats 30 --warmups 3`, every configuration
bit-exact. `run_bench.py` now records `n_trials`, `iqr_s`, `p25_s`, `p75_s`.
The new PCIe host reports `SYS` topology with **P2P available** — a different
machine from the archived `PHB`/no-P2P pod, and the middle point
[docs/analysis.md](docs/analysis.md) said was missing.

| | NVLink `NV12` | PCIe `SYS`, P2P ok | PCIe `PHB`, no P2P |
| --- | --- | --- | --- |
| trials | 30 | 30 | 5 (archived) |
| AllReduce, `f64` sync 2 GPU | 0.479 ms | 9.958 ms | 36.743 ms |
| communication share | 11.6% | 76.0% | 91.1% |
| `f64` → `packed` | **−4.7%** | **−23.5%** | **−55.5%** |
| sync → async at `f64` | +19.3% | +9.7% | −4.3% |
| best configuration | `sync packed` | `sync packed` | `async packed` |

Compression tracks communication share monotonically across all three —
11.6% / 76% / 91% of the iteration buys 4.7% / 23.5% / 55.5%. That is the
paper's claim, now on three points instead of two.

Async does not track it. It still *loses* at 76% communication and only turns
positive on the host-staged link, so "async pays when the collective is the
bottleneck" is too coarse: at 76% comm the overlap ceiling is 23.7% and
chunking still costs more than it returns.

### The controlled version of that comparison

The table above still compares *hosts*. `NCCL_P2P_DISABLE=1` makes the same
comparison *within* one host — same binary, same GPUs, same session, with the
transport as the only variable. 30 trials after 3 warm-ups, every trial
validated (`engine/bench/p2p_ab.py`, data in `results/bench/p2p_ab.json`):

| | P2P on | P2P off (host-staged) | Δ |
| --- | --- | --- | --- |
| `sync f64` | 12.477 ms | 14.884 ms | +19.3% |
| `sync packed` | 9.331 ms | 9.940 ms | +6.5% |
| AllReduce, `f64` | 9.409 ms | 11.889 ms | +26.4% |
| **compression, `f64` → `packed`** | **−25.2%** | **−33.2%** | |
| `sync` → `async` at `f64` | +7.2% | +0.1% | |

Both trends reproduce with one variable moved: slowing the transport raises
what compression buys (−25.2% → −33.2%) and pushes async from a clear loss
toward break-even. This is the claim without the cross-host confound.

It also shows the archived `PHB` pod's 36.7 ms was **not** mostly about P2P.
Disabling P2P here costs 26.4% on the collective; that pod was ~4x slower
than this one's P2P-off number. Topology and host memory path dominate, which
is one more reason to read the three-host table as regimes rather than as a
controlled sweep.

**The 2-GPU rows are the noisy ones.** IQR as a fraction of median, 30 trials:

| | 1 GPU | sync 2 GPU | async 2 GPU |
| --- | --- | --- | --- |
| NVLink | 0.24–0.88% | 0.58–1.10% | **13.4–17.6%** |
| PCIe P2P | 0.34–4.15% | 5.11–5.42% | **6.5–13.0%** |

Async-2-GPU spread is 15–70x the single-GPU spread, which is why the archived
5-trial `nccl_async_g2_packed` (5.188 ms) and this run (4.306 ms) differ by
17%. Single-GPU and sync rows reproduce tightly; async 2-GPU medians should
be read with their IQR attached. On the CPU side the same re-run moved
`mpi_r4` −19.5% and `openmp_t16_static` −17.8% against their 5-trial values
while every within-run IQR stayed at 0.5–2.8%.

Data: `results/bench/bench_20260904_200900.json` (NVLink),
`bench_20260904_211956.json` (PCIe P2P), `bench_20260904_123357.json` (M5
CPU), environments in `results/gpu_env_20260904_*.txt`. Redrawn CPU figures
in `results/figures/20260904_30trials/`; the ones above are untouched.

### Configuration-matched Nsight traces (NVLink)

Both gaps that [docs/analysis.md](docs/analysis.md) recorded are closed: a
`packed` NVLink trace now exists, and all four captures use the default chunk
count instead of 18. Per GPU0, warm-up excluded:

| trace | collective | compute | overlapped |
| --- | --- | --- | --- |
| `sync f64` | 0.430 ms | 3.430 ms | **0.000 ms** |
| `sync packed` | 0.197 ms | 3.461 ms | **0.000 ms** |
| `async f64` | 1.468 ms (**3.41x**) | 3.531 ms (1.03x) | 0.161 ms |
| `async packed` | 1.087 ms (**5.52x**) | 3.685 ms (1.06x) | 0.769 ms |

Sync still measures exactly 0.000 ms — the control holds on new hardware.
Chunking inflates the collective **3.41x** at the benchmarked chunk count,
not the ~1.8x that document extrapolated from its 18-chunk capture; the
extrapolation understated the effect about twofold. Compute inflation is
negligible on NVLink (1.03–1.06x), so the NVLink async penalty is entirely a
collective-side cost, as claimed.

One archived inference does *not* survive. That document predicted the async
penalty would worsen under compression (−9.3% → −23.4%); at 30 trials it is
+19.3% at `f64` and +9.6% at `packed`, the opposite ordering. Given the
13–18% IQR on those rows, neither ordering is established — the supportable
claim is that async loses on NVLink at every payload.

### PCIe traces (`SYS`, P2P available)

The same four captures on the PCIe host, GPU0, warm-up excluded:

| trace | collective | compute | overlapped |
| --- | --- | --- | --- |
| `sync f64` | 9.091 ms | 2.807 ms | **0.000 ms** |
| `sync packed` | 6.560 ms | 2.834 ms | **0.000 ms** |
| `async f64` | 11.179 ms (**1.23x**) | 2.869 ms (1.02x) | 0.023 ms |
| `async packed` | 9.102 ms (**1.39x**) | 3.532 ms (1.25x) | 0.901 ms |

Sync measures 0.000 ms on a third independent host. The chunking penalty is
far smaller here than on NVLink — 1.23x against 3.41x at `f64` — which is
what a less latency-bound link should show. Yet async still loses, because
the overlap actually achieved on GPU0 is nearly nothing (0.023 ms, 0.2% of
the collective). GPU1 hides much more (55% of its compute) but its collective
time is inflated by spin-wait, so GPU0 is the binding side. **On this link
async fails for a different reason than on NVLink:** there the chunking cost
dominates, here the pipeline simply does not overlap on the critical GPU —
consistent with GPU0 also running finalize on its communication stream.

Traces are not shipped. `nsys` records the profiled process's environment,
and these again captured a live `RUNPOD_API_KEY` despite unsetting it in the
launching shell — Runpod injects it into the container's init environment,
where `nsys` reads it from `/proc`. Unsetting in the shell is not sufficient;
verify with `strings <trace> | grep rpa_` before sharing any capture.

### Async on PCIe was an implementation bug, not a property of the link

The PCIe traces above showed GPU0 hiding 0.2% of its collective while GPU1 hid
55%. The only structural difference between them is that GPU0 also runs the
finalize kernel — on its **communication stream**. `--finalize-stream separate`
moves it onto its own stream, chained to the collective by an event so the
ordering is unchanged; bit-exact at every chunk size from 16k to 524k pairs.

| PCIe, async 2 GPU, 30 trials | finalize on `comm` | on `separate` | |
| --- | --- | --- | --- |
| `f64` | 6.554 ms | 5.830 ms | **−11.1%** |
| `packed` | 4.777 ms | 3.992 ms | **−16.4%** |

Against `sync packed` on the same host (4.152 ms), async goes from **losing
15%** to **winning 3.9%**. The traces confirm the mechanism rather than just
the outcome — GPU0's share of compute hidden rises 4x, and the two GPUs become
symmetric, which is what a stream-occupancy explanation predicts:

| PCIe, `async packed` | GPU0 compute hidden | GPU1 compute hidden |
| --- | --- | --- |
| finalize on `comm` | **8.9%** | 40.6% |
| finalize on `separate` | **35.9%** | 37.1% |

**It does not help on NVLink** (30 trials): `sync f64` 4.089 ms against async
4.782 (`comm`) and 4.819 (`separate`); at `packed`, 3.905 against 4.563 and
4.470, inside a 0.6–0.75 ms IQR. That is the prediction, not a disappointment —
NVLink's collective is 0.25 ms, so there is nothing for a freed stream to
overlap with. The two links really do fail for different reasons: chunking cost
on NVLink, stream occupancy on PCIe. Only the second was fixable.

### Chunk size, swept for time rather than correctness

The chunk sweep had only ever been run for correctness. Both curves are
U-shaped, and the fix moves the whole PCIe curve down and its optimum:

| chunks | `comm` f64 | `separate` f64 | `comm` packed | `separate` packed |
| --- | --- | --- | --- | --- |
| sync | 6.146 | 6.137 | 4.152 | 4.157 |
| 72 | 8.927 | — | 6.963 | — |
| 36 | 7.362 | 7.034 | 5.666 | 5.242 |
| 18 | 6.636 | 6.309 | 4.902 | 4.540 |
| 9 | 6.546 | **5.751** | 4.646 | 4.090 |
| 5 | 6.459 | 5.771 | 4.648 | **3.838** |
| 3 | 7.615 | 7.073 | 4.519 | 5.131 |
| 2 | 6.854 | 7.082 | **4.425** | 4.516 |

Over-chunking costs 38% at 72 chunks; under-chunking gives the pipeline nothing
to overlap. The optimum sits at 5–9 chunks, and the default (262144 pairs, 5
chunks) was already in the right place.

### Where the GPU starts paying

Every GPU number in this project came from one problem size. Five fixtures on
one PCIe host, 20 trials:

| fixture | pairs | serial | 1 GPU | 2 GPU | GPU speedup |
| --- | --- | --- | --- | --- | --- |
| `item_p1e3` | 726 | 0.470 ms | 0.219 ms | 0.365 ms | **2.1x** |
| `item_p1e4` | 9,054 | 7.657 ms | 0.250 ms | 0.416 ms | 30.6x |
| `item_medium` | 74,904 | 64.451 ms | 0.464 ms | 0.627 ms | 138.9x |
| `item_full` | 1,171,857 | 1017.453 ms | 3.928 ms | 4.145 ms | **259.0x** |
| `user_full` | 1,411,864 | 489.276 ms | 2.946 ms | 3.960 ms | 166.1x |

The 262x headline holds only at the top. Kernel time is nearly flat from 726 to
74,904 pairs — 0.219 to 0.464 ms for 103x the work — so everything below ~10⁵
pairs is launch latency, not throughput. **And two GPUs are slower than one at
every size measured here**, because the collective's fixed cost exceeds what
halving the rating dimension saves; the second GPU only paid for itself on
NVLink.

### The workload matrix, on CPU

Three dimensions the brief specified had no number against them. All are cheap
and none needed a GPU (M5, 30 trials; `tools/make_overlap_fixtures.py`,
`bench_20260904_17*.json`).

**Problem size at fixed row length.** `item_tiny` is not a small `item_full` —
it carries 713 ratings per entity against 48 — so two new subsets hold row
length constant instead:

| pairs | serial | omp16 speedup |
| --- | --- | --- |
| 726 | 0.389 ms | **0.57x** (a net loss) |
| 9,054 | 6.069 ms | 4.16x |
| 74,904 | 49.998 ms | 6.56x |
| 1,171,857 | 760.023 ms | 7.08x |

Per-pair cost stays at 0.54–0.67 µs, confirming the control. Sixteen threads
are *worse than one* below ~10³ pairs.

**MPI has a sharper cliff.** Local compute scales (5.90 → 2.03 ms at 9,054
pairs) but the AllReduce jumps from 0.057 ms at 35 KB to 10.3 ms at 435 KB — a
180x step for 12x the payload, the signature of an eager-to-rendezvous switch.
Between 10⁴ and 10⁵ pairs the collective is 72–83% of the iteration and MPI is
net slower than serial.

**Overlap skew is not why dynamic scheduling wins.** Bands cut from
`item_full` share its CSR byte for byte, so the distribution is the only
variable. Dynamic vs static at 16 threads:

| band | stdev of n | dynamic vs static |
| --- | --- | --- |
| `n == 3` only | 0.00 | **−10.0%** |
| `n >= 11` | 9.39 | **−16.7%** |
| natural mix | 4.10 | −4.0% |

Skew helps, in the expected direction. But dynamic still wins by 10% where
every pair has *identical* work and static should be optimal, and the benefit
appears exactly at 8 threads — the M5's P-core/E-core boundary. On a
heterogeneous CPU most of what dynamic scheduling buys is hardware imbalance,
not workload skew. Parallel efficiency itself is insensitive to the
distribution (7.08–7.15x across all three).

`user_full` had also never been benchmarked, only validated. It is
communication-heavier than `item_full`: more pairs, 2.7x cheaper each
(0.244 vs 0.649 µs), so OpenMP matches (7.03x vs 7.08x) while MPI falls off
(1.77x vs 2.95x).

### Still open

- **A `PHB`/no-P2P host at 30 trials.** The 91%-communication column is still
  5-trial. Not a CUDA problem, as first assumed: the A100 PCIe hosts offered
  on 2026-09-04 advertise a driver capped at CUDA 12.5, but
  `NVIDIA_DISABLE_REQUIRE=1` starts the CUDA 12.8 image on them anyway and
  every correctness gate passes bit-exact — 12.x minor-version compatibility
  covers the gap, so toolkit and NCCL still match the archived runs. The real
  obstacle is that the pool only yielded `SYS`-topology machines with P2P
  available; `PHB` with P2P unavailable was never offered. The within-host
  A/B above is the closest substitute and shows that configuration is not
  reachable by disabling P2P alone.
- **The headline table predates the finalize fix.** Every async row in
  *Results* above, and in the three-regime table, was measured with finalize
  on the communication stream. On PCIe those rows now understate async by
  11–16%; on NVLink they are unaffected. Re-running the full matrix under
  `--finalize-stream separate` — and deciding whether it should become the
  default — is the obvious next pass.
- **Overlap skew on the GPU.** The band fixtures showed dynamic scheduling
  matters more for CPU core heterogeneity than for workload skew. The GPU
  equivalent — whether warp-per-pair suffers on `_ovl_skew`, where overlaps
  run to 264 — was not measured.
