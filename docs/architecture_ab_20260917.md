# Which axis should the work be divided on?

**2026-09-17 · one host · `8c8833e` · binary `6c5040db`**

## The question

Both GPUs already hold the entire input — the CSR ratings, the candidate
pairs, the order. Given that, the engine's original decomposition is not the
only one available:

* **split the rating dimension** — every device computes every pair over its
  own slice of the dimensions, and an AllReduce sums six partial statistics
  per pair. This is what the engine has always done, and what the compressed
  payload work is about.
* **split the pairs** — every device computes a disjoint run of pairs to
  completion over the full dimension range. No collective at all.

The repository had never asked which is faster. It answers a question the
compression study presupposes: the compression only matters if the collective
should be there in the first place.

## The four configurations

| | GPUs | split | payload | collective |
| --- | --- | --- | --- | --- |
| **A** | 1 | pairs | f64 | none |
| **B** | 2 | dimension | f64 | 1 × 48 B/pair |
| **C** | 2 | dimension | packed | 1 × 16 B/pair |
| **D** | 2 | pairs | f64 | none |

Everything else is held fixed: one binary, sync mode, `--group 4`,
`--pair-order source`, hoist off, plan metrics off, every batch validated.

`D/B` isolates the splitting strategy at one representation and is the primary
comparison. `C/B` isolates the representation at one splitting strategy.
`D/C` compares the two real candidates but moves both factors, so it cannot
attribute the difference to either alone.

## What was measured

`resident_host_complete_ms`: one host clock around statistics, the collective
where there is one, finalize, and the **full copy back to host memory**, with
the input already resident and the buffers already allocated. It excludes
process launch, fixture load, allocation, the first H2D and communicator
setup. It is a batch latency for a warm, resident engine — not a cold start,
not a per-user online request, and not a service throughput.

500 timed batches per block; a block is one run of all four configurations in
a seeded random order; 30 blocks per workload in three segments separated by
180 s. **The unit of analysis is the block, not the iteration** — 30 paired
observations, not 15,000.

The estimator is the paired geometric mean `R = exp(mean(log(t_x/t_y)))` over
blocks, with a 95 % percentile interval from a 5,000-draw block bootstrap
stratified by segment. This is not the ratio of the two medians and is not
reported as one.

## Results

![Full batch latency by work division](../results/figures/architecture_latency.png)

| | item_full (1,171,857 pairs) | | user_full (1,411,864 pairs) | |
| --- | --- | --- | --- | --- |
| | median ms | IQR | median ms | IQR |
| A 1 GPU, pairs, f64 | 4.461 | 0.030 | 3.064 | 0.041 |
| B 2 GPU, dim, f64 | 3.284 | 0.036 | 2.852 | 0.058 |
| C 2 GPU, dim, packed | 3.054 | 0.424 | 2.599 | 0.050 |
| **D 2 GPU, pairs, f64** | **3.012** | 0.190 | **2.401** | 0.281 |

![Paired block ratios with 95% intervals](../results/figures/architecture_ratios.png)

| | D/B | C/B | D/C |
| --- | --- | --- | --- |
| item_full, segments 1+2 | **0.888** [0.841, 0.918] | 0.936 [0.864, 1.032] *n.e.* | 0.948 [0.879, 0.986] |
| item_full, all 30 blocks | **0.809** [0.711, 0.901] | 0.840 [0.728, 0.957] | 0.963 [0.903, 1.016] *n.e.* |
| user_full, all 30 blocks | **0.749** [0.648, 0.828] | 0.833 [0.726, 0.910] | 0.900 [0.861, 0.937] |

*n.e.* = interval crosses 1, direction not established.

**Splitting pairs is faster than splitting the rating dimension**, by 11 % to
25 %, on every analysis of both workloads. Since both devices already hold
the whole input, the AllReduce the engine was built around buys nothing that
the pair split does not get for free.

**The compression result does not survive uniformly.** `C/B` is established
on user_full and on all 30 item_full blocks, and not established on the two
settled item_full segments. Both are reported. Which one is quoted cannot be
chosen after seeing them.

This does **not** contradict the README's compression figures, which are on
`device_total`, where the AllReduce is a large share of a ~3.2 ms number.
Here the metric includes the copy back to host, so the collective is a
smaller fraction of it and a smaller gain is what the same mechanism predicts.

## Two measurement defects, both found before any formal data

**Validation between batches cost 1.8×.** Validating each iteration inside the
timed loop left both GPUs idle ~5 ms per batch; the driver dropped the SM
clock from 1410 MHz to ~795 MHz and every later batch ran 1.77× slower, as a
clean step mid-run. `1410/795 = 1.77` matched the step for one GPU and for
two. Each iteration now writes to its own output slot and all slots are
checked after the timed loop: identical coverage, no host work between
batches. Clock locking is refused inside the container.

**item_full's segment 0 is campaign warm-up.** Its two fastest configurations
drift +19.9 % (C) and +12.9 % (D) from segment 0 to segment 2 while A and B
move +0.1 % and +0.2 %. user_full ran immediately afterwards from 30 minutes
of sustained load and is flat to within 2.5 % in every configuration — the
prediction that explanation makes, which could have failed.

## Correctness

Every number above comes from a run that validated 500/500 batches against
golden. Across the whole matrix: **240 config-blocks, 120,000 timed batches,
0 failures.**

| gate | result |
| --- | --- |
| pair-count boundaries × 4 configs (N = 1,2,3,63,64,65,127,128,129,511,512,513) | 48/48 |
| both formal workloads × 4 configs | 8/8 |
| 20 resident batches reused in one process, each validated | 4/4 |
| combinations that must be refused (packed, bylen, async, warmup without resident) | 4/4 refused |
| compute-sanitizer, A and D | 0 errors |
| compute-sanitizer, B and C | 94, all `cudaErrorPeerAccessAlreadyEnabled` in libnccl bootstrap |
| host test suite on the pod | 13/13 |

The sanitizer asymmetry is itself evidence: A and D never initialise NCCL, so
their zero is what shows those 94 belong to NCCL rather than to this kernel —
the same count the warp-packing campaign found, reproduced on a different pod.

Fault injection (`engine/tests/`): the output checker rejects NaN, ±Inf into
either side, short arrays, a swap that preserves the emitted count, and a
retained-set change within tolerance — 60 checks, plus 31 end-to-end against
the real binary's exit code and JSON. The pair-partition write path is
replayed on the host with three wrong implementations, each required to be
caught.

## Limitations

* **One host, one GPU generation, one interconnect.** EPYC 7763 + 2 × A100-SXM4-80GB
  NV12. Three time segments on one machine are not three machines. Nothing
  here generalises to PCIe, to more than two GPUs, or to other chips.
* **A is the weakest row.** Its batch time is bimodal between two clock states
  and its median did not reproduce across processes in pre-flight
  (4.44 / 6.05 / 5.89 ms). `D/A` and `C/A` inherit that; `D/B` and `C/B` do
  not, since those configurations were stable.
* **Four of 240 config-blocks show host interference** (medians 16–18 ms
  against a ~3 ms typical, max 41 ms), hitting A and B. They are kept in the
  headline analysis. Excluding them changes no conclusion and makes every
  effect *smaller*: item_full D/B 0.809 → 0.853, user_full D/B 0.749 → 0.794.
* **`max_abs_diff = 0` is numeric equality, not a bitwise comparison.** No
  bit-exactness claim is made from it here.
* **The collective sizes are representation sizes**, not measured link
  traffic: 48 B/pair for B and 16 B/pair for C, giving 56,249,136 and
  18,749,712 bytes per rank on item_full. No network counter was read.
* Not tested: async mode, `bylen` ordering, i32, more than two GPUs, or any
  pair split other than a contiguous even division.

## Reproduction

```bash
python3 engine/bench/architecture_ab.py run \
    --binary engine/build/pearson_engine_nccl \
    --fixture data/fixtures/item_full \
    --out results/bench/architecture_formal_20260917/formal_item_full.json \
    --warmup 50 --repeat 500 --segment-pause 180 --seed 20260917
python3 engine/bench/architecture_ab.py analyze <that file>
.venv/bin/python engine/bench/make_architecture_figures.py
```

Raw records, environment, topology and binary hashes are in
`results/bench/architecture_formal_20260917/`; the pre-flight ladder is in
`results/bench/architecture_pilot_20260917/`.
