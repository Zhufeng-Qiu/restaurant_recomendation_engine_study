# Claim ledger

Every number this repository publishes, with the version it was measured at,
the configuration, **what the timer actually covered**, the raw record it can
be recomputed from, and where the claim stops.

Two timing bases appear here and they are not interchangeable:

* **`device_total`** — stats + AllReduce + finalize on the device. Excludes
  load, setup, H2D and the copy back to host. Every number in the README's
  headline table is on this basis.
* **`resident_host_complete`** — one host clock around stats, the collective
  where there is one, finalize **and the full copy back to host memory**, with
  input already resident. Only the 2026-09-17 work-division study uses it.

A speedup on one basis cannot be compared to a speedup on the other. The
collective is a large share of a 3.2 ms `device_total` and a smaller share of
the same batch once the copy back is included, so the same mechanism produces
a smaller number under `resident_host_complete`. That is expected, not a
contradiction.

**Verified** means recomputed from the raw record during the 2026-09-17
session. **Carried** means the number is as written up at the time and was not
re-derived here.

## Headline backend table — `device_total`, item_full

Source record: `results/bench/bench_20260906_044118.json` · commit `174bb1c` ·
EPYC 7742 + 2 × A100-SXM4-80GB NV12 · 30 trials after 3 warm-ups.

| Claim | Value | Config key in the record | Status |
| --- | --- | --- | --- |
| Serial C++ oracle | 1253.179 ms | `serial` | verified |
| OpenMP 16t dynamic | 94.366 ms | `openmp_t16_dynamic` | verified |
| MPI 16 ranks | 278.642 ms | `mpi_r16` | verified |
| CUDA 1 GPU | 4.207 ms | `cuda_1gpu` | verified |
| NCCL sync packed, 2 GPU | 3.170 ms | `nccl_sync_g2_packed` | verified |
| NCCL async packed, 2 GPU | 3.707 ms | `nccl_async_g2_packed` | verified |
| Headline speedup | 395.26× | serial ÷ nccl_sync_g2_packed | verified |

**Boundary.** Steady state on one machine, excluding load, setup and H2D/D2H.
Not a cold start and not an end-to-end recommendation latency.

> ⚠ `engine/bench/headline_table.py` with no argument reads the **newest**
> `bench_*.json`, which is `bench_20260906_044505.json` (390.64×) — a
> different replicate from the one the README prints. Pass the file
> explicitly to reproduce the published table.

## Three-host replication

| Host | Speedup | Record | Status |
| --- | --- | --- | --- |
| replicate A | 411.29× | `bench_20260906_034430.json` | verified |
| replicate B | 395.26× | `bench_20260906_044118.json` | verified |
| replicate C | 390.64× | `bench_20260906_044505.json` | verified |
| spread | 5.29 % | (411.29 − 390.64) ÷ 390.64 | verified |

**Boundary.** Comparable builds from the same sources under a shared timing
contract; **no build hash was kept**, so an identical binary is intended
rather than evidenced. CPU rows agree to 0.2–0.9 %.

## Compression

| Claim | Value | Basis | Status |
| --- | --- | --- | --- |
| f64 collective payload, item_full | 56,249,136 B (56.2 MB) | 48 B/pair × 1,171,857 | verified |
| packed collective payload, item_full | 18,749,712 B (18.7 MB) | 16 B/pair × 1,171,857 | verified |
| ratio | 3× | representation size | verified |
| content vs wire | 85 bits in 384 | six fields, 21 bits each | carried |
| gain vs communication share | 11 / 75 / 93 % → 5.1 / 25.2 / 55.4 % | `device_total`, three hosts | carried |

**Boundary.** 3× is the **size of the AllReduce input representation**, not a
measured reduction in link traffic and not a latency claim. No network
counter was read. The three-host trend is descriptive; the controlled version
is the single-host `NCCL_P2P_DISABLE` toggle.

## Second GPU

| Claim | Value | Derivation | Status |
| --- | --- | --- | --- |
| "a second GPU adds 1.46×, not 2×" | 1.4625× | `nccl_sync_g1_packed` 4.636 ms ÷ `nccl_sync_g2_packed` 3.170 ms | verified |

**Boundary.** This is NCCL 1-GPU → NCCL 2-GPU on the same binary and path.
It is **not** `cuda_1gpu` → NCCL 2-GPU, which is 1.327×. The README states the
ratio without naming the baseline; that ambiguity is resolved here.

## Warp packing

Source: `docs/warp_packing_experiment_20260905.md`, records under
`results/bench/warp_packing_*.json`.

| Claim | Value | Status |
| --- | --- | --- |
| default `--group 4 --pair-order source`, 1 GPU | −17.59 % [−19.75, −15.38] | carried |
| same, 2 GPU | −29.43 % [−35.68, −22.58] | carried |
| the sort, 2 GPU | −16.39 % [−22.79, −9.46] | carried |
| sort planning cost | ~100 ms | carried |

**Boundary.** `device_total`, one host. Two mechanism hypotheses (the
lane-slot model, and redundant slice-bound searches) were falsified by their
own controls; the gain is real and its cause is not established.

## Work division — `resident_host_complete`, 2026-09-17

Source records: `results/bench/architecture_formal_20260917/` · commit
`8c8833e` · binary `6c5040db` · EPYC 7763 + 2 × A100-SXM4-80GB NV12 ·
CUDA 12.4, NCCL 2.21.5, driver 580.126.20 · 30 paired blocks × 500 timed
batches. **All verified in this session.**

| Claim | item_full | user_full |
| --- | --- | --- |
| A — 1 GPU, pairs, f64 | 4.461 ms (IQR 0.030) | 3.064 ms (IQR 0.041) |
| B — 2 GPU, dim, f64 | 3.284 ms (IQR 0.036) | 2.852 ms (IQR 0.058) |
| C — 2 GPU, dim, packed | 3.054 ms (IQR 0.424) | 2.599 ms (IQR 0.050) |
| D — 2 GPU, pairs, f64 | 3.012 ms (IQR 0.190) | 2.401 ms (IQR 0.281) |
| **D/B** pair vs dim split | 0.809 [0.711, 0.901] (all 30)<br>0.888 [0.841, 0.918] (seg 1+2) | 0.749 [0.648, 0.828] |
| **C/B** packed vs f64 | 0.840 [0.728, 0.957] (all 30)<br>0.936 [0.864, 1.032] **n.e.** (seg 1+2) | 0.833 [0.726, 0.910] |
| **D/C** the two candidates | 0.963 [0.903, 1.016] **n.e.** (all 30)<br>0.948 [0.879, 0.986] (seg 1+2) | 0.900 [0.861, 0.937] |

**Boundary.**

* The unit is the **block** — 30 paired observations, not 15,000 iterations.
  The estimator is the paired geometric mean, which is **not** the ratio of
  the two medians.
* `D/B` is established on every analysis of both workloads. **`C/B` is not:**
  established on user_full and on all 30 item_full blocks, not established on
  the two settled item_full segments. Both are published; neither is chosen
  after the fact.
* item_full segment 0 is campaign warm-up (C drifts +19.9 %, D +12.9 %, while
  A and B move ≤0.2 %); user_full, run from sustained load, is flat to 2.5 %.
* Four of 240 config-blocks carry host interference (medians 16–18 ms). They
  are **kept**. Excluding them changes no conclusion and shrinks every effect:
  item_full D/B 0.809 → 0.853, user_full D/B 0.749 → 0.794.
* `D/A` and `C/A` inherit A's clock bimodality — its median did not reproduce
  across processes in pre-flight (4.44 / 6.05 / 5.89 ms). Treat those two
  rows as weaker than the rest.
* One host, one GPU generation, one interconnect, two workloads that are two
  views of the same Yelp corpus — not two independent datasets.

## Correctness

| Claim | Value | Status |
| --- | --- | --- |
| formal matrix batches validated | 120,000 of 120,000, 0 failures | verified |
| `max_abs_diff` vs golden, both workloads, serial | 0.000e+00 | verified |
| non-finite outputs / goldens | 0 / 0 | verified |
| retained-set mismatches | 0 | verified |
| end-to-end RMSE vs archived Spark model | 0.8652 vs 0.8657 | carried |

**Boundary.** `max_abs_diff = 0` is **numeric equality, not a bitwise
comparison**. No bit-exactness claim is derived from it. The only escape a
numeric zero permits is +0.0 against −0.0; a bitwise check was not run.
