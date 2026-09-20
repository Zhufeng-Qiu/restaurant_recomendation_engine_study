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

A speedup on one basis cannot be compared to a speedup on the other, and no
mechanism is offered here for why they differ. An earlier draft explained the
difference by the copy back alone and predicted a *smaller* compression gain
under `resident_host_complete`. The data say the opposite — ~5-7 % on NVLink
under `device_total`, ~16 % here — and the two runs also differ in machine,
commit and execution model, so a single-cause account is not available. The
claim is only that the numbers are not comparable.

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

**Boundary.** One machine. Each trial is its own process measuring one
batch's device stages, excluding load, setup and H2D/D2H. Not a cold start,
not an end-to-end recommendation latency, and **not** the resident batch
latency measured in the 2026-09-17 study.

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

> ⚠ `git_sha`, `git_dirty` and `image_digest` are **null inside these JSON
> records** — the pod received a `git archive` and had no `.git`. The version
> link comes from `host_manifest.txt` and `binary_hash.txt` beside them, not
> from the record itself. The pilot and formal runs are different builds and
> are hashed separately. `run_bench.environment()` now falls back to a shipped
> `GIT_SHA` file, but that fix post-dates these records.

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
* `D/B` is established on every analysis of both workloads. **`C/B` is
  established in the pre-registered 30-block analysis of both workloads**
  (0.840 and 0.833). It stops being established in a post-hoc subset of
  item_full that drops segment 0, leaving n=20. That subset is a
  time-sensitivity analysis reported alongside the main result; it does not
  replace it, and nothing shows it to be a steady-state measurement.
* item_full segment 0 differs from its other two: C drifts +19.9 % and
  D +12.9 % from segment 0 to segment 2, while A and B move ≤0.2 %. user_full,
  run from sustained load, is flat to 2.5 %. A settling explanation fits both
  and predicted the second, but **no clock or power trace was recorded**, so
  the cause is not established. Segment 0 is part of the pre-registered
  sample and stays in the main analysis.
* Four of 240 config-blocks ran 5–6× slow (medians 16–18 ms). The cause was
  not diagnosed; "interference" is a guess. They are **kept**. Excluding
  blocks with any configuration above 10 ms changes no conclusion and shrinks
  every effect: item_full D/B 0.809 → 0.853, user_full D/B 0.749 → 0.794.
  Both the threshold and the exclusion were chosen after seeing the data.
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
| compute-sanitizer, A and D (no NCCL init) | `ERROR SUMMARY: 0 errors` | verified |
| compute-sanitizer, B and C | 94 reports, every one `cudaErrorPeerAccessAlreadyEnabled` (704) with a libnccl backtrace; target completed and passed validation | verified |

**Boundary.** `max_abs_diff = 0` is **numeric equality, not a bitwise
comparison**. No bit-exactness claim is derived from it. The only escape a
numeric zero permits is +0.0 against −0.0; a bitwise check was not run.

B and C are **not** "sanitizer clean with zero reports" — they have 94
reports each, reclassified rather than absent. Two independent reasons
support the reclassification: NCCL 2.21.5 explicitly handles and clears the
duplicate-peer-access return code in `src/transport/p2p.cc`, and A and D,
which never initialise NCCL, report zero on the same fixture and binary.

Per-config detail in the JSON records is the **last** output slot's
`max_abs_diff`; the other 499 slots are recorded as pass/fail booleans. So
"all 120,000 batches passed the tolerance check" is supported; "all 120,000
batches recorded a zero error" is not.
