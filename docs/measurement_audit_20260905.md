# Update: Measurement audit and headline rebuild — 20260905

Companion to [README.md](../README.md). This is the full record of a one-day
audit that found several published numbers did not mean what they said,
corrected them, unified how everything is timed, and only then rebuilt the
headline table.

It is kept out of the README deliberately. The README should say what the
project measured and what is still open; the reasoning, the wrong turns and
the arithmetic behind each correction belong here, where they can be checked
without pushing the results down the page.

**Corrections that changed a published claim**

| Claim | Was | Is |
| --- | --- | --- |
| payload content | "sixteen bits" | **85 bits** — that was the widest single field, not the total |
| interconnect A/B | "only the interconnect changed" | two hosts differing in CPU too, with GPU and CPU paths moving in opposite directions |
| lane utilisation | 87.07% | **85.58%** — a scan-weighted mean of per-pair ratios where the aggregate was wanted |
| why async loses on PCIe | chunking cost | **finalize occupying the communication stream** — GPU0's hidden compute goes 8.9% → 35.9% once moved |
| overlap skew on the GPU | "n=3 idles 29 of 32 lanes" | 29.89 of 32 are **busy**; lanes stride over the shorter row, not the intersection |
| NCCL sync totals | stats + allreduce | **+ finalize** — the only backend that excluded it |

**Everything below is the working record**, in the order it was found.

Corrections, new tests, and re-measurement on fresh hardware. Most of this
section is appended rather than rewritten, but **some text above was edited in
place** once the errata started contradicting it; that list is at the end of
this section.

### What was edited above, and why

Appending everything preserved the record but left the first screen asserting
things the errata disprove — a reader going top-down met three uncorrected
claims before reaching any correction. These were changed in place rather than
only annotated:

| Where | Was | Now |
| --- | --- | --- |
| *Start here* | "Only the interconnect changed" | says two hosts, two regimes, and points at the within-host P2P A/B |
| *Key takeaways* | "async turns positive" on PCIe | keeps it, flags the 12–44% IQR on every async 2-GPU row |
| *Layout* | "4 exported workloads", 4 tools | 6 workloads + 5 domain fixtures, 10 tools |
| *Three regimes* | blamed async's shortfall on chunking cost | superseded note: chunking is 1.23x here; the cause was stream occupancy |
| this section's finalize result | quoted −11.1% / −16.4% as settled | ten A/B measurements, sign reproducible, magnitude not |
| all six figures | drawn from 5-trial runs (Aug 28–29), no error bars on the GPU charts | redrawn from the 30-trial runs; `gpu_comparison` and `payload_comparison` now carry IQR whiskers, and `gpu_comparison` moved to a log axis |
| *Backend comparison* | speedups measured against two different serial baselines (M5 for MPI, pods for GPU) | left as measured, with a same-machine table added under *Four gaps the audit found* |
| *Four gaps* | called OpenMP-vs-MPI a shared-memory/message-passing result; said the skewed GPU band was 2.1x more efficient at comparable work because `n = 3` idles 29 of 32 lanes | both corrected: the two backends also differ in decomposition, and the lane claim was simply false — the kernel strides over the shorter *row* (65 elements, 29.89 lanes busy), so normalised properly the gap is 1.21x |

The same two corrections were applied to [docs/analysis.md](analysis.md).
Nothing in the *Results* tables was restated: those numbers stand as measured,
with their caveats now attached rather than buried.

The last row is a correction to this document's own earlier draft. A full
re-run of the matrix on two fresh hosts — 30 configurations each including
`--finalize-stream separate`, `bench_20260905_*.json` — did not reproduce the
effect size, and the reason turned out to be more interesting than the effect:
async 2-GPU medians carry 12–44% IQR on every host measured, while `sync` rows
reproduce to under 0.3%.

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

[docs/analysis.md](analysis.md) already said so under *Caveats on scope*.
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
sits in [`engine/src/common/payload_domain.hpp`](../engine/src/common/payload_domain.hpp),
host-compilable, taking its field width from `engine_cuda::kPackMaxField`.

> Those counts are this audit's, on 2026-09-05. The gate was extended to 39
> checks on 2026-09-17 after a hole was found in it: the single capacity bound
> `vmax²·max_row ≤ L` is vacuous at `vmax = 0`. See
> [pearson_contract.md §9.4](pearson_contract.md).

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
pass at `tol_failures=0`** with unchanged emitted counts (557,478 / 634,993),
single-GPU CUDA
unchanged on all three fixtures, and no timing regression — `sync f64`
4.137 ms and `sync packed` 3.933 ms against the 4.122 / 3.928 ms measured
before it.

### Three interconnect regimes, re-measured at 30 trials

Re-run on fresh pods at `--repeats 30 --warmups 3`, every configuration
passing its tolerance gate. `run_bench.py` now records `n_trials`, `iqr_s`,
`p25_s`, `p75_s`.
The new PCIe host reports `SYS` topology with **P2P available** — a different
machine from the archived `PHB`/no-P2P pod, and the middle point
[docs/analysis.md](analysis.md) said was missing.

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

Async does not track it. It *loses* at 76% communication and only turns
positive on the host-staged link, so "async pays when the collective is the
bottleneck" is too coarse: at 76% comm the overlap ceiling is 23.7% and the
pipeline still does not clear it.

> **Superseded in part.** This paragraph originally blamed the per-chunk
> cost. The traces below show that on this link chunking is cheap (1.23x)
> and the real cause was GPU0 running finalize on its communication stream —
> see *Async on PCIe was an implementation bug*. The `+9.7%` in the row above
> was also measured before that change, and async 2-GPU rows carry a 12–44%
> IQR, so treat it as a sign rather than a magnitude.

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

Both gaps that [docs/analysis.md](analysis.md) recorded are closed: a
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

### Async on PCIe: a real mechanism, an effect size the hosts cannot resolve

The PCIe traces showed GPU0 hiding 0.2% of its collective while GPU1 hid 55%
on identical work. The only structural difference is that GPU0 also runs the
finalize kernel — on its **communication stream**. `--finalize-stream separate`
moves it onto its own stream, chained to the collective by an event so ordering
is unchanged; `tol_failures=0` at every chunk size from 16k to 524k pairs.

**The mechanism is measured and unambiguous**, because it comes from the trace
rather than from a wall-clock median:

| PCIe, `async packed` | GPU0 compute hidden | GPU1 compute hidden |
| --- | --- | --- |
| finalize on `comm` | **8.9%** | 40.6% |
| finalize on `separate` | **35.9%** | 37.1% |

GPU0's overlap rises 4x and the two GPUs become symmetric (70.2% and 69.8% of
their collectives overlapped). GPU1, which never ran finalize, is unchanged.
That is what a stream-occupancy explanation predicts and what a link-property
explanation does not.

**The end-to-end effect is another matter.** Ten A/B measurements across four
host-runs put the direction where the mechanism says it should be — PCIe
benefits, NVLink does not — but not one of them clears its own noise:

| host | payload | `comm` | `separate` | effect | IQR |
| --- | --- | --- | --- | --- | --- |
| PCIe A | `f64` | 6.554 | 5.830 | −11.1% | 13.0% |
| PCIe A | `packed` | 4.777 | 3.992 | −16.4% | 12.5% |
| PCIe B | `f64` | 14.034 | 12.939 | −7.8% | 26.4% |
| PCIe B | `i32` | 11.471 | 7.633 | −33.5% | 28.7% |
| PCIe B | `packed` | 8.797 | 9.665 | +9.9% | 43.9% |
| NVLink A | `f64` | 4.782 | 4.819 | +0.8% | 14.5% |
| NVLink A | `packed` | 4.563 | 4.470 | −2.0% | 16.4% |
| NVLink B | `f64` | 4.470 | 4.830 | +8.1% | 19.3% |
| NVLink B | `packed` | 4.253 | 4.730 | +11.2% | 15.8% |

Four of five PCIe measurements improve; four of five NVLink measurements do
not. The sign is reproducible across independent hosts; the magnitude is not,
so **no percentage here should be quoted as the effect size**. The stronger
end-to-end evidence is the chunk curve below, where `separate` sits under
`comm` at every one of four chunk counts on PCIe — four consistent points beat
one median.

**The async 2-GPU rows are the least trustworthy numbers in this study.** They
carry 12–44% IQR on every host measured, while `sync` rows on the same runs
reproduce across independent hosts to within 0.3%:

| NVLink, independent runs | first | second | drift |
| --- | --- | --- | --- |
| `nccl_sync_g2_packed` | 3.928 ms | 3.928 ms | **0.00%** |
| `nccl_sync_g2_f64` | 4.122 ms | 4.110 ms | −0.27% |
| `cuda_1gpu` | 4.726 ms | 4.731 ms | +0.10% |

So the measurement method is sound and it is the async pipeline itself that is
unstable — which is the more useful finding, and the reason the earlier draft
of this section quoted −11.1% / −16.4% as if they were settled. They are not.

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

### Four gaps the audit found, now measured

An audit of what had never been run turned up two methodological gaps and two
open questions from the brief. All four were measured on one host — the first
in this project to run **MPI and NCCL together** (EPYC 7742, 2x A100 NV12, 30
trials).

**Every backend on one machine.** MPI had only ever run on the Apple M5 and
NCCL only on pods, so the headline table compared speedups against two
different serial baselines. Measured together:

| backend | median | vs same-machine serial |
| --- | --- | --- |
| serial | 1257.13 ms | 1.0x |
| MPI, 16 ranks | 270.04 ms | 4.7x |
| OpenMP, 16 threads | 94.12 ms | 13.4x |
| CUDA, 1 GPU | 4.727 ms | 266x |
| NCCL sync `packed`, 2 GPU | 3.917 ms | 321x |

Two caveats keep this from being more than it is. The OpenMP/MPI gap (2.87x at
16-way) is **not** a controlled shared-memory-versus-message-passing A/B: the
OpenMP backend shards *candidate pairs* and needs no communication at all,
while MPI shards the *rating dimension* and AllReduces a 56 MB tensor. Two
variables move together, and the decomposition is the larger one. And the
`321x` is not the same timing basis as the `266x` — see the note below.

**The NCCL sync rows exclude their finalize kernel.** `cuda` and `mpi` both
include finalize in their totals; the sync NCCL path runs it *after* the timed
region closes, so it was never counted. The kernel is 0.018–0.044 ms against a
~3.9 ms iteration, so **0.5–1.1%** — no conclusion in this document turns on
it, but `321x` versus `266x` is not a clean comparison and the compression
percentages below are a *stats-kernel + AllReduce* subtotal rather than full
device time. `run_bench.py` now adds `t_finalize_s` when the binary reports it;
the binary does not yet report it, which is the one-line follow-up.

**MPI strong scaling, past 8 ranks for the first time.** The M5 has 10 cores so
the curve stopped at 8. This host exposes **128 logical CPUs but a cgroup
CPU-time quota of 27.2 CPU-equivalents** (`cpu.max` = 2720000/100000):

| ranks | median | speedup | efficiency | AllReduce share |
| --- | --- | --- | --- | --- |
| 1 | 1359.92 ms | 1.00x | 100% | 0% |
| 2 | 777.13 ms | 1.75x | 87.5% | 3.5% |
| 4 | 491.46 ms | 2.77x | 69.2% | 9.4% |
| 8 | 345.82 ms | 3.93x | 49.2% | 15.8% |
| **16** | **270.04 ms** | **5.04x** | 31.5% | 23.1% |
| 24 | 288.74 ms | 4.71x | 19.6% | 38.6% |
| 32 | 287.93 ms | 4.72x | 14.8% | 35.5% |
| 48 | 578.55 ms | 2.35x | 4.9% | **51.0%** |

Peak at 16, flat through 32, collapse at 48. The turnover is *consistent* with
the quota but does not by itself prove it: the AllReduce share climbs from 15.8%
to 51% over the same range, so communication growth and CPU-time starvation are
confounded here. Separating them needs a host without a quota.

**Compression is not an `item_full` artefact.** Every compression number in this
study came from one workload. `user_full` is structurally different — LSH
candidate pairs, 1.41M of them, with a pair-weighted shorter row about 36% as
long — and the same host gives:

| `sync` 2 GPU (stats + AllReduce) | `item_full` | `user_full` |
| --- | --- | --- |
| `f64` | 4.111 ms | 3.737 ms |
| `i32` | 3.947 ms | 3.555 ms |
| `packed` | 3.917 ms | 3.488 ms |
| **`f64` → `packed`** | **−4.7%** | **−6.7%** |

It reproduces, and is slightly larger on the second workload. The claim no
longer rests on a single fixture. Note the subtotal caveat above: these are
stats-kernel plus AllReduce, and the collective is where the effect lives, so
the direction is solid and the exact percentage will shift slightly once
finalize is counted.

**Overlap skew: the measurement stands, my explanation of it did not.** The
brief asked whether skewed pair costs need bucketing. Measured on two bands cut
from `item_full`:

| band | pairs | short-row elements scanned | CUDA | ns per element |
| --- | --- | --- | --- | --- |
| `n == 3` (stdev 0) | 504,958 | 33.05 M | 1.819 ms | 0.0550 |
| `n >= 11` (stdev 9.4) | 69,771 | 14.83 M | 0.675 ms | 0.0455 |

An earlier draft of this section normalised by *intersection* elements, called
the two bands comparable in work, and concluded the skewed band was 2.1x more
efficient because an `n = 3` pair leaves 29 of a warp's 32 lanes idle. **All
three of those are wrong.** `pair_lane_stats` strides its lanes over the
*shorter rating row*, not over the intersection, so the work is 33.05 M against
14.83 M — a 2.2x difference, not comparable — and the `n == 3` band's shorter
row averages 65.45 elements, which keeps **29.89 of 32 lanes busy** in the first
round rather than 3. Normalised against what the kernel actually scans, the gap
is **1.21x**, not 2.1x.

What survives: the skewed band is modestly more efficient per element scanned,
and warp packing for short rows is a **hypothesis worth testing**, not a
conclusion this experiment supports.

### Unified timing contract, randomised run order, and a second headline basis

Steps 1–2 of the corrected plan, plus the nccl-tests baseline. One host
(EPYC 7742, 2x A100 NV12), 30 rounds, **round-robin with a per-round reshuffle**
(seed 20260905) rather than draining each configuration in turn.

**Randomisation changed the noise picture, but not where expected.** With
configurations interleaved, `sync` and single-GPU rows tighten to 0.22–1.12%
IQR. The async 2-GPU rows stay at **13–16%**. Interleaving rules out a simple
ordering effect — a spread that came from draining one configuration before the
next would have collapsed — but not every temporal or thermal mechanism. The
spread **persists under randomised interleaving and across these hosts**
(16.3% / 16.4% / 15.6% on three independent pods); its magnitude remains
protocol- and host-dependent, and later runs on the current default show
10.9–11.2%.

**The headline speedups were always steady-state.** Every backend now reports
`device_total = stats + allreduce + finalize` and, alongside it, a cold-path
total.

> ⚠ **The cold-path numbers in this subsection are superseded.** They were
> measured against `one_shot_total_s`, which excluded `ncclCommInitAll` and
> `MPI_Init` — both are inside the timed region now, and the field was renamed
> `cold_data_path_s` because process launch still is not (and cannot be, from
> inside the binary). The *ratio* between the two bases is the point and it
> survives; the absolute cold figures below understate themselves. They have
> not been re-measured.

| backend | steady-state | one-shot *(superseded basis)* | steady speedup | one-shot speedup |
| --- | --- | --- | --- | --- |
| serial | 1318.88 ms | 1333.7 ms | 1.0x | 1.0x |
| OpenMP 16t | 86.67 ms | 102.4 ms | 15.2x | 13.0x |
| MPI 16 ranks | 275.70 ms | 291.5 ms | 4.8x | 4.6x |
| CUDA 1 GPU | 4.75 ms | 23.8 ms | **277.5x** | **55.9x** |
| NCCL sync `packed` 2 GPU | 3.98 ms | 80.9 ms | **331.6x** | **16.5x** |

**331x becomes 16.5x** once a cold caller pays for staging, and on that basis
two GPUs are *worse than one* (80.9 vs 23.8 ms) because NCCL init and a second
device's transfers dominate. The CPU backends barely move, because their only
extra cost is reading the fixture. Neither number is wrong; the headline table
above is the steady-state one, which is the right basis for a resident
workload and the wrong one for a one-shot tool. Every run records both:
`device_total_s` and, on the cold side, `cold_data_path_s`.

Sync `packed` also moves 3.917 → 3.977 ms, the 1.5% that was missing while
NCCL sync excluded its finalize kernel.

**How far from the hardware ceiling.** `all_reduce_perf` from nccl-tests, on
the same host and GPUs, at this project's actual message sizes:

| payload | bytes | this engine | nccl-tests | ratio |
| --- | --- | --- | --- | --- |
| `packed` | 18.75 MB | 250.0 µs | 173.0 µs | 1.45x |
| `i32` | 28.12 MB | 305.0 µs | 233.3 µs | 1.31x |
| `f64` | 56.25 MB | 459.5 µs | 392.3 µs | 1.17x |

⚠ **Indicative, not calibrated.** `all_reduce_perf -b 16M -e 64M -f 2` measures
16/32/64 **MiB**, not this engine's 18,749,712 / 28,124,568 / 56,249,136 bytes,
so the nccl-tests column is interpolated across the wrong sizes; and it reads
the **out-of-place** column while `ncclAllReduce(slot, slot, ...)` is in-place.
Both are corrected in the next GPU session — exact byte counts, matching
dtypes, in-place column, and the raw command and stdout committed.

The collective runs within **1.2–1.5x** of what the library achieves on the
same link with nothing else in flight, and the gap widens as the payload
shrinks — consistent with the smaller message being more latency-bound, which
is the same effect that caps what compression can return.

Environment capture now records logical CPUs, scheduler affinity, cpuset,
`cpu.max` and its CPU-equivalents, GPU UUIDs and driver, and the shuffle seed.
This host: 128 logical CPUs, **27.2 CPU-equivalents of quota** — the number
that makes the earlier rank sweep interpretable, and which no previous run
stored.

### Steps 3 and 7: the lane model, and three-session replication

**Lane utilisation drives the kernel, and the earlier explanation of it did
not.** Two fixtures were built to differ in the warp tail and nothing else:
rows paired to need the same number of warp rounds (64/33, 96/65, 128/97),
matched on pair count, binary-search depth, hit-rate decile and emission order,
sharing the source CSR (`tools/make_lane_bands.py`).

| band | pairs | lane-slots | effective elements | utilisation |
| --- | --- | --- | --- | --- |
| `lane_full` | 55,002 | 4,987,168 | 4,940,039 | **99.05%** |
| `lane_tail` | 55,002 | 4,987,168 | 3,334,396 | **66.86%** |

Identical lane-slots, 32% less real work in the tail band. A balanced
crossover (15 blocks each direction, 30 total, unprofiled):

| | predicted | measured |
| --- | --- | --- |
| if time tracks **lane-slots** | 0% | |
| if time tracks **useful elements** | −32.5% | |
| **observed, tail vs full** | | **−2.08%**, 95% CI [−3.25%, −0.90%] |

Doing a third less useful work at the same lane-slot count buys 2%. **Kernel
time mostly tracks lane-slots** — the model predicts 0% and the CI excludes it,
so this bounds the useful-work model out, not a perfect fit — and the tail is
therefore real cost that warp packing targets — with a ceiling of **14.42%** of scan time at one GPU (85.58%
utilisation on `item_full`) and 25.32% at two (74.68%; splitting the dimension
shortens every slice and raises total lane-slots from 126M to 144M). Packing is
**not implemented**: that ceiling is the prize, and the kernel rewrite is
scoped as separate work rather than folded in here.

Nsight Compute could not corroborate it. `ERR_NVGPUCTRPERM` — performance
counters are disabled at the driver level on these pods and `cap_sys_admin`
inside the container is not enough. The crossover is the stronger evidence
anyway: it is a behavioural test of the model rather than a proxy metric.

**The compression effect replicates across independent sessions.** Three
separate pods, three pre-recorded seeds, balanced crossover with the estimator
`log(T_packed / T_f64)` per block:

| session | seed | `f64` | `packed` | compression | 95% CI |
| --- | --- | --- | --- | --- | --- |
| s1 | 101 | 4.2120 ms | 4.0125 ms | **−4.81%** | [−5.03%, −4.59%] |
| s2 | 202 | 4.1760 ms | 3.9670 ms | **−4.99%** | [−5.15%, −4.83%] |
| s3 | 303 | 4.1810 ms | 3.9700 ms | **−5.08%** | [−5.23%, −4.93%] |

Three replication units, not 90 samples. They span **0.27 percentage points**,
and the randomised headline matrix's −5.11% sits in the same range. The
compression number is reproducible.

The async noise reproduces too: 16.3%, 16.4%, 15.6% IQR on `async packed`
across the three sessions. It **persists under randomised interleaving and
across these hosts**, so it is not an artefact of one session; its magnitude
remains protocol- and host-dependent, and is not established as a property of
the pipeline as such.

**The nccl-tests baseline, redone properly** — exact byte counts, matching
dtypes, in-place column (the engine's collective is `ncclAllReduce(slot, slot,
...)`), raw command and stdout in `results/nccl_tests/`:

| payload | bytes | dtype | engine | nccl-tests (in-place) | ratio |
| --- | --- | --- | --- | --- | --- |
| `packed` | 18,749,712 | `uint64` | 250.0 µs | 157.13 µs | **1.59x** |
| `i32` | 28,124,568 | `int32` | 305.0 µs | 218.10 µs | **1.40x** |
| `f64` | 56,249,136 | `double` | 459.5 µs | 378.28 µs | **1.21x** |

Worse than the 1.45/1.31/1.17 first published, which had swept 16/32/64 MiB
and read the out-of-place column. The gap still widens as the payload shrinks.

### Steps 5 and 6: the third regime, and the MPI confound separated

**The `PHB` / no-P2P host was found and measured.** Probe-then-decide on the
inventory: create, read `nvidia-smi topo`, keep only on an exact hit. The first
attempt in CA-MTL-3 SECURE returned `PHB` with `P2P = NS` on an EPYC 7763 —
the same class as the archived pod. Full randomised matrix on the
timing-correct build, 30 rounds:

| | NVLink `NV12` | PCIe `SYS` + P2P | PCIe `PHB`, no P2P |
| --- | --- | --- | --- |
| `sync f64` | 4.191 ms | 13.101 ms | **48.719 ms** |
| `sync packed` | 3.977 ms | 10.020 ms | **21.723 ms** |
| **compression, paired** | **−4.96%** (3 sessions) | — | **−51.67%** [−54.61%, −48.55%] |
| AllReduce vs nccl-tests | 1.59x | — | **2.48x** |

All three regimes share a **timing contract and an intended synced working
tree**. Identical source or build artefacts are *not* independently evidenced:
the NVLink matrix predates the provenance fields and carries no `git_sha` at
all, the PHB run records the public clone's SHA with `git_dirty=true` because
the engine was rsynced over it, no run records an `image_digest`, and no dirty
patch or binary hash was kept. The compression trend holds across a 10x span in
communication share, and the third point is a paired estimate with a CI rather
than a single 5-trial median.

The distance from the library's own ceiling *widens* as the link slows and the
payload shrinks: 1.35x at `f64` on this host, 2.48x at `packed`. On a
host-staged link `all_reduce_perf` moves 18.75 MB in 7.47 ms; this engine
takes 18.56 ms for the same bytes. That gap is the clearest remaining
optimisation target in the project, and it is not one compression can close.

**Read as regimes, not as a controlled interconnect sweep.** Three different
physical hosts with different CPUs, topologies and P2P capability. Same code,
same protocol, descriptively comparable — but the differences cannot be
attributed causally to the interconnect alone.

**The MPI turnover had two candidate causes and they are now separated.** The
16-rank peak sat on a 27.2-CPU-equivalent quota while the AllReduce share
climbed 15.8% → 51% over the same range. Running the identical sweep on a
fixture with 6% of the payload, on the same machine in the same session:

| fixture | AllReduce payload | peak speedup | at ranks |
| --- | --- | --- | --- |
| `item_full` | 56.2 MB | 4.40x | **16** |
| `item_full_ovl_skew` | 3.35 MB | **7.11x** | **24** |

A 16.8x smaller payload raises the peak 62% and moves the turnover from 16 to
24. Pure quota starvation would have peaked at the same rank in both, so **the
collective contributes materially to the turnover**. It is not a clean
isolation: the smaller fixture also changes pair count, per-rank compute and
cache behaviour, so this bounds the quota-only explanation out rather than
attributing the turnover to the collective alone.

An unrestricted host was not found. RunPod's CPU pods cap at 32 vCPU across
the CPU3/CPU5 families, and the GPU hosts expose 128 physical cores (2 sockets
x 64, no SMT, 2 NUMA nodes) behind a 27.2-equivalent quota. Scoped as: not
found in this time window, this region, and the inventory checked — not a
claim about the platform.

### Still open

*Scope: these are open **for the host class and protocol used in this
document** — 2x A100-SXM4-80GB, 5- or 30-trial runs on the pre-warp-packing
default. Several were closed by later work; where that is so, it says which.*

Rewritten 2026-09-05 after the baseline campaign closed. Entries removed
because they are done: the `PHB` host, the nccl-tests redo, the cross-session
replication, and the finalize labelling. The async-spread entry is gone for a
different reason: the IQR reproduced across three independent pods (16.3% /
15.6% / 16.4% here; 10.9–11.2% in later runs on the current default), so it is
**reproducible under this protocol and host class**, not an open question. It
is not established as a property of the pipeline as such: the magnitude is
protocol- and host-dependent, and the later figure is the one to quote.

- **Warp packing is implemented and measured** (2026-09-05, after this audit
  was written): see
  [docs/warp_packing_experiment_20260905.md](warp_packing_experiment_20260905.md).
  It works, and **the lane-slot model in this document is not why**. The prize
  quoted here — 14.42% at one GPU, 25.32% at two, 19.22% after the shared-order
  constraint — was directionally right and quantitatively short: measured
  critical lane slots fell 18.3% at two GPUs, close to the model, while stats
  time fell far more. The `lane_full` fixture settles it — its lane slots vary
  by 0.02% across group sizes and its stats time still drops 13.5%.
  A replacement hypothesis (four redundant `lower_bound` searches per thread)
  was implemented as a hoist and **measured at +0.15% [−0.08, +0.38] — no
  effect**, so it is not that either. A cost proportional to the thread count
  is real and remains unattributed. The numbers in this section are what the
  model predicted, not what the hardware did.
- **The AllReduce is 1.2–2.5x off the library's own ceiling**, and the gap
  widens as the link slows and the payload shrinks — 2.48x at `packed` on the
  host-staged link, where `all_reduce_perf` moves the same 18.75 MB in 7.47 ms
  against this engine's 18.56 ms. That is a larger prize than warp packing and
  nothing in this project has looked at it.
- **Provenance is not closed *for the three-regime experiments in this
  document*.** The NVLink headline matrix here predates the provenance fields
  (no `git_sha`); the PHB run records the public clone's SHA with
  `git_dirty=true` because the engine tree was rsynced over it; and none of
  these runs records an `image_digest`. That limitation is specific to this
  campaign — later work records `git_sha`, a tracked-vs-untracked dirty split
  and a real `image_digest`, and the headline was re-measured on a clean
  checkout. See the warp-packing document.
  The three replication sessions (`../results/bench/repro_s*.json`) carry seed,
  block count and results but no host, SHA, image or timestamp —
  `../engine/bench/repro_session.py` records those now, but s1–s3 predate it and
  their independence rests on this text rather than on their files.
- **No PHB traces.** The `PHB` session captured the matrix, the paired A/B,
  nccl-tests and the topology, but not the four Nsight traces. Nothing in this
  document depends on them; the async mechanism on a host-staged link is
  described from the earlier `SYS`+P2P captures, which is a different machine.
  Marked not-done rather than approximated.
- **`../results/nccl_tests/all_reduce_perf_phb.txt` is an excerpt**, not full
  stdout — the run was tailed. The NVLink file is complete.
- **MPI ranks beyond 32, unquota'd.** RunPod CPU pods cap at 32 vCPU across
  CPU3/CPU5; GPU hosts expose 128 physical cores behind a 27.2-equivalent
  quota. Not found in this window, region and inventory — not a claim about
  the platform. `item_full` also OOMs at 32 ranks, each rank holding its own
  copy of the fixture.
- **`cold_data_path_s` is not a CLI one-shot.** It covers entry to data-ready
  including `MPI_Init` and `ncclCommInitAll`; `mpirun`'s spawn and the CUDA
  driver's first touch precede any code that could time them.
