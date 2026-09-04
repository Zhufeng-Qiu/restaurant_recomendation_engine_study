# Analysis — mechanism behind the results

Companion to [README.md](../README.md), which reports *what* was measured.
This file is *how we know*: where each gain and each loss actually came from,
and which measurements rule out the alternative explanations.

Every number here comes from `results/bench/*.json` and from Nsight traces
read with `tools/nsys_overlap.py`. The traces themselves are not in the
repository — `nsys` captures the profiled process's environment, and these
were taken on hosts that had credentials in theirs.
Two hosts: **2x A100-SXM4-80GB (NV12 NVLink)** and **2x A100 80GB PCIe (PHB,
no P2P)**, both CUDA 12.8 / NCCL 2.25.1, `item_full` (1,171,857 pairs),
median of 5 trials after 2 warm-ups.

---

## Where the NVLink gain went

Measured on 2x A100 SXM (NV12), `item_full`, median of 5 trials after 2
warm-ups:

| `--payload` | AllReduce moves | AllReduce time | Total device time | Comm share |
| --- | --- | --- | --- | --- |
| `f64` | 56.25 MB | 0.471 ms | 4.396 ms | 10.7% |
| `i32` | 28.12 MB | 0.318 ms | 4.254 ms | 7.5% |
| `packed` | 18.75 MB | 0.246 ms | 4.205 ms | 5.9% |

**The 3x payload cut bought a 1.9x faster collective and 4.3% end to end.**
Two separate reasons for the gap. The collective scales sublinearly because
a ring AllReduce pays latency and per-launch cost that no amount of byte
reduction touches. And even a free collective could only have returned the
10.7% it occupied — Amdahl set the ceiling before the first byte was saved.

The single-GPU rows are the control that makes this readable: with no peer
to reduce against, AllReduce costs 0.004 ms at every payload and the totals
sit within 0.6% of each other (5.003 / 5.008 / 5.033 ms). So the packing
itself is free at this scale — `pack_six` in the stats kernel and
`unpack_six` in finalize cost nothing measurable — and the entire effect
above lives in the collective, exactly where the design put it.

Async did not benefit: 4.80 ms at `f64` but 5.28 / 5.19 ms at `i32` /
`packed`. Its cost is per-chunk launch overhead, which compression does not
address, so shrinking the payload only removes work that was already hidden.
Sync remains the A/B winner on NVLink, now for two independent reasons.

Read together with the async result, this is a coherent negative finding
rather than two disappointments: **on a fast interconnect this workload is
not communication-bound, so neither hiding communication nor shrinking it
moves the needle.** Both techniques target the same regime, and NVLink is
not it.

---

## Which regime the collective is in

The single most useful number in this study is not a latency — it is what
happens to *effective bandwidth* as the payload shrinks.

![NVLink vs PCIe payload comparison](../results/figures/interconnect_comparison.png)

Effective collective bandwidth *falls* on NVLink as the payload
shrinks — 119 → 88 → 76 GB/s — because at 0.47 ms the collective is
latency- and launch-bound and smaller messages cannot fill the link. On PCIe
it stays flat at 1.53 → 1.54 → 1.30 GB/s, because there the link is
genuinely saturated. A bandwidth-bound collective converts saved bytes into
saved time almost one-for-one (3x fewer bytes, 2.55x faster); a
latency-bound one does not (3x fewer bytes, 1.91x faster). That distinction,
not the interconnect's name, is what decides whether either technique is
worth its complexity.

---

## Why overlap is capped on *both* links

The two measurements together also correct this project's original
explanation of the async result. "Async targets slow interconnects" turns out
to be only half right. Overlap can hide at most the smaller of compute and
communication, so its ceiling is `min(compute, comm) / total`:

| Link | Payload | Compute | Comm | Overlap ceiling | Achieved | Binding side |
| --- | --- | --- | --- | --- | --- | --- |
| NVLink | `f64` | 3.92 ms | 0.47 ms | 10.7% | -9.3% | comm |
| NVLink | `packed` | 3.96 ms | 0.25 ms | 5.9% | -23.4% | comm |
| PCIe | `f64` | 3.58 ms | 36.74 ms | 8.9% | +4.3% | compute |
| PCIe | `packed` | 3.54 ms | 14.40 ms | 19.7% | **+11.3%** | compute |

The ceiling is low on both links, but for opposite reasons: on NVLink there
is almost no communication to hide, and on PCIe there is almost no
computation to hide it behind. Async wins only where that ceiling clears its
own per-chunk launch overhead — which it does on PCIe (capturing about half
the available headroom) and does not on NVLink, where it goes backwards.
**Overlap pays when compute and communication are comparable, not when
communication is large.**

That also explains why `async packed` is the fastest configuration measured
anywhere in this study. Compression pulls comm down from 36.7 ms toward the
3.5 ms of compute, which more than doubles the overlap ceiling (8.9% →
19.7%) — so the two optimisations are not independent. Compression moves the
workload into the balanced regime where overlap becomes useful. On NVLink the
same coupling runs the other way: compression pushes comm further below
compute, halving the ceiling (10.7% → 5.9%) and making async worse still.

## What the async result actually consists of (Nsight, both hosts)

Wall-clock timings say async is worth +11.3% at `packed`; they cannot say
whether that is good overlap of a small quantity or poor overlap of a large
one. Nsight Systems traces answer it. `tools/nsys_overlap.py` classifies every
device kernel as communication (`ncclDevKernel_AllReduce_*`) or computation
(`pair_stats_kernel_*`), then measures, per GPU, how much communication time
has a compute kernel genuinely running inside it.

| Trace | Comm | Compute | Overlapped | Share of compute hidden |
| --- | --- | --- | --- | --- |
| `sync f64` / `sync packed` | 35.9 / 18.4 ms | 2.8 ms | **0.000 ms** | 0% |
| `async f64`, GPU0 | 34.07 ms | 12.55 ms | 11.28 ms | **89.9%** |
| `async f64`, GPU1 | 33.99 ms | 11.45 ms | 10.25 ms | **89.6%** |
| `async packed`, GPU0 | 12.75 ms | 5.46 ms | 3.54 ms | 64.9% |
| `async packed`, GPU1 | 13.88 ms | 7.10 ms | 6.49 ms | **91.5%** |

Sync measures exactly 0.000 ms of overlap on every GPU and payload, which is
the control that makes the rest credible: the method finds no overlap where
the design says there is none. Async then hides close to 90% of the compute
it could possibly hide. **The pipeline is not the weak part** — the modest
end-to-end gain is because compute was only ~9% of the iteration to start
with, exactly the ceiling computed above. (GPU0 sits lower at `packed`
because it also runs the finalize kernel on its communication stream.)

The same analysis run on the archived NVLink traces (GPU0, warm-up excluded)
shows sync at exactly 0.000 ms there too, and explains the NVLink penalty —
with a mechanism that is **not** the one this README previously inferred:

| NVLink, GPU0 | Collective | Compute | Overlapped |
| --- | --- | --- | --- |
| `sync` (1 fused collective) | 0.371 ms | 3.425 ms | 0.000 ms |
| `async` (18 chunks) | **2.428 ms** | 3.873 ms | 0.830 ms (34.2% of comm) |

Chunking makes the *collective itself* 6.5x more expensive on NVLink, because
18 small collectives pay 18 latencies on a link whose transfer time is already
negligible. Concurrency costs only 1.13x on the compute side there. On PCIe
the two effects are exactly reversed: chunking leaves the collective
essentially unchanged (35.96 → 34.07 ms, since a bandwidth-bound link cares
only about total bytes), while concurrency inflates compute 4.47x.

**So async loses on NVLink and wins on PCIe for two different measured
reasons, not one.** The penalty on a latency-bound link is paid in the
collective; the penalty on a bandwidth-bound link is paid in the compute
kernel, where it can be hidden. This also checks out against wall time on
PCIe: 1.89 ms saved on communication plus 1.54 ms of compute made invisible
predicts 3.43 ms, and the traces show 3.49 ms.

One measurement gap: the archived NVLink async trace was captured at 18
chunks, while the benchmarked async row uses the default 5. It therefore
demonstrates the mechanism but overstates its magnitude for that row (roughly
1.8x rather than 6.5x, scaling with chunk count). The PCIe traces are
configuration-matched to their benchmark rows at 5 chunks.

Now the PCIe contention cost in detail, comparing the aggregate time of the
statistics kernel on GPU0:

| Configuration | Launches | Stats-kernel time | vs its sync baseline |
| --- | --- | --- | --- |
| sync, **1 GPU** | 1 | 3.723 ms | — |
| async, **1 GPU** | 5 | 3.772 ms | **+1.3%** |
| sync, 2 GPU `packed` | 1 | 2.834 ms | — |
| async, 2 GPU `packed` | 5 | 5.457 ms | **1.93x** |
| sync, 2 GPU `f64` | 1 | 2.806 ms | — |
| async, 2 GPU `f64` | 5 | 12.549 ms | **4.47x** |

On one GPU, where the collective is a no-op, splitting the work into five
chunked launches costs 1.3% — so chunking itself is free and is not the
explanation. On two GPUs the same kernel takes 1.9x to 4.5x longer, and the
inflation grows with the volume communicated (`f64` moves 3x the bytes of
`packed` and suffers 2.3x the inflation). That is the signature of SM
contention: NCCL's persistent `RING_LL` kernels hold streaming multiprocessors
while they wait on data, so a concurrent compute kernel gets fewer of them.

So async's balance sheet is *compute hidden* minus *chunking paid*, and which
term dominates is a property of the link, not of the code. On PCIe the
chunking is free and the hidden compute is real profit. On NVLink the
chunking is the whole story: there is only 0.47 ms of communication to hide
behind, and splitting it into chunks costs more than the hiding returns.
Consistent with this, NVLink's async penalty *worsens* under compression
(-9.3% → -23.4%) — compression shrinks the communication that overlap needs
while leaving the per-chunk latency count untouched. That last step is an
inference: no `packed` NVLink trace exists, because the payload flag postdates
those captures.

A methodological caveat that applies to both hosts: an NCCL kernel's duration
includes spin-waiting on peers, so "comm" here is kernel occupancy, not pure
transfer. It matters on NVLink, where real transfer is sub-millisecond — GPU1
reports 7.37 ms of collective against GPU0's 2.45 ms in the same run, the
difference being wait — and much less on PCIe, where the two GPUs agree to
within 0.3% because genuine transfer dominates. Per-GPU figures are reported
rather than averaged for this reason.

Trace timings carry profiling overhead and come from single runs, so they are
used here only to attribute mechanism; every speedup number in this README
comes from the benchmark medians.

## Caveats on scope

P2P is unavailable on the PCIe host, so the 78x AllReduce gap between the two
machines (0.47 vs 36.74 ms) measures *PCIe with host staging* — a real
multi-tenant configuration, but slower than direct PCIe P2P. A P2P-capable
host would sit somewhere between the two.

The PCIe pod's kernel is also 14% faster than the SXM pod's (4.31 vs 5.00 ms
for identical code), so absolute totals are not comparable across machines.
Every ratio in this document and in the README is therefore computed *within*
a single machine, against a baseline measured in the same session.

## Sep. 4th 2026 - Update

Re-measured on fresh pods at 30 trials, with configuration-matched traces.
Two things above are corrected by measurement; the mechanism findings hold.

### The cross-host caveat is the binding one

The README claimed "Same binary, same data, same GPU model. Only the
interconnect changed." The *Caveats on scope* paragraph above is the correct
account. The pods differ in CPU (EPYC 7742 vs 7763) and in direction: the
PCIe pod's GPU path is 13.8% faster while its `serial` is 1.1% slower. Two
opposite signs cannot both be the link. Every ratio here was already
within-host, so nothing below changes.

### A third regime: PCIe with P2P

The missing middle point now exists. A `SYS`-topology A100 pair with P2P
available sits between NVLink and the host-staged `PHB` pod:

| | NVLink `NV12` | PCIe `SYS`, P2P | PCIe `PHB`, no P2P |
| --- | --- | --- | --- |
| AllReduce, `f64` sync 2 GPU | 0.479 ms | 9.958 ms | 36.743 ms |
| communication share | 11.6% | 76.0% | 91.1% |
| `f64` → `packed` | −4.7% | −23.5% | −55.5% |
| sync → async at `f64` | +19.3% | +9.7% | −4.3% |

Compression tracks communication share monotonically over all three points,
which is the strongest form the central claim has been in.

Overlap does not. It still loses at 76% communication even though the ceiling
there — `min(compute, comm)/total` = 3.1/13.1 — is **23.7%**, nearly three
times the 8.9% ceiling on the host-staged link where it wins. So the ceiling
does not predict the sign, and the account above ("overlap pays when compute
and communication are comparable") is necessary but not sufficient: on the
P2P link the per-chunk cost still exceeds the hiding. The crossing point lies
between 76% and 91% communication and is not localised, because no trace was
obtained on the P2P host.

### Chunking cost, measured at the benchmarked chunk count

The 18-chunk NVLink capture is superseded. All four configurations were
recaptured at the default chunk count (5 chunks for `item_full`), GPU0,
warm-up excluded:

| trace | collective | compute | overlapped |
| --- | --- | --- | --- |
| `sync f64` | 0.430 ms | 3.430 ms | **0.000 ms** |
| `sync packed` | 0.197 ms | 3.461 ms | **0.000 ms** |
| `async f64` | 1.468 ms (3.41x) | 3.531 ms (1.03x) | 0.161 ms |
| `async packed` | 1.087 ms (5.52x) | 3.685 ms (1.06x) | 0.769 ms |

Sync measures exactly 0.000 ms overlap on new hardware — the control that
makes the rest readable still holds. Chunking inflates the collective
**3.41x** at `f64`, against the ~1.8x this document extrapolated from the
18-chunk trace: the extrapolation understated it roughly twofold, and the
direct measurement replaces it. Compute inflation stays negligible
(1.03–1.06x), confirming the NVLink penalty is paid entirely on the
collective side.

Per-GPU, the spin-wait asymmetry noted above persists and is larger in async:
GPU1 reports 2.980 ms of collective against GPU0's 1.453 ms in the same
`async f64` run, so per-GPU figures remain the honest unit.

### One inference did not survive

This document inferred that NVLink's async penalty *worsens* under
compression (−9.3% → −23.4%), flagged as an inference because no `packed`
NVLink trace existed. That trace now exists. At 30 trials the penalty is
+19.3% at `f64` and +9.6% at `packed` — the opposite ordering. But those rows
carry 13–18% IQR, so neither ordering is established. The supportable claim
is the weaker one: **async loses on NVLink at every payload.**

### Measurement noise is concentrated in the 2-GPU rows

IQR as a fraction of median, 30 trials:

| | 1 GPU | sync 2 GPU | async 2 GPU |
| --- | --- | --- | --- |
| NVLink | 0.24–0.88% | 0.58–1.10% | **13.4–17.6%** |
| PCIe P2P | 0.34–4.15% | 5.11–5.42% | **6.5–13.0%** |

Async-2-GPU spread is 15–70x the single-GPU spread. This is why the archived
5-trial `nccl_async_g2_packed` and the 30-trial value differ by 17%, and it
is a reason to read every async median with its IQR attached. Trace timings
remain single-run and are used only to attribute mechanism.

### Credential hazard, restated

`nsys` again captured a live `RUNPOD_API_KEY` even though the launching shell
unset it: Runpod injects the key into the container's init environment and
`nsys` reads the target's environment from `/proc`, below the shell. Shell-level
unsetting is not a control. Verify with `strings <trace> | grep rpa_` before
sharing any capture; traces stay out of this repository.
