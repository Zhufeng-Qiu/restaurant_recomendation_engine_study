# Warp packing: implementation and measurement — 20260905

The one optimisation [the measurement audit](measurement_audit_20260905.md)
left unimplemented. It works — and it works for a reason the audit's own model
did not predict, which is the more useful half of this document.

**One-line answer.** Giving a pair four lanes instead of thirty-two, with pairs
sorted by shorter-slice length, cuts single-GPU `device_total` by
**−34.68% [−34.98, −34.37]** against the pre-packing binary and two-GPU NCCL
`packed` by **−47.62% [−47.99, −47.24]**. The lane-tail waste the audit
predicted recovering accounts for well under half of that; the rest tracks the
*thread count*, a cost neither model counted.

Two limits on that sentence, both load-bearing. It is **steady state only** —
on the cold path packing loses, because building the sorted plan costs about
thirty times the kernel it accelerates (§7). And this is **warp packing v1**:
`G=4` is optimal for a kernel that still repeats its slice-bound search in
every lane, and would very likely move if that search were hoisted (§6).

Everything below is from one session on an EPYC 7742 with 2x A100-SXM4-80GB
(NV12): nvcc 12.8.93, NCCL 2.25.1, driver 580.126.16, `sm_80`,
`CMAKE_BUILD_TYPE=Release`. Raw per-trial data, run order and seeds are in
`results/bench/warp_packing_*`.

**Provenance, stated exactly.** Engine source at `ccbfe4e`; the legacy baseline
binary from `df29ca1`, built in a detached worktree. The runs record
`git_dirty=true`: the session script `engine/bench/gpu_session_warp_packing.sh`
was patched twice mid-session (a `GROUPS` name collision with the bash builtin,
and the sanitizer classification in §2). No file under `engine/src` was touched
after `ccbfe4e`, so the measured binaries correspond to that commit — but that
is a statement about what was edited, not something the records prove on their
own. `image_digest` is null, as it is for every result in this repository.

---

## 1. What was built

`pair_stats_kernel` is templated on a group size `G ∈ {1,2,4,8,16,32}`. `G`
lanes cooperate on one pair, a warp carries `32/G` pairs, and the pair each
group works on comes from a plan array. `--group` and `--pair-order` select the
mapping on both the CUDA and NCCL binaries; the defaults (`32`, `source`) are
the pre-existing mapping.

Two structural consequences:

* **Statistics are written at the slot index, not the pair index**, so the
  collective payload stays contiguous and chunkable exactly as before. The
  permutation is undone in `finalize`, which scatters to `sims[order[slot]]`.
* **Every GPU must be handed the same order**, or the AllReduce would sum slots
  standing for different pairs. So the order is built over the full dimension
  range — nobody's own optimum. Each device then runs it against its own
  `[dim_lo, dim_hi)`, and the JSON reports that per device rather than
  reporting the full-range figure as if a GPU had executed it.

## 2. Correctness, before any timing

The ladder runs first and the session aborts if it fails
(`results/bench/warp_packing_correctness_20260905_210121.txt`).

| check | count | result |
| --- | --- | --- |
| CUDA validate: 5 fixtures x 6 groups x 2 orders | 60 | `tol_failures=0`, `max_abs_diff=0.000e+00` |
| NCCL sync: 1 and 2 GPUs, f64/i32/packed, all groups and orders | 88 | `tol_failures=0` |
| Async: chunks 16384 / 262144 / 100003 x `comm`/`separate` x 2 mappings | 12 | `tol_failures=0` |
| Output byte-compared against the `g32/source` reference | 12 | 12 identical |
| CLI rejections (bad group, order, mode, gpus, chunk, unknown flag, missing value, sync+separate) | 11 | 11 rejected, 0 accepted |
| `compute-sanitizer memcheck`, CUDA, every group and order | 12 | no errors |
| `compute-sanitizer memcheck`, NCCL 2 GPU, every group | 6 | no **memory** errors |

`100003` is deliberately not a divisor of the pair count, and every group size
leaves a trailing partial warp.

Two things worth stating plainly rather than burying:

**The NCCL sanitizer runs report 94 "errors" each, and they are not ours.**
Every one is `cudaErrorPeerAccessAlreadyEnabled` (704), raised inside
`libnccl.so.2`'s own bootstrap threads on a `cudaGetLastError` call and
swallowed there. The count is *identical* — 94 — at `G=32` with the source
order, which is the mapping that predates packing entirely. The pass condition
is therefore the absence of memory errors, with the 704 count recorded rather
than hidden.

**`ncu` could not be used.** `ERR_NVGPUCTRPERM` on this host, as on every
previous pod in this project. No NCU number appears anywhere below, and none of
the timings come from a profiled run.

## 3. Register pressure and occupancy

Every `pair_stats_kernel<G>` instantiation, all three payload variants and all
six group sizes — twelve kernels — compiles to **37 registers, 0 bytes stack
frame, 0 spill stores, 0 spill loads** (`-Xptxas=-v`, `sm_80`, separate build;
`results/bench/warp_packing_build_20260905_210121.txt`).

What that supports, exactly: **register allocation and spilling do not explain
the differences between group sizes.** The count is identical across all of
them and nothing spills. It does *not* support "the accumulators cost nothing"
— 37 registers per thread can still cap occupancy below 100%, and **absolute
occupancy was not measured**: `ncu` is unavailable here. The binaries now emit
a theoretical figure from `cudaFuncGetAttributes` and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` (active blocks and warps per
SM, the limiting resource), which is an upper bound on the achieved value, not
a measurement of it. That field postdates the runs below.

## 4. Screening

All arms, 10 trials each after 3 warm-ups, round-robin with a fresh shuffle
every round, seed 20260905. `vs base` is against `cuda:32:source`, the same
tree's unpacked mapping.

**`item_full`, single GPU** (`device_total`, median ms):

| arm | median | IQR% | stats | vs base |
| --- | --- | --- | --- | --- |
| `cuda:4:bylen` | **3.099** | 1.45 | 3.056 | **−36.6%** |
| `cuda:8:bylen` | 3.190 | 0.98 | 3.138 | −34.8% |
| `cuda:2:bylen` | 3.456 | 1.29 | 3.366 | −29.3% |
| `cuda:16:bylen` | 3.681 | 1.03 | 3.586 | −24.7% |
| `cuda:8:source` | 3.964 | 0.79 | 3.904 | −18.9% |
| `cuda:4:source` | 4.103 | 0.88 | 4.087 | −16.1% |
| `legacy` | 4.723 | 0.25 | 4.664 | −3.4% |
| `cuda:32:bylen` | 4.852 | 0.56 | 4.768 | −0.8% |
| `cuda:32:source` | 4.891 | 0.40 | 4.829 | 0.0% |
| `cuda:1:source` | 5.152 | 1.28 | 5.132 | +5.3% |
| `cuda:1:bylen` | 6.877 | 1.68 | 6.664 | +40.6% |

Three things fall out immediately, and two of them contradict the model.

**`G=1` is the worst arm, not the best.** The lane-slot model says `G=1` reaches
99.96% utilisation and should win by 14.39%. It loses by 40.6%. The likely
reason is that a lane handling a whole pair alone no longer shares a row with
its neighbours, so the coalesced reads that 32 lanes striding one row were
getting are lost — likely, not shown; see §6.

**Sorting alone buys almost nothing.** `cuda:32:bylen` vs `cuda:32:source` is
−0.8%. At `G=32` there is one pair per warp, so ordering cannot affect lockstep;
it can only affect locality, and locality turns out to be worth under a percent.
Sorting matters *because it makes small `G` viable* — at `G=4` it is worth
−24.4% (`4:source` 4.103 → `4:bylen` 3.099) — not on its own.

**Packing helps even without sorting.** `cuda:8:source` beats `cuda:32:source`
by 18.9%. The model predicts unsorted packing should be catastrophically
*worse*. Section 6 explains why it is not.

**Other fixtures** (medians, ms):

| arm | `user_full` | `lane_full` | `lane_tail` |
| --- | --- | --- | --- |
| `cuda:32:source` | 3.750 | 0.4455 | 0.4390 |
| `legacy` | 3.557 | 0.4015 | 0.3900 |
| `cuda:8:bylen` | 2.016 | **0.3865** | **0.3565** |
| `cuda:4:bylen` | **1.847** | 0.3980 | 0.3625 |
| `cuda:1:bylen` | 2.900 | 0.5735 | 0.4950 |
| *best vs base* | *−50.8%* | *−13.2%* | *−18.8%* |
| *`cuda:4:bylen` vs legacy* | *−48.1%* | *−0.9%* | *−7.1%* |

The best group size is 4 on both real fixtures and 8 on both synthetic lane
bands, and the two are within 3.4% of each other on `item_full`. Note the last
row: on `lane_full` — short rows, no tail — packing barely beats the
pre-packing binary at all.

## 5. Paired A/B

30 balanced crossover blocks each — 15 baseline-first, 15 candidate-first,
block order shuffled from a recorded seed. Estimator is the mean of
`log(candidate/baseline)` over blocks; the CI is `1.96 x SE` over the 30 block
log-ratios and describes this session.

| candidate | baseline | effect | 95% CI |
| --- | --- | --- | --- |
| `cuda:4:bylen` | `cuda:32:source` | **−36.78%** | [−37.08, −36.47] |
| `cuda:4:bylen` | `legacy` | **−34.68%** | [−34.98, −34.37] |
| `cuda:32:source` | `legacy` | **+3.28%** | [+2.86, +3.71] |
| `cuda:4:bylen` | `cuda:8:bylen` | −3.35% | [−3.59, −3.11] |
| `nccl2:4:bylen:packed` | `nccl2:32:source:packed` | **−47.62%** | [−47.99, −47.24] |
| `nccl1:4:bylen:packed` | `nccl1:32:source:packed` | −35.41% | [−36.88, −33.90] |

Every interval excludes zero.

**The third row is the one that would have been easy not to publish.** The
instrumented tree at its default mapping is **3.28% slower than the binary that
predates packing**, because every pair now goes through `order[slot]` and
finalize scatters instead of writing straight to `sims[k]`. That is the cost of
the plumbing, and it is charged whether or not packing is switched on. It is
above the 1% threshold set before the run. Section 8 says what follows from it.

**The collective is untouched, as it should be.** Across every arm of the
two-GPU screen, AllReduce sits at 0.229–0.251 ms with no trend in `G`; packing
changes lane assignment, not payload bytes. **Finalize regresses**, 0.049 →
0.073 ms — the scatter — and that is a real stage regression, not a rounding
artefact; it is simply small against the stats collapse of 3.784 → 1.847 ms,
which is where the entire two-GPU gain lives.

## 6. The model that predicted it does not explain it

The audit's model says time tracks **lane slots**: `32 * ceil(L/G)` per pair,
with the partly-filled last round as the waste. On that model the two-GPU prize
was 25.32%, and 19.22% after the shared-order constraint.

Measured, two-GPU critical lane slots fall 72,771,808 → 59,488,032, which is
**−18.3%** — almost exactly what the model promised. Measured stats time falls
**−51.2%**. The model is directionally right and quantitatively nowhere near.

`lane_full` settles it. That fixture was built for the audit's lane-band
experiment so that its warps are essentially full, and its lane slots barely
move across group sizes:

| arm | lane slots | stats ms |
| --- | --- | --- |
| `cuda:32:source` | 4,987,168 | 0.4230 |
| `cuda:16:bylen` | 4,987,200 | 0.3720 |
| `cuda:8:bylen` | 4,987,520 | 0.3660 |
| `cuda:4:bylen` | 4,988,160 | 0.3780 |

The slot count varies by 0.02% — there is no tail to recover — and stats time
still drops 13.5%. Whatever is being saved, it is not lane-tail waste.

**Leading mechanism hypothesis.** `pair_lane_stats` opens with **four `lower_bound`
searches per thread**, to locate each row's slice within `[dim_lo, dim_hi)`.
Every lane of a group repeats them for the same pair. That cost is proportional
to the number of *threads*, i.e. to `n_pairs * G` — and it is invisible to a
model that counts only the scan loop. Adding one term for it:

    H0:  t_stats = b * lane_slots                (the published model)
    H1:  t_stats = a * G  +  b * lane_slots      (plus redundant per-thread setup)

Fitted over the sorted arms with `G >= 2`:

| fixture | H0 R² | H1 R² |
| --- | --- | --- |
| `item_full` | 0.517 | **0.926** |
| `user_full` | 0.732 | **0.976** |
| `lane_tail` | 0.526 | **0.885** |
| `lane_full` | −0.022 | **0.330** |

`lane_full` is the case where H0 has no explanatory power at all — it predicts a
constant — and where H1 still recovers a third of the variance from the `G`
term alone.

**What the fit does and does not establish.** It establishes that a cost
proportional to the *thread count* exists and is large. It does **not**
establish that the four `lower_bound` searches are that cost: any other
per-thread setup — the launch itself, the slot and lane arithmetic, the
zeroing of six accumulators, the `order[slot]` load — lands in the same `a*G`
term. The searches are the largest identifiable candidate, not a demonstrated
cause. The A/B that would settle it is hoisting them (below); until that runs,
this is a hypothesis with supporting evidence, not a finding.

This is also consistent with the two screening surprises. Unsorted packing
helps because a per-thread saving does not depend on the sort. And `G=1` is a
separate regime: it sits 28–56% *above* H1's prediction on every fixture, which
is **consistent with a loss of coalescing** when each lane walks its own row —
consistent with, not demonstrated; that too would need a memory-transaction
count, and `ncu` is unavailable.

So the mechanism ledger is: a large term proportional to `G` (redundant slice
searches), a smaller genuine lane-tail term (visible as `lane_tail` gaining
18.9% in stats against `lane_full`'s 13.5%, at the same `G=8`), and a
memory-behaviour penalty that ends the descent at `G=4`. `G=4` is where those three meet on this hardware and this
workload; nothing here says 4 is universal.

**The obvious follow-up, now quantified.** If the dominant cost is four
redundant searches per thread, compute the slice bounds *once per group* and
broadcast them with a shuffle. That is a contained change to
`subwarp_pair_stats`, and this data says it is worth more than the packing that
found it. It is not implemented here: it is a different optimisation, and
folding it in would have meant re-running the whole ladder and every A/B above.

## 7. Cold path — provisional, and currently mismeasured

Packing is a **steady-state** optimisation, and this document originally
reported only steady state. The cold path points the other way, and the
numbers below are provisional for two independent reasons.

Under the timing in force during these runs, on `item_full`:

| arm | `device_total` | `t_plan` | `cold_data_path` |
| --- | --- | --- | --- |
| `cuda:32:source` | 4.891 ms | 50.28 ms | 77.72 ms |
| `cuda:4:bylen` | 3.099 ms | 102.89 ms | 128.44 ms |

Building the plan costs roughly thirty times the kernel it accelerates, so a
caller that runs the binary once is worse off, not better.

**Why these are not publishable numbers yet.**

1. **The `t_plan_s` they were measured under was wrong.** `plan_pairs`
   unconditionally computed shorter-slice lengths and lane-slot metrics, even
   for `--pair-order source`, which needs neither. That descriptive pass is
   most of the 50.28 ms on the baseline arm, and it sat inside
   `cold_data_path_s` in **both** backends — NCCL's separately-timed
   `t_plan_metrics_s` covered only the per-device diagnostics added later, not
   the full-dimension metrics. Required planning and diagnostics are now split
   in the common planner, so the baseline's cold path should fall sharply and
   the candidate's relative disadvantage should get **worse**, not better.
2. **They are single samples.** The harness kept only the first record per
   arm, so every `t_plan` and `cold_data_path` figure above is one observation
   with no median, IQR or interval, while the `device_total` columns elsewhere
   are medians of ten. The harness now stores all cold-path fields per trial
   and the paired runner reports a per-stage effect with a CI.

**No break-even figure is stated here.** The arithmetic on the numbers above
gives about 28 queries, and that is the correct arithmetic on the wrong
baseline: it compares against the instrumented arm, which itself carries both
the 3.28% plumbing regression and ~50 ms of diagnostics. The product
comparison is against the pre-packing binary, whose cold path this schema never
recorded. A rough re-estimate lands nearer 56 queries against a corrected
instrumented baseline and higher still against legacy, but every input to that
is being re-measured, so the honest statement is: **packing loses on the cold
path, by a margin yet to be measured.**

## 8. What the per-device reporting changed

The full-dimension plan's utilisation is not what a GPU executes, and on
`item_tiny` at 2 GPUs the gap is plain: `global_full_dimension_lane_utilisation`
0.94344 against per-device 0.92569 and 0.92827. Splitting the dimensions
shortens every row and gives each device its own tail, so the shared full-range
figure flatters the machine. Both are now reported, under names that cannot be
confused, along with each device's `counterfactual_per_device_sorted_*` —
what that slice would cost if it could sort its own pairs, which it cannot
while the collective is elementwise.

## 9. Decision

Against the gates fixed before the numbers were seen:

* **Correctness gate — passed.** Every check in §2, no memory errors, all three
  payloads bit-exact, output order preserved.
* **Effect gate — passed, far above the 3% bar.** `cuda:4:bylen` beats the
  pre-packing binary by 34.68% with the CI entirely on the favourable side, and
  beats it on every fixture tested.
* **Control gate — passed, with one honest caveat.** Every fixture improves
  against base: `user_full` −50.8%, `lane_tail` −18.8% (at `G=8`), `lane_full`
  −13.2% (at `G=8`), and at the proposed `G=4` default −10.7%. Measured against
  `legacy` rather than base, though, the lane bands are much thinner:
  `cuda:4:bylen` is −7.1% on `lane_tail` and only **−0.9%** on `lane_full`. On
  a fixture with no tail to recover and short rows, the 3.28% plumbing cost
  eats nearly all of the gain. That is consistent with §6 rather than
  unexplained, but it is the case where packing nearly fails to pay.
* **Two-GPU gate — passed on `device_total` (−47.62%), with one stage
  regressing.** AllReduce stayed stable (0.229–0.251 ms, no trend in `G`), but
  finalize rose 0.049 → 0.073 ms, about +49%, because it now scatters through
  `order`. The stats collapse from 3.784 to 1.847 ms covers it many times over,
  which is why `device_total` improves — but "no regression in any stage" would
  have been false, and this document said it.
* **Default-path gate — NOT passed as written.** The instrumented default is
  3.28% slower than legacy, above the 1% threshold.

**Recommendation: make `--group 4 --pair-order bylen` the default.** The
default-path regression is real but it is an argument about a mapping that
would no longer be the default; anyone on the new default is 34.68% faster than
legacy, and a separate legacy fast path would mean maintaining a second kernel
for a configuration that is 34.68% slower.

**The default has NOT been changed in this commit.** Changing it obliges
re-running the same-machine headline matrix and the NVLink `f64→packed`
compression A/B on the frozen SHA — compute makes up a larger share of a faster
kernel, so the compression percentage will move — and this session stopped at
the spending cap before that could be done. Flipping the default while the
README still quotes numbers from the old mapping would reintroduce exactly the
inconsistency the September audit spent its time removing.

## 10. Still open

* **Cold-path re-measurement.** Required, and blocking any break-even claim:
  the planner split changes `t_plan_s` for every arm.
* **The headline re-run.** Required before a new default is merged or
  published — the experiment branch sets the proposed default first, then a
  clean SHA is frozen and the headline re-run with the real default command. Same-machine
  matrix plus the compression paired experiment, on a frozen SHA.
* **Hoisting the slice-bound search** (§6). Larger prize than the packing that
  revealed it.
* **Async.** Correctness is verified at three chunk sizes and both finalize
  streams. Performance is deliberately *not* measured: chunks are contiguous
  slot runs, so an ascending-by-length order hands early chunks the short pairs
  and late chunks the long ones, changing pipeline balance for reasons that
  have nothing to do with lane packing. A chunk-balanced order is a separate
  piece of work.
* **`G=4` is not established as universal.** It won on two real fixtures; `G=8`
  won on both synthetic lane bands. A workload-aware choice, or simply leaving
  the flag exposed, may be better than a constant.
* **`ncu` remains unavailable** (`ERR_NVGPUCTRPERM`), so the coalescing
  explanation for `G=1` is inference from the fit's residuals, not a measured
  memory-transaction count.
* **One session.** Every number here comes from a single pod. The compression
  effect in the audit needed three sessions before it was worth trusting to
  0.3 percentage points; these effects are far larger than that spread, but
  they have not been replicated across hosts.
