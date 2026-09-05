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
them and nothing spills.

It does *not* support "the accumulators cost nothing", and the driver says so.
From `cudaFuncGetAttributes` and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on this host:

| | |
| --- | --- |
| registers / thread | 37 |
| local bytes / thread | 0 |
| block size | 128 (4 warps) |
| active blocks / SM | 12 |
| active warps / SM | 48 of 64 |
| **theoretical occupancy** | **0.75** |
| limiter | **registers** |

So the six double accumulators do cost something: they hold the kernel to 75%
of the machine's warp slots, and registers are the binding constraint (12
blocks fit; a 13th would need 66,560 of the SM's 65,536 registers once
allocation granularity is applied). What they do not do is vary with `G`, which
is the only thing the group-size comparison needed them not to do.

This is **theoretical** occupancy — an upper bound the driver will state
without any counters. Achieved occupancy is still unmeasured: `ncu` fails with
`ERR_NVGPUCTRPERM` on this host as on every previous one.

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

**Session 1** (pod `1d532d19`, GPUs `c3aab2f9` / `7ef491a4`):

| candidate | baseline | effect | 95% CI |
| --- | --- | --- | --- |
| `cuda:4:bylen` | `cuda:32:source` | **−36.78%** | [−37.08, −36.47] |
| `cuda:4:bylen` | `legacy` | **−34.68%** | [−34.98, −34.37] |
| `cuda:32:source` | `legacy` | **+3.28%** | [+2.86, +3.71] |
| `cuda:4:bylen` | `cuda:8:bylen` | −3.35% | [−3.59, −3.11] |
| `nccl2:4:bylen:packed` | `nccl2:32:source:packed` | **−47.62%** | [−47.99, −47.24] |
| `nccl1:4:bylen:packed` | `nccl1:32:source:packed` | −35.41% | [−36.88, −33.90] |

**Session 2**, a different pod (`0fd9…`, GPUs `5749bb50` / `7826451f`), after
the planner and harness fixes:

| candidate | baseline | effect | 95% CI |
| --- | --- | --- | --- |
| `cuda:4:bylen` | `cuda:32:source` | **−36.18%** | [−38.73, −33.52] |
| `cuda:4:bylen` | `legacy` | −33.20% | [−33.62, −32.77] |
| `cuda:4:source` | `cuda:32:source` | −17.59% | [−19.75, −15.38] |
| `cuda:4:source` | `legacy` | −11.43% | [−11.84, −11.02] |
| `cuda:4:bylen:hoist` | `cuda:4:bylen` | +0.15% | [−0.08, +0.38] |

Every interval excludes zero except the last, which is the point of it.

**The primary effect replicated across two independent pods** — −36.78% and
−36.18% for the same comparison, on different physical GPUs. That is one
crossing of the boundary the audit's compression result needed three of, and
it is worth exactly that much: the effect is not an artefact of one machine.

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

### The hypothesis, tested and rejected

If the dominant cost is four redundant searches per thread, computing the slice
bounds once per group and broadcasting them with a shuffle should recover most
of it. That was implemented (`--hoist on`: `pair_slice_bounds` in one lane,
`__shfl_sync` to the rest, verified bit-identical to per-lane computation over
10,917 host checks) and measured as a full 2x2 against the group size.

**It does nothing.** Paired A/B at `G=4`, 30 blocks:

    hoist on vs hoist off      +0.15%   95% CI [-0.08%, +0.38%]

The interval contains zero, and the screen agrees at every group size — hoisted
arms land within about 1% of their unhoisted twins and, uniformly, very
slightly *slower*:

| G (bylen) | hoist off | hoist on | delta |
| --- | --- | --- | --- |
| 32 | 4.9030 | 4.9780 | +1.53% |
| 16 | 3.7380 | 3.7885 | +1.35% |
| 8 | 3.2460 | 3.2575 | +0.35% |
| 4 | 3.1330 | 3.1695 | +1.17% |
| 2 | 3.5025 | 3.5250 | +0.64% |
| 1 | 6.9120 | 6.9425 | +0.44% |

So the `a*G` term is real and large, and **the four `lower_bound` searches are
not it.** The named hypothesis is dead; what survives is only the weaker claim
the fit actually supported — a cost proportional to the thread count, of
unknown composition. Candidates that remain, none of them tested here: the
shuffle reduction tree, whose per-pair cost is `G * log2(G) * 6` shuffles and
which shrinks 20-fold from `G=32` to `G=4`; the launch and scheduling of
`n_pairs * G` threads rather than `n_pairs * G / 8`; the per-thread zeroing of
six accumulators; the `order[slot]` load.

That the negative result is *cheap to state* is the point of having run it. It
also has a practical consequence: since hoisting does not move the optimum, the
best group size is stable with respect to it, which is what makes it safe to
freeze a default without waiting for the mechanism to be settled.

## 7. Cold path, correctly measured

Packing's gain is **steady state**. The cold path points the other way, and
until the planner was fixed it pointed there by the wrong amount.

`plan_pairs` used to compute shorter-slice lengths and lane-slot metrics
unconditionally, even for `--pair-order source`, which needs neither. That
descriptive pass sat inside `cold_data_path_s` in **both** backends. Split out
(`t_plan_metrics_s`), the baseline's planning cost collapses and the gap widens
exactly as predicted:

| | `t_plan` before the fix | `t_plan` after | `cold_data_path` after |
| --- | --- | --- | --- |
| `cuda:32:source` | 50.28 ms | **1.92 ms** | 24.53 ms |
| `cuda:4:bylen` | 102.89 ms | 101.88 ms | 127.10 ms |

Paired, 30 blocks, per stage:

| stage | `cuda:32:source` | `cuda:4:bylen` | effect | 95% CI |
| --- | --- | --- | --- | --- |
| `device_total` | 4.953 ms | 3.149 ms | **−36.18%** | [−38.73, −33.52] |
| `t_stats` | 4.894 ms | 3.071 ms | −36.97% | [−39.53, −34.30] |
| `t_finalize` | 0.060 ms | 0.077 ms | **+28.88%** | [+28.43, +29.33] |
| `t_plan` | 1.917 ms | 101.883 ms | +5314% | [+5114, +5522] |
| `cold_data_path` | 24.534 ms | 127.101 ms | **+372.62%** | [+327.31, +422.73] |

Sorting the pairs costs about a hundred milliseconds against a kernel of about
three. **Break-even is roughly 57 queries** — 102.6 ms of extra planning
divided by 1.80 ms saved per query — so `bylen` belongs to a resident engine
and not to a single invocation of this binary. The earlier "28 queries" was
correct arithmetic on a baseline inflated by its own diagnostics.

The `t_finalize` row is the scatter, and it is a genuine stage regression: it
is simply small.

**The way out is not to choose between the two bases.** `--group 4` without the
sort improves steady state *and* leaves the cold path alone, because it builds
no sorted order:

| | vs `cuda:32:source` | 95% CI |
| --- | --- | --- |
| `cuda:4:source` `device_total`, `item_full` | **−17.59%** | [−19.75, −15.38] |
| `cuda:4:source` `device_total`, `user_full` | **−42.23%** | [−42.54, −41.92] |
| `cuda:4:source` `cold_data_path`, `item_full` | −3.08% | [−9.75, +4.08] |
| `cuda:4:source` `cold_data_path`, `user_full` | −5.60% | [−14.52, +4.24] |
| `cuda:4:source` vs **legacy**, `item_full` | −11.43% | [−11.84, −11.02] |

Both cold-path intervals span zero: there is no penalty to find. That
configuration dominates the phase-3 mapping on every basis measured, which is
what makes it the default rather than a trade.

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
* **Effect gate — passed, far above the 3% bar.** The chosen default beats the
  pre-packing binary by 11.43% [−11.84, −11.02] on `item_full` and the phase-3
  mapping by 17.59% and 42.23% on the two real fixtures. The opt-in `bylen`
  reaches 33.20% [−33.62, −32.77] against legacy.
* **Control gate — passed, with one honest caveat.** Every fixture improves
  against base: `user_full` −50.8%, `lane_tail` −18.8% (at `G=8`), `lane_full`
  −13.2% (at `G=8`), and −10.7% at `G=4`. Measured against `legacy` rather than
  base, though, the lane bands are much thinner: `cuda:4:bylen` is −7.1% on
  `lane_tail` and only **−0.9%** on `lane_full`. On a fixture with no tail to
  recover and short rows, the plumbing cost eats nearly all of the gain — the
  case where packing nearly fails to pay.
* **Hoist gate — the mechanism hypothesis failed.** +0.15% [−0.08, +0.38];
  see §6. The group-size result is unaffected, which is why a default could be
  frozen anyway.
* **Two-GPU gate — passed on `device_total` (−47.62%), with one stage
  regressing.** AllReduce stayed stable (0.229–0.251 ms, no trend in `G`), but
  finalize rose 0.049 → 0.073 ms, about +49%, because it now scatters through
  `order`. The stats collapse from 3.784 to 1.847 ms covers it many times over,
  which is why `device_total` improves — but "no regression in any stage" would
  have been false, and this document said it.
* **Default-path gate — NOT passed as written.** The instrumented default is
  3.28% slower than legacy, above the 1% threshold.

**Decision: the default is now `--group 4 --pair-order source`.**

That is not the configuration this document originally proposed. `bylen` is
faster in steady state — another 25% — but it costs ~100 ms of sorting against
a ~3 ms kernel, so it makes every one-shot caller pay for an order they never
amortise (§7). `--group 4` alone takes most of the win and leaves the cold path
untouched: it is better than the phase-3 mapping on `device_total`, on
`cold_data_path`, on both real fixtures, and against the pre-packing binary,
with no interval on the wrong side. A default should not be a trade the user
has to know about.

`--pair-order bylen` stays available and documented as the resident-engine
setting, with its break-even stated.

**On the default-path gate.** The 3.28% plumbing regression is still there and
still real, but it is now moot as a *default-path* question: nobody lands on
`--group 32 --pair-order source` unless they ask for it, and the new default is
11.43% faster than legacy with the plumbing included. A separate legacy fast
path would mean maintaining a second kernel to serve a configuration that is
slower than the default on every measure. Not added.

**What this obliges.** The headline matrix and the NVLink compression A/B are
re-run on the frozen SHA with the real default command, because the CUDA and
NCCL rows now describe a different kernel and a faster kernel changes the
communication share the compression result depends on.

## 10. Still open

* **The mechanism.** The `a*G` term is real, large, and unattributed. Hoisting
  the slice-bound searches — the leading candidate — was implemented and came
  back null (§6). Untested candidates: the shuffle reduction tree
  (`G * log2(G) * 6` shuffles per pair, a 20-fold difference between `G=32` and
  `G=4`), launch and scheduling of `n_pairs * G` threads, the per-thread
  accumulator zeroing.
* **Async.** Correctness is verified at three chunk sizes and both finalize
  streams. Performance is deliberately *not* measured: chunks are contiguous
  slot runs, so an ascending-by-length order hands early chunks the short pairs
  and late chunks the long ones, changing pipeline balance for reasons that
  have nothing to do with lane packing. A chunk-balanced order is a separate
  piece of work — and it only matters for `bylen`, which is no longer the
  default.
* **`G=4` is not established as universal.** It won on both real fixtures;
  `G=8` won on both synthetic lane bands, and the two are within 3.4% of each
  other. This is one GPU generation, one block size, one workload family.
* **Achieved occupancy** is still unmeasured (`ncu`, `ERR_NVGPUCTRPERM`). The
  theoretical figure is 0.75, register-limited, and identical across group
  sizes — so it cannot be what separates them, but the achieved figure could
  still differ.
* **The `G=1` coalescing explanation** remains inference from residuals, for
  the same reason.
* **Two sessions, one host class.** The primary effect reproduced on two
  independent pods (−36.78%, −36.18%), both 2x A100-SXM4-80GB with NV12. No
  other GPU, interconnect or CUDA version has seen this code.
