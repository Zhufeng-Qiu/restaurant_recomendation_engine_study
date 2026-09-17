"""The four-way work-division comparison, run and analysed as blocks.

The question: with the whole input already resident on every GPU, is giving
each device half the PAIRS better than giving it half the rating DIMENSIONS
and reducing partial statistics? And if the dimension split is kept, does
shrinking the collective's representation from 48 to 16 bytes per pair reduce
the full batch latency?

    A   1 GPU,  pairs to completion, f64      no collective
    B   2 GPUs, dimension split,     f64      the original path
    C   2 GPUs, dimension split,     packed   the compressed path
    D   2 GPUs, pair split,          f64      no collective

  D/B  isolates the splitting strategy at one representation -- the main
       comparison
  C/B  isolates the representation at one splitting strategy
  D/C  compares the two real candidates, but moves both at once, so it
       cannot attribute the difference to either on its own
  D/A, C/A  describe what a second GPU buys, in latency and in GPU-seconds

Everything else is held fixed: one binary, sync mode, G4, source order, hoist
off, plan metrics off, every iteration validated.

Why this is not run_bench.py: that harness restarts the process per trial and
measures a single shot, which answers a different question and cannot be made
to answer this one by raising a warm-up count. Here each configuration is one
process that sets up once and then times 20 resident batches, and the paired
unit of analysis is the BLOCK -- one run of all four configurations, in a
randomised order -- not the iteration. 30 blocks per fixture, in three
segments of ten, is 600 timed iterations per configuration and still only 30
paired observations. Reporting 600 as the sample size would be claiming 600
independent experiments, which it is not.

  architecture_ab.py run --fixture data/fixtures/item_full --out results/...
  architecture_ab.py run ... --pilot 3       # pre-flight, not formal data
  architecture_ab.py analyze <record.json>
  architecture_ab.py --selftest
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_bench  # noqa: E402  -- provenance, timeout-safe launch, atomic write

# Held fixed across all four, so a difference cannot come from anywhere else.
COMMON = ["--mode", "sync", "--group", "4", "--pair-order", "source",
          "--hoist", "off", "--plan-metrics", "off", "--validate",
          "--resident-bench"]

CONFIGS = [
    ("A", 1, "pair", "f64"),
    ("B", 2, "dim", "f64"),
    ("C", 2, "dim", "packed"),
    ("D", 2, "pair", "f64"),
]

# Which ratios get reported, and what each one is allowed to be read as.
COMPARISONS = [
    ("D", "B", "splitting strategy, one representation"),
    ("C", "B", "representation, one splitting strategy"),
    ("D", "C", "the two candidates -- both factors move"),
    ("D", "A", "second GPU under the pair split"),
    ("C", "A", "second GPU under the dimension split"),
]

SEGMENTS = 3
BLOCKS_PER_SEGMENT = 10

# A record that is missing one of these is a failed trial, not a slow one.
REQUIRED = {
    "schema_version": str,
    "partition": str,
    "timing_basis": str,
    "durations_ms": list,
    "resident_host_complete_ms_median": (int, float),
    "warmup_count": int,
    "repeat_count": int,
    "setup_count": int,
    "pid": int,
    "collective_ops_per_iteration": int,
    "collective_input_bytes_per_rank": int,
    "host_output_bytes": int,
    "pair_ranges": list,
    "n_pairs": int,
    "iterations_validated": int,
    "iterations_passed": int,
    "validation_position": str,
    "output_slots": int,
}


def command(binary, fixture, gpus, partition, payload, warmup, repeat):
    return ([binary, fixture, "--gpus", str(gpus), "--partition", partition,
             "--payload", payload] + COMMON +
            ["--warmup", str(warmup), "--repeat", str(repeat)])


def schema_problems(rec, warmup, repeat):
    """Complaints about one configuration's record; empty means usable."""
    out = []
    for field, kinds in REQUIRED.items():
        if field not in rec:
            out.append(f"missing {field!r}")
        elif not isinstance(rec[field], kinds):
            out.append(f"{field!r} is {type(rec[field]).__name__}")
    if out:
        return out

    if rec["schema_version"] != "resident-v1":
        out.append(f"unexpected schema_version {rec['schema_version']!r}")
    if rec["timing_basis"] != "resident_host_complete":
        out.append(f"timing_basis is {rec['timing_basis']!r}, not resident")
    if rec["warmup_count"] != warmup or rec["repeat_count"] != repeat:
        out.append("warmup/repeat do not match what was asked for")
    if rec["setup_count"] != 1:
        out.append(f"setup ran {rec['setup_count']} times, not once")
    # Validation inside the timed loop is not a slower measurement, it is a
    # different one: on 2xA100 the ~5 ms host gap between batches drops the SM
    # clock from 1410 to ~795 MHz and every later batch runs 1.77x slower, as
    # a clean step mid-run. A record that validated in the loop is not
    # comparable to one that did not, so it is rejected rather than merged.
    if rec["validation_position"] != "after_timed_loop":
        out.append(f"validation ran {rec['validation_position']!r}, "
                   "which changes the GPU clock between batches")
    if rec["output_slots"] < repeat:
        out.append(f"{rec['output_slots']} output slots for {repeat} "
                   "iterations -- some batch was not kept to be checked")
    if len(rec["durations_ms"]) != repeat:
        out.append(f"{len(rec['durations_ms'])} durations for {repeat} repeats")
    # Every duration finite and strictly positive. A zero would sail through a
    # median and make a configuration look infinitely fast.
    for i, d in enumerate(rec["durations_ms"]):
        if not isinstance(d, (int, float)) or not math.isfinite(d) or d <= 0:
            out.append(f"durations_ms[{i}] is {d!r}")
            break
    # Validation state, not just validation presence.
    if rec["iterations_validated"] != repeat:
        out.append(f"{rec['iterations_validated']} iterations validated "
                   f"of {repeat}")
    if rec["iterations_passed"] != rec["iterations_validated"]:
        out.append("some iteration did not match golden")
    if rec.get("validation_passed") is not True:
        out.append("final validation_passed is not true")
    if rec.get("tol_failures", 0) != 0:
        out.append(f"tol_failures = {rec.get('tol_failures')}")
    if rec.get("nonfinite_output", 0) or rec.get("nonfinite_golden", 0):
        out.append("non-finite similarities")
    if rec.get("emitted_mismatches", 0):
        out.append("retained set differs from golden")

    # The partition's defining property, checked rather than assumed.
    covered = sorted((r["pair_begin"], r["pair_count"]) for r in rec["pair_ranges"])
    cursor = 0
    for begin, count in covered:
        if rec["partition"] == "pair":
            if begin != cursor:
                out.append(f"pair shards do not tile: gap or overlap at {begin}")
                break
            cursor = begin + count
    if rec["partition"] == "pair":
        if cursor != rec["n_pairs"]:
            out.append(f"pair shards cover {cursor} of {rec['n_pairs']}")
        if rec["collective_ops_per_iteration"] != 0:
            out.append("a pair-split run reported a collective")
        if rec["collective_input_bytes_per_rank"] != 0:
            out.append("a pair-split run reported collective bytes")
    else:
        if rec["collective_ops_per_iteration"] != 1:
            out.append("a dimension-split run did not report one collective")
        for r in rec["pair_ranges"]:
            if r["pair_count"] != rec["n_pairs"]:
                out.append("a dimension-split device did not get every pair")
                break
    if rec["host_output_bytes"] != 8 * rec["n_pairs"]:
        out.append("host output is not 8 bytes per pair")
    return out


def block_plan(seed, n_blocks):
    """The order each block runs its four configurations in.

    Randomised per block and seeded, so position in the block cannot be
    confounded with configuration, and the whole schedule is reproducible from
    one integer.
    """
    rng = random.Random(seed)
    plan = []
    for b in range(n_blocks):
        names = [c[0] for c in CONFIGS]
        rng.shuffle(names)
        plan.append(names)
    return plan


def paired_log_ratio(blocks, x, y):
    """Geometric mean of per-block ratios, as a ratio.

    Not the same estimator as dividing two medians, and reported as what it
    is: R = exp(mean(log(t_x / t_y))) over blocks where both ran.
    """
    rs = [math.log(b[x] / b[y]) for b in blocks if x in b and y in b]
    if not rs:
        return None, []
    return math.exp(statistics.fmean(rs)), rs


def bootstrap_ci(blocks, x, y, segments, draws=5000, seed=20260917):
    """Percentile CI from a paired block bootstrap, stratified by segment.

    Resampling whole blocks keeps the four configurations paired; stratifying
    keeps each replicate balanced across the three time segments, so a
    replicate cannot be drawn entirely from one of them. The interval
    describes this host under these conditions -- three segments of one
    machine are not evidence about other machines.
    """
    usable = [b for b in blocks if x in b and y in b]
    if len(usable) < 2:
        return None, None
    by_seg = {}
    for b in usable:
        by_seg.setdefault(b["segment"], []).append(b)
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        rs = []
        for seg in segments:
            pool = by_seg.get(seg, [])
            if not pool:
                continue
            for _ in range(len(pool)):
                b = pool[rng.randrange(len(pool))]
                rs.append(math.log(b[x] / b[y]))
        if rs:
            out.append(math.exp(statistics.fmean(rs)))
    if not out:
        return None, None
    out.sort()
    lo = out[int(0.025 * (len(out) - 1))]
    hi = out[int(0.975 * (len(out) - 1))]
    return lo, hi


def run_matrix(args, log=print):
    fixture = args.fixture.rstrip("/")
    n_blocks = args.pilot if args.pilot else SEGMENTS * BLOCKS_PER_SEGMENT
    plan = block_plan(args.seed, n_blocks)
    doc = {
        "kind": "architecture_ab",
        "schema_version": "architecture-ab-v1",
        "pilot": bool(args.pilot),
        "fixture": fixture,
        "binary": args.binary,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "n_blocks": n_blocks,
        "segments": SEGMENTS if not args.pilot else 1,
        "blocks_per_segment": BLOCKS_PER_SEGMENT if not args.pilot else n_blocks,
        "configs": {n: {"gpus": g, "partition": p, "payload": y}
                    for n, g, p, y in CONFIGS},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": run_bench.environment(),
        "fixture_meta": None,
        "blocks": [],
        "failures": [],
    }
    meta_path = os.path.join(fixture, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            doc["fixture_meta"] = json.load(f)

    # A pilot exists to find out whether the output parses, the units are
    # what they claim, and the times are plausible. It is explicitly not
    # formal data, and is written to its own file so it cannot be mistaken
    # for any later.
    if args.pilot:
        log(f"PILOT: {n_blocks} blocks. Not formal data.")

    by_config = {n: [] for n, _, _, _ in CONFIGS}
    spec = {n: (g, p, y) for n, g, p, y in CONFIGS}

    for b, order in enumerate(plan):
        seg = 0 if args.pilot else b // BLOCKS_PER_SEGMENT
        block = {"block": b, "segment": seg, "order": list(order),
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": {}}
        for name in order:
            gpus, partition, payload = spec[name]
            cmd = command(args.binary, fixture, gpus, partition, payload,
                          args.warmup, args.repeat)
            try:
                rec = run_bench.run_once(cmd, {}, timeout=args.timeout)
            except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
                doc["failures"].append(
                    {"block": b, "config": name, "error": repr(exc),
                     "cmd": cmd})
                log(f"  block {b:>2} {name}: FAILED {exc!r}")
                rec = None
            if rec is None:
                continue
            problems = schema_problems(rec, args.warmup, args.repeat)
            if problems:
                doc["failures"].append(
                    {"block": b, "config": name, "problems": problems,
                     "cmd": cmd})
                log(f"  block {b:>2} {name}: REJECTED {problems[:3]}")
                continue
            block["runs"][name] = rec
            by_config[name].append(rec["resident_host_complete_ms_median"])
            log(f"  block {b:>2} {name}: "
                f"{rec['resident_host_complete_ms_median']:8.3f} ms median "
                f"of {args.repeat}")

        # A block is only complete when all four ran; an incomplete block is
        # kept in the record but excluded from every paired comparison, and
        # dropping it silently is how a matrix ends up unbalanced.
        block["complete"] = len(block["runs"]) == len(CONFIGS)
        block["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        doc["blocks"].append(block)
        # Written after every block. A failure in block 29 must not cost the
        # first 28.
        run_bench.atomic_write_json(args.out, doc)

        # Stop on a correctness failure rather than collecting 30 blocks of
        # data from a path that does not compute the right answer.
        if args.failures and len(doc["failures"]) >= args.failures:
            log(f"stopping: {len(doc['failures'])} failures reached the limit")
            break

    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_bench.atomic_write_json(args.out, doc)
    complete = sum(1 for b in doc["blocks"] if b["complete"])
    log(f"\n{complete} complete blocks of {len(doc['blocks'])} "
        f"({len(doc['failures'])} failures) -> {args.out}")
    return 1 if doc["failures"] else 0


def summarise(doc, log=print):
    blocks = []
    for b in doc["blocks"]:
        if not b["complete"]:
            continue
        row = {"segment": b["segment"]}
        for name, rec in b["runs"].items():
            row[name] = rec["resident_host_complete_ms_median"]
        blocks.append(row)

    if not blocks:
        log("no complete blocks")
        return 1

    segments = sorted({b["segment"] for b in blocks})
    n_pairs = None
    for b in doc["blocks"]:
        for rec in b["runs"].values():
            n_pairs = rec["n_pairs"]
            break
        if n_pairs:
            break

    label = "PILOT -- not formal data" if doc.get("pilot") else "formal matrix"
    log(f"\n{doc['fixture']}  ({label})")
    log(f"{len(blocks)} complete blocks over {len(segments)} segment(s), "
        f"{doc['repeat']} timed iterations each\n")

    log(f"{'cfg':<4}{'gpus':>5}{'partition':>11}{'payload':>9}"
        f"{'median ms':>12}{'IQR ms':>10}{'pair/s':>14}{'GPU-s/batch':>13}")
    for name, gpus, partition, payload in CONFIGS:
        vals = sorted(b[name] for b in blocks if name in b)
        if not vals:
            continue
        med = statistics.median(vals)
        q1, q3 = (statistics.quantiles(vals, n=4)[0],
                  statistics.quantiles(vals, n=4)[2]) if len(vals) >= 4 else (med, med)
        pps = (n_pairs / (med / 1e3)) if n_pairs and med > 0 else 0.0
        log(f"{name:<4}{gpus:>5}{partition:>11}{payload:>9}"
            f"{med:>12.3f}{q3 - q1:>10.3f}{pps:>14,.0f}"
            f"{gpus * med / 1e3:>13.4f}")

    log("\npaired block ratios, geometric mean with a 95% bootstrap interval")
    log("(R < 1 means the first is faster; the unit is the block, not the "
        "iteration)\n")
    for x, y, meaning in COMPARISONS:
        R, rs = paired_log_ratio(blocks, x, y)
        if R is None:
            continue
        lo, hi = bootstrap_ci(blocks, x, y, segments)
        span = ("crosses 1 -- direction not established"
                if lo is not None and lo < 1.0 < hi else "")
        ci = f"[{lo:.4f}, {hi:.4f}]" if lo is not None else "[n/a]"
        log(f"  {x}/{y}  R={R:.4f}  {ci}  "
            f"{(R - 1) * 100:+6.2f}%   {meaning}")
        if span:
            log(f"          {span}")

    # Per-segment, because three segments of one machine can disagree and
    # reporting only the pooled number would hide it.
    if len(segments) > 1:
        log("\nby segment (same host, different times -- not three machines)")
        for x, y, _ in COMPARISONS[:3]:
            parts = []
            for seg in segments:
                sub = [b for b in blocks if b["segment"] == seg]
                R, _ = paired_log_ratio(sub, x, y)
                parts.append(f"{R:.4f}" if R else "n/a")
            log(f"  {x}/{y}  " + "  ".join(parts))
    return 0


def selftest():
    fails = 0

    def expect(ok, what):
        nonlocal fails
        if not ok:
            fails += 1
            print(f"  FAIL {what}")

    # --- the schedule ---
    plan = block_plan(7, 30)
    expect(len(plan) == 30, "thirty blocks")
    expect(all(sorted(b) == ["A", "B", "C", "D"] for b in plan),
           "every block runs all four configurations exactly once")
    expect(plan == block_plan(7, 30), "the same seed gives the same schedule")
    expect(plan != block_plan(8, 30), "a different seed gives a different one")
    first = [b[0] for b in plan]
    expect(len(set(first)) == 4, "no single configuration always goes first")

    # --- the estimator ---
    blocks = [{"segment": i // 10, "B": 100.0, "D": 50.0} for i in range(30)]
    R, _ = paired_log_ratio(blocks, "D", "B")
    expect(abs(R - 0.5) < 1e-12, "a uniform 2x speedup gives R = 0.5")
    lo, hi = bootstrap_ci(blocks, "D", "B", [0, 1, 2], draws=200)
    expect(abs(lo - 0.5) < 1e-9 and abs(hi - 0.5) < 1e-9,
           "zero-variance data gives a zero-width interval")
    # The paired geometric mean is NOT the ratio of the two medians.
    skew = [{"segment": 0, "B": 100.0, "D": 50.0},
            {"segment": 0, "B": 100.0, "D": 200.0}]
    R2, _ = paired_log_ratio(skew, "D", "B")
    expect(abs(R2 - 1.0) < 1e-12, "paired geometric mean of 0.5 and 2.0 is 1.0")
    expect(abs(statistics.median([50.0, 200.0]) /
               statistics.median([100.0, 100.0]) - 1.25) < 1e-12,
           "...while the ratio of medians is 1.25 -- different estimators")

    # --- the schema gate ---
    good = {
        "schema_version": "resident-v1", "partition": "pair",
        "timing_basis": "resident_host_complete",
        "durations_ms": [1.5, 1.6], "resident_host_complete_ms_median": 1.55,
        "warmup_count": 20, "repeat_count": 2, "setup_count": 1, "pid": 42,
        "collective_ops_per_iteration": 0,
        "collective_input_bytes_per_rank": 0, "host_output_bytes": 80,
        "pair_ranges": [{"device": 0, "pair_begin": 0, "pair_count": 5},
                        {"device": 1, "pair_begin": 5, "pair_count": 5}],
        "n_pairs": 10, "iterations_validated": 2, "iterations_passed": 2,
        "validation_passed": True, "tol_failures": 0,
        "validation_position": "after_timed_loop", "output_slots": 2,
        "nonfinite_output": 0, "nonfinite_golden": 0, "emitted_mismatches": 0,
    }
    expect(schema_problems(good, 20, 2) == [], "a well-formed record passes")

    def broken(**kw):
        r = dict(good)
        r.update(kw)
        return schema_problems(r, 20, 2)

    expect(broken(durations_ms=[1.5, 0.0]), "a zero duration is rejected")
    expect(broken(durations_ms=[1.5, float("nan")]),
           "a non-finite duration is rejected")
    expect(broken(durations_ms=[1.5]), "too few durations is rejected")
    expect(broken(setup_count=2), "setting up twice is rejected")
    expect(broken(iterations_passed=1),
           "an iteration that did not match golden is rejected")
    expect(broken(validation_passed=False), "a failed validation is rejected")
    expect(broken(tol_failures=3), "a tolerance failure is rejected")
    expect(broken(emitted_mismatches=1), "a retained-set change is rejected")
    expect(broken(collective_ops_per_iteration=1),
           "a pair split claiming a collective is rejected")
    expect(broken(pair_ranges=[{"device": 0, "pair_begin": 0, "pair_count": 5},
                               {"device": 1, "pair_begin": 6, "pair_count": 4}]),
           "pair shards with a gap are rejected")
    expect(broken(pair_ranges=[{"device": 0, "pair_begin": 0, "pair_count": 6},
                               {"device": 1, "pair_begin": 5, "pair_count": 5}]),
           "overlapping pair shards are rejected")
    expect(broken(host_output_bytes=40),
           "a host output that is not 8 bytes per pair is rejected")
    expect(broken(schema_version="v0"), "an old-schema record is rejected")
    expect(broken(validation_position="in_timed_loop"),
           "a record that validated between batches is rejected")
    expect(broken(output_slots=1),
           "fewer output slots than iterations is rejected")
    expect(broken(timing_basis="device_total"),
           "a non-resident timing basis is rejected")
    dim = dict(good, partition="dim", collective_ops_per_iteration=1,
               collective_input_bytes_per_rank=160,
               pair_ranges=[{"device": 0, "pair_begin": 0, "pair_count": 10},
                            {"device": 1, "pair_begin": 0, "pair_count": 10}])
    expect(schema_problems(dim, 20, 2) == [],
           "a well-formed dimension-split record passes")
    expect(schema_problems(dict(dim, collective_ops_per_iteration=0), 20, 2),
           "a dimension split claiming no collective is rejected")

    print("architecture_ab selftest: " + ("FAIL" if fails else "all pass"))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("run")
    r.add_argument("--binary", default="engine/build/pearson_engine_nccl")
    r.add_argument("--fixture", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--repeat", type=int, default=20)
    r.add_argument("--seed", type=int, default=20260917)
    r.add_argument("--timeout", type=float, default=900.0)
    r.add_argument("--pilot", type=int, default=0,
                   help="run N blocks as a pre-flight; not formal data")
    r.add_argument("--failures", type=int, default=1,
                   help="stop after this many failures (0 = never)")

    a = sub.add_parser("analyze")
    a.add_argument("record")

    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.cmd == "run":
        return run_matrix(args)
    if args.cmd == "analyze":
        with open(args.record) as f:
            return summarise(json.load(f))
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
