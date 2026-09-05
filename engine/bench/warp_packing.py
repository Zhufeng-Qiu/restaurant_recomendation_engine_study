"""Warp-packing experiments: screening, then a balanced paired A/B.

Three separate baselines, because two things changed at once and only one of
them is the hypothesis:

  legacy       the binary from before packing existed (one warp per pair, no
               order array, finalize writes straight to sims[k]).
  instrumented this tree at --group 32 --pair-order source. Same mapping, but
               every pair now goes through order[slot] and finalize scatters.
  candidate    --group < 32, with or without the length sort.

legacy -> instrumented is the cost of the plumbing; instrumented -> candidate
is the packing effect. Reporting only the second would hide a default-path
regression, and reporting only legacy -> candidate would credit packing with
whatever the plumbing did.

Two phases:

  screen   all arms, few trials, round-robin with a fresh shuffle each round.
           Cheap, and only good enough to drop the obvious losers.
  paired   one candidate against one baseline, 30 balanced crossover blocks --
           15 A->B and 15 B->A, block order shuffled from a recorded seed. The
           estimator is mean log(t_candidate / t_baseline) over blocks, so
           each block cancels its own drift; the CI is over blocks and
           describes THIS session, not the hardware.

Nothing here profiles. NCU replay timings must never enter a performance
table; use profiles to explain an effect that measurement already found.

Usage:
  warp_packing.py screen --fixture DIR [--arms SPEC,...] [--repeats 10]
                         [--seed N] [--tag NAME]
  warp_packing.py paired --fixture DIR --baseline SPEC --candidate SPEC
                         [--blocks 30] [--seed N] [--tag NAME]

SPEC is one of
  legacy                      the pre-packing binary (see --legacy-bin)
  cuda:<group>:<order>        single GPU
  nccl<gpus>:<group>:<order>:<payload>[:<mode>]
"""

import argparse
import json
import math
import os
import random
import statistics as st
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(ROOT, "engine", "build", "pearson_engine")
ENGINE_NCCL = os.path.join(ROOT, "engine", "build", "pearson_engine_nccl")
LEGACY = os.path.join(ROOT, "engine", "build-legacy", "pearson_engine")


def parse_arm(spec, legacy_bin):
    """SPEC -> (label, argv-builder). Rejects anything it cannot run exactly."""
    parts = spec.split(":")
    kind = parts[0]
    if kind == "legacy":
        return spec, lambda fx: [legacy_bin, fx, "--backend", "cuda", "--validate"]
    if kind == "cuda":
        if len(parts) != 3:
            raise SystemExit(f"cuda spec needs group and order: {spec}")
        g, order = parts[1], parts[2]
        return spec, lambda fx: [ENGINE, fx, "--backend", "cuda", "--validate",
                                 "--group", g, "--pair-order", order]
    if kind.startswith("nccl"):
        gpus = kind[4:] or "2"
        if len(parts) not in (4, 5):
            raise SystemExit(f"nccl spec needs group, order, payload: {spec}")
        g, order, payload = parts[1], parts[2], parts[3]
        mode = parts[4] if len(parts) == 5 else "sync"
        return spec, lambda fx: [ENGINE_NCCL, fx, "--gpus", gpus, "--mode", mode,
                                 "--payload", payload, "--validate",
                                 "--group", g, "--pair-order", order]
    raise SystemExit(f"unknown arm: {spec}")


def run_once(argv):
    """One invocation -> the merged JSON record. Raises on any failure."""
    p = subprocess.run(argv, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)}\n{p.stderr[-800:]}")
    rec = {}
    for line in p.stdout.strip().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        rec.update(d.pop("cuda_detail", {}))
        rec.update(d)
    if not rec:
        raise RuntimeError(f"no JSON from {' '.join(argv)}")
    # Correctness is a precondition for a timing, not a separate report: a
    # fast wrong answer is not a data point.
    if rec.get("tol_failures", 0) != 0:
        raise RuntimeError(f"tol_failures={rec['tol_failures']} from {' '.join(argv)}")
    return rec


def time_of(rec):
    if rec.get("timing_basis") == "pipeline_total":
        return rec["t_pipeline_s"]
    return rec["device_total_s"]


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def environment():
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "git_sha": sh("git rev-parse HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        "legacy_sha": sh("git -C engine/build-legacy-src rev-parse HEAD"),
        "image_digest": os.environ.get("BENCH_IMAGE_DIGEST"),
        "gpus": sh("nvidia-smi --query-gpu=name,uuid --format=csv,noheader"),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        "nvcc": sh("nvcc --version | tail -1"),
        "nccl_version": sh("python3 -c \"import ctypes;l=ctypes.CDLL('libnccl.so.2');"
                           "v=ctypes.c_int();l.ncclGetVersion(ctypes.byref(v));print(v.value)\""),
        "topo": sh("nvidia-smi topo -m"),
        "p2p_r": sh("nvidia-smi topo -p2p r"),
        "p2p_w": sh("nvidia-smi topo -p2p w"),
    }


def screen(args):
    arms = [parse_arm(s, args.legacy_bin) for s in args.arms.split(",")]
    rng = random.Random(args.seed)
    for label, build in arms:                      # warm-ups, discarded
        for _ in range(args.warmups):
            run_once(build(args.fixture))
    trials = {label: [] for label, _ in arms}
    records = {}
    order_log = []
    for r in range(args.repeats):
        shuffled = arms[:]
        rng.shuffle(shuffled)
        order_log.append([label for label, _ in shuffled])
        for label, build in shuffled:
            rec = run_once(build(args.fixture))
            trials[label].append(time_of(rec))
            records.setdefault(label, rec)
    out = {
        "phase": "screen", "fixture": args.fixture, "seed": args.seed,
        "repeats": args.repeats, "warmups": args.warmups,
        "round_order": order_log, "environment": environment(),
        "arms": {},
    }
    for label, _ in arms:
        ts = trials[label]
        q = st.quantiles(ts, n=4) if len(ts) >= 4 else [min(ts), st.median(ts), max(ts)]
        rec = records[label]
        out["arms"][label] = {
            "trials_s": ts,
            "median_s": st.median(ts), "min_s": min(ts), "max_s": max(ts),
            "iqr_pct": (q[2] - q[0]) / st.median(ts) * 100,
            "t_stats_s": rec.get("t_stats_s"),
            "t_finalize_s": rec.get("t_finalize_s"),
            "t_allreduce_s": rec.get("t_allreduce_s"),
            "t_plan_s": rec.get("t_plan_s"),
            "cold_data_path_s": rec.get("cold_data_path_s"),
            "max_abs_diff": rec.get("max_abs_diff"),
            "plan_lane_slots": rec.get("plan_lane_slots"),
            "plan_lane_utilisation": rec.get("plan_lane_utilisation"),
            "plan_aggregate_lane_slots": rec.get("plan_aggregate_lane_slots"),
            "plan_aggregate_lane_utilisation": rec.get("plan_aggregate_lane_utilisation"),
            "plan_critical_lane_slots": rec.get("plan_critical_lane_slots"),
            "plan_per_device": rec.get("plan_per_device"),
        }
    write(out, args.tag or "screen")
    base = min(out["arms"], key=lambda k: out["arms"][k]["median_s"])
    print(f"\n{'arm':<24s} {'median ms':>10s} {'IQR%':>6s} {'stats ms':>9s} "
          f"{'vs best':>8s}")
    for label in sorted(out["arms"], key=lambda k: out["arms"][k]["median_s"]):
        a = out["arms"][label]
        stats_ms = f"{a['t_stats_s'] * 1e3:9.4f}" if a["t_stats_s"] else " " * 9
        print(f"{label:<24s} {a['median_s'] * 1e3:10.4f} {a['iqr_pct']:6.2f} "
              f"{stats_ms} {a['median_s'] / out['arms'][base]['median_s'] - 1:+7.2%}")


def paired(args):
    (blab, bbuild) = parse_arm(args.baseline, args.legacy_bin)
    (clab, cbuild) = parse_arm(args.candidate, args.legacy_bin)
    rng = random.Random(args.seed)
    for build in (bbuild, cbuild):
        for _ in range(args.warmups):
            run_once(build(args.fixture))

    half = args.blocks // 2
    order = ["bc"] * half + ["cb"] * (args.blocks - half)
    rng.shuffle(order)
    rows, diffs = [], []
    for first in order:
        if first == "bc":
            tb = time_of(run_once(bbuild(args.fixture)))
            tc = time_of(run_once(cbuild(args.fixture)))
        else:
            tc = time_of(run_once(cbuild(args.fixture)))
            tb = time_of(run_once(bbuild(args.fixture)))
        rows.append({"first": first, "baseline_s": tb, "candidate_s": tc})
    lr = [math.log(r["candidate_s"] / r["baseline_s"]) for r in rows]
    m = st.mean(lr)
    se = st.stdev(lr) / math.sqrt(len(lr)) if len(lr) > 1 else 0.0
    ci = 1.96 * se
    out = {
        "phase": "paired", "fixture": args.fixture, "seed": args.seed,
        "baseline": blab, "candidate": clab, "blocks": len(rows),
        "block_order": [r["first"] for r in rows], "rows": rows,
        "estimator": "mean of log(candidate/baseline) over blocks, exponentiated",
        "ci_method": "1.96 x SE, normal approximation over the block log-ratios",
        "ci_level": 0.95,
        "effect_pct": (math.exp(m) - 1) * 100,
        "ci_lo_pct": (math.exp(m - ci) - 1) * 100,
        "ci_hi_pct": (math.exp(m + ci) - 1) * 100,
        "baseline_median_s": st.median(r["baseline_s"] for r in rows),
        "candidate_median_s": st.median(r["candidate_s"] for r in rows),
        "environment": environment(),
    }
    write(out, args.tag or "paired")
    print(f"\n{clab} vs {blab} on {os.path.basename(args.fixture)}")
    print(f"  baseline  {out['baseline_median_s'] * 1e3:9.4f} ms")
    print(f"  candidate {out['candidate_median_s'] * 1e3:9.4f} ms")
    print(f"  effect {out['effect_pct']:+.2f}%  "
          f"95% CI [{out['ci_lo_pct']:+.2f}%, {out['ci_hi_pct']:+.2f}%]  "
          f"({'excludes' if out['ci_lo_pct'] * out['ci_hi_pct'] > 0 else 'INCLUDES'} zero)")


def write(out, tag):
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(ROOT, "results", "bench"), exist_ok=True)
    path = os.path.join(ROOT, "results", "bench", f"warp_packing_{tag}_{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("screen", "paired"):
        s = sub.add_parser(name)
        s.add_argument("--fixture", required=True)
        s.add_argument("--seed", type=int, default=20260905)
        s.add_argument("--warmups", type=int, default=3)
        s.add_argument("--tag")
        s.add_argument("--legacy-bin", default=LEGACY)
        if name == "screen":
            s.add_argument("--arms", required=True)
            s.add_argument("--repeats", type=int, default=10)
        else:
            s.add_argument("--baseline", required=True)
            s.add_argument("--candidate", required=True)
            s.add_argument("--blocks", type=int, default=30)
    args = ap.parse_args()
    (screen if args.cmd == "screen" else paired)(args)


if __name__ == "__main__":
    main()
