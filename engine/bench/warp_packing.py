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
  cuda:<group>:<order>[:hoist]        single GPU
  nccl<gpus>:<group>:<order>:<payload>[:<mode>][:hoist]

Appending `hoist` to any arm computes each pair's slice bounds once per lane
group and broadcasts them, instead of repeating the four searches in every
lane. It is orthogonal to the group size on purpose: the leading hypothesis for
why small groups win is a per-thread cost, and this is the arm that tests it.
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
    hoist = "off"
    if parts[-1] in ("hoist", "nohoist"):
        hoist = "on" if parts[-1] == "hoist" else "off"
        parts = parts[:-1]
    kind = parts[0]
    if kind == "legacy":
        return spec, lambda fx: [legacy_bin, fx, "--backend", "cuda", "--validate"]
    if kind == "cuda":
        if len(parts) != 3:
            raise SystemExit(f"cuda spec needs group and order: {spec}")
        g, order = parts[1], parts[2]
        return spec, lambda fx: [ENGINE, fx, "--backend", "cuda", "--validate",
                                 "--group", g, "--pair-order", order,
                                 "--hoist", hoist]
    if kind.startswith("nccl"):
        gpus = kind[4:] or "2"
        if len(parts) not in (4, 5):
            raise SystemExit(f"nccl spec needs group, order, payload: {spec}")
        g, order, payload = parts[1], parts[2], parts[3]
        mode = parts[4] if len(parts) == 5 else "sync"
        return spec, lambda fx: [ENGINE_NCCL, fx, "--gpus", gpus, "--mode", mode,
                                 "--payload", payload, "--validate",
                                 "--group", g, "--pair-order", order,
                                 "--hoist", hoist]
    raise SystemExit(f"unknown arm: {spec}")


def run_once(argv):
    """One invocation -> the merged JSON record. Raises on any failure."""
    t0 = time.perf_counter()
    p = subprocess.run(argv, capture_output=True, text=True)
    wall = time.perf_counter() - t0
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
    # Wall clock around the process: launch, linking and the CUDA driver's
    # first touch, none of which the binary can time. These runs pass
    # --validate, so it is a VALIDATED CLI wall time, not a production one-shot.
    rec["t_process_wall_s"] = wall
    rec["t_process_wall_includes_validate"] = True
    if rec.get("tol_failures", 0) != 0:
        raise RuntimeError(f"tol_failures={rec['tol_failures']} from {' '.join(argv)}")
    return rec


def time_of(rec):
    if rec.get("timing_basis") == "pipeline_total":
        return rec["t_pipeline_s"]
    return rec["device_total_s"]


# Every per-trial number worth a median. Keeping only the first record made the
# cold-path figures single samples with no spread, which is how a 50 ms
# diagnostic pass sat in the cold path unnoticed.
TRIAL_FIELDS = ("device_total_s", "t_plan_s", "t_plan_metrics_s", "t_h2d_s",
                "t_setup_s", "t_d2h_s", "cold_data_path_s", "t_process_wall_s",
                "t_stats_s", "t_allreduce_s", "t_finalize_s", "max_abs_diff")


def trial_of(rec):
    t = {k: rec[k] for k in TRIAL_FIELDS if k in rec}
    t["time_s"] = time_of(rec)
    return t


def summarise(trials, key):
    vals = [t[key] for t in trials if t.get(key) is not None]
    if not vals:
        return None
    med = st.median(vals)
    q = st.quantiles(vals, n=4) if len(vals) >= 4 else [min(vals), med, max(vals)]
    return {"median": med, "min": min(vals), "max": max(vals),
            "iqr_pct": (q[2] - q[0]) / med * 100 if med else 0.0,
            "n": len(vals), "values": vals}


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def environment():
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "git_sha": sh("git rev-parse HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        "git_dirty_tracked":len([l for l in (sh("git status --porcelain") or "").splitlines()
                                 if l.strip().split(" ")[0] != "??"]),
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
            trials[label].append(trial_of(rec))
            records.setdefault(label, rec)
    out = {
        "phase": "screen", "fixture": args.fixture, "seed": args.seed,
        "repeats": args.repeats, "warmups": args.warmups,
        "round_order": order_log, "environment": environment(),
        "arms": {},
    }
    for label, _ in arms:
        ts = trials[label]
        times = [t["time_s"] for t in ts]
        q = st.quantiles(times, n=4) if len(times) >= 4 else [min(times), st.median(times), max(times)]
        rec = records[label]
        out["arms"][label] = {
            "trials": ts,                       # every field, every trial
            "trials_s": times,
            "median_s": st.median(times), "min_s": min(times), "max_s": max(times),
            "iqr_pct": (q[2] - q[0]) / st.median(times) * 100,
            # Distributions, not the first sample. Cold path especially: it is
            # dominated by host work whose variance is nothing like the
            # kernel's.
            "summary": {k: summarise(ts, k) for k in TRIAL_FIELDS},
            "t_stats_s": (summarise(ts, "t_stats_s") or {}).get("median"),
            "t_finalize_s": (summarise(ts, "t_finalize_s") or {}).get("median"),
            "t_allreduce_s": (summarise(ts, "t_allreduce_s") or {}).get("median"),
            "t_plan_s": (summarise(ts, "t_plan_s") or {}).get("median"),
            "t_plan_metrics_s": (summarise(ts, "t_plan_metrics_s") or {}).get("median"),
            "cold_data_path_s": (summarise(ts, "cold_data_path_s") or {}).get("median"),
            "occupancy": rec.get("occupancy"),
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
    hdr = (f"\n{'arm':<24s}{'median ms':>11s}{'IQR%':>7s}{'stats ms':>10s}"
           f"{'plan ms':>10s}{'metrics ms':>12s}{'cold ms':>10s}{'coldIQR%':>10s}"
           f"{'vs best':>9s}")
    print(hdr)

    def ms(v, w=10, p=4):
        return f"{v * 1e3:{w}.{p}f}" if v is not None else " " * w

    for label in sorted(out["arms"], key=lambda k: out["arms"][k]["median_s"]):
        a = out["arms"][label]
        cold = a["summary"].get("cold_data_path_s")
        cold_med = cold["median"] if cold else None
        cold_iqr = f"{cold['iqr_pct']:10.2f}" if cold else " " * 10
        rel = a["median_s"] / out["arms"][base]["median_s"] - 1
        print(f"{label:<24s}{a['median_s'] * 1e3:11.4f}{a['iqr_pct']:7.2f}"
              f"{ms(a['t_stats_s'])}{ms(a['t_plan_s'], 10, 2)}"
              f"{ms(a['t_plan_metrics_s'], 12, 2)}{ms(cold_med)}{cold_iqr}"
              f"{rel:+9.2%}")


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
    rows = []
    for first in order:
        if first == "bc":
            rb, rc = run_once(bbuild(args.fixture)), run_once(cbuild(args.fixture))
        else:
            rc, rb = run_once(cbuild(args.fixture)), run_once(bbuild(args.fixture))
        rows.append({"first": first,
                     "baseline_s": time_of(rb), "candidate_s": time_of(rc),
                     "baseline": trial_of(rb), "candidate": trial_of(rc)})

    def effect(getter):
        """Paired log-ratio effect over blocks, or None if a side lacks data."""
        pairs = [(getter(r["baseline"]), getter(r["candidate"])) for r in rows]
        pairs = [(b, c) for b, c in pairs if b and c]
        if len(pairs) < 2:
            return None
        lr = [math.log(c / b) for b, c in pairs]
        m, se = st.mean(lr), st.stdev(lr) / math.sqrt(len(lr))
        return {"blocks": len(lr),
                "effect_pct": (math.exp(m) - 1) * 100,
                "ci_lo_pct": (math.exp(m - 1.96 * se) - 1) * 100,
                "ci_hi_pct": (math.exp(m + 1.96 * se) - 1) * 100,
                "baseline_median_s": st.median(b for b, _ in pairs),
                "candidate_median_s": st.median(c for _, c in pairs)}

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
        # A steady-state win and a cold-path loss are both real and they point
        # opposite ways for this change, so both get an interval.
        "effect_by_stage": {k: effect(lambda t, k=k: t.get(k))
                            for k in ("cold_data_path_s", "t_plan_s",
                                      "t_stats_s", "t_finalize_s",
                                      "t_allreduce_s")},
        "environment": environment(),
    }
    write(out, args.tag or "paired")
    print(f"\n{clab} vs {blab} on {os.path.basename(args.fixture)}")
    print(f"  baseline  {out['baseline_median_s'] * 1e3:9.4f} ms")
    print(f"  candidate {out['candidate_median_s'] * 1e3:9.4f} ms")
    print(f"  effect {out['effect_pct']:+.2f}%  "
          f"95% CI [{out['ci_lo_pct']:+.2f}%, {out['ci_hi_pct']:+.2f}%]  "
          f"({'excludes' if out['ci_lo_pct'] * out['ci_hi_pct'] > 0 else 'INCLUDES'} zero)")
    for k, e in out["effect_by_stage"].items():
        if e:
            print(f"    {k:<20s} {e['baseline_median_s']*1e3:9.3f} -> "
                  f"{e['candidate_median_s']*1e3:9.3f} ms  {e['effect_pct']:+8.2f}%  "
                  f"[{e['ci_lo_pct']:+.2f}, {e['ci_hi_pct']:+.2f}]")


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
