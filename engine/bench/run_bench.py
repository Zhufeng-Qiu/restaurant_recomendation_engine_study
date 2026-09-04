"""Benchmark runner for pearson_engine backends.

Runs each configuration with warm-ups and repeated trials, parses the JSON
line each binary prints, and writes raw results + a median summary to
machine-readable JSON and CSV under results/bench/.

Usage:
    python3 engine/bench/run_bench.py [--fixture data/fixtures/item_full]
                                      [--repeats 5] [--warmups 2]
                                      [--out results/bench]
                                      [--gpu] [--payloads f64,i32,packed]

Configurations run (all on the chosen fixture):
    serial
    openmp  threads in {1,2,4,8,16} x schedule in {static, dynamic,1024}
    mpi     ranks in {1,2,4,8}            (skipped if pearson_engine_mpi absent)
    --gpu adds:
    cuda    single GPU (pearson_engine --backend cuda)
    nccl    gpus in {1,2} x mode in {sync, async} x payload in --payloads,
            named nccl_{mode}_g{gpus}_{payload}      (pearson_engine_nccl)

Runs before the --payload feature are named nccl_{mode}_g{gpus} with no
suffix; those are f64 runs. make_figures.py resolves both spellings.
"""

import argparse
import csv
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(ROOT, "engine", "build", "pearson_engine")
ENGINE_MPI = os.path.join(ROOT, "engine", "build", "pearson_engine_mpi")
ENGINE_NCCL = os.path.join(ROOT, "engine", "build", "pearson_engine_nccl")


def time_of(trial):
    """Comparable compute time for a trial, whatever backend produced it.

    GPU trials count device-side work (kernels + collectives), excluding
    one-time context creation and H2D/D2H staging, mirroring what the CPU
    numbers measure (compute only, not fixture load).
    """
    if "t_kernel_stats_s" in trial:                  # cuda (single GPU)
        return trial["t_kernel_stats_s"] + trial["t_kernel_finalize_s"]
    if "t_compute_s" in trial:                       # serial / openmp
        return trial["t_compute_s"]
    if trial.get("mode") == "async":                 # nccl async pipeline
        return trial["t_pipeline_s"]
    if "t_kernel_s" in trial:                        # nccl sync
        return trial["t_kernel_s"] + trial["t_allreduce_s"]
    # mpi
    return trial["t_local_s"] + trial["t_allreduce_s"] + trial["t_finalize_s"]


def sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def run_once(cmd, env_extra):
    env = dict(os.environ, **env_extra)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{p.stderr[-2000:]}")
    # Backends may print auxiliary JSON lines (e.g. cuda_detail) before the
    # summary line; merge them all into one flat record.
    rec = {}
    for line in p.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            rec.update(d.pop("cuda_detail", {}))
            rec.update(d)
    return rec


def bench(cmd, env_extra, warmups, repeats):
    for _ in range(warmups):
        run_once(cmd, env_extra)
    trials = [run_once(cmd, env_extra) for _ in range(repeats)]
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=os.path.join(ROOT, "data/fixtures/item_full"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/bench"))
    ap.add_argument("--gpu", action="store_true",
                    help="add cuda + nccl configurations (GPU host only)")
    ap.add_argument("--payloads", default="f64,i32,packed",
                    help="comma-separated AllReduce payload representations to "
                         "sweep for the nccl backend (contract §9)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    configs = [{"name": "serial", "cmd": [ENGINE, args.fixture, "--backend", "serial",
                                          "--validate"], "env": {}, "threads": 1}]
    for t in (1, 2, 4, 8, 16):
        for sched in ("static", "dynamic,1024"):
            configs.append({
                "name": f"openmp_t{t}_{sched.split(',')[0]}",
                "cmd": [ENGINE, args.fixture, "--backend", "openmp", "--validate"],
                "env": {"OMP_NUM_THREADS": str(t), "OMP_SCHEDULE": sched},
                "threads": t,
            })
    if os.path.exists(ENGINE_MPI):
        for r in (1, 2, 4, 8):
            configs.append({
                "name": f"mpi_r{r}",
                "cmd": ["mpirun", "-np", str(r), ENGINE_MPI, args.fixture, "--validate"],
                "env": {}, "threads": r,
            })
    if args.gpu:
        configs.append({"name": "cuda_1gpu",
                        "cmd": [ENGINE, args.fixture, "--backend", "cuda",
                                "--validate"], "env": {}, "threads": 1})
        for payload in args.payloads.split(","):
            for mode in ("sync", "async"):
                for g in (1, 2):
                    configs.append({
                        "name": f"nccl_{mode}_g{g}_{payload}",
                        "cmd": [ENGINE_NCCL, args.fixture, "--gpus", str(g),
                                "--mode", mode, "--payload", payload,
                                "--validate"],
                        "env": {}, "threads": g,
                    })

    results = []
    for cfg in configs:
        # A config that fails must not cost the whole session: the GPU
        # configurations run last, and an exception there would discard the CPU
        # results measured before it. Record the failure and carry on.
        try:
            trials = bench(cfg["cmd"], cfg["env"], args.warmups, args.repeats)
            times = [time_of(t) for t in trials]
        except Exception as exc:
            results.append({"config": cfg["name"],
                            "threads_or_ranks": cfg["threads"],
                            "error": f"{type(exc).__name__}: {exc}"})
            print(f"{cfg['name']:26s} FAILED: {type(exc).__name__}: {exc}")
            continue
        # Quartiles need enough samples to mean anything; below 4 trials
        # report them as None rather than a number computed from 2 points.
        if len(times) >= 4:
            q1, _, q3 = statistics.quantiles(times, n=4)
            iqr, p25, p75 = q3 - q1, q1, q3
        else:
            iqr = p25 = p75 = None
        rec = {
            "config": cfg["name"],
            "threads_or_ranks": cfg["threads"],
            "median_s": statistics.median(times),
            "min_s": min(times),
            "max_s": max(times),
            "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
            "p25_s": p25,
            "p75_s": p75,
            "iqr_s": iqr,
            "n_trials": len(times),
            "max_abs_diff": max(t["max_abs_diff"] for t in trials),
            "tol_failures": max(t["tol_failures"] for t in trials),
            "trials": trials,
        }
        for field in ("payload", "payload_bytes_per_pair", "allreduce_bytes"):
            if field in trials[0]:
                rec[field] = trials[0][field]
        # The async pipeline reports t_allreduce_s = 0: its collectives run
        # inside the timed pipeline and are not separately measurable. Leave
        # the field out rather than recording a 0 that reads as "free".
        if any(t.get("t_allreduce_s", 0.0) > 0.0 for t in trials):
            rec["median_allreduce_s"] = statistics.median(t["t_allreduce_s"] for t in trials)
            if "comm_fraction" in trials[0]:
                rec["comm_fraction"] = statistics.median(t["comm_fraction"] for t in trials)
            elif rec["median_s"] > 0:
                rec["comm_fraction"] = rec["median_allreduce_s"] / rec["median_s"]
        results.append(rec)
        print(f"{cfg['name']:26s} median={rec['median_s']:.4f}s "
              f"stdev={rec['stdev_s']:.4f}s max_diff={rec['max_abs_diff']:.1e}")

    doc = {
        "schema": "pearson-bench-v1",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "fixture": args.fixture,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
            "ncpu": sysctl("hw.ncpu"),
            "python": sys.version.split()[0],
        },
        "results": results,
    }
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out, f"bench_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(doc, f, indent=2)

    csv_path = os.path.join(args.out, f"bench_{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "threads_or_ranks", "n_trials", "median_s",
                    "min_s", "max_s", "stdev_s", "iqr_s", "p25_s", "p75_s",
                    "payload", "allreduce_bytes", "median_allreduce_s",
                    "comm_fraction", "max_abs_diff", "tol_failures", "error"])
        for r in results:
            if "error" in r:
                w.writerow([r["config"], r["threads_or_ranks"]] + [""] * 14
                           + [r["error"]])
                continue
            ar = r.get("median_allreduce_s")
            fmt = lambda v: f"{v:.6f}" if v is not None else ""
            w.writerow([r["config"], r["threads_or_ranks"], r.get("n_trials", ""),
                        f"{r['median_s']:.6f}",
                        f"{r['min_s']:.6f}", f"{r['max_s']:.6f}", f"{r['stdev_s']:.6f}",
                        fmt(r.get("iqr_s")), fmt(r.get("p25_s")), fmt(r.get("p75_s")),
                        r.get("payload", ""), r.get("allreduce_bytes", ""),
                        fmt(ar),
                        f"{r.get('comm_fraction', '')}", f"{r['max_abs_diff']:.3e}",
                        r["tol_failures"], ""])
    failed = [r["config"] for r in results if "error" in r]
    print(f"\nwrote {json_path}\nwrote {csv_path}")
    if failed:
        print(f"{len(failed)} config(s) FAILED and are recorded as such: "
              + ", ".join(failed))


if __name__ == "__main__":
    main()
