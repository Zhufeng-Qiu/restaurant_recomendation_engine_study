"""Benchmark runner for pearson_engine backends.

Runs each configuration with warm-ups and repeated trials, parses the JSON
line each binary prints, and writes raw results + a median summary to
machine-readable JSON and CSV under results/bench/.

Usage:
    python3 engine/bench/run_bench.py [--fixture data/fixtures/item_full]
                                      [--repeats 5] [--warmups 2]
                                      [--out results/bench]

Configurations run (all on the chosen fixture):
    serial
    openmp  threads in {1,2,4,8,16} x schedule in {static, dynamic,1024}
    mpi     ranks in {1,2,4,8}            (skipped if pearson_engine_mpi absent)
    --gpu adds:
    cuda    single GPU (pearson_engine --backend cuda)
    nccl    gpus in {1,2} x mode in {sync, async}   (pearson_engine_nccl)
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
        for mode in ("sync", "async"):
            for g in (1, 2):
                configs.append({
                    "name": f"nccl_{mode}_g{g}",
                    "cmd": [ENGINE_NCCL, args.fixture, "--gpus", str(g),
                            "--mode", mode, "--validate"],
                    "env": {}, "threads": g,
                })

    results = []
    for cfg in configs:
        trials = bench(cfg["cmd"], cfg["env"], args.warmups, args.repeats)
        times = [time_of(t) for t in trials]
        rec = {
            "config": cfg["name"],
            "threads_or_ranks": cfg["threads"],
            "median_s": statistics.median(times),
            "min_s": min(times),
            "max_s": max(times),
            "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
            "max_abs_diff": max(t["max_abs_diff"] for t in trials),
            "tol_failures": max(t["tol_failures"] for t in trials),
            "trials": trials,
        }
        if "t_allreduce_s" in trials[0]:
            rec["median_allreduce_s"] = statistics.median(t["t_allreduce_s"] for t in trials)
            if "comm_fraction" in trials[0]:
                rec["comm_fraction"] = statistics.median(t["comm_fraction"] for t in trials)
            elif rec["median_s"] > 0:
                rec["comm_fraction"] = rec["median_allreduce_s"] / rec["median_s"]
        results.append(rec)
        print(f"{cfg['name']:24s} median={rec['median_s']:.4f}s "
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
        w.writerow(["config", "threads_or_ranks", "median_s", "min_s", "max_s",
                    "stdev_s", "comm_fraction", "max_abs_diff", "tol_failures"])
        for r in results:
            w.writerow([r["config"], r["threads_or_ranks"], f"{r['median_s']:.6f}",
                        f"{r['min_s']:.6f}", f"{r['max_s']:.6f}", f"{r['stdev_s']:.6f}",
                        f"{r.get('comm_fraction', '')}", f"{r['max_abs_diff']:.3e}",
                        r["tol_failures"]])
    print(f"\nwrote {json_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
