"""Emit the README's headline table from a bench JSON, instead of typing it.

The table in the README is the one number anyone quotes, and every previous
version of it drifted from the run it claimed to describe -- rows from
different machines, speedups against two different serial baselines, a figure
plotting a payload the table did not report. Deriving it removes the step where
that happens.

Usage:  python3 engine/bench/headline_table.py [results/bench/bench_*.json]
        (defaults to the newest bench JSON by modification time)
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (config name, label). The headline reports the best-of-breed configuration
# per backend, and `packed` for NCCL because that is the production payload.
ROWS = [
    ("serial", "Serial C++ (oracle)"),
    ("openmp_t16_dynamic", "OpenMP 16t dynamic"),
    ("mpi_r16", "MPI 16 ranks"),
    ("cuda_1gpu", "CUDA 1 GPU"),
    ("nccl_sync_g2_packed", "**NCCL sync `packed`, 2 GPU**"),
    ("nccl_async_g2_packed", "NCCL async `packed`, 2 GPU"),
]


def main():
    path = (sys.argv[1] if len(sys.argv) > 1
            else max(glob.glob(os.path.join(ROOT, "results/bench/bench_*.json")),
                     key=os.path.getmtime))
    d = json.load(open(path))
    by = {r["config"]: r for r in d["results"] if "median_s" in r}
    if "serial" not in by:
        raise SystemExit("no serial row: speedups need a same-run baseline")
    base = by["serial"]["median_s"]

    env = d.get("environment", {})
    print(f"source: {os.path.relpath(path, ROOT)}")
    print(f"seed:   {d.get('seed')}   repeats: {d.get('repeats')}   "
          f"warmups: {d.get('warmups')}")
    tracked = env.get("git_dirty_tracked")
    dirty = (f"tracked-clean (untracked artifacts only)" if tracked == 0
             else f"{tracked} tracked file(s) modified" if tracked
             else f"dirty={env.get('git_dirty')} (tracked/untracked not split)")
    print(f"git:    {env.get('git_sha')}  {dirty}")
    print(f"image:  {env.get('image_digest')}")
    print(f"cpu:    {env.get('cpu')}")
    for g in (env.get("gpus") or []):
        print(f"gpu:    {g}")
    print()
    print("| Backend | Median | Speedup vs same-machine serial |")
    print("| --- | --- | --- |")
    missing = [c for c, _ in ROWS if c not in by]
    for cfg, label in ROWS:
        r = by.get(cfg)
        if r is None:
            print(f"| {label} | **ABSENT FROM THIS RUN** | |")
            continue
        ms = r["median_s"] * 1e3
        basis = r.get("timing_basis", "?")
        star = "" if basis in ("device_total", "pipeline_total") else f" [{basis}]"
        print(f"| {label} | {ms:.3f} ms{star} | {base / r['median_s']:.2f}x |")

    print()
    lo = min(by[c]["median_s"] for c, _ in ROWS if c in by) * 1e3
    print(f"caption span: {base * 1e3:.1f} ms to {lo:.1f} ms")
    for cfg, label in ROWS:
        r = by.get(cfg)
        if r and r.get("iqr_s") and r.get("median_s"):
            print(f"  {cfg:<24s} IQR {r['iqr_s'] / r['median_s'] * 100:5.2f}%  "
                  f"basis={r.get('timing_basis')}  "
                  f"group={r.get('group')} order={r.get('pair_order')}")
    bad = [r["config"] for r in d["results"]
           if r.get("timing_identity_ok") is False]
    print(f"\ntiming identity failures: {bad or 'none'}")
    # A headline with a hole is not a headline. cuda_1gpu once dropped out at
    # warm-up and this generator still exited 0, so the table could have been
    # published with a row reading "(absent from this run)".
    if missing or bad:
        print(f"\nREFUSING to emit a headline: "
              + (f"missing rows {missing} " if missing else "")
              + (f"timing identity failures {bad}" if bad else ""))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
