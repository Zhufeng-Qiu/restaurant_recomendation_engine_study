"""Generate CPU scaling figures from a run_bench.py JSON file.

Usage:
    python engine/bench/make_figures.py results/bench/bench_<stamp>.json

Writes results/figures/*.png. GPU figures are added in the GPU phases.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    bench_path = sys.argv[1]
    with open(bench_path) as f:
        doc = json.load(f)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(bench_path)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    by_name = {r["config"]: r for r in doc["results"]}
    fixture = os.path.basename(doc["fixture"])
    cpu = doc["environment"]["cpu"]
    src = f"{fixture} · {cpu} · median of {doc['repeats']} trials after {doc['warmups']} warm-ups"
    threads = [1, 2, 4, 8, 16]
    ranks = [1, 2, 4, 8]

    # --- Figure 1: OpenMP speedup and efficiency -----------------------------
    serial_t = by_name["serial"]["median_s"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for sched, marker in (("static", "o"), ("dynamic", "s")):
        sp = [serial_t / by_name[f"openmp_t{t}_{sched}"]["median_s"] for t in threads]
        ax1.plot(threads, sp, marker=marker, label=f"schedule({sched})")
        ax2.plot(threads, [s / t for s, t in zip(sp, threads)], marker=marker,
                 label=f"schedule({sched})")
    ax1.plot(threads, threads, "k:", linewidth=1, label="ideal")
    ax1.set_xlabel("OpenMP threads")
    ax1.set_ylabel("Speedup vs serial  S(p) = T(1)/T(p)")
    ax1.set_title("OpenMP speedup — Pearson pair evaluation")
    ax1.set_xticks(threads)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("OpenMP threads")
    ax2.set_ylabel("Parallel efficiency  E(p) = S(p)/p")
    ax2.set_title("OpenMP parallel efficiency")
    ax2.set_xticks(threads)
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle(src, y=0.02, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p1 = os.path.join(out_dir, "openmp_scaling.png")
    fig.savefig(p1, dpi=150)

    # --- Figure 2: MPI strong scaling and communication fraction --------------
    if "mpi_r1" in by_name:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        t1 = by_name["mpi_r1"]["median_s"]
        sp = [t1 / by_name[f"mpi_r{r}"]["median_s"] for r in ranks]
        ax1.plot(ranks, sp, marker="o", label="measured")
        ax1.plot(ranks, ranks, "k:", linewidth=1, label="ideal")
        ax1.set_xlabel("MPI ranks")
        ax1.set_ylabel("Speedup vs 1 rank  S(p) = T(1)/T(p)")
        ax1.set_title("MPI strong scaling — partial-stats AllReduce")
        ax1.set_xticks(ranks)
        ax1.legend()
        ax1.grid(alpha=0.3)
        cf = [100 * by_name[f"mpi_r{r}"].get("comm_fraction", 0.0) for r in ranks]
        ax2.bar([str(r) for r in ranks], cf)
        ax2.set_xlabel("MPI ranks")
        ax2.set_ylabel("AllReduce share of iteration time (%)")
        ax2.set_title("Communication fraction (56 MB six-stat tensor)")
        ax2.grid(alpha=0.3, axis="y")
        fig.suptitle(src, y=0.02, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        p2 = os.path.join(out_dir, "mpi_scaling.png")
        fig.savefig(p2, dpi=150)

    # --- Figure 3: runtime overview -------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = (["serial"] + [f"openmp_t{t}_dynamic" for t in threads]
             + [f"mpi_r{r}" for r in ranks if f"mpi_r{r}" in by_name])
    labels = (["serial"] + [f"OMP {t}t" for t in threads]
              + [f"MPI {r}r" for r in ranks if f"mpi_r{r}" in by_name])
    med = [by_name[n]["median_s"] for n in names]
    err = [[by_name[n]["median_s"] - by_name[n]["min_s"] for n in names],
           [by_name[n]["max_s"] - by_name[n]["median_s"] for n in names]]
    ax.bar(labels, med, yerr=err, capsize=3)
    ax.set_ylabel("Compute time (s)")
    ax.set_title(f"Pearson kernel compute time by backend — {fixture}")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle(src, y=0.02, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p3 = os.path.join(out_dir, "cpu_runtime.png")
    fig.savefig(p3, dpi=150)

    print("wrote:", p1)
    if "mpi_r1" in by_name:
        print("wrote:", p2)
    print("wrote:", p3)

    # --- Figure 4 (GPU host only): backend latency comparison ------------------
    gpu_names = [n for n in ("cuda_1gpu", "nccl_sync_g1", "nccl_sync_g2",
                             "nccl_async_g2") if n in by_name]
    if gpu_names:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        best_omp = min((by_name[f"openmp_t{t}_dynamic"]["median_s"], t)
                       for t in threads if f"openmp_t{t}_dynamic" in by_name)
        names = ["serial", f"openmp_t{best_omp[1]}_dynamic"] + gpu_names
        labels = ["serial", f"OMP best ({best_omp[1]}t)"] + [
            {"cuda_1gpu": "CUDA 1 GPU", "nccl_sync_g1": "NCCL sync 1 GPU",
             "nccl_sync_g2": "NCCL sync 2 GPU",
             "nccl_async_g2": "NCCL async 2 GPU"}[n] for n in gpu_names]
        med = [by_name[n]["median_s"] for n in names]
        bars = ax.bar(labels, med)
        for b, m in zip(bars, med):
            ax.text(b.get_x() + b.get_width() / 2, m, f"{m*1e3:.1f} ms",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Compute time (s)")
        ax.set_title(f"CPU vs GPU backend latency — {fixture}")
        ax.grid(alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.suptitle(src, y=0.02, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        p4 = os.path.join(out_dir, "gpu_comparison.png")
        fig.savefig(p4, dpi=150)
        print("wrote:", p4)


if __name__ == "__main__":
    main()
