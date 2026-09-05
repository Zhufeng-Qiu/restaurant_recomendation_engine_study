"""Generate scaling and comparison figures from a run_bench.py JSON file.

Usage:
    python engine/bench/make_figures.py results/bench/bench_<stamp>.json

    # interconnect A/B — writes only interconnect_comparison.png, so it cannot
    # overwrite figures whose documented source is a different run
    python engine/bench/make_figures.py <a.json> --compare <b.json> \
        --labels "NVLink (NV12),PCIe (PHB, no P2P)"

Writes results/figures/*.png. CPU figures always; the GPU comparison and the
AllReduce payload comparison only when the JSON contains those configs.
"""

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAYLOADS = ("f64", "i32", "packed")
PAYLOAD_LABEL = {"f64": "f64 (6x float64)", "i32": "i32 (6x int32)",
                 "packed": "packed (2x uint64)"}


def iqr_err(rec):
    """Asymmetric [lower, upper] error from p25/p75 when the run recorded them.

    Runs before 2026-09-04 have no quartiles; fall back to min/max so older
    bench files still plot, and return None only when neither exists. Without
    this the async 2-GPU bars look as settled as the sync ones, and they are
    not -- they carry 12-44% IQR against sync's 0.3%.
    """
    m = rec["median_s"]
    lo = rec.get("p25_s")
    hi = rec.get("p75_s")
    if lo is None or hi is None:
        lo, hi = rec.get("min_s"), rec.get("max_s")
    if lo is None or hi is None:
        return None
    return [max(0.0, m - lo), max(0.0, hi - m)]


def err_pair(recs):
    """Column-wise [[lower...],[upper...]] for a list of records, or None."""
    es = [iqr_err(r) for r in recs]
    if any(e is None for e in es):
        return None
    return [[e[0] for e in es], [e[1] for e in es]]


def nccl_name(by_name, mode, gpus, payload):
    """Resolve an nccl config across the pre/post --payload naming.

    Runs predating --payload are named nccl_{mode}_g{gpus} and were f64, so
    the bare spelling resolves only for f64. Returns None if absent.
    """
    candidates = [f"nccl_{mode}_g{gpus}_{payload}"]
    if payload == "f64":
        candidates.append(f"nccl_{mode}_g{gpus}")
    for n in candidates:
        if n in by_name:
            return n
    return None


def load_configs(path):
    with open(path) as f:
        doc = json.load(f)
    return {r["config"]: r for r in doc["results"] if "median_s" in r}, doc


def interconnect_comparison(path_a, path_b, labels, out_dir):
    """A/B the same payload sweep across two interconnects.

    Left panel is the mechanism (what the collective costs, log scale because
    the two links differ by ~2 orders of magnitude); right panel is the
    consequence, normalised per link so the regime flip is visible despite
    that gap.
    """
    a, doc_a = load_configs(path_a)
    b, _ = load_configs(path_b)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    width = 0.38
    xs = range(len(PAYLOADS))
    for i, (D, label) in enumerate(((a, labels[0]), (b, labels[1]))):
        off = (i - 0.5) * width
        vals, tags = [], []
        for p in PAYLOADS:
            n = nccl_name(D, "sync", 2, p)
            r = D[n] if n else None
            ar = r.get("median_allreduce_s") if r else None
            vals.append(ar * 1e3 if ar else float("nan"))
            # Effective algorithmic bandwidth: bytes moved / wall time.
            gb = (r["allreduce_bytes"] / 1e9 / ar) if (r and ar) else None
            tags.append(f"{ar*1e3:.2f} ms\n{gb:.2f} GB/s" if gb else "")
        bars = ax1.bar([x + off for x in xs], vals, width, label=label)
        for bar, v, t in zip(bars, vals, tags):
            if t:
                ax1.text(bar.get_x() + bar.get_width() / 2, v, t, ha="center",
                         va="bottom", fontsize=7)
    ax1.set_yscale("log")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([f"{p}\n{48 if p=='f64' else 24 if p=='i32' else 16} B/pair"
                         for p in PAYLOADS])
    ax1.set_ylabel("AllReduce time (ms, log scale)")
    ax1.set_title("Collective cost — sync, 2 GPU")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")
    ax1.margins(y=0.45)

    combos = [("sync", "f64"), ("sync", "packed"),
              ("async", "f64"), ("async", "packed")]
    for i, (D, label) in enumerate(((a, labels[0]), (b, labels[1]))):
        off = (i - 0.5) * width
        base = D[nccl_name(D, "sync", 2, "f64")]["median_s"]
        vals = [100.0 * D[nccl_name(D, m, 2, p)]["median_s"] / base
                for m, p in combos]
        abs_ms = [D[nccl_name(D, m, 2, p)]["median_s"] * 1e3 for m, p in combos]
        bars = ax2.bar([x + off for x in range(len(combos))], vals, width,
                       label=label)
        for bar, v, ms in zip(bars, vals, abs_ms):
            ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}%\n{ms:.1f} ms",
                     ha="center", va="bottom", fontsize=7)
    ax2.axhline(100, color="black", lw=0.8, ls="--")
    ax2.set_xticks(list(range(len(combos))))
    ax2.set_xticklabels([f"{m}\n{p}" for m, p in combos])
    ax2.set_ylabel("Total device time (% of that link's sync f64)")
    ax2.set_title("Same techniques, opposite verdicts")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    ax2.margins(y=0.28)

    fixture = os.path.basename(doc_a["fixture"])
    fig.suptitle(f"{fixture} · 2 GPU · median of {doc_a['repeats']} trials "
                 f"after {doc_a['warmups']} warm-ups", y=0.02, fontsize=8,
                 va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(out_dir, "interconnect_comparison.png")
    fig.savefig(p, dpi=150)
    print("wrote:", p)


def main():
    bench_path = sys.argv[1]
    out_dir = os.path.join(os.path.dirname(os.path.dirname(bench_path)), "figures")
    os.makedirs(out_dir, exist_ok=True)

    if "--compare" in sys.argv:
        other = sys.argv[sys.argv.index("--compare") + 1]
        labels = ("A", "B")
        if "--labels" in sys.argv:
            labels = tuple(sys.argv[sys.argv.index("--labels") + 1].split(",", 1))
        interconnect_comparison(bench_path, other, labels, out_dir)
        return

    with open(bench_path) as f:
        doc = json.load(f)
    # Configs that failed are recorded with an "error" and no timings; drop
    # them so the "is this config present" guards below skip them.
    by_name = {r["config"]: r for r in doc["results"] if "median_s" in r}
    fixture = os.path.basename(doc["fixture"])
    cpu = doc["environment"]["cpu"]
    src = f"{fixture} · {cpu} · median of {doc['repeats']} trials after {doc['warmups']} warm-ups"
    threads = [1, 2, 4, 8, 16]
    # Rank points come from the data: --ranks is configurable, and a many-core
    # host sweeps well past the 1,2,4,8 a laptop can use.
    ranks = sorted(int(m.group(1)) for m in
                   (re.match(r"mpi_r(\d+)$", c) for c in by_name) if m)

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
    # Must match the README's headline table, which reports `packed` -- the
    # best configuration -- and includes MPI. This figure previously showed
    # f64 and omitted MPI, so it disagreed with the table directly beneath it.
    gpu_specs = [("mpi_r16", "MPI 16 ranks"),
                 ("cuda_1gpu", "CUDA 1 GPU"),
                 (nccl_name(by_name, "sync", 1, "packed"), "NCCL sync 1 GPU"),
                 (nccl_name(by_name, "sync", 2, "packed"), "NCCL sync 2 GPU"),
                 (nccl_name(by_name, "async", 2, "packed"), "NCCL async 2 GPU")]
    gpu_specs = [(n, lab) for n, lab in gpu_specs if n and n in by_name]
    if gpu_specs:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        best_omp = min((by_name[f"openmp_t{t}_dynamic"]["median_s"], t)
                       for t in threads if f"openmp_t{t}_dynamic" in by_name)
        names = ["serial", f"openmp_t{best_omp[1]}_dynamic"] + [n for n, _ in gpu_specs]
        labels = (["serial", f"OMP best ({best_omp[1]}t)"]
                  + [lab for _, lab in gpu_specs])
        med = [by_name[n]["median_s"] for n in names]
        e = err_pair([by_name[n] for n in names])
        bars = ax.bar(labels, med, yerr=e, capsize=4,
                      error_kw={"ecolor": "0.25", "lw": 1.2})
        for b, m, n in zip(bars, med, names):
            top = m + (iqr_err(by_name[n]) or [0, 0])[1]
            ax.text(b.get_x() + b.get_width() / 2, top, f"{m*1e3:.1f} ms",
                    ha="center", va="bottom", fontsize=8)
        # Log scale is not decoration here: the span is 1258 ms to 4.1 ms, so a
        # linear axis flattens every GPU bar to the baseline and hides the IQR
        # whiskers that are the point of showing them.
        ax.set_yscale("log")
        ax.set_ylabel("Compute time (s, log scale)")
        ax.set_title(f"CPU vs GPU backend latency — {fixture}")
        ax.grid(alpha=0.3, axis="y", which="both")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.suptitle(src, y=0.02, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        p4 = os.path.join(out_dir, "gpu_comparison.png")
        fig.savefig(p4, dpi=150)
        print("wrote:", p4)

    # --- Figure 5 (GPU host only): AllReduce payload comparison ----------------
    # Left: total device time per (mode, gpus) for each payload — does a 3x
    # smaller collective show up end to end? Right: the AllReduce itself, which
    # is only separately measurable in sync mode (the async pipeline overlaps
    # it into a single timed region by construction).
    combos = [(m, g) for m in ("sync", "async") for g in (1, 2)]
    present = [p for p in PAYLOADS
               if any(nccl_name(by_name, m, g, p) for m, g in combos)]
    combos = [(m, g) for m, g in combos
              if all(nccl_name(by_name, m, g, p) for p in present)]
    if len(present) > 1 and combos:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
        width = 0.8 / len(present)
        xs = range(len(combos))
        for i, p in enumerate(present):
            off = (i - (len(present) - 1) / 2) * width
            recs = [by_name[nccl_name(by_name, m, g, p)] for m, g in combos]
            vals = [r["median_s"] * 1e3 for r in recs]
            e = err_pair(recs)
            if e is not None:
                e = [[v * 1e3 for v in e[0]], [v * 1e3 for v in e[1]]]
            bars = ax1.bar([x + off for x in xs], vals, width,
                           label=PAYLOAD_LABEL[p], yerr=e, capsize=3,
                           error_kw={"ecolor": "0.25", "lw": 1.0})
            for b, v in zip(bars, vals):
                ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                         ha="center", va="bottom", fontsize=7)
        ax1.set_xticks(list(xs))
        ax1.set_xticklabels([f"{m}\n{g} GPU" for m, g in combos])
        ax1.set_ylabel("Total device time (ms)")
        ax1.set_title("Device time by AllReduce payload")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3, axis="y")

        # AllReduce time vs payload, sync only, widest GPU count available.
        sync_g = [g for g in (2, 1)
                  if all(nccl_name(by_name, "sync", g, p) for p in present)]
        drawn = False
        if sync_g:
            g = sync_g[0]
            names = [nccl_name(by_name, "sync", g, p) for p in present]
            ar = [by_name[n].get("median_allreduce_s") for n in names]
            if all(v is not None for v in ar):
                bars = ax2.bar([PAYLOAD_LABEL[p].split(" ")[0] for p in present],
                               [v * 1e3 for v in ar], color="tab:orange")
                for b, v, n in zip(bars, ar, names):
                    mb = by_name[n].get("allreduce_bytes")
                    tag = f"{v*1e3:.2f} ms"
                    if mb:
                        tag += f"\n{mb / 1e6:.1f} MB"
                    ax2.text(b.get_x() + b.get_width() / 2, v * 1e3, tag,
                             ha="center", va="bottom", fontsize=8)
                ax2.set_ylabel("AllReduce time (ms)")
                ax2.set_title(f"Collective cost vs payload — sync, {g} GPU")
                ax2.grid(alpha=0.3, axis="y")
                ax2.margins(y=0.2)
                drawn = True
        if not drawn:
            ax2.axis("off")
            ax2.text(0.5, 0.5, "no separable AllReduce timing\nin this run",
                     ha="center", va="center", fontsize=9)
        fig.suptitle(src, y=0.02, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        p5 = os.path.join(out_dir, "payload_comparison.png")
        fig.savefig(p5, dpi=150)
        print("wrote:", p5)


if __name__ == "__main__":
    main()
