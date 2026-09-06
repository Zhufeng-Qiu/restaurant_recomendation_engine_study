"""Compression gain against how much of the iteration is communication.

The figure this replaces plotted two hosts as bars and was captioned as three
regimes -- it never contained the `PHB` point at all, and a two-bar chart
invites "the link causes this", which separate machines measured at different
times cannot support.

Every point below is read from its own result file rather than transcribed,
because the previous version hard-coded an archived 5-trial pair (91.1%, 55.5%)
that later work had already superseded (93.3%, 55.41%).

The y-axis is one consistent statistic: the ratio of the two medians in the
same run. Two of the points also have a paired estimate with a CI, noted in the
caption; mixing the two estimators on one axis is how the numbers drifted last
time.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bench_point(name):
    d = json.load(open(os.path.join(ROOT, "results/bench", name)))
    by = {r["config"]: r for r in d["results"] if "median_s" in r}
    f, p = by["nccl_sync_g2_f64"], by["nccl_sync_g2_packed"]
    return (f["median_allreduce_s"] / f["median_s"] * 100,
            (1 - p["median_s"] / f["median_s"]) * 100)


def p2p_point(suffix):
    d = json.load(open(os.path.join(ROOT, "results/bench/p2p_ab.json")))
    f, p = d[f"sync_f64_{suffix}"], d[f"sync_packed_{suffix}"]
    return (f["allreduce_ms"] / f["median_ms"] * 100,
            (1 - p["median_ms"] / f["median_ms"]) * 100)


def main():
    nv_old = bench_point("bench_20260905_064736.json")     # NVLink, pre-packing
    nv_now = bench_point("bench_20260906_044118.json")     # NVLink, current default
    phb = bench_point("bench_20260905_082024.json")        # PCIe PHB, no P2P
    on, off = p2p_point("p2pon"), p2p_point("p2poff")      # one host, flag toggled

    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    xs, ys = zip(nv_old, on, phb)
    ax.plot(xs, ys, "o", ms=11, color="#1f77b4", zorder=3,
            label="three hosts, comparable protocol (descriptive)")
    for (x, y), lab, off_xy in ((nv_old, "NVLink NV12", (4, -20)),
                                (on, "PCIe SYS, P2P available", (-6, -22)),
                                (phb, "PCIe PHB, no P2P", (0, 14))):
        ax.annotate(lab, (x, y), xytext=off_xy, textcoords="offset points",
                    ha="center", fontsize=9, color="#1f77b4")

    ax.plot([on[0], off[0]], [on[1], off[1]], "s-", ms=8, color="#d62728",
            zorder=4, label="one host, NCCL_P2P_DISABLE toggled (controlled)")
    ax.annotate("P2P off", (off[0], off[1]), xytext=(12, -2),
                textcoords="offset points", fontsize=9, color="#d62728")

    ax.plot([nv_now[0]], [nv_now[1]], "^", ms=12, color="#2ca02c", zorder=5,
            label="NVLink, current default (after warp packing)")

    ax.set_xlabel("AllReduce as a share of the iteration, `f64` (%)")
    ax.set_ylabel("what 3x lossless compression buys (%)")
    ax.set_title("Compression tends to pay more as the collective occupies\n"
                 "more of the iteration — not according to the link's name")
    ax.grid(alpha=.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 64)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.text(0.5, -0.155,
            "Ratio of medians within each run. The three blue points are "
            "different physical machines run under a comparable protocol — not\n"
            "a verified identical binary, since those runs kept no build hash — "
            "so read them as regimes, not a swept variable. The joined pair is\n"
            "one host with a single flag changed and moves the same way. Paired "
            "estimates exist for two: NVLink −4.96% (3 sessions), PHB −51.67%\n"
            "[−54.61, −48.55]; they agree in sign and magnitude with the "
            "median ratios plotted.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            color="#444444")

    fig.tight_layout(rect=[0, 0.14, 1, 1])
    out = os.path.join(ROOT, "results/figures/compression_regimes.png")
    fig.savefig(out, dpi=110)
    print("wrote:", os.path.relpath(out, ROOT))
    for lab, pt in (("NVLink pre-packing", nv_old), ("NVLink current", nv_now),
                    ("PCIe SYS P2P on", on), ("PCIe SYS P2P off", off),
                    ("PCIe PHB no P2P", phb)):
        print(f"  {lab:<22s} share {pt[0]:5.1f}%   compression {pt[1]:5.2f}%")


if __name__ == "__main__":
    main()
