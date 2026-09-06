"""The warp-packing figure: what the group size buys, and what the sort costs.

Two panels, because the optimisation has two faces and quoting either alone
misleads. Left: steady-state latency against lanes-per-pair, with and without
the length sort -- where G=4 is chosen, and where G=1 refutes the model that
predicted it. Right: the cold path, where the sort's ~100 ms plan against a
~3 ms kernel is why `bylen` is opt-in rather than default.

The break-even on the right holds the group size fixed at 4 on both arms, so it
is the cost of the SORT. Comparing bylen-at-G4 against source-at-G32 instead
would fold the group change into the sort's payback.

Usage: make_warp_figure.py [group-sweep JSON] [cold-path paired JSON]
"""

import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUPS = [1, 2, 4, 8, 16, 32]


def newest(pattern):
    pat = re.compile(pattern)
    fs = [f for f in glob.glob(os.path.join(ROOT, "results/bench/warp_packing_*.json"))
          if pat.search(os.path.basename(f))]
    if not fs:
        raise SystemExit(f"no file matching {pattern}")
    return sorted(fs)[-1]


def left_panel(ax, arms):
    def series(order):
        return [arms[f"cuda:{g}:{order}"]["median_s"] * 1e3
                if f"cuda:{g}:{order}" in arms else None for g in GROUPS]

    src, byl = series("source"), series("bylen")
    legacy = arms["legacy"]["median_s"] * 1e3 if "legacy" in arms else None
    i4 = GROUPS.index(4)

    # The default is marked with a band, not an arrow: an arrow into the middle
    # of a two-line chart lands on something wherever it starts.
    ax.axvspan(3.4, 4.75, color="#fff2cc", zorder=0)
    ax.text(4, 0.985, "default\nG=4", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color="#8a6d00")

    if legacy:
        ax.axhline(legacy, ls="--", lw=1, color="#d62728", zorder=1)
        ax.text(0.02, legacy + 0.06, f" pre-packing binary {legacy:.2f} ms",
                transform=ax.get_yaxis_transform(), color="#d62728",
                va="bottom", ha="left", fontsize=8.5)

    ax.plot(GROUPS, src, "o-", color="#888888", zorder=3)
    ax.plot(GROUPS, byl, "o-", color="#1f77b4", zorder=3)
    ax.text(1.25, src[0] + 0.05, "fixture order\n(`source`)", color="#666666",
            fontsize=9, ha="left", va="bottom")
    ax.text(1.25, byl[0] + 0.05, "sorted by shorter row\n(`bylen`)",
            color="#1f77b4", fontsize=9, ha="left", va="bottom")
    ax.annotate(f"{src[i4]:.2f}", xy=(4, src[i4]), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9, color="#444444")
    ax.annotate(f"{byl[i4]:.2f}", xy=(4, byl[i4]), xytext=(-30, 3),
                textcoords="offset points", ha="center", fontsize=9, color="#1f77b4")

    ax.text(0.40, 0.02,
            "the lane-slot model calls G=1 optimal (99.96% lane utilisation);\n"
            "measured, it is the worst of the twelve arms",
            transform=ax.transAxes, fontsize=8.5, color="#555555",
            va="bottom", ha="left")

    ax.set_xscale("log", base=2)
    ax.set_xticks(GROUPS)
    ax.set_xticklabels([str(g) for g in GROUPS])
    ax.set_xlim(0.88, 42)
    lo, hi = min(min(src), min(byl)), max(max(src), max(byl))
    ax.set_ylim(lo - 0.85, hi + 0.45)
    ax.set_xlabel("lanes per pair (G)  —  a warp carries 32/G pairs")
    ax.set_ylabel("device_total (ms, steady state)")
    ax.set_title("Steady state: what the group size buys")
    ax.grid(alpha=.3, zorder=0)


def right_panel(bx, cold):
    st = cold["effect_by_stage"]
    base_cold = st["cold_data_path_s"]["baseline_median_s"] * 1e3
    cand_cold = st["cold_data_path_s"]["candidate_median_s"] * 1e3
    base_plan = st["t_plan_s"]["baseline_median_s"] * 1e3
    cand_plan = st["t_plan_s"]["candidate_median_s"] * 1e3
    saved = (cold["baseline_median_s"] - cold["candidate_median_s"]) * 1e3

    labels = ["default\n(`source`)", "opt-in\n(`bylen`)"]
    rest = [base_cold - base_plan, cand_cold - cand_plan]
    plan = [base_plan, cand_plan]
    bx.bar(labels, rest, color="#c6dbef", label="load, staging, compute, readback")
    bx.bar(labels, plan, bottom=rest, color="#e6550d", label="building the lane plan")
    for i, (r, p) in enumerate(zip(rest, plan)):
        bx.text(i, r + p + 2, f"{r + p:.0f} ms", ha="center", va="bottom", fontsize=9)
    bx.text(1, rest[1] + plan[1] / 2, f"{plan[1]:.0f} ms\nsorting", ha="center",
            va="center", fontsize=9.5, color="white", weight="bold")
    bx.set_ylim(0, max(base_cold, cand_cold) * 1.22)
    bx.set_ylabel("cold_data_path (ms)")
    bx.set_title("Cold path: what the sort costs")
    bx.legend(fontsize=8, loc="upper left")
    bx.grid(alpha=.3, axis="y")
    bx.text(0.5, -0.19,
            f"saves {saved:.2f} ms per query, costs {cand_plan - base_plan:.0f} ms once\n"
            f"→ pays back after ~{(cand_plan - base_plan) / saved:.0f} queries on "
            f"1 GPU, ~244 on 2",
            transform=bx.transAxes, ha="center", va="top", fontsize=9)


def main():
    sweep = sys.argv[1] if len(sys.argv) > 1 else newest(r"^warp_packing_cuda_item_full_\d")
    cold = sys.argv[2] if len(sys.argv) > 2 else newest(r"^warp_packing_final_bylen_coldpath_\d")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                 gridspec_kw={"width_ratios": [1.8, 1]})
    left_panel(ax, json.load(open(sweep))["arms"])
    right_panel(bx, json.load(open(cold)))
    fig.suptitle("Warp packing on item_full — a sub-warp per pair instead of a warp",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = os.path.join(ROOT, "results/figures/warp_packing.png")
    fig.savefig(out, dpi=110)
    print("wrote:", os.path.relpath(out, ROOT))
    print(f"  sweep: {os.path.basename(sweep)}\n  cold:  {os.path.basename(cold)}")


if __name__ == "__main__":
    main()
