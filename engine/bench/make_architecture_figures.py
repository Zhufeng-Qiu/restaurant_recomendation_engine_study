"""The two figures for the work-division comparison.

Left to itself, a bar chart of four medians would hide the two things that
actually matter here: that item_full's first ten blocks measured a GPU which
had not reached steady state, and that the paired ratio -- not the ratio of
medians -- is the estimator. So:

  architecture_latency.png   every block, coloured by time segment, so the
                             settling in item_full's segment 0 is visible
                             rather than averaged away
  architecture_ratios.png    the paired geometric mean with its 95% interval,
                             for both the pre-registered all-30 analysis and
                             the settled subset, side by side

Every number is read from the run's own JSON. Nothing is transcribed, and a
missing configuration is an error rather than a gap in a bar chart.

Usage: make_architecture_figures.py [results/bench/architecture_formal_20260917]
"""

import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import architecture_ab as ab

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS = ["A", "B", "C", "D"]
LABEL = {
    "A": "A\n1 GPU\npair split\nf64",
    "B": "B\n2 GPU\ndim split\nf64",
    "C": "C\n2 GPU\ndim split\npacked",
    "D": "D\n2 GPU\npair split\nf64",
}
SEG_COLOUR = ["#c0392b", "#2c6fbb", "#1f8a52"]
SCOPE = ("input already resident on every GPU; one host clock around stats, "
         "the collective where there is one, finalize and the full copy back "
         "to host memory;\nevery batch validated against golden AFTER the "
         "timed loop, never between batches; 500 timed batches per block; "
         "one host, EPYC 7763 + 2x A100-SXM4-80GB NV12")


def load(d, fixture):
    path = os.path.join(d, f"formal_{fixture}.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    doc = json.load(open(path))
    blocks = [b for b in doc["blocks"] if b["complete"]]
    if not blocks:
        raise SystemExit(f"{path}: no complete blocks")
    for b in blocks:
        for c in CONFIGS:
            if c not in b["runs"]:
                raise SystemExit(f"{path}: block {b['block']} has no {c}")
    return doc, blocks


def rows(blocks, keep):
    out = []
    for b in blocks:
        if b["segment"] not in keep:
            continue
        r = {"segment": b["segment"]}
        for c, rec in b["runs"].items():
            r[c] = rec["resident_host_complete_ms_median"]
        out.append(r)
    return out


def latency_figure(data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=False)
    for ax, (fixture, doc, blocks) in zip(axes, data):
        n_pairs = blocks[0]["runs"]["A"]["n_pairs"]
        for i, c in enumerate(CONFIGS):
            vals = [b["runs"][c2]["resident_host_complete_ms_median"]
                    for b in blocks for c2 in CONFIGS]
            typical = statistics.median(vals)
            for b in blocks:
                v = b["runs"][c]["resident_host_complete_ms_median"]
                hot = v > 3 * typical
                ax.plot(i + (b["segment"] - 1) * 0.16, v,
                        "o", ms=6.5 if hot else 4.5,
                        alpha=0.9 if hot else 0.75,
                        mfc="none" if hot else SEG_COLOUR[b["segment"]],
                        mew=1.6 if hot else 0,
                        color=SEG_COLOUR[b["segment"]], zorder=3)
                if hot:
                    ax.annotate(f"block {b['block']}", (i + (b["segment"]-1)*0.16, v),
                                textcoords="offset points", xytext=(9, -1),
                                fontsize=7, color="#444")
            med = statistics.median(
                [b["runs"][c]["resident_host_complete_ms_median"] for b in blocks])
            ax.hlines(med, i - 0.34, i + 0.34, color="#222", lw=1.6, zorder=4)
            ax.text(i + 0.38, med, f"{med:.2f}", va="center", fontsize=8.5,
                    color="#222")
        ax.set_xticks(range(len(CONFIGS)))
        ax.set_xticklabels([LABEL[c] for c in CONFIGS], fontsize=8.5)
        ax.set_title(f"{fixture}  ({n_pairs:,} pairs)", fontsize=11)
        ax.set_ylabel("full batch latency (ms), median of 500")
        # Log scale, because four of the 240 config-blocks ran 5-6x slow and a
        # linear axis spends two thirds of its height on them. Their cause was
        # not diagnosed -- nothing external was being recorded at the time --
        # so they are marked rather than explained, and kept in the figure and
        # in the analysis, since dropping them only makes every effect smaller.
        ax.set_yscale("log")
        ax.set_yticks([2, 3, 4, 5, 7, 10, 15, 20])
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        # A log axis still labels its minor ticks as "6 x 10^0", which is
        # noise next to explicit millisecond ticks.
        ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(axis="y", alpha=0.25, lw=0.6, which="both")
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=SEG_COLOUR[s],
                          label=f"segment {s} (blocks {s*10}-{s*10+9})")
               for s in range(3)]
    handles.append(plt.Line2D([], [], color="#222", lw=1.6,
                              label="median of 30 blocks"))
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mew=1.6,
                              color="#444",
                              label="slower config-blocks, cause undiagnosed (4 of 240)"))
    # Below the axes, not inside them: the legend box sat on top of one of the
    # marked blocks in the left panel.
    fig.legend(handles=handles, fontsize=8, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.075), frameon=False)
    fig.suptitle("Full batch latency by work division, every block shown",
                 fontsize=12.5, y=0.99)
    fig.text(0.5, 0.012, SCOPE, ha="center", fontsize=7.4, color="#555")
    fig.tight_layout(rect=[0, 0.155, 1, 0.96])
    fig.savefig(out_path, dpi=170)
    print(f"wrote {out_path}")


def ratio_figure(data, out_path):
    analyses = []
    for fixture, doc, blocks in data:
        segs = sorted({b["segment"] for b in blocks})
        n_all = len([b for b in blocks if b["segment"] in set(segs)])
        analyses.append((f"{fixture}\nall {n_all} blocks (pre-registered)",
                         blocks, set(segs), segs))
        if fixture == "item_full":
            n_sub = len([b for b in blocks if b["segment"] in {1, 2}])
            # Not "settled": that names a conclusion the label is supposed to
            # stay neutral about. It is a subset chosen after seeing the data.
            analyses.append((f"{fixture}\nsegments 1+2, n={n_sub} (post-hoc)",
                             blocks, {1, 2}, [1, 2]))

    comps = ab.COMPARISONS[:3]
    fig, axes = plt.subplots(1, len(comps), figsize=(14.5, 4.9), sharex=False)
    for ax, (x, y, meaning) in zip(axes, comps):
        ys, labels, spans = [], [], []
        for k, (label, blocks, keep, segs) in enumerate(analyses):
            bl = rows(blocks, keep)
            R, _ = ab.paired_log_ratio(bl, x, y)
            lo, hi = ab.bootstrap_ci(bl, x, y, segs)
            crosses = lo < 1.0 < hi
            colour = "#999" if crosses else "#2c6fbb"
            ax.plot([lo, hi], [k, k], lw=2.4, color=colour, solid_capstyle="round")
            ax.plot([R], [k], "o", ms=7, color=colour, zorder=3)
            ax.text(hi + 0.012, k, f"{(R-1)*100:+.1f}%" + ("  n.e." if crosses else ""),
                    va="center", fontsize=8.4, color=colour)
            ys.append(k)
            labels.append(label)
            spans.append((lo, hi))
        ax.axvline(1.0, color="#c0392b", lw=1.1, ls="--")
        # Room for the per-row percentage labels, which otherwise run off the
        # right edge and collide with the axis frame.
        lo_all = min(lo for lo, _ in spans)
        hi_all = max(hi for _, hi in spans)
        pad = (hi_all - lo_all) * 0.58
        ax.set_xlim(lo_all - (hi_all - lo_all) * 0.06, hi_all + pad)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7.6)
        ax.set_ylim(-0.6, len(analyses) - 0.2)
        ax.set_xlabel(f"{x}/{y}   (R < 1: {x} faster)")
        ax.set_title(meaning, fontsize=9)
        ax.grid(axis="x", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Paired block ratios, geometric mean with 95% bootstrap interval",
                 fontsize=12.5, y=0.995)
    fig.text(0.5, 0.015,
             "The unit is the block, not the iteration: n is shown per row "
             "(30 pre-registered, 20 for the post-hoc item_full subset).\n"
             "Grey with 'n.e.' = interval crosses 1, direction not "
             "established.  " + SCOPE.split(";\n")[1],
             ha="center", fontsize=7.4, color="#555")
    fig.tight_layout(rect=[0, 0.12, 1, 0.94], w_pad=2.6)
    fig.savefig(out_path, dpi=170)
    print(f"wrote {out_path}")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "results/bench/architecture_formal_20260917")
    data = []
    for fixture in ("item_full", "user_full"):
        doc, blocks = load(d, fixture)
        data.append((fixture, doc, blocks))
        print(f"{fixture}: {len(blocks)} complete blocks, "
              f"{len(doc['failures'])} failures")
    outdir = os.path.join(ROOT, "results/figures")
    latency_figure(data, os.path.join(outdir, "architecture_latency.png"))
    ratio_figure(data, os.path.join(outdir, "architecture_ratios.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
