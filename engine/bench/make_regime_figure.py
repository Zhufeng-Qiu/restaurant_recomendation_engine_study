"""Compression gain against how much of the iteration is communication.

The old interconnect figure plotted two hosts side by side and was captioned as
three regimes -- it never contained the `PHB` point at all. More importantly, a
bar chart of two hosts invites the reading "the link causes this", which the
data cannot support: those are different physical machines measured at
different times.

Plotting the gain against the *communication share* says what is actually
supported. The three cross-host points are descriptive, and they line up. The
two joined points are the controlled version -- one host, `NCCL_P2P_DISABLE`
toggled, everything else fixed -- and they move the same way, which is the
evidence that the trend is about the share and not about which machine it was.

Sources: docs/measurement_audit_20260905.md (three-regime table),
docs/analysis.md (within-host P2P A/B), results/bench/repro_finalsha.json.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (communication share %, compression gain %, label)
CROSS = [(11.6, 4.7, "NVLink NV12"),
         (76.0, 23.5, "PCIe SYS, P2P ok"),
         (91.1, 55.5, "PCIe PHB, no P2P")]
# same host, one variable moved: NCCL_P2P_DISABLE on/off
P2P = [(75.4, 25.2, "P2P on"), (79.9, 33.2, "P2P off")]
# the current default, after warp packing made the compute half smaller
NOW = (9.1, 7.70, "NVLink NV12,\ncurrent default")


def main():
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    xs = [p[0] for p in CROSS]
    ys = [p[1] for p in CROSS]
    ax.plot(xs, ys, "o", ms=11, color="#1f77b4", zorder=3,
            label="three hosts, one binary (descriptive)")
    offsets = {"NVLink NV12": (2, -20), "PCIe SYS, P2P ok": (-8, -20),
               "PCIe PHB, no P2P": (0, 14)}
    for x, y, lab in CROSS:
        dx, dy = offsets[lab]
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", fontsize=9, color="#1f77b4")

    px = [p[0] for p in P2P]
    py = [p[1] for p in P2P]
    ax.plot(px, py, "s-", ms=8, color="#d62728", zorder=4,
            label="one host, P2P toggled (controlled)")
    ax.annotate("NCCL_P2P_DISABLE\non one machine", (px[1], py[1]),
                xytext=(14, -6), textcoords="offset points", fontsize=9,
                color="#d62728", va="top")

    ax.plot([NOW[0]], [NOW[1]], "^", ms=12, color="#2ca02c", zorder=5,
            label="NVLink after warp packing")
    ax.annotate(NOW[2], (NOW[0], NOW[1]), xytext=(8, 12),
                textcoords="offset points", fontsize=9, color="#2ca02c")

    ax.set_xlabel("AllReduce as a share of the iteration (%)")
    ax.set_ylabel("what 3x lossless compression buys (%)")
    ax.set_title("Compression pays in proportion to how much of the iteration\n"
                 "is communication — not according to the link's name")
    ax.grid(alpha=.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 64)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.5, -0.155,
            "Cross-host points are different physical machines measured at "
            "different times: read them as regimes, not as a swept variable.\n"
            "The joined pair is the same host with one flag changed, and it "
            "moves the same way. NVLink points differ because warp packing\n"
            "shrank the compute half, which raises communication's share.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color="#444444")

    fig.tight_layout(rect=[0, 0.11, 1, 1])
    out = os.path.join(ROOT, "results/figures/compression_regimes.png")
    fig.savefig(out, dpi=110)
    print("wrote:", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
