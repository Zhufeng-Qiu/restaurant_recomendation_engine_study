"""Quantify compute/communication overlap in an nsys SQLite export.

Classifies device kernels into communication (NCCL collectives) and
computation (the pair-statistics kernel), then per GPU measures how much of
the communication time is covered by a concurrently running compute kernel.
The untimed NCCL warm-up collective is excluded by restricting the analysis
to the window that starts with the first statistics kernel.

Usage: python3 tools/nsys_overlap.py <export.sqlite> [<export.sqlite> ...]
"""

import sqlite3
import sys


def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def analyse(path):
    con = sqlite3.connect(path)
    rows = list(con.execute(
        "select s.value, k.deviceId, k.start, k.end "
        "from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.shortName"))
    devs = sorted({r[1] for r in rows})
    print(f"{path}")
    for d in devs:
        comm = [(r[2], r[3]) for r in rows if r[1] == d and "nccl" in r[0]]
        comp = [(r[2], r[3]) for r in rows if r[1] == d and "pair_stats" in r[0]]
        if not comm or not comp:
            continue
        t0 = min(s for s, _ in comp)          # drops the pre-timing warm-up
        comm = union([(s, e) for s, e in comm if e > t0])
        comp = union(comp)
        ov = total(intersect(comm, comp))
        c, k = total(comm), total(comp)
        span = max(max(e for _, e in comm), max(e for _, e in comp)) - t0
        print("  GPU%d  comm=%7.3f ms  compute=%7.3f ms  overlapped=%6.3f ms "
              "(%5.1f%% of comm, %5.1f%% of compute)  wall=%7.3f ms  "
              "serial_sum/wall=%.2f"
              % (d, c / 1e6, k / 1e6, ov / 1e6,
                 100 * ov / c if c else 0, 100 * ov / k if k else 0,
                 span / 1e6, (c + k) / span if span else 0))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyse(p)
