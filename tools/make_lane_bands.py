"""Build two fixtures that differ in warp lane utilisation and nothing else.

The earlier overlap experiment split pairs by intersection size `n`, concluded
the skewed band was 2.1x more efficient, and explained it by claiming an n=3
pair idles 29 of a warp's 32 lanes. All of that was wrong: `pair_lane_stats`
strides its lanes over the shorter rating ROW, so `n` does not set the trip
count at all, and the two bands differed in scan volume by 2.2x.

This builds the controlled version. Lane utilisation is `short / (32 *
ceil(short/32))` -- purely a tail effect of the shorter row length -- so the
bands pair rows that need the SAME number of warp rounds but fill the last one
differently:

    rounds   full        tail
      2      64          33
      3      96          65
      4      128         97

Same lane-slots per pair, roughly half the useful elements in the tail band.
If the kernel's time tracks lane-slots, the two fixtures take the same time
despite one doing far less real work. That is the whole hypothesis.

Everything else is matched rather than assumed away:

  * pair count, exactly, and per round bucket
  * binary-search depth, via a log2(long row) stratum
  * hit rate (n / short), via a decile stratum
  * emission order, so cache locality is not confounded

Both fixtures share the source's CSR arrays byte for byte.

Usage:  python3 tools/make_lane_bands.py [data/fixtures/item_full]
"""

import hashlib
import json
import os
import shutil
import sys

from array import array
from collections import defaultdict

# (rounds, full-lane short length, worst-tail short length). The +/- window
# keeps enough pairs per stratum while staying inside the same round bucket.
BUCKETS = [(2, 64, 33), (3, 96, 65), (4, 128, 97)]
WINDOW = 2


def read_bin(path, tc):
    a = array(tc)
    n = os.path.getsize(path) // a.itemsize
    with open(path, "rb") as f:
        a.fromfile(f, n)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def intersect_size(offsets, dims, i, j):
    a, ae = offsets[i], offsets[i + 1]
    b, be = offsets[j], offsets[j + 1]
    c = 0
    while a < ae and b < be:
        if dims[a] < dims[b]:
            a += 1
        elif dims[b] < dims[a]:
            b += 1
        else:
            c += 1
            a += 1
            b += 1
    return c


def main():
    src = (sys.argv[1] if len(sys.argv) > 1 else "data/fixtures/item_full").rstrip("/")
    offsets = read_bin(f"{src}/offsets.bin", "q")
    dims = read_bin(f"{src}/dims.bin", "i")
    pairs = read_bin(f"{src}/pairs.bin", "i")
    golden = read_bin(f"{src}/golden.bin", "d")
    rowlen = [offsets[k + 1] - offsets[k] for k in range(len(offsets) - 1)]
    n_pairs = len(golden)
    print(f"source {src}: {n_pairs:,} pairs")

    # Candidate pool per (bucket, band), keyed by matching stratum.
    pool = {(r, side): defaultdict(list) for r, _, _ in BUCKETS for side in ("full", "tail")}
    for k in range(n_pairs):
        i, j = pairs[2 * k], pairs[2 * k + 1]
        s, l = min(rowlen[i], rowlen[j]), max(rowlen[i], rowlen[j])
        for r, full, tail in BUCKETS:
            side = ("full" if full - WINDOW <= s <= full
                    else "tail" if tail <= s <= tail + WINDOW else None)
            if side is None:
                continue
            n = intersect_size(offsets, dims, i, j)
            depth = int(l).bit_length()              # ~log2 of the searched row
            hit = min(int((n / s) * 10), 9)          # hit-rate decile
            pool[(r, side)][(depth, hit)].append(k)
            break

    keep = {"full": [], "tail": []}
    print(f"\n{'rounds':>7s} {'stratum':>8s} {'full':>8s} {'tail':>8s} {'matched':>8s}")
    for r, _, _ in BUCKETS:
        f_by, t_by = pool[(r, "full")], pool[(r, "tail")]
        matched = 0
        for st in sorted(set(f_by) | set(t_by)):
            take = min(len(f_by.get(st, [])), len(t_by.get(st, [])))
            if not take:
                continue
            keep["full"] += f_by[st][:take]
            keep["tail"] += t_by[st][:take]
            matched += take
        print(f"{r:>7d} {'all':>8s} {sum(len(v) for v in f_by.values()):>8,} "
              f"{sum(len(v) for v in t_by.values()):>8,} {matched:>8,}")

    for side in ("full", "tail"):
        keep[side].sort()                            # preserve source order
        dst = f"{src}_lane_{side}"
        os.makedirs(dst, exist_ok=True)
        for name in ("offsets.bin", "dims.bin", "vals.bin", "entities.txt", "dims.txt"):
            shutil.copyfile(f"{src}/{name}", f"{dst}/{name}")
        po, go = array("i"), array("d")
        eff = slots = 0
        for k in keep[side]:
            i, j = pairs[2 * k], pairs[2 * k + 1]
            po.append(i)
            po.append(j)
            go.append(golden[k])
            s = min(rowlen[i], rowlen[j])
            eff += s
            slots += 32 * -(-s // 32)
        if sys.byteorder != "little":
            po.byteswap()
            go.byteswap()
        with open(f"{dst}/pairs.bin", "wb") as f:
            po.tofile(f)
        with open(f"{dst}/golden.bin", "wb") as f:
            go.tofile(f)

        meta = dict(json.load(open(f"{src}/meta.json")))
        meta["mode"] = "lane-utilisation band"
        meta["derived_from"] = src
        meta["lane_band"] = {
            "side": side,
            "buckets": [{"rounds": r, "short_len": (full if side == "full" else tail),
                         "window": WINDOW} for r, full, tail in BUCKETS],
            "matched_on": ["pair count", "round bucket", "log2(long row)",
                           "hit-rate decile", "emission order"],
            "effective_elements": eff,
            "lane_slots": slots,
            "lane_utilisation": round(eff / slots, 5),
        }
        meta["counts"] = dict(meta["counts"])
        meta["counts"]["n_pairs"] = len(go)
        meta["files"] = {n: sha256_file(f"{dst}/{n}") for n in
                         ("entities.txt", "dims.txt", "offsets.bin", "dims.bin",
                          "vals.bin", "pairs.bin", "golden.bin")}
        with open(f"{dst}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  {os.path.basename(dst):26s} pairs={len(go):>7,}  "
              f"effective={eff:>10,}  lane_slots={slots:>10,}  "
              f"utilisation={eff / slots * 100:5.2f}%")


if __name__ == "__main__":
    main()
