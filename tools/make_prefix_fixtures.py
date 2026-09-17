"""Boundary fixtures for the pair split: a prefix of a real candidate list.

The pair partition's failure modes live at the edges -- a single pair, an odd
split, a shard boundary landing mid-warp or mid-block, an empty shard. None of
those exist in the shipped fixtures, which all have six-figure pair counts.

These take the first N candidate pairs of a real fixture with its CSR arrays
byte for byte unchanged, so the golden similarities come from the same
pipeline as everything else rather than from a hand-written expectation.

N is chosen to bracket every boundary the kernels have: 1, 2 and 3 pairs; the
warp edge at 64; the G=4 sub-warp block tail at 128; the finalize block tail
at 512; and one either side of each.

It also refuses to emit a fixture whose two halves could hide a misplacement.
If the pairs on either side of the split carry the same similarity, swapping
them is invisible, and a test that cannot distinguish them is not testing the
split. Same for the last element, which is where an off-by-one lands.

Usage: python3 tools/make_prefix_fixtures.py [data/fixtures/item_tiny]
"""

import json
import os
import shutil
import sys
from array import array

SIZES = [1, 2, 3, 63, 64, 65, 127, 128, 129, 511, 512, 513]


def read_bin(path, tc):
    a = array(tc)
    n = os.path.getsize(path) // a.itemsize
    with open(path, "rb") as f:
        a.fromfile(f, n)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def main():
    src = (sys.argv[1] if len(sys.argv) > 1 else "data/fixtures/item_tiny").rstrip("/")
    pairs = read_bin(f"{src}/pairs.bin", "i")
    golden = read_bin(f"{src}/golden.bin", "d")
    available = len(golden)
    print(f"source {src}: {available:,} pairs")

    made, skipped = 0, 0
    for n in SIZES:
        if n > available:
            print(f"  skip N={n}: source has only {available}")
            skipped += 1
            continue
        dst = f"{src}_n{n}"
        os.makedirs(dst, exist_ok=True)
        for name in ("offsets.bin", "dims.bin", "vals.bin", "entities.txt",
                     "dims.txt"):
            shutil.copyfile(f"{src}/{name}", f"{dst}/{name}")

        po = array("i", pairs[:2 * n])
        go = array("d", golden[:n])

        # Where a two-way split falls, and whether it is detectable.
        mid = n // 2
        notes = []
        if n >= 2:
            if go[mid - 1] == go[mid]:
                notes.append(f"pairs {mid-1} and {mid} share a similarity "
                             f"({go[mid]!r}) -- a swap across the shard "
                             f"boundary would be invisible here")
            if n >= 3 and go[-1] == go[-2]:
                notes.append(f"the last two pairs share a similarity "
                             f"({go[-1]!r}) -- an off-by-one at the end would "
                             f"be invisible here")
        if notes:
            print(f"  REFUSED N={n}: " + "; ".join(notes))
            shutil.rmtree(dst)
            skipped += 1
            continue

        if sys.byteorder != "little":
            po.byteswap()
            go.byteswap()
        with open(f"{dst}/pairs.bin", "wb") as f:
            po.tofile(f)
        with open(f"{dst}/golden.bin", "wb") as f:
            go.tofile(f)

        meta = dict(json.load(open(f"{src}/meta.json")))
        meta["mode"] = "pair-count boundary prefix"
        meta["derived_from"] = src
        meta["prefix"] = {
            "n_pairs": n,
            "two_way_split": [mid, n - mid],
            "boundary_similarities": [golden[mid - 1], golden[mid]] if n >= 2 else [],
            "last_similarity": golden[n - 1],
        }
        meta["counts"] = dict(meta["counts"])
        meta["counts"]["n_pairs"] = n
        meta.pop("files", None)  # the hash manifest no longer describes these
        with open(f"{dst}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  {os.path.basename(dst):22s} N={n:>4}  split {mid}/{n-mid}  "
              f"boundary sims {golden[mid-1]:+.6f} | {golden[mid]:+.6f}"
              if n >= 2 else
              f"  {os.path.basename(dst):22s} N={n:>4}  single pair")
        made += 1

    print(f"\n{made} fixtures written, {skipped} skipped")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
