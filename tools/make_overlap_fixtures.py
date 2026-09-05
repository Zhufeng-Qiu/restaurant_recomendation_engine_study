"""Split a fixture's candidate pairs into overlap bands, keeping ratings fixed.

The project brief lists "overlap distribution (low / medium / high common-user
counts)" as a workload dimension, because per-pair cost is proportional to the
overlap n and a skewed distribution is what makes dynamic scheduling worth its
overhead. Nothing measured it: every benchmark ran the natural mix.

item_full's mix is heavily skewed -- 43% of pairs sit at the n=3 minimum while
the tail reaches n=264 -- so the two extremes can be cut out of it directly.
The derived fixtures share the SOURCE's CSR arrays byte for byte and differ
only in which pairs are listed, so overlap distribution is the single variable:

  <src>_ovl_flat   n == 3 only          zero variance, perfectly balanced
  <src>_ovl_skew   n >= <hi> (11)       high variance, long tail

golden.bin is carried over per pair, so the derived fixtures validate against
exactly the same reference values as their source.

Usage:
  python3 tools/make_overlap_fixtures.py data/fixtures/item_full [hi]
"""

import hashlib
import json
import os
import shutil
import sys

from array import array


def read_bin(path, typecode):
    a = array(typecode)
    n = os.path.getsize(path) // a.itemsize
    with open(path, "rb") as f:
        a.fromfile(f, n)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def overlap_counts(offsets, dims, pairs):
    """Sorted two-pointer intersection size for every pair."""
    ns = array("i", [0]) * 0
    n_pairs = len(pairs) // 2
    for k in range(n_pairs):
        i, j = pairs[2 * k], pairs[2 * k + 1]
        a, a_end = offsets[i], offsets[i + 1]
        b, b_end = offsets[j], offsets[j + 1]
        c = 0
        while a < a_end and b < b_end:
            da, db = dims[a], dims[b]
            if da < db:
                a += 1
            elif db < da:
                b += 1
            else:
                c += 1
                a += 1
                b += 1
        ns.append(c)
    return ns


def write_band(src, dst, pairs, golden, keep, band, ns):
    os.makedirs(dst, exist_ok=True)
    # Ratings side is shared verbatim -- that is the point of the experiment.
    for name in ("offsets.bin", "dims.bin", "vals.bin", "entities.txt", "dims.txt"):
        shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))

    p_out, g_out = array("i"), array("d")
    kept_ns = []
    for k in keep:
        p_out.append(pairs[2 * k])
        p_out.append(pairs[2 * k + 1])
        g_out.append(golden[k])
        kept_ns.append(ns[k])
    if sys.byteorder != "little":
        p_out.byteswap()
        g_out.byteswap()
    with open(os.path.join(dst, "pairs.bin"), "wb") as f:
        p_out.tofile(f)
    with open(os.path.join(dst, "golden.bin"), "wb") as f:
        g_out.tofile(f)

    src_meta = json.load(open(os.path.join(src, "meta.json")))
    files = {}
    for name in ("entities.txt", "dims.txt", "offsets.bin", "dims.bin",
                 "vals.bin", "pairs.bin", "golden.bin"):
        files[name] = sha256_file(os.path.join(dst, name))
    mean = sum(kept_ns) / len(kept_ns)
    var = sum((x - mean) ** 2 for x in kept_ns) / len(kept_ns)
    meta = dict(src_meta)
    meta["mode"] = "overlap-band"
    meta["derived_from"] = src
    meta["overlap_band"] = {
        "rule": band,
        "n_min": min(kept_ns),
        "n_max": max(kept_ns),
        "n_mean": round(mean, 3),
        "n_stdev": round(var ** 0.5, 3),
        "work_share_of_source": round(sum(kept_ns) / sum(ns), 4),
    }
    meta["counts"] = dict(src_meta["counts"])
    meta["counts"]["n_pairs"] = len(g_out)
    meta["counts"]["n_emitted"] = sum(1 for k in keep if golden[k] > src_meta["policy"]["eps"])
    meta["files"] = files
    with open(os.path.join(dst, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {os.path.basename(dst):22s} pairs={len(g_out):>9,}  n in [{min(kept_ns)},{max(kept_ns)}]  "
          f"mean={mean:6.2f}  stdev={var ** 0.5:6.2f}  work={sum(kept_ns) / sum(ns) * 100:5.1f}% of source")


def main():
    src = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "data/fixtures/item_full"
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 11

    offsets = read_bin(f"{src}/offsets.bin", "q")
    dims = read_bin(f"{src}/dims.bin", "i")
    pairs = read_bin(f"{src}/pairs.bin", "i")
    golden = read_bin(f"{src}/golden.bin", "d")
    print(f"source {src}: {len(golden):,} pairs")
    ns = overlap_counts(offsets, dims, pairs)

    flat = [k for k, n in enumerate(ns) if n == 3]
    skew = [k for k, n in enumerate(ns) if n >= hi]
    write_band(src, src + "_ovl_flat", pairs, golden, flat, "n == 3", ns)
    write_band(src, src + "_ovl_skew", pairs, golden, skew, f"n >= {hi}", ns)


if __name__ == "__main__":
    main()
