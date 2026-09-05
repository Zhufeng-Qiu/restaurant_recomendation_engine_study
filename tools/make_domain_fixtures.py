"""Generate tiny synthetic fixtures that probe the compressed-payload domain gate.

docs/pearson_contract.md section 9.4 defines when `--payload i32` / `packed` are
valid: ratings must be non-negative integers, and `vmax^2 * N` (the loosest of
the six statistic bounds) must fit the field width. Every shipped fixture is
comfortably inside that domain, so the gate's rejection branches never fire --
the README lists this as the one unexercised piece of the contract.

These fixtures fire them. Each is a handful of ratings over 3 entities, written
in the same pearson-fixture-v1 layout the exporter produces, so
`engine/tests/payload_domain_test.cpp` (and, on a GPU host,
`pearson_engine_nccl --payload ...`) can load them like any other fixture.

  ok               in-domain control            f64 ok    i32 ok      packed ok
  fractional       one rating = 3.5             f64 ok    i32 REJECT  packed REJECT
  negative         one rating = -2              f64 ok    i32 REJECT  packed REJECT
  packed_overflow  vmax=1000, N=5  -> 5.0e6     f64 ok    i32 ok      packed REJECT
  i32_overflow     vmax=100000, N=3 -> 3.0e10   f64 ok    i32 REJECT  packed REJECT

Usage:  python tools/make_domain_fixtures.py [output_root]
        (default output root: data/fixtures/domain)
"""

import hashlib
import json
import math
import os
import sys

from array import array

MIN_OVERLAP = 3
EPS = 1e-14
TOL = 1e-12


def pearson_from_stats(n, sx, sy, sxx, syy, sxy):
    """Contract section 3, sufficient-statistics form."""
    num = n * sxy - sx * sy
    vx = n * sxx - sx * sx
    vy = n * syy - sy * sy
    if num == 0.0 or vx <= 0.0 or vy <= 0.0:
        return 0.0
    return num / math.sqrt(vx * vy)


def pair_sim(ri, rj):
    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    for d, xv in ri.items():
        yv = rj.get(d)
        if yv is not None:
            n += 1
            sx += xv
            sy += yv
            sxx += xv * xv
            syy += yv * yv
            sxy += xv * yv
    if n < MIN_OVERLAP:
        return n, None
    return n, pearson_from_stats(n, sx, sy, sxx, syy, sxy)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_fixture(out_dir, rows, note, expect):
    """rows: list of {dim: rating} dicts, one per entity (dims will be sorted)."""
    os.makedirs(out_dir, exist_ok=True)

    offsets = array("q", [0])
    dcol = array("i")
    vals = array("d")
    for r in rows:
        for d in sorted(r):
            dcol.append(d)
            vals.append(float(r[d]))
        offsets.append(len(dcol))

    # All unordered pairs whose overlap reaches the contract minimum.
    pairs = array("i")
    golden = array("d")
    n_emitted = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            n, sim = pair_sim(rows[i], rows[j])
            if sim is None:
                continue
            pairs.append(i)
            pairs.append(j)
            golden.append(sim)
            if sim > EPS:
                n_emitted += 1

    if not golden:
        raise SystemExit(f"{out_dir}: no candidate pair reached overlap {MIN_OVERLAP}")

    if sys.byteorder != "little":
        for a in (offsets, dcol, vals, pairs, golden):
            a.byteswap()

    for name, arr in (("offsets.bin", offsets), ("dims.bin", dcol),
                      ("vals.bin", vals), ("pairs.bin", pairs),
                      ("golden.bin", golden)):
        with open(os.path.join(out_dir, name), "wb") as f:
            arr.tofile(f)

    entities = [f"e{i}" for i in range(len(rows))]
    all_dims = sorted({d for r in rows for d in r})
    with open(os.path.join(out_dir, "entities.txt"), "w") as f:
        f.write("\n".join(entities) + "\n")
    with open(os.path.join(out_dir, "dims.txt"), "w") as f:
        f.write("\n".join(f"d{d}" for d in all_dims) + "\n")

    files = {}
    for name in ("entities.txt", "dims.txt", "offsets.bin", "dims.bin",
                 "vals.bin", "pairs.bin", "golden.bin"):
        files[name] = sha256_file(os.path.join(out_dir, name))

    max_row = max(len(r) for r in rows)
    vmax = max(v for r in rows for v in r.values())
    meta = {
        "schema": "pearson-fixture-v1",
        "mode": "synthetic",
        "source": "tools/make_domain_fixtures.py",
        "purpose": note,
        "subset": None,
        "policy": {
            "min_overlap": MIN_OVERLAP,
            "eps": EPS,
            "tolerance": TOL,
            "duplicate_policy": "n/a (synthetic, unique keys by construction)",
            "jaccard_threshold": None,
        },
        "counts": {
            "n_entities": len(rows),
            "n_dims": len(all_dims),
            "nnz": len(vals),
            "n_pairs": len(golden),
            "n_emitted": n_emitted,
        },
        "domain": {
            "max_row": max_row,
            "max_rating": vmax,
            "loosest_statistic_bound": vmax * vmax * max_row,
            "expect_accept": expect,
        },
        "layout": {
            "endianness": "little",
            "offsets.bin": "int64[n_entities+1]",
            "dims.bin": "int32[nnz], ascending within each entity row",
            "vals.bin": "float64[nnz]",
            "pairs.bin": "int32[2*n_pairs], (i,j) with i<j, lexicographically sorted",
            "golden.bin": "float64[n_pairs], unfiltered Pearson sim per pair",
        },
        "files": files,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"{os.path.basename(out_dir):18s} entities={len(rows)} pairs={len(golden)} "
          f"max_row={max_row} vmax={vmax:g} bound={vmax * vmax * max_row:.3g} "
          f"accept={','.join(k for k, v in expect.items() if v) or 'f64 only'}")


CASES = [
    (
        "ok",
        [{0: 1, 1: 2, 2: 3, 3: 4},
         {0: 2, 1: 3, 2: 4, 4: 5},
         {1: 5, 2: 4, 3: 3, 4: 2}],
        "in-domain control: integer ratings 1-5, every payload must be accepted",
        {"f64": True, "i32": True, "packed": True},
    ),
    (
        "fractional",
        [{0: 1, 1: 2, 2: 3.5, 3: 4},
         {0: 2, 1: 3, 2: 4, 4: 5},
         {1: 5, 2: 4, 3: 3, 4: 2}],
        "one non-integer rating (3.5): compressed payloads must refuse",
        {"f64": True, "i32": False, "packed": False},
    ),
    (
        "negative",
        [{0: 1, 1: 2, 2: 3, 3: 4},
         {0: -2, 1: 3, 2: 4, 4: 5},
         {1: 5, 2: 4, 3: 3, 4: 2}],
        "one negative rating (-2): compressed payloads must refuse",
        {"f64": True, "i32": False, "packed": False},
    ),
    (
        "packed_overflow",
        [{0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000},
         {0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 999},
         {0: 999, 1: 1000, 2: 1000, 3: 1000, 4: 1000}],
        "integer but vmax^2*N = 5.0e6 > 2^21-1: packed must refuse, i32 still fits",
        {"f64": True, "i32": True, "packed": False},
    ),
    (
        "i32_overflow",
        [{0: 100000, 1: 100000, 2: 100000},
         {0: 100000, 1: 100000, 2: 99999},
         {0: 99999, 1: 100000, 2: 100000}],
        "integer but vmax^2*N = 3.0e10 > 2^31-1: both compressed payloads refuse",
        {"f64": True, "i32": False, "packed": False},
    ),
]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "data/fixtures/domain"
    os.makedirs(root, exist_ok=True)
    for name, rows, note, expect in CASES:
        write_fixture(os.path.join(root, name), rows, note, expect)
    print(f"\n{len(CASES)} domain fixtures written under {root}")


if __name__ == "__main__":
    main()
