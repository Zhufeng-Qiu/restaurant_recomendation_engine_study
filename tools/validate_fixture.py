"""Validate a Pearson workload fixture without Spark.

Recomputes the similarity for every candidate pair from the CSR arrays using
the contract's sufficient-statistics form and compares against golden.bin.
Also verifies file hashes, CSR structure, ID map ordering, and pair ordering.

Usage:
    python3 tools/validate_fixture.py <fixture_dir> [--sample N]

--sample N checks every ceil(n_pairs/N)-th pair instead of all of them.
Exit code 0 iff everything passes. Requires only the Python standard library.
"""

import hashlib
import json
import math
import os
import sys
from array import array


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_array(path, typecode):
    a = array(typecode)
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def pearson_from_stats(n, sx, sy, sxx, syy, sxy):
    num = n * sxy - sx * sy
    vx = n * sxx - sx * sx
    vy = n * syy - sy * sy
    if num == 0.0 or vx <= 0.0 or vy <= 0.0:
        return 0.0
    return num / math.sqrt(vx * vy)


def main():
    fixture = sys.argv[1]
    sample = None
    if "--sample" in sys.argv:
        sample = int(sys.argv[sys.argv.index("--sample") + 1])

    with open(os.path.join(fixture, "meta.json")) as f:
        meta = json.load(f)
    pol = meta["policy"]
    min_overlap, eps, tol = pol["min_overlap"], pol["eps"], pol["tolerance"]
    counts = meta["counts"]
    problems = []

    # 1. File integrity.
    for name, expected in meta["files"].items():
        actual = sha256_file(os.path.join(fixture, name))
        if actual != expected:
            problems.append(f"sha256 mismatch for {name}")

    # 2. Load arrays and ID maps.
    offsets = load_array(os.path.join(fixture, "offsets.bin"), "q")
    dcol = load_array(os.path.join(fixture, "dims.bin"), "i")
    vals = load_array(os.path.join(fixture, "vals.bin"), "d")
    pairs = load_array(os.path.join(fixture, "pairs.bin"), "i")
    golden = load_array(os.path.join(fixture, "golden.bin"), "d")
    with open(os.path.join(fixture, "entities.txt")) as f:
        entities = f.read().splitlines()
    with open(os.path.join(fixture, "dims.txt")) as f:
        dims = f.read().splitlines()

    # 3. Structural checks.
    n_e, n_d = counts["n_entities"], counts["n_dims"]
    if len(entities) != n_e or entities != sorted(entities):
        problems.append("entities.txt count/order wrong")
    if len(dims) != n_d or dims != sorted(dims):
        problems.append("dims.txt count/order wrong")
    if len(offsets) != n_e + 1 or offsets[0] != 0 or offsets[-1] != counts["nnz"]:
        problems.append("offsets.bin shape wrong")
    if any(offsets[k] > offsets[k + 1] for k in range(n_e)):
        problems.append("offsets not monotonic")
    if len(dcol) != counts["nnz"] or len(vals) != counts["nnz"]:
        problems.append("dims.bin/vals.bin length wrong")
    for k in range(n_e):
        row = dcol[offsets[k]:offsets[k + 1]]
        if any(row[t] >= row[t + 1] for t in range(len(row) - 1)):
            problems.append(f"row {k}: dim indices not strictly ascending")
            break
    n_p = counts["n_pairs"]
    if len(pairs) != 2 * n_p or len(golden) != n_p:
        problems.append("pairs.bin/golden.bin length wrong")
    prev = (-1, -1)
    for k in range(n_p):
        ij = (pairs[2 * k], pairs[2 * k + 1])
        if not ij[0] < ij[1]:
            problems.append(f"pair {k} not canonical (i<j)")
            break
        if not prev < ij:
            problems.append(f"pairs not sorted at {k}")
            break
        prev = ij

    if problems:
        print("FAIL (structure):")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    # 4. Recompute similarities (two-pointer sorted intersection).
    idxs = range(n_p)
    if sample and n_p > sample:
        step = (n_p + sample - 1) // sample
        idxs = range(0, n_p, step)

    max_diff = 0.0
    checked = 0
    n_emitted = 0
    boundary = 0
    for k in idxs:
        i, j = pairs[2 * k], pairs[2 * k + 1]
        a0, a1 = offsets[i], offsets[i + 1]
        b0, b1 = offsets[j], offsets[j + 1]
        n = 0
        sx = sy = sxx = syy = sxy = 0.0
        while a0 < a1 and b0 < b1:
            da, db = dcol[a0], dcol[b0]
            if da < db:
                a0 += 1
            elif db < da:
                b0 += 1
            else:
                xv, yv = vals[a0], vals[b0]
                n += 1
                sx += xv
                sy += yv
                sxx += xv * xv
                syy += yv * yv
                sxy += xv * yv
                a0 += 1
                b0 += 1
        if n < min_overlap:
            problems.append(f"pair {k}: overlap {n} < {min_overlap}")
            continue
        sim = pearson_from_stats(n, sx, sy, sxx, syy, sxy)
        d = abs(sim - golden[k])
        if d > max_diff:
            max_diff = d
        if d > tol:
            problems.append(f"pair {k}: recomputed {sim!r} vs golden {golden[k]!r}")
        if abs(sim - eps) <= tol:
            boundary += 1
        checked += 1

    for k in range(n_p):
        if golden[k] > eps:
            n_emitted += 1
    if n_emitted != counts["n_emitted"]:
        problems.append(f"emitted count {n_emitted} != meta {counts['n_emitted']}")

    if problems:
        print(f"FAIL: {len(problems)} problem(s); checked {checked}/{n_p} pairs")
        for p in problems[:20]:
            print("  -", p)
        sys.exit(1)
    print(
        f"OK: {fixture}  pairs={n_p} checked={checked} emitted={n_emitted} "
        f"max_abs_diff={max_diff:.3e} boundary_band={boundary}"
    )


if __name__ == "__main__":
    main()
