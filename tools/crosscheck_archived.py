"""Cross-check an item-mode fixture against the archived Spark golden model
(gate 2 of the correctness ladder, docs/pearson_contract.md section 8).

Compares emitted pairs (sim > EPS) from the fixture, mapped back to original
string IDs, against original_code/python/result/task2_model_item. Per the
contract, membership may differ only inside the EPS boundary band; common
pairs must agree within TOL.

Usage:
    python3 tools/crosscheck_archived.py <fixture_dir> <archived_model_json>
"""

import json
import os
import sys
from array import array


def load_array(path, typecode):
    a = array(typecode)
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def main():
    fixture, model_path = sys.argv[1], sys.argv[2]
    with open(os.path.join(fixture, "meta.json")) as f:
        meta = json.load(f)
    eps, tol = meta["policy"]["eps"], meta["policy"]["tolerance"]
    # Membership at the sim>EPS boundary is decided by values within rounding
    # noise of EPS; allow a modest multiple of TOL for the band.
    band = 10 * tol

    with open(os.path.join(fixture, "entities.txt")) as f:
        entities = f.read().splitlines()
    pairs = load_array(os.path.join(fixture, "pairs.bin"), "i")
    golden = load_array(os.path.join(fixture, "golden.bin"), "d")

    ours = {}
    for k in range(len(golden)):
        if golden[k] > eps:
            ours[(entities[pairs[2 * k]], entities[pairs[2 * k + 1]])] = golden[k]

    theirs = {}
    with open(model_path) as f:
        for line in f:
            j = json.loads(line)
            a, b = j["b1"], j["b2"]
            if a > b:
                a, b = b, a
            theirs[(a, b)] = j["sim"]

    ok, nk = set(ours), set(theirs)
    common = ok & nk
    max_diff, worst = 0.0, None
    val_fail = 0
    for k in common:
        d = abs(ours[k] - theirs[k])
        if d > max_diff:
            max_diff, worst = d, k
        if d > tol:
            val_fail += 1

    member_fail = 0
    for k in ok - nk:
        if ours[k] > eps + band:
            member_fail += 1
    for k in nk - ok:
        # Archived model used filter sim > 0; near-zero members are boundary.
        if theirs[k] > eps + band:
            member_fail += 1

    print(f"fixture emitted: {len(ours)}  archived: {len(theirs)}  common: {len(common)}")
    print(f"only-fixture: {len(ok - nk)}  only-archived: {len(nk - ok)}")
    print(f"max abs diff on common: {max_diff:.3e} at {worst}")
    if val_fail or member_fail:
        print(f"FAIL: {val_fail} value mismatches > tol, "
              f"{member_fail} membership differences outside boundary band")
        sys.exit(1)
    print("OK: archived model matches fixture within contract tolerances")


if __name__ == "__main__":
    main()
