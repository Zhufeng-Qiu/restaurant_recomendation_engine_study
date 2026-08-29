"""Pure-Python reference implementation of the frozen Pearson contract.

No Spark, no third-party dependencies. This is the normative executable
form of docs/pearson_contract.md and the validator for the hand-computed
microcases. Native backends replicate `pearson_from_stats`.

Usage:
    python3 tools/pearson_reference.py            # run microcase validation
"""

import json
import math
import os
import sys

MIN_OVERLAP = 3
EPS = 1e-14
TOL = 1e-12


def sufficient_stats(x, y):
    """Six additive statistics over the co-rated dimensions of two rating
    maps. Returns (n, sum_x, sum_y, sum_xx, sum_yy, sum_xy)."""
    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    for d in x:
        if d in y:
            xv, yv = float(x[d]), float(y[d])
            n += 1
            sx += xv
            sy += yv
            sxx += xv * xv
            syy += yv * yv
            sxy += xv * yv
    return n, sx, sy, sxx, syy, sxy


def pearson_from_stats(n, sx, sy, sxx, syy, sxy):
    """Finalization used by all native backends (contract §3)."""
    num = n * sxy - sx * sy
    vx = n * sxx - sx * sx
    vy = n * syy - sy * sy
    if num == 0.0 or vx <= 0.0 or vy <= 0.0:
        return 0.0
    return num / math.sqrt(vx * vy)


def pearson_two_pass(x, y):
    """Literal transcription of cf_train.py's two-pass form (contract §3),
    kept to prove numerical agreement between the two forms."""
    common = [d for d in x if d in y]
    if not common:
        return None
    xs = [float(x[d]) for d in common]
    ys = [float(y[d]) for d in common]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs)) * math.sqrt(
        sum((b - my) ** 2 for b in ys)
    )
    if num == 0.0 or den == 0.0:
        return 0.0
    return num / den


def evaluate_pair(x, y, min_overlap=MIN_OVERLAP):
    """Full kernel: returns (stats, sim, emit) or None if not a candidate."""
    stats = sufficient_stats(x, y)
    if stats[0] < min_overlap:
        return None
    sim = pearson_from_stats(*stats)
    return stats, sim, sim > EPS


def run_microcases(path):
    with open(path) as f:
        doc = json.load(f)
    assert doc["policy"]["min_overlap"] == MIN_OVERLAP
    assert doc["policy"]["eps"] == EPS
    assert doc["policy"]["tolerance"] == TOL

    failures = []
    for case in doc["cases"]:
        name = case["name"]
        result = evaluate_pair(case["x"], case["y"])
        expect = case["expect"]

        if expect is None:
            if result is not None:
                failures.append(f"{name}: expected non-candidate, got {result}")
            n = sufficient_stats(case["x"], case["y"])[0]
            if n != case["n"]:
                failures.append(f"{name}: expected n={case['n']}, got {n}")
            continue

        if result is None:
            failures.append(f"{name}: expected candidate, kernel rejected it")
            continue

        stats, sim, emit = result
        n, sx, sy, sxx, syy, sxy = stats
        exp_stats = (
            expect["n"], expect["sum_x"], expect["sum_y"],
            expect["sum_xx"], expect["sum_yy"], expect["sum_xy"],
        )
        if (n,) + tuple(round(v, 9) for v in (sx, sy, sxx, syy, sxy)) != exp_stats:
            failures.append(f"{name}: stats {stats} != expected {exp_stats}")
        if abs(sim - expect["sim"]) > TOL:
            failures.append(f"{name}: sim {sim!r} != expected {expect['sim']!r}")
        if emit != expect["emit"]:
            failures.append(f"{name}: emit {emit} != expected {expect['emit']}")

        # Cross-check the two algebraic forms against each other.
        sim2 = pearson_two_pass(case["x"], case["y"])
        if abs(sim - sim2) > TOL:
            failures.append(f"{name}: stats-form {sim!r} vs two-pass {sim2!r}")

    return failures, len(doc["cases"])


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    micro = os.path.join(here, "..", "data", "fixtures", "microcases.json")
    failures, total = run_microcases(micro)
    if failures:
        print(f"FAIL: {len(failures)} problem(s) in {total} cases")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"OK: all {total} microcases pass (both formula forms, EPS={EPS}, TOL={TOL})")
