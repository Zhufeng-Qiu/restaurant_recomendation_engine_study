"""Create a small but dense review file for the tiny fixture.

Keeps every review of the K most-reviewed businesses (ties broken by
business_id ascending) from the input, preserving original line order so the
last-line-wins duplicate policy is unaffected. Deterministic by construction.

Usage:
    python3 tools/make_tiny_source.py <train_file> <output_file> [K=50]
"""

import json
import sys
from collections import Counter


def main():
    train, out = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    counts = Counter()
    with open(train) as f:
        for line in f:
            counts[json.loads(line)["business_id"]] += 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    keep = {b for b, _ in top}

    n = 0
    with open(train) as f, open(out, "w") as g:
        for line in f:
            if json.loads(line)["business_id"] in keep:
                g.write(line)
                n += 1
    print(f"kept {n} reviews across {len(keep)} businesses -> {out}")


if __name__ == "__main__":
    main()
