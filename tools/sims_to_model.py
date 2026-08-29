"""Convert a native backend's similarity output into a cf_predict.py model.

Reads a fixture (for entity IDs, mode, and policy) and a sims .bin produced by
`pearson_engine --out`, then writes the task3-format JSONL model containing
the pairs with sim > EPS. This is how native backends plug back into the
Python prediction/RMSE layer.

Usage:
    python3 tools/sims_to_model.py <fixture_dir> <sims.bin> <output_model>
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
    fixture, sims_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(os.path.join(fixture, "meta.json")) as f:
        meta = json.load(f)
    eps = meta["policy"]["eps"]
    tag1, tag2 = ("b1", "b2") if meta["mode"] == "item" else ("u1", "u2")

    with open(os.path.join(fixture, "entities.txt")) as f:
        entities = f.read().splitlines()
    pairs = load_array(os.path.join(fixture, "pairs.bin"), "i")
    sims = load_array(sims_path, "d")
    assert 2 * len(sims) == len(pairs), "sims length does not match fixture pairs"

    n = 0
    with open(out_path, "w") as out:
        for k, sim in enumerate(sims):
            if sim > eps:
                json.dump({tag1: entities[pairs[2 * k]],
                           tag2: entities[pairs[2 * k + 1]],
                           "sim": sim}, out)
                out.write("\n")
                n += 1
    print(f"wrote {n} pairs (sim > {eps}) to {out_path}")


if __name__ == "__main__":
    main()
