"""Export a backend-neutral Pearson workload fixture from review JSON.

This is the Python/Spark side of the contract boundary (docs/pearson_contract.md
section 6): Spark decides the candidate pairs and golden similarities; every
native backend only evaluates those same pairs from the exported arrays.

Usage:
  python export_fixture.py <train_file> <output_dir> <mode> [subset_mod subset_rem]

  <mode>       item | user
  subset_mod/rem  optional: keep only entities whose sorted index satisfies
                  idx % subset_mod == subset_rem (used for the medium fixture)

Output directory layout (all binaries little-endian):
  meta.json        counts, policy, layouts, sha256 of every file
  entities.txt     entity original IDs, sorted, one per line (fixture row order)
  dims.txt         dimension original IDs, sorted, one per line
  offsets.bin      int64  [n_entities + 1]   CSR row offsets
  dims.bin         int32  [nnz]              dim indices, ascending per row
  vals.bin         float64[nnz]              ratings
  pairs.bin        int32  [2 * n_pairs]      candidate pairs (i, j), i < j,
                                             sorted lexicographically
  golden.bin       float64[n_pairs]          Pearson sim for EVERY candidate
                                             pair (no sign filter)
"""

import hashlib
import json
import math
import os
import sys
import time

from array import array
from pyspark import SparkContext, SparkConf

MIN_OVERLAP = 3
EPS = 1e-14
TOL = 1e-12
JACCARD_THRESHOLD = 0.01  # user mode LSH verification threshold (from cf_train)

# LSH constants for user mode, copied verbatim from cf_train.py (first 50).
N_HASH = 50
LSH_P = 24251
A1 = [5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441]
A2 = [9127,9133,9137,9151,9157,9161,9173,9181,9187,9199,9203,9209,9221,9227,9239,9241,9257,9277,9281,9283,9293,9311,9319,9323,9337,9341,9343,9349,9371,9377,9391,9397,9403,9413,9419,9421,9431,9433,9437,9439,9461,9463,9467,9473,9479,9491,9497,9511,9521,9533]
B1 = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229]
B2 = [1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pearson_from_stats(n, sx, sy, sxx, syy, sxy):
    num = n * sxy - sx * sy
    vx = n * sxx - sx * sx
    vy = n * syy - sy * sy
    if num == 0.0 or vx <= 0.0 or vy <= 0.0:
        return 0.0
    return num / math.sqrt(vx * vy)


def pair_sim(ri, rj):
    """Stats-form Pearson over the key intersection of two rating dicts.
    Returns (n, sim)."""
    if len(rj) < len(ri):
        ri, rj = rj, ri
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


def min_hash_signature(dim_indices, m):
    """cf_train.py's min_hash, restated: 50 permutation minima over a set of
    dimension indices. Order-independent."""
    sig = []
    for itr in range(N_HASH):
        itr1 = itr + 1
        h1a, h1b = A1[itr], B1[itr]
        h2a, h2b = A2[itr], B2[itr]
        best = None
        for idx in dim_indices:
            v = ((itr1 * (h1a * idx + h1b) + itr1 * (h2a * idx + h2b) + itr1 * itr1) % LSH_P) % m
            if best is None or v < best:
                best = v
        sig.append(best)
    return sig


def main():
    train_file = sys.argv[1]
    output_dir = sys.argv[2]
    mode = sys.argv[3]
    subset = None
    if len(sys.argv) > 5:
        subset = (int(sys.argv[4]), int(sys.argv[5]))
    assert mode in ("item", "user"), mode

    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    conf = (
        SparkConf()
        .setAppName("pearson-fixture-export")
        .setMaster("local[*]")
        .setAll((("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
    )
    sc = SparkContext(conf=conf)

    # --- 1. Load + dedup (last review in file line order wins, contract §4) --
    def parse(line_with_idx):
        line, idx = line_with_idx
        j = json.loads(line)
        return ((j["user_id"], j["business_id"]), (idx, float(j["stars"])))

    deduped = (
        sc.textFile(train_file)
        .zipWithIndex()
        .map(parse)
        .reduceByKey(lambda a, b: a if a[0] > b[0] else b)
        .map(lambda kv: (kv[0][0], kv[0][1], kv[1][1]))
        .collect()
    )  # [(user_id, business_id, stars)] with unique (user, business)

    # --- 2. Entity/dimension orientation ------------------------------------
    if mode == "item":
        triples = [(b, u, s) for (u, b, s) in deduped]
    else:
        triples = [(u, b, s) for (u, b, s) in deduped]

    # --- 3. Optional entity subset (medium fixture) --------------------------
    all_entities = sorted({t[0] for t in triples})
    if subset is not None:
        keep = {e for i, e in enumerate(all_entities) if i % subset[0] == subset[1]}
        triples = [t for t in triples if t[0] in keep]

    entities = sorted({t[0] for t in triples})
    dims = sorted({t[1] for t in triples})
    e_idx = {e: i for i, e in enumerate(entities)}
    d_idx = {d: i for i, d in enumerate(dims)}

    # --- 4. CSR (entity-major, dim indices ascending per row) ----------------
    rows = [[] for _ in entities]
    for e, d, s in triples:
        rows[e_idx[e]].append((d_idx[d], s))
    offsets = array("q", [0])
    dcol = array("i")
    vals = array("d")
    rating_dicts = []
    for r in rows:
        r.sort()
        for di, s in r:
            dcol.append(di)
            vals.append(s)
        offsets.append(len(dcol))
        rating_dicts.append(dict(r))

    # --- 5. Candidate pairs ---------------------------------------------------
    eligible = [i for i, rd in enumerate(rating_dicts) if len(rd) >= MIN_OVERLAP]

    if mode == "item":
        ratings_b = sc.broadcast(rating_dicts)
        rdd = sc.parallelize(eligible, 64)

        def item_pairs(i):
            rds = ratings_b.value
            ri = rds[i]
            out = []
            for j in eligible_b.value:
                if j <= i:
                    continue
                n, sim = pair_sim(ri, rds[j])
                if sim is not None:
                    out.append((i, j, sim))
            return out

        eligible_b = sc.broadcast(eligible)
        results = rdd.flatMap(item_pairs).collect()
    else:
        m = len(dims)
        sets = {i: frozenset(rating_dicts[i].keys()) for i in eligible}

        def bands(i):
            sig = min_hash_signature(sets_b.value[i], m)
            # 50 bands x 1 row: bucket key is (band index, value).
            return [((band, v), [i]) for band, v in enumerate(sig)]

        sets_b = sc.broadcast(sets)
        ratings_b = sc.broadcast(rating_dicts)

        def verify(pair):
            i, j = pair
            si, sj = sets_b.value[i], sets_b.value[j]
            inter = len(si & sj)
            if inter / (len(si) + len(sj) - inter) < JACCARD_THRESHOLD:
                return None
            n, sim = pair_sim(ratings_b.value[i], ratings_b.value[j])
            if sim is None:
                return None
            return (i, j, sim)

        import itertools
        results = (
            sc.parallelize(eligible, 64)
            .flatMap(bands)
            .reduceByKey(lambda a, b: a + b)
            .filter(lambda kv: len(kv[1]) > 1)
            .flatMap(lambda kv: itertools.combinations(sorted(kv[1]), 2))
            .distinct()
            .map(verify)
            .filter(lambda r: r is not None)
            .collect()
        )

    results.sort()
    pairs = array("i")
    golden = array("d")
    n_emitted = 0
    for i, j, sim in results:
        pairs.append(i)
        pairs.append(j)
        golden.append(sim)
        if sim > EPS:
            n_emitted += 1
    sc.stop()

    # --- 6. Write fixture ------------------------------------------------------
    if sys.byteorder != "little":
        for a in (offsets, dcol, vals, pairs, golden):
            a.byteswap()

    def write_bin(name, arr):
        with open(os.path.join(output_dir, name), "wb") as f:
            arr.tofile(f)

    write_bin("offsets.bin", offsets)
    write_bin("dims.bin", dcol)
    write_bin("vals.bin", vals)
    write_bin("pairs.bin", pairs)
    write_bin("golden.bin", golden)
    with open(os.path.join(output_dir, "entities.txt"), "w") as f:
        f.write("\n".join(entities) + "\n")
    with open(os.path.join(output_dir, "dims.txt"), "w") as f:
        f.write("\n".join(dims) + "\n")

    files = {}
    for name in ("entities.txt", "dims.txt", "offsets.bin", "dims.bin",
                 "vals.bin", "pairs.bin", "golden.bin"):
        files[name] = sha256_file(os.path.join(output_dir, name))

    meta = {
        "schema": "pearson-fixture-v1",
        "mode": mode,
        "source": train_file,
        "source_sha256": sha256_file(train_file),
        "subset": {"mod": subset[0], "rem": subset[1]} if subset else None,
        "policy": {
            "min_overlap": MIN_OVERLAP,
            "eps": EPS,
            "tolerance": TOL,
            "duplicate_policy": "last-line-wins",
            "jaccard_threshold": JACCARD_THRESHOLD if mode == "user" else None,
        },
        "counts": {
            "n_entities": len(entities),
            "n_dims": len(dims),
            "nnz": len(vals),
            "n_pairs": len(golden),
            "n_emitted": n_emitted,
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
        "export_duration_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"fixture written to {output_dir}")
    print(json.dumps(meta["counts"]))
    print(f"Duration: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
