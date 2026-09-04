"""Audit how many bits the six sufficient statistics actually need.

Written for the Sep. 4 2026 correction: the README claimed "Sixteen bits of
content in 384 bits of wire format", which is the width of the WIDEST single
field (Sxx/Syy/Sxy <= 25N), not the total across all six. The total is the sum
of the six field widths.

Two numbers per field:

  UPPER BOUND    what docs/pearson_contract.md section 9.1 guarantees from the
                 domain alone (n <= N, Sx/Sy <= vmax*N, Sxx/Syy/Sxy <= vmax^2*N).
                 This is what check_payload_domain() enforces, so it is the
                 number the packing must be safe against.

  OBSERVED MAX   the largest value each statistic actually reaches over every
                 candidate pair in the fixture. Always well below the bound,
                 and it is what tells you the real headroom inside a 21-bit
                 packed field.

Stdlib only (same convention as tools/validate_fixture.py); no Spark, no numpy.

Usage:
  python3 tools/payload_bit_audit.py <fixture_dir> [...]         bounds only (fast)
  python3 tools/payload_bit_audit.py --observed <fixture_dir>    also scan all pairs
"""

import os
import sys

from array import array

PACK_BITS = 21  # engine_cuda::kPackBits
FIELDS = ("n", "Sx", "Sy", "Sxx", "Syy", "Sxy")


def read_bin(path, typecode):
    a = array(typecode)
    n = os.path.getsize(path) // a.itemsize
    with open(path, "rb") as f:
        a.fromfile(f, n)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def load(d):
    return (read_bin(f"{d}/offsets.bin", "q"),
            read_bin(f"{d}/dims.bin", "i"),
            read_bin(f"{d}/vals.bin", "d"),
            read_bin(f"{d}/pairs.bin", "i"))


def bits_for(x):
    """Bits needed to hold integer values 0..x inclusive."""
    return int(x).bit_length()


def observed_maxima(offsets, dims, vals, pairs):
    """Largest value each of the six statistics reaches over all pairs."""
    obs = [0.0] * 6
    n_pairs = len(pairs) // 2
    for k in range(n_pairs):
        i = pairs[2 * k]
        j = pairs[2 * k + 1]
        a, a_end = offsets[i], offsets[i + 1]
        b, b_end = offsets[j], offsets[j + 1]
        n = 0
        sx = sy = sxx = syy = sxy = 0.0
        while a < a_end and b < b_end:      # sorted two-pointer intersection
            da, db = dims[a], dims[b]
            if da < db:
                a += 1
            elif db < da:
                b += 1
            else:
                x, y = vals[a], vals[b]
                n += 1
                sx += x
                sy += y
                sxx += x * x
                syy += y * y
                sxy += x * y
                a += 1
                b += 1
        for idx, v in enumerate((n, sx, sy, sxx, syy, sxy)):
            if v > obs[idx]:
                obs[idx] = v
    return obs


def audit(d, want_observed):
    offsets, dims, vals, pairs = load(d)
    max_row = max(offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1))
    vmax = max(vals)
    integral = all(v >= 0 and v == int(v) for v in vals)

    print(f"\n=== {d} ===")
    print(f"  entities={len(offsets) - 1}  nnz={len(vals)}  pairs={len(pairs) // 2}")
    print(f"  longest rating row N={max_row}  max rating={vmax:g}  "
          f"non-negative integers={integral}")

    bounds = (max_row, vmax * max_row, vmax * max_row,
              vmax * vmax * max_row, vmax * vmax * max_row, vmax * vmax * max_row)

    print(f"\n  {'field':6s} {'upper bound':>14s} {'bits':>5s}")
    total = 0
    for name, b in zip(FIELDS, bounds):
        w = bits_for(b)
        total += w
        print(f"  {name:6s} {int(b):>14,} {w:>5d}")
    print(f"  {'TOTAL':6s} {'':>14s} {total:>5d}   <- content bits per pair")
    print(f"  wire format 6 x float64 = 384 bits;  packed 2 x uint64 = 128 bits "
          f"({6 * PACK_BITS} used)")

    widest = max(bits_for(b) for b in bounds)
    print(f"  widest single field = {widest} bits  "
          f"(this is the number the README called 'sixteen bits of content')")

    for f, b in zip(FIELDS, bounds):
        if bits_for(b) > PACK_BITS:
            print(f"  !! {f} needs {bits_for(b)} bits, exceeds the "
                  f"{PACK_BITS}-bit packed field")

    if not want_observed:
        return
    obs = observed_maxima(offsets, dims, vals, pairs)
    print(f"\n  {'field':6s} {'observed max':>14s} {'bits':>5s} {'headroom':>11s}")
    total_obs = 0
    for name, v in zip(FIELDS, obs):
        w = bits_for(v)
        total_obs += w
        head = (1 << PACK_BITS) / v if v else float("inf")
        print(f"  {name:6s} {int(v):>14,} {w:>5d} {head:>10.1f}x")
    print(f"  {'TOTAL':6s} {'':>14s} {total_obs:>5d}")


def main():
    args = [a for a in sys.argv[1:] if a != "--observed"]
    want_observed = "--observed" in sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    for d in args:
        audit(d.rstrip("/"), want_observed)


if __name__ == "__main__":
    main()
