"""End-to-end fault injection: corrupt a fixture, run the real binary, check
the exit code and the JSON it printed.

engine/tests/result_check_test.cpp proves the validator's decision function
rejects bad input. This proves the PROGRAM does -- that the verdict reaches
the process exit code and the benchmark record, which is the part a harness
actually consumes. A validator that returns the right answer into a variable
nobody checks is not a gate.

The output side of the injection (a NaN in the computed similarities) stays at
unit level on purpose: it can no longer be produced from a fixture, because
non-finite ratings are refused before the kernel runs. That is the intended
design, so the two tests cover the two halves rather than duplicating one.

Usage: fault_injection_e2e.py <binary> <fixture_dir> [--backend serial]
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

NAN = struct.pack("<d", float("nan"))
POS_INF = struct.pack("<d", float("inf"))
NEG_INF = struct.pack("<d", float("-inf"))


def poke(path, index, raw):
    """Overwrite the index-th float64 of a binary file."""
    with open(path, "r+b") as f:
        f.seek(index * 8)
        f.write(raw)


def swap(path, a, b):
    with open(path, "r+b") as f:
        f.seek(a * 8); va = f.read(8)
        f.seek(b * 8); vb = f.read(8)
        f.seek(a * 8); f.write(vb)
        f.seek(b * 8); f.write(va)


def truncate_pairs(fixture, n_drop):
    """Drop trailing pairs from golden.bin only, so it disagrees with pairs.bin."""
    g = os.path.join(fixture, "golden.bin")
    size = os.path.getsize(g)
    with open(g, "r+b") as f:
        f.truncate(size - 8 * n_drop)


def run(binary, fixture, backend):
    p = subprocess.run([binary, fixture, "--backend", backend, "--validate"],
                       capture_output=True, text=True, timeout=300)
    obj = None
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                pass
    return p.returncode, obj, p.stderr


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    binary, source = sys.argv[1], sys.argv[2].rstrip("/")
    backend = "serial"
    if "--backend" in sys.argv:
        backend = sys.argv[sys.argv.index("--backend") + 1]

    failures = 0
    checks = 0

    def expect(ok, what, detail=""):
        nonlocal failures, checks
        checks += 1
        if ok:
            print(f"ok   {what}")
        else:
            failures += 1
            print(f"FAIL {what}{('  -- ' + detail) if detail else ''}")

    with tempfile.TemporaryDirectory(prefix="fault_inject_") as tmp:
        def variant(name, mutate):
            d = os.path.join(tmp, name)
            shutil.copytree(source, d)
            mutate(d)
            return d

        # ---- positive control: an untouched copy must still pass ----
        clean = variant("clean", lambda d: None)
        rc, obj, _ = run(binary, clean, backend)
        expect(rc == 0, "clean copy exits 0", f"rc={rc}")
        expect(obj is not None and obj.get("validation_passed") is True,
               "clean copy reports validation_passed")
        expect(obj is not None and obj.get("nonfinite_golden") == 0,
               "clean copy reports no non-finite golden")

        # ---- non-finite golden: the case that used to exit 0 silently ----
        for label, raw in (("NaN", NAN), ("+Inf", POS_INF), ("-Inf", NEG_INF)):
            d = variant(f"golden_{label}",
                        lambda d, r=raw: poke(os.path.join(d, "golden.bin"), 7, r))
            rc, obj, err = run(binary, d, backend)
            expect(rc == 1, f"golden[7] = {label} exits 1", f"rc={rc}")
            expect(obj is not None and obj.get("validation_passed") is False,
                   f"golden[7] = {label} reports validation_passed false")
            expect(obj is not None and obj.get("nonfinite_golden", 0) >= 1,
                   f"golden[7] = {label} counts the non-finite value")
            expect("validation FAILED" in err,
                   f"golden[7] = {label} says so on stderr")
            # A NaN must never reach the JSON as a bare token.
            expect(obj is not None,
                   f"golden[7] = {label} still emits parseable JSON")

        # ---- non-finite ratings: refused before the kernel runs ----
        for label, raw in (("NaN", NAN), ("+Inf", POS_INF), ("-Inf", NEG_INF)):
            d = variant(f"vals_{label}",
                        lambda d, r=raw: poke(os.path.join(d, "vals.bin"), 11, r))
            rc, obj, err = run(binary, d, backend)
            expect(rc == 2, f"vals[11] = {label} exits 2 (input rejected)",
                   f"rc={rc}")
            expect("input rejected" in err,
                   f"vals[11] = {label} explains the rejection on stderr")
            expect(obj is None,
                   f"vals[11] = {label} prints no result record at all")

        # ---- misplacement: same values, wrong positions ----
        d = variant("swapped", lambda d: swap(os.path.join(d, "golden.bin"), 0, 2))
        rc, obj, _ = run(binary, d, backend)
        expect(rc == 1, "two golden values swapped exits 1", f"rc={rc}")
        expect(obj is not None and obj.get("tol_failures", 0) >= 1,
               "two golden values swapped is caught by tolerance")

        # ---- length disagreement ----
        d = variant("short", lambda d: truncate_pairs(d, 1))
        rc, obj, err = run(binary, d, backend)
        expect(rc != 0, "golden one pair short is refused", f"rc={rc}")
        expect("mismatch" in err or "length" in err,
               "golden one pair short names the length problem", err.strip()[:80])

    print(f"\n{checks} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
