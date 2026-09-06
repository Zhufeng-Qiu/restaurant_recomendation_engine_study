"""Decide whether a compute-sanitizer run actually proved anything.

`compute-sanitizer ... || true` followed by a grep for memory errors reports
"no memory errors" for a run that never executed the kernel: a binary that
cannot load, a sanitizer that failed to attach, a launch failure. Absence of
evidence read as evidence of absence.

A run counts as clean only if all of these hold:

  * the sanitizer reached its own summary ("ERROR SUMMARY: N errors")
  * the target actually ran and validated -- its summary JSON is present with
    tol_failures == 0
  * no memory error, launch failure or internal sanitizer error appears
  * the only tolerated errors are NCCL's cudaErrorPeerAccessAlreadyEnabled
    (704), raised inside libnccl's own bootstrap threads and swallowed there;
    the count is identical under the pre-packing mapping, so it is not ours

  check_sanitizer.py [--allow-nccl-peer] < combined-output
  check_sanitizer.py --selftest
"""

import json
import re
import sys

MEMORY = re.compile(r"Invalid __(global|shared|local)__ (read|write)|"
                    r"Invalid managed|misaligned|Leaked|"
                    r"Program hit cudaErrorIllegal|Out-of-bounds", re.I)
FATAL = re.compile(r"launch fail|internal error|could not load|cannot execute|"
                   r"error while loading shared libraries|Unable to attach|"
                   r"TARGET APPLICATION TERMINATED|failed to initiali[sz]e", re.I)
PEER = "cudaErrorPeerAccessAlreadyEnabled"
SUMMARY = re.compile(r"ERROR SUMMARY:\s*(\d+)\s+error")


def problems(text, allow_nccl_peer=False):
    out = []
    m = SUMMARY.search(text)
    if not m:
        out.append("sanitizer never printed its ERROR SUMMARY -- it did not "
                   "complete, so nothing was proved")
    total = int(m.group(1)) if m else None

    ran = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "tol_failures" in d:
                ran = True
                if d["tol_failures"] != 0:
                    out.append(f"target reported tol_failures={d['tol_failures']}")
    if not ran:
        out.append("target produced no validated summary JSON -- it did not run "
                   "to completion under the sanitizer")

    if FATAL.search(text):
        out.append("sanitizer or target reported a fatal condition "
                   f"({FATAL.search(text).group(0)!r})")

    mem = [l for l in text.splitlines() if MEMORY.search(l)]
    if mem:
        out.append(f"{len(mem)} memory error line(s), first: {mem[0].strip()[:120]}")

    # Every counted error must be accounted for by the allowlist.
    if total:
        peer = text.count(PEER)
        if allow_nccl_peer and peer >= total and not mem:
            pass                       # all of them are the known NCCL 704
        elif not (allow_nccl_peer and peer):
            out.append(f"ERROR SUMMARY reports {total} error(s) and none are "
                       f"allow-listed")
        elif peer < total:
            out.append(f"ERROR SUMMARY reports {total} error(s) but only {peer} "
                       f"are the allow-listed NCCL peer-access case")
    return out


def selftest():
    fails = 0

    def expect(ok, what):
        nonlocal fails
        if not ok:
            fails += 1
            print(f"  FAIL {what}")

    ok_json = '{"n_pairs":10,"tol_failures":0}'
    clean = f"========= COMPUTE-SANITIZER\n{ok_json}\n========= ERROR SUMMARY: 0 errors\n"
    expect(problems(clean) == [], "a clean run passes")

    # the false positive this file exists for
    expect(problems("bash: ./pearson_engine: cannot execute binary file\n"),
           "a binary that cannot execute is not 'clean'")
    expect(problems(""), "empty output is not 'clean'")
    expect(problems(f"========= ERROR SUMMARY: 0 errors\n"),
           "sanitizer summary without the target running is not 'clean'")
    expect(problems(f"{ok_json}\n"), "target output without a sanitizer summary fails")

    bad = (f"========= Invalid __global__ write of size 4\n{ok_json}\n"
           "========= ERROR SUMMARY: 1 errors\n")
    expect(problems(bad), "a real memory error fails")

    peer = ("========= Program hit cudaErrorPeerAccessAlreadyEnabled\n" * 3
            + ok_json + "\n========= ERROR SUMMARY: 3 errors\n")
    expect(problems(peer), "NCCL 704 fails unless explicitly allowed")
    expect(problems(peer, allow_nccl_peer=True) == [],
           "NCCL 704 passes when allow-listed")
    mixed = ("========= Program hit cudaErrorPeerAccessAlreadyEnabled\n"
             "========= Invalid __global__ read of size 8\n" + ok_json
             + "\n========= ERROR SUMMARY: 2 errors\n")
    expect(problems(mixed, allow_nccl_peer=True),
           "a memory error is not excused by the allowlist")
    fail_json = '{"n_pairs":10,"tol_failures":3}'
    expect(problems(f"{fail_json}\n========= ERROR SUMMARY: 0 errors\n"),
           "a wrong answer fails even with no memory errors")

    print(f"check_sanitizer selftest: {'PASS' if fails == 0 else 'FAIL'}")
    return fails


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    bad = problems(sys.stdin.read(), allow_nccl_peer="--allow-nccl-peer" in sys.argv)
    for b in bad:
        print(f"  {b}")
    raise SystemExit(1 if bad else 0)
