"""Type-check the CUDA sources on a machine with no CUDA toolkit.

nccl_main.cu is the file that produced every published GPU number here, and
it keeps being edited on a laptop that cannot compile it. This catches what a
host compiler can catch -- type errors, argument counts, undeclared names,
and printf format mismatches, which is the defect class that has actually
shipped -- so the first thing a rented GPU does is not find a typo.

Kernel launch syntax is the only thing a host compiler cannot parse, so
`kernel<<<grid, block, shm, stream>>>(args)` is rewritten to `kernel(args)`
in a scratch copy. The signatures stay real, so a call with the wrong
arguments still fails.

It proves nothing about device semantics. See cuda_stubs/README.md.

Usage: cuda_syntax_check.py <engine_src_dir> [--cxx c++]
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

LAUNCH = re.compile(r"<<<.*?>>>", re.S)
# Sources to check, and the headers that must be shadowed because they carry
# launch syntax of their own.
TARGETS = ["nccl/nccl_main.cu", "cuda/cuda_backend.cu"]
SHADOW = ["cuda/pair_kernel.cuh"]


def strip_launches(text):
    return LAUNCH.sub("", text), len(LAUNCH.findall(text))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = os.path.abspath(sys.argv[1].rstrip("/"))
    cxx = sys.argv[sys.argv.index("--cxx") + 1] if "--cxx" in sys.argv else "c++"
    stubs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_stubs")

    failures = 0
    with tempfile.TemporaryDirectory(prefix="cuda_syntax_") as tmp:
        shadow_dir = os.path.join(tmp, "shadow")
        for rel in SHADOW:
            dst = os.path.join(shadow_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(os.path.join(src, rel)) as f:
                text, n = strip_launches(f.read())
            with open(dst, "w") as f:
                f.write(text)
            print(f"  shadowed {rel} ({n} launches rewritten)")

        for rel in TARGETS:
            path = os.path.join(src, rel)
            if not os.path.exists(path):
                print(f"  skip {rel} (absent)")
                continue
            with open(path) as f:
                text, n = strip_launches(f.read())
            scratch = os.path.join(tmp, os.path.basename(rel) + ".cpp")
            with open(scratch, "w") as f:
                f.write(text)

            # Both build variants. ENGINE_PROFILING is off by default, so
            # its code is the most likely to rot unnoticed -- and it is the
            # apparatus that measures the timing boundary, which makes a
            # silent breakage there particularly expensive.
            for variant in ([], ["-DENGINE_PROFILING"]):
              cmd = [cxx, "-std=c++17", "-fsyntax-only",
                     "-Wall", "-Wformat=2", "-Werror=format",
                     "-Werror=format-extra-args",
                     "-D__CUDACC__"] + variant + [
                     "-include", os.path.join(stubs, "shim.h"),
                     "-I", stubs, "-I", shadow_dir, "-I", src, scratch]
              p = subprocess.run(cmd, capture_output=True, text=True)
            # The launch rewrite leaves the grid/block variables unused; that
            # is an artefact of this check, not of the source.
              noise = [ln for ln in p.stderr.splitlines()
                       if "unused variable" not in ln
                       and not ln.strip().startswith(("|", "^", "~"))
                       and "warning generated" not in ln
                       and ln.strip()]
              tag = "profiling" if variant else "default "
              if p.returncode != 0:
                  failures += 1
                  print(f"FAIL {rel} [{tag}] ({n} launches rewritten)")
                  print("\n".join(noise[:30]))
              else:
                  print(f"ok   {rel} [{tag}] ({n} launches rewritten, "
                        f"type-checks clean)")

    if failures:
        print(f"\n{failures} source(s) failed to type-check")
        return 1
    print("\nCUDA syntax check: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
