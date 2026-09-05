#!/usr/bin/env bash
# Warp-packing GPU session: build, correctness ladder, resource capture, then
# the experiments. Run from the repository root on a 2x GPU host.
#
# Phases are separate on purpose:
#   * the ptxas/-lineinfo build exists only to report register usage. Every
#     timing comes from the plain Release build.
#   * correctness and compute-sanitizer run BEFORE anything is timed. A fast
#     wrong answer is not a result.
#   * the legacy binary is built from a detached worktree, so the working tree
#     under test is never checked out from under itself.
set -euo pipefail
trap 'echo "ABORTED at line $LINENO: $BASH_COMMAND" >&2' ERR
cd "$(dirname "$0")/../.."
TS=$(date -u +%Y%m%d_%H%M%S)
BUILDLOG="results/bench/warp_packing_build_${TS}.txt"
LADDER="results/bench/warp_packing_correctness_${TS}.txt"
mkdir -p results/bench
# NOT "GROUPS": that is a bash builtin array holding the caller's group
# IDs. Assigning to it is ignored and reading it back yields the primary
# GID -- 0 as root -- which the binaries then correctly rejected as an
# invalid group size.
LANE_GROUPS="1 2 4 8 16 32"
HOISTS="off on"

say() { printf '\n=== %s ===\n' "$*"; }

say "environment" | tee "$BUILDLOG"
{
  date -u; uname -a; nvidia-smi; nvidia-smi topo -m; nvidia-smi topo -p2p r
  nvcc --version
  echo "git_sha=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
  echo "image_digest=${BENCH_IMAGE_DIGEST:-unset}"
} >> "$BUILDLOG" 2>&1

say "legacy worktree (pre-packing baseline)"
git worktree list | grep -q build-legacy-src || \
  git worktree add --detach engine/build-legacy-src "$(git merge-base HEAD main)"
cmake -S engine/build-legacy-src/engine -B engine/build-legacy \
      -DCMAKE_BUILD_TYPE=Release -DENGINE_OPENMP=ON -DENGINE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80 >>"$BUILDLOG" 2>&1
cmake --build engine/build-legacy -j"$(nproc)" >>"$BUILDLOG" 2>&1
echo "legacy_sha=$(git -C engine/build-legacy-src rev-parse HEAD)" >> "$BUILDLOG"

say "release build (this is what every timing comes from)"
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release \
      -DENGINE_OPENMP=ON -DENGINE_CUDA=ON -DENGINE_NCCL=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80 >>"$BUILDLOG" 2>&1
cmake --build engine/build -j"$(nproc)" >>"$BUILDLOG" 2>&1
ctest --test-dir engine/build --output-on-failure 2>&1 | tail -8 | tee -a "$BUILDLOG"

# Always from scratch: ptxas only prints on an actual compile, and a cached
# build would silently report nothing.
say "register / spill report (separate build; NOT used for timing)"
rm -rf engine/build-ptxas
cmake -S engine -B engine/build-ptxas -DCMAKE_BUILD_TYPE=Release \
      -DENGINE_OPENMP=ON -DENGINE_CUDA=ON -DENGINE_NCCL=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_CUDA_FLAGS="-lineinfo -Xptxas=-v" >>"$BUILDLOG" 2>&1
cmake --build engine/build-ptxas -j"$(nproc)" >engine/build-ptxas/compile.log 2>&1
grep -E "Compiling entry|registers|spill|stack frame" engine/build-ptxas/compile.log \
  | tee -a "$BUILDLOG" | tail -20

say "lane-band fixtures (derived; gitignored)"
[ -d data/fixtures/item_full_lane_full ] || python3 tools/make_lane_bands.py data/fixtures/item_full

# --------------------------------------------------------------------------
say "CORRECTNESS LADDER" | tee "$LADDER"
fail=0
for fx in item_tiny item_full_lane_full item_full_lane_tail item_full user_full; do
  [ -d "data/fixtures/$fx" ] || { echo "SKIP $fx (absent)" | tee -a "$LADDER"; continue; }
  for g in $LANE_GROUPS; do
    for o in source bylen; do
      for h in $HOISTS; do
        line=$(./engine/build/pearson_engine "data/fixtures/$fx" --backend cuda \
               --validate --group "$g" --pair-order "$o" --hoist "$h" | tail -1)
        tf=$(python3 -c "import json,sys;d=json.loads(sys.argv[1]);print(d['tol_failures'],d['max_abs_diff'])" "$line")
        echo "cuda  $fx g=$g order=$o hoist=$h -> $tf" | tee -a "$LADDER"
        case "$tf" in 0\ *) ;; *) fail=1 ;; esac
      done
    done
  done
done

# Every mapping must produce output in ORIGINAL pair order, not slot order.
say "output permutation check" | tee -a "$LADDER"
./engine/build/pearson_engine data/fixtures/item_full --backend cuda \
    --group 32 --pair-order source --out /tmp/ref.bin >/dev/null
for g in $LANE_GROUPS; do
  for o in source bylen; do
    for h in $HOISTS; do
      ./engine/build/pearson_engine data/fixtures/item_full --backend cuda \
          --group "$g" --pair-order "$o" --hoist "$h" --out /tmp/try.bin >/dev/null
      r=$(cmp -s /tmp/ref.bin /tmp/try.bin && echo identical || echo DIFFERENT)
      echo "  g=$g order=$o hoist=$h vs g32/source: $r" | tee -a "$LADDER"
      [ "$r" = identical ] || fail=1
    done
  done
done

say "NCCL correctness" | tee -a "$LADDER"
for gpus in 1 2; do
  for pl in f64 i32 packed; do
    for g in $LANE_GROUPS; do
      for o in source bylen; do
        line=$(./engine/build/pearson_engine_nccl data/fixtures/item_tiny \
               --gpus "$gpus" --mode sync --payload "$pl" --validate \
               --group "$g" --pair-order "$o" | tail -1)
        tf=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['tol_failures'])" "$line")
        echo "nccl$gpus $pl g=$g order=$o -> tol_failures=$tf" | tee -a "$LADDER"
        [ "$tf" = 0 ] || fail=1
      done
    done
  done
done
# Big fixture, representative mappings only.
for gpus in 1 2; do
  for pl in f64 packed; do
    for spec in "32 source" "8 bylen" "4 bylen" "1 bylen"; do
      set -- $spec
      line=$(./engine/build/pearson_engine_nccl data/fixtures/item_full \
             --gpus "$gpus" --mode sync --payload "$pl" --validate \
             --group "$1" --pair-order "$2" | tail -1)
      tf=$(python3 -c "import json,sys;d=json.loads(sys.argv[1]);print(d['tol_failures'],d['max_abs_diff'])" "$line")
      echo "nccl$gpus item_full $pl g=$1 order=$2 -> $tf" | tee -a "$LADDER"
      case "$tf" in 0\ *) ;; *) fail=1 ;; esac
    done
  done
done

say "async correctness (chunk boundaries, both finalize streams)" | tee -a "$LADDER"
for chunk in 16384 262144 100003; do
  for fs in comm separate; do
    for spec in "32 source" "8 bylen"; do
      set -- $spec
      line=$(./engine/build/pearson_engine_nccl data/fixtures/item_full \
             --gpus 2 --mode async --payload packed --validate --chunk "$chunk" \
             --finalize-stream "$fs" --group "$1" --pair-order "$2" | tail -1)
      tf=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['tol_failures'])" "$line")
      echo "async chunk=$chunk fs=$fs g=$1 order=$2 -> tol_failures=$tf" | tee -a "$LADDER"
      [ "$tf" = 0 ] || fail=1
    done
  done
done

say "CLI rejection checks" | tee -a "$LADDER"
for bad in "--group 3" "--group 0" "--group 33" "--pair-order sideways" \
           "--hoist maybe" "--hoist" \
           "--mode fast" "--gpus 0" "--gpus 99" "--chunk 0" "--bogus" "--group"; do
  if ./engine/build/pearson_engine_nccl data/fixtures/item_tiny $bad >>"$LADDER" 2>&1; then
    echo "  NOT REJECTED: $bad" | tee -a "$LADDER"; fail=1
  else
    echo "  rejected: $bad" | tee -a "$LADDER"
  fi
done
if ./engine/build/pearson_engine_nccl data/fixtures/item_tiny --mode sync \
     --finalize-stream separate >>"$LADDER" 2>&1; then
  echo "  NOT REJECTED: sync + --finalize-stream separate" | tee -a "$LADDER"; fail=1
else
  echo "  rejected: sync + --finalize-stream separate" | tee -a "$LADDER"
fi

say "compute-sanitizer (memcheck, small fixture, both ends of G)" | tee -a "$LADDER"
if command -v compute-sanitizer >/dev/null; then
  for g in $LANE_GROUPS; do
    for o in source bylen; do
      out=$(compute-sanitizer --tool memcheck --error-exitcode 9 \
            ./engine/build/pearson_engine data/fixtures/item_tiny --backend cuda \
            --validate --group "$g" --pair-order "$o" --hoist on 2>&1 | tail -3) && r=clean || { r=ERRORS; fail=1; }
      echo "memcheck cuda g=$g order=$o hoist=on: $r" | tee -a "$LADDER"
      [ "$r" = clean ] || echo "$out" | tee -a "$LADDER"
    done
  done
  # NCCL's own bootstrap threads call cudaGetLastError while peer access is
  # already enabled, and compute-sanitizer counts each cudaErrorPeerAccess-
  # AlreadyEnabled (704) as an "error". They are API status reports NCCL then
  # swallows, not memory faults, and the count is identical at g=32 with the
  # source order -- the mapping that predates packing entirely. So the pass
  # condition is the absence of MEMORY errors, and the 704s are counted and
  # recorded rather than silently ignored.
  for g in $LANE_GROUPS; do
    out=$(compute-sanitizer --tool memcheck \
          ./engine/build/pearson_engine_nccl data/fixtures/item_tiny --gpus 2 \
          --mode sync --payload packed --validate --group "$g" \
          --pair-order bylen 2>&1 || true)
    mem=$(printf '%s' "$out" | grep -cE "Invalid __(global|shared|local)__ (read|write)|Invalid managed|misaligned|Leaked" || true)
    peer=$(printf '%s' "$out" | grep -c "cudaErrorPeerAccessAlreadyEnabled" || true)
    if [ "$mem" -eq 0 ]; then
      echo "memcheck nccl2 g=$g: no memory errors (${peer}x NCCL peer-access 704)" | tee -a "$LADDER"
    else
      echo "memcheck nccl2 g=$g: $mem MEMORY ERRORS" | tee -a "$LADDER"
      printf '%s\n' "$out" | grep -E "Invalid|misaligned|Leaked" | head -10 | tee -a "$LADDER"
      fail=1
    fi
  done
else
  echo "compute-sanitizer NOT AVAILABLE" | tee -a "$LADDER"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "CORRECTNESS LADDER FAILED -- stopping before any timing." | tee -a "$LADDER"
  exit 1
fi
echo "CORRECTNESS LADDER PASSED" | tee -a "$LADDER"
echo "build report: $BUILDLOG"
echo "ladder:       $LADDER"
