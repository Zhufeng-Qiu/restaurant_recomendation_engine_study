#!/usr/bin/env bash
# Pilot session for the four-way work-division comparison.
#
# Order matters and is not negotiable: correctness first, on every boundary,
# with a memory checker, and only then three pilot blocks. A pilot exists to
# find out whether the output parses, the units are what they claim and the
# times are plausible -- it is NOT formal data and is written to its own file.
#
# Every stage writes its own log. Nothing here decides anything about
# performance; the formal matrix is a separate invocation after the code is
# frozen.
#
# Usage (on the pod):  bash engine/bench/gpu_pilot_session.sh <out_dir>
set -uo pipefail

OUT="${1:?usage: gpu_pilot_session.sh <out_dir>}"
mkdir -p "$OUT"
BUILD=engine/build
NCCL=$BUILD/pearson_engine_nccl
FAIL=0

say() { printf '\n=== %s ===\n' "$*"; }
step() { printf '  %-58s' "$*"; }
ok()   { printf 'ok\n'; }
bad()  { printf 'FAIL\n'; FAIL=$((FAIL+1)); }

say "provenance"
{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || cat GIT_SHA 2>/dev/null || echo unknown)"
  echo "git_dirty=$(git status --porcelain 2>/dev/null | wc -l)"
  echo "nvcc=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader
  echo "--- topology ---"; nvidia-smi topo -m 2>/dev/null | head -8
  echo "--- nccl ---"; ls /usr/lib/x86_64-linux-gnu/libnccl.so.2.* 2>/dev/null
} | tee "$OUT/provenance.txt"

say "build"
cmake -S engine -B $BUILD -DCMAKE_BUILD_TYPE=Release \
      -DENGINE_OPENMP=ON -DENGINE_CUDA=ON -DENGINE_NCCL=ON -DENGINE_MPI=OFF \
      > "$OUT/cmake.log" 2>&1 || { echo "cmake FAILED"; tail -20 "$OUT/cmake.log"; exit 1; }
cmake --build $BUILD -j"$(nproc)" > "$OUT/build.log" 2>&1 \
  || { echo "build FAILED"; grep -E 'error' "$OUT/build.log" | head -20; exit 1; }
echo "built: $(ls -la $NCCL | awk '{print $5}') bytes"
sha256sum $NCCL $BUILD/pearson_engine | tee "$OUT/binary_hashes.txt"

say "host test suite"
ctest --test-dir $BUILD --output-on-failure > "$OUT/ctest.log" 2>&1 \
  && echo "ctest: $(grep -c 'Passed' "$OUT/ctest.log") passed" \
  || { echo "ctest FAILED"; tail -30 "$OUT/ctest.log"; FAIL=$((FAIL+1)); }

# ---------------------------------------------------------------- correctness
# Four configurations, as frozen in the protocol. A is a single GPU computing
# every pair with no collective, which --partition pair --gpus 1 expresses
# exactly.
run_cfg() {  # name gpus partition payload fixture extra...
  local name=$1 gpus=$2 part=$3 pay=$4 fx=$5; shift 5
  $NCCL "$fx" --gpus "$gpus" --partition "$part" --payload "$pay" \
        --mode sync --group 4 --pair-order source --hoist off \
        --plan-metrics off --validate "$@"
}

say "correctness: pair-count boundaries (every config, every N)"
python3 tools/make_prefix_fixtures.py data/fixtures/item_tiny > "$OUT/prefix_fixtures.log" 2>&1
for N in 1 2 3 63 64 65 127 128 129 511 512 513; do
  FX="data/fixtures/item_tiny_n$N"
  [ -d "$FX" ] || continue
  for CFG in "A 1 pair f64" "B 2 dim f64" "C 2 dim packed" "D 2 pair f64"; do
    set -- $CFG
    step "N=$N $1 (gpus=$2 $3 $4)"
    if run_cfg "$@" "$FX" >> "$OUT/boundary_$1.jsonl" 2>>"$OUT/boundary_err.log"; then ok; else bad; fi
  done
done

say "correctness: full workloads (every config)"
for FX in data/fixtures/item_full data/fixtures/user_full; do
  for CFG in "A 1 pair f64" "B 2 dim f64" "C 2 dim packed" "D 2 pair f64"; do
    set -- $CFG
    step "$(basename $FX) $1"
    if run_cfg "$@" "$FX" >> "$OUT/full_$1.jsonl" 2>>"$OUT/full_err.log"; then ok; else bad; fi
  done
done

say "correctness: repeat/reuse stress (20 batches in one process, each validated)"
for CFG in "A 1 pair f64" "B 2 dim f64" "C 2 dim packed" "D 2 pair f64"; do
  set -- $CFG
  step "reuse x20 $1"
  if run_cfg "$@" data/fixtures/item_tiny \
       --resident-bench --warmup 5 --repeat 20 >> "$OUT/reuse.jsonl" 2>>"$OUT/reuse_err.log"; then ok; else bad; fi
done

say "memory check (compute-sanitizer, tiny, every config)"
if command -v compute-sanitizer >/dev/null; then
  # --allow-nccl-peer only for the configurations that initialise NCCL. The
  # pair split does not, and must therefore show a clean zero rather than the
  # 94 cudaErrorPeerAccessAlreadyEnabled that libnccl's bootstrap raises and
  # swallows -- which is also the check that the 94 are in fact NCCL's.
  for CFG in "A 1 pair f64 " "D 2 pair f64 " "B 2 dim f64 --allow-nccl-peer" "C 2 dim packed --allow-nccl-peer"; do
    set -- $CFG
    step "sanitizer $1"
    compute-sanitizer --tool memcheck \
      $NCCL data/fixtures/item_tiny_n129 --gpus $2 --partition $3 --payload $4 \
      --mode sync --group 4 --pair-order source --hoist off --plan-metrics off \
      --validate > "$OUT/sanitizer_$1.log" 2>&1
    # check_sanitizer.py reads the combined output on STDIN. Passing the log
    # as an argument made it judge an empty string, which it correctly called
    # "never printed its ERROR SUMMARY" -- a checker that refused to pass a
    # run it had not seen, which is the behaviour it was written for.
    python3 engine/bench/check_sanitizer.py ${5:-} < "$OUT/sanitizer_$1.log" \
      >> "$OUT/sanitizer_verdict.txt" 2>&1 && ok || bad
  done
else
  echo "  compute-sanitizer not installed -- SKIPPED (recorded, not passed)"
  echo "compute-sanitizer absent" > "$OUT/sanitizer_verdict.txt"
fi

say "refusals (the combinations that must not run)"
for BAD in "--partition pair --payload packed" \
           "--partition pair --pair-order bylen" \
           "--partition pair --mode async" \
           "--warmup 5"; do
  step "refuses: $BAD"
  if $NCCL data/fixtures/item_tiny --gpus 2 $BAD > /dev/null 2>&1; then bad; else ok; fi
done

say "timing boundary (profiling build, delay in band vs out of band)"
echo "requires -DENGINE_PROFILING=ON; run separately" > "$OUT/timing_boundary.txt"

if [ "$FAIL" -ne 0 ]; then
  say "STOPPING: $FAIL correctness failures -- no pilot data will be collected"
  exit 1
fi

# ---------------------------------------------------------------------- pilot
say "pilot: 3 blocks on item_full (NOT formal data)"
python3 engine/bench/architecture_ab.py run \
  --binary "$NCCL" --fixture data/fixtures/item_full \
  --out "$OUT/pilot_item_full.json" --pilot 3 \
  --warmup 20 --repeat 20 2>&1 | tee "$OUT/pilot.log"

python3 engine/bench/architecture_ab.py analyze "$OUT/pilot_item_full.json" \
  2>&1 | tee "$OUT/pilot_summary.txt"

say "done -- $FAIL failures"
