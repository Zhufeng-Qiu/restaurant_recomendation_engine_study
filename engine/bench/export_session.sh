#!/usr/bin/env bash
# Package everything a pod is holding, so terminating it loses nothing.
#
# Run this LAST, after the benchmarks, not after the correctness ladder. The
# verify script used to write its own archive at the end of the ladder, which
# is before any benchmark exists -- so the archive contained four verification
# files and the run that mattered was still only in /tmp when the pod died.
#
# Usage:  bash engine/bench/export_session.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG="${1:-$(date -u +%Y%m%d_%H%M%S)}"
OUT="results/bench"
mkdir -p "$OUT"

MANIFEST="$OUT/session_manifest_${TAG}.txt"
{
  echo "# session manifest ${TAG}"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(uname -n)"
  echo "git_sha=$(git rev-parse HEAD)"
  # Tracked and untracked separately: a pod always carries build directories
  # and results, and calling that "dirty" hides whether a source file changed.
  echo "git_dirty_tracked=$(git status --porcelain | grep -v '^??' | wc -l | tr -d ' ')"
  echo "git_dirty_untracked=$(git status --porcelain | grep -c '^??' | tr -d ' ')"
  echo "image_digest=${BENCH_IMAGE_DIGEST:-unset}"
  echo
  echo "# tracked files modified, if any:"
  git status --porcelain | grep -v '^??' || echo "  (none)"
  echo
  echo "# full diff of tracked changes:"
  git diff
  echo
  echo "# gpus:"
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader 2>/dev/null || echo "  (none)"
} > "$MANIFEST" 2>&1

# Copy loose logs into the results tree so one rsync of results/bench/ is
# sufficient. /tmp is what gets lost when a pod is terminated.
# Only logs this session plausibly wrote: /tmp on a shared or reused host can
# hold anything, and sweeping it wholesale drags unrelated files into the
# repository (it already did once, locally).
for f in /tmp/headline.log /tmp/final.log /tmp/verify.log /tmp/sess.log /tmp/build.log; do
  [ -e "$f" ] || continue
  cp -n "$f" "$OUT/session_${TAG}_$(basename "$f")" 2>/dev/null || true
done

ARCHIVE="$OUT/session_${TAG}.tar.gz"
tar czf "$ARCHIVE" -C "$OUT" $(cd "$OUT" && ls | grep -v '\.tar\.gz$') 2>/dev/null || true

echo "manifest: $MANIFEST"
echo "archive:  $ARCHIVE  ($(du -h "$ARCHIVE" | cut -f1))"
echo "contents: $(tar tzf "$ARCHIVE" | wc -l | tr -d ' ') files"
echo
echo "Retrieve before terminating the pod:"
echo "  rsync -az <pod>:$(pwd)/results/bench/ ./results/bench/"
