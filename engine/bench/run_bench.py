"""Benchmark runner for pearson_engine backends.

Runs each configuration with warm-ups and repeated trials, parses the JSON
line each binary prints, and writes raw results + a median summary to
machine-readable JSON and CSV under results/bench/.

Usage:
    python3 engine/bench/run_bench.py [--fixture data/fixtures/item_full]
                                      [--repeats 5] [--warmups 2]
                                      [--out results/bench]
                                      [--gpu] [--ranks 1,2,4,8]
                                      [--payloads f64,i32,packed]

Configurations run (all on the chosen fixture):
    serial
    openmp  threads in {1,2,4,8,16} x schedule in {static, dynamic,1024}
    mpi     ranks from --ranks (default 1,2,4,8; skipped if pearson_engine_mpi
            absent, and note Open MPI refuses to run as root without
            OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1)
    --gpu adds:
    cuda    single GPU (pearson_engine --backend cuda)
    nccl    gpus in {1,2} x mode in {sync, async} x payload in --payloads,
            named nccl_{mode}_g{gpus}_{payload}      (pearson_engine_nccl)

Runs before the --payload feature are named nccl_{mode}_g{gpus} with no
suffix; those are f64 runs. make_figures.py resolves both spellings.
"""

import argparse
import csv
import datetime
import json
import os
import platform
import random
import statistics
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(ROOT, "engine", "build", "pearson_engine")
ENGINE_MPI = os.path.join(ROOT, "engine", "build", "pearson_engine_mpi")
ENGINE_NCCL = os.path.join(ROOT, "engine", "build", "pearson_engine_nccl")


def time_of(trial):
    """Comparable steady-state time, whatever backend produced it.

    Every backend now emits device_total_s = stats + allreduce + finalize, and
    async emits pipeline_total (its stages overlap, so they cannot be summed).
    `timing_basis` says which one is comparable. Everything excludes fixture
    load, device setup, plan construction and H2D/D2H -- see
    cold_data_path_of for the cold-start view.

    The fallbacks reconstruct the same quantity from pre-2026-09-05 runs, which
    predate the schema. NCCL sync is the one that changes: those runs never
    timed their finalize kernel, so they understate it by ~0.5-1%.
    """
    if "device_total_s" in trial and trial.get("timing_basis") == "device_total":
        return trial["device_total_s"]
    if trial.get("mode") == "async":
        return trial["t_pipeline_s"]
    if "t_kernel_stats_s" in trial:                  # legacy cuda
        return trial["t_kernel_stats_s"] + trial["t_kernel_finalize_s"]
    if "t_compute_s" in trial:                       # legacy serial / openmp
        return trial["t_compute_s"]
    if "t_kernel_s" in trial:                        # legacy nccl sync
        return (trial["t_kernel_s"] + trial["t_allreduce_s"]
                + trial.get("t_finalize_s", 0.0))
    return trial["t_local_s"] + trial["t_allreduce_s"] + trial["t_finalize_s"]


def cold_data_path_of(trial):
    """Entry-to-result total: load + runtime init + setup/H2D + compute + D2H.

    NOT a CLI one-shot. Process launch -- mpirun's spawn, the CUDA driver's
    first-touch, the shell -- is outside every binary that could measure it,
    so a true one-shot has to be timed by an outer process. What this captures
    is everything from `main` entry to results in host memory, which is the
    part the engine controls. Returns None for runs predating the schema.
    """
    if "cold_data_path_s" in trial:
        return trial["cold_data_path_s"]
    if "one_shot_total_s" in trial:       # pre-rename runs, and they excluded
        return trial["one_shot_total_s"]  # NCCL/MPI init entirely
    if "t_h2d_s" in trial and "t_load_s" in trial:   # legacy cuda
        return (trial["t_load_s"] + trial["t_h2d_s"] + time_of(trial)
                + trial.get("t_d2h_s", 0.0))
    return None


def split_git_status(porcelain):
    """(tracked_paths, untracked_count) from `git status --porcelain`.

    Written without positional slicing because the callers strip the command's
    output, which removes the leading space of the FIRST line only -- so a
    fixed l[3:] eats a character from exactly one path and nothing else, which
    is the kind of defect that survives review.
    """
    tracked, untracked = [], 0
    for line in (porcelain or "").splitlines():
        status, _, path = line.strip().partition(" ")
        if status == "??":
            untracked += 1
        elif path.strip():
            tracked.append(path.strip())
    return tracked, untracked


def timing_identity(trial):
    """Re-derive a trial's cold path from its own stages, where that is defined.

    The plan-construction stage was added for warp packing and was initially
    left out of both GPU cold paths, so a cold caller looked cheaper than it
    was. This catches a term that stops being summed.

    It is NOT a universal formula. Each backend publishes a different
    decomposition, and mpi's in particular is not re-derivable from what it
    prints: its cold_data_path_s is (entry -> data-ready, MPI_MAX over ranks)
    plus device_total, and that first term already contains the fixture load,
    so adding t_load_s again would double-count it and flag every MPI run.
    Backends whose contract is not known here are skipped explicitly, with a
    reason, rather than silently or wrongly.

    Returns None when nothing can be checked.
    """
    cold = trial.get("cold_data_path_s")
    if cold is None:
        return None
    backend = trial.get("backend")
    if backend == "mpi":
        return {"checked": False, "ok": True, "backend": backend,
                "reason": "mpi publishes entry-to-ready, which already "
                          "contains t_load_s; not re-derivable from its fields"}
    if backend not in ("serial", "openmp", "cuda", "nccl"):
        return {"checked": False, "ok": True, "backend": backend,
                "reason": f"no cold-path contract known for backend {backend!r}"}

    # The binary reports whether its diagnostic window overlapped its timed
    # setup window. It did once, silently, and no re-summation could catch it:
    # both sides of the identity read the same inflated t_setup_s.
    if trial.get("stage_windows_disjoint") is False:
        return {"checked": True, "ok": False, "backend": backend,
                "recorded_s": cold, "resummed_s": float("nan"),
                "tolerance_s": 0.0,
                "reason": "diagnostic timing window overlaps the setup window; "
                          "cold_data_path_s includes t_plan_metrics_s"}
    parts = {k: trial[k] for k in
             ("t_load_s", "t_plan_s", "t_comm_init_s", "t_setup_s", "t_h2d_s",
              "t_d2h_s") if k in trial}
    if "t_load_s" not in parts:
        return None
    # t_setup_s on the cuda line repeats t_h2d_s (the staging IS the setup), so
    # counting both would double it.
    staging = parts.get("t_h2d_s", parts.get("t_setup_s", 0.0))
    if ("t_h2d_s" in parts and "t_setup_s" in parts
            and abs(parts["t_h2d_s"] - parts["t_setup_s"]) > 1e-9):
        staging = parts["t_h2d_s"] + parts["t_setup_s"]
    want = (parts["t_load_s"] + parts.get("t_plan_s", 0.0)
            + parts.get("t_comm_init_s", 0.0) + staging + time_of(trial)
            + parts.get("t_d2h_s", 0.0))
    # Loose: the stages use different clocks (CUDA events for kernels,
    # steady_clock for transfers) and the binary adds a little unmeasured glue
    # between them. A dropped stage is orders of magnitude bigger than that.
    tol = max(1e-4, 0.02 * cold)
    return {"checked": True, "ok": abs(cold - want) <= tol, "backend": backend,
            "recorded_s": cold, "resummed_s": want, "tolerance_s": tol}


def timing_identity_warning(config_name, bad):
    """The message for a failed check. A function so it can be tested.

    It exists because the first version of this line referred to an undefined
    variable, which raised only on the failure path -- after a 30-round matrix
    had finished measuring and before anything was written. A diagnostic must
    not be able to destroy the measurement it is describing.
    """
    if bad.get("reason"):
        return f"  WARNING {config_name}: {bad['reason']}"
    return (f"  WARNING {config_name}: cold_data_path_s "
            f"{bad['recorded_s']:.6f} != resummed {bad['resummed_s']:.6f} "
            f"(tolerance {bad['tolerance_s']:.6f})")


def selftest():
    """Exercise both branches of the timing check, including the message."""
    fails = 0

    def check(cond, what):
        nonlocal fails
        if not cond:
            fails += 1
            print(f"  FAIL {what}")

    base = {"backend": "cuda", "timing_basis": "device_total",
            "t_load_s": 0.30, "t_plan_s": 0.10, "t_h2d_s": 0.05,
            "t_setup_s": 0.05, "t_d2h_s": 0.01, "device_total_s": 0.005}
    ok = dict(base, cold_data_path_s=0.30 + 0.10 + 0.05 + 0.005 + 0.01)
    check(timing_identity(ok)["ok"], "consistent cuda record passes")
    bad = dict(base, cold_data_path_s=0.30 + 0.05 + 0.005 + 0.01)  # plan dropped
    r = timing_identity(bad)
    check(not r["ok"], "a dropped plan stage is caught")
    # The failure path must produce a message, not an exception.
    msg = timing_identity_warning("cuda_1gpu", r)
    check("cuda_1gpu" in msg and "resummed" in msg, "warning renders")
    # mpi is skipped rather than wrongly flagged: its cold path already
    # contains t_load, so the generic formula would double-count it.
    mpi = {"backend": "mpi", "timing_basis": "device_total", "t_load_s": 0.4,
           "device_total_s": 0.2, "cold_data_path_s": 0.9}
    m = timing_identity(mpi)
    check(m is not None and not m["checked"] and m["ok"], "mpi is skipped, not failed")
    check(timing_identity({"backend": "cuda"}) is None, "no cold path -> None")
    # The stripped-first-line trap, which ate a character from exactly one path.
    tr, un = split_git_status(" M a/b.py\n M c/d.py\n?? e/f.txt")
    check(tr == ["a/b.py", "c/d.py"] and un == 1, "porcelain parses with leading space")
    tr, un = split_git_status("M  a/b.py\n M c/d.py\n?? e/f.txt")
    check(tr == ["a/b.py", "c/d.py"] and un == 1, "porcelain parses when stripped")
    # An overlapping diagnostic window is a failure even when the arithmetic
    # is self-consistent -- which is exactly how it escaped the first time.
    nested = dict(base, cold_data_path_s=0.465, backend="nccl",
                  stage_windows_disjoint=False)
    n = timing_identity(nested)
    check(n["checked"] and not n["ok"], "overlapping timing windows are caught")
    check("overlaps" in timing_identity_warning("nccl_sync_g2_packed", n),
          "the overlap warning renders")

    # --- the failure modes that cost an hour of GPU time -------------------
    import tempfile, time as _time
    good = ('echo \'{"n_pairs":1,"tol_failures":0,"max_abs_diff":0.0,'
            '"device_total_s":0.1,"timing_basis":"device_total"}\'')

    t0 = _time.time()
    try:
        run_once(["/bin/sh", "-c", "sleep 30"], {}, timeout=1.0)
        check(False, "a hung trial is killed")
    except TrialTimeout:
        check(_time.time() - t0 < 5, "a hung trial is killed by its timeout")

    try:                                   # OOM/SIGKILL: no traceback at all
        run_once(["/bin/sh", "-c", "kill -9 $$"], {})
        check(False, "a killed trial raises")
    except RuntimeError as exc:
        check("signal 9" in str(exc), "a killed trial names the signal")

    try:                                   # exit 0, no JSON: used to return {}
        run_once(["/bin/sh", "-c", "echo not json; exit 0"], {})
        check(False, "a trial with no JSON raises")
    except RuntimeError as exc:
        check("no usable record" in str(exc), "exit 0 without JSON is rejected")

    check(run_once(["/bin/sh", "-c", good], {})["n_pairs"] == 1,
          "a valid record still parses")
    # A record with a summary line but no timing field -- what a vanished
    # cuda_detail line looks like -- must fail here, not after thirty rounds.
    try:
        run_once(["/bin/sh", "-c",
                  'echo \'{"n_pairs":1,"tol_failures":0,"max_abs_diff":0.0}\''], {})
        check(False, "a record with no timing field raises")
    except RuntimeError as exc:
        check("one of" in str(exc), "a record with no timing field is rejected")

    d = tempfile.mkdtemp()
    ck = os.path.join(d, "ck.json")
    cfgs = [{"name": "a", "threads": 1, "cmd": ["/bin/sh", "-c", good], "env": {}}]
    bench_round_robin(cfgs, 0, 3, 7, log=lambda m: None, checkpoint=ck)
    saved = json.load(open(ck))
    check(saved["rounds_done"] == 3 and len(saved["trials_by_config"]["a"]) == 3,
          "every round is checkpointed")
    check(not os.path.exists(ck + ".tmp"), "atomic write leaves no temp file")
    saved["rounds_done"] = 1
    saved["trials_by_config"]["a"] = saved["trials_by_config"]["a"][:1]
    atomic_write_json(ck, saved)
    tr, _ = bench_round_robin(cfgs, 0, 3, 7, log=lambda m: None, resume_from=ck)
    check(len(tr["a"]) == 3, "resume completes the remaining rounds")

    print(f"run_bench selftest: {'PASS' if fails == 0 else 'FAIL'}")
    return fails


def sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _cmd(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def environment():
    """Everything needed to know what machine a number came from.

    A cgroup CPU quota is the difference between a 128-core host and 27 cores
    of actual CPU time, and a rank sweep is uninterpretable without it. The
    earlier schema recorded `ncpu` via sysctl, which is empty on Linux, so
    every pod run was stored with no core count at all.
    """
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "logical_cpus": os.cpu_count(),
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "ncpu": sysctl("hw.ncpu") or str(os.cpu_count() or ""),
    }
    model = _cmd(["sh", "-c", "lscpu | sed -n 's/^Model name: *//p'"])
    if model:
        env["cpu"] = model.splitlines()[0]
    try:
        env["affinity_cpus"] = len(os.sched_getaffinity(0))
    except AttributeError:
        env["affinity_cpus"] = None
    env["cpuset"] = _read("/sys/fs/cgroup/cpuset.cpus.effective")
    quota = _read("/sys/fs/cgroup/cpu.max")
    env["cpu_max"] = quota
    if quota and quota.split()[0] != "max":
        q, period = quota.split()[:2]
        env["cpu_quota_equivalents"] = round(int(q) / int(period), 2)
    env["git_sha"] = _cmd(["git", "rev-parse", "HEAD"])
    dirty = _cmd(["git", "status", "--porcelain"])
    env["git_dirty"] = bool(dirty) if dirty is not None else None
    # Split, because "dirty" alone is unreadable: a pod always carries
    # untracked build directories and result files, and that is not the same
    # as a modified source file. A run was once described as having no tracked
    # modifications when it had one, because only the aggregate was recorded.
    if dirty is not None:
        tracked, untracked = split_git_status(dirty)
        env["git_dirty_tracked"] = len(tracked)
        env["git_dirty_tracked_files"] = tracked[:20]
        env["git_dirty_untracked"] = untracked
    # Passed in by whatever launched the container; there is no reliable way to
    # read your own image digest from inside it.
    env["image_digest"] = os.environ.get("BENCH_IMAGE_DIGEST")
    nvcc = _cmd(["sh", "-c", "nvcc --version | sed -n 's/.*release \\([0-9.]*\\).*/\\1/p'"])
    if nvcc:
        env["cuda_toolkit"] = nvcc.strip()
    nccl = _cmd(["sh", "-c",
                 "ls /usr/lib/x86_64-linux-gnu/libnccl.so.2.* 2>/dev/null | head -1"])
    if nccl:
        env["nccl"] = nccl.rsplit(".so.", 1)[-1]
    p2p = _cmd(["sh", "-c", "nvidia-smi topo -p2p rw 2>/dev/null | head -6"])
    if p2p:
        env["p2p_matrix"] = p2p
    gpus = _cmd(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                 "--format=csv,noheader"])
    if gpus:
        env["gpus"] = gpus.splitlines()
    topo = _cmd(["sh", "-c", "nvidia-smi topo -m 2>/dev/null | head -4"])
    if topo:
        env["gpu_topology"] = topo
    return env


# A trial that prints no usable JSON is a failed trial, not an empty one.
# Returning {} let a broken configuration travel all the way to the summary,
# where it crashed an hour of measurement instead of one trial.
# Everything the summariser will later need. Requiring only n_pairs and
# tol_failures let a run whose entire cuda_detail line vanished complete all
# thirty rounds and fail at the end for want of a timing field.
REQUIRED_FIELDS = ("n_pairs", "tol_failures", "max_abs_diff")
REQUIRED_TIMING = ("device_total_s", "t_pipeline_s", "t_compute_s")


class TrialTimeout(RuntimeError):
    pass


def run_once(cmd, env_extra, timeout=None):
    """One invocation -> merged JSON record. Raises on anything unusable.

    Started in its own session so a timeout can kill the whole process group.
    mpirun spawns children that survive a bare kill() and would then compete
    with every later trial for the same GPUs.
    """
    env = dict(os.environ, **env_extra)
    # Wall clock around the whole process. This is the only true CLI one-shot
    # available: process launch, dynamic linking and the CUDA driver's first
    # touch all happen before any code inside the binary could time them, and
    # cold_data_path_s explicitly excludes them.
    t_process0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            p.kill()
        p.communicate()
        raise TrialTimeout(f"{' '.join(cmd)} exceeded {timeout}s")
    if p.returncode != 0:
        # OOM and SIGKILL arrive as a negative return code and no traceback,
        # so say which signal rather than printing empty stderr.
        why = (f"killed by signal {-p.returncode}" if p.returncode < 0
               else f"exit {p.returncode}")
        raise RuntimeError(f"{' '.join(cmd)} failed ({why}):\n{err[-2000:]}")
    # Backends may print auxiliary JSON lines (e.g. cuda_detail) before the
    # summary line; merge them all into one flat record.
    rec = {}
    for line in out.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            rec.update(d.pop("cuda_detail", {}))
            rec.update(d)
    # Named for what it is: these invocations pass --validate, so the wall
    # clock includes the golden comparison as well as launch, linking and the
    # CUDA driver's first touch. It is a validated-CLI wall time, not the
    # latency of a production one-shot, which would not validate.
    rec["t_process_wall_s"] = time.perf_counter() - t_process0
    rec["t_process_wall_includes_validate"] = True
    missing = [f for f in REQUIRED_FIELDS if f not in rec]
    if not any(f in rec for f in REQUIRED_TIMING):
        missing.append(f"one of {REQUIRED_TIMING}")
    if missing:
        raise RuntimeError(f"{' '.join(cmd)} produced no usable record "
                           f"(missing {missing}); stdout was:\n{out[-800:]}")
    return rec


def atomic_write_json(path, doc):
    """Write via a temporary file and rename, so a checkpoint is never half a
    file. A crash mid-write must not also destroy the previous checkpoint."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def bench_round_robin(configs, warmups, repeats, seed, log=print,
                      timeout=None, checkpoint=None, resume_from=None,
                      fixture=None):
    """Interleave configurations instead of draining each one in turn.

    Running 30 consecutive trials of `f64` and then 30 of `packed` confounds
    the payload with whatever drifts over those two minutes -- clock/thermal
    state, a noisy neighbour, page cache. The compression effect being measured
    is 4-7%, comfortably inside that. Round-robin with a per-round reshuffle
    spreads every configuration across the whole session, so drift hits all of
    them alike and cancels in the medians.

    Returns {name: [trial, ...]} plus {name: error} for configs that failed.
    A failure disables that config for the remaining rounds but never aborts
    the session -- the GPU half of a run should survive a broken MPI.
    """
    rng = random.Random(seed)
    trials = {c["name"]: [] for c in configs}
    errors = {}
    live = list(configs)
    first_round = 0

    if resume_from:
        # Continue an interrupted session. The shuffle sequence is a function
        # of the seed alone, so replaying the completed rounds' shuffles puts
        # the generator back where it was -- the resumed half sees the order it
        # would have seen.
        prev = json.load(open(resume_from))
        # A checkpoint records the experiment it came from. Resuming with
        # different parameters would silently splice two different experiments
        # into one result file.
        for key, now in (("seed", seed), ("repeats", repeats),
                         ("warmups", warmups)):
            was = prev.get(key)
            if was is not None and was != now:
                raise SystemExit(
                    f"--resume refused: checkpoint has {key}={was}, this run "
                    f"has {key}={now}. Re-run with the original value or start "
                    f"a fresh session.")
        was_cfgs = prev.get("all_configs")
        now_cfgs = [c["name"] for c in configs]
        if was_cfgs is not None and was_cfgs != now_cfgs:
            raise SystemExit(
                "--resume refused: the configuration set differs from the "
                f"checkpoint ({len(was_cfgs)} vs {len(now_cfgs)} configs).")
        was_fx = prev.get("fixture")
        if was_fx is not None and fixture is not None and was_fx != fixture:
            raise SystemExit(
                f"--resume refused: checkpoint fixture {was_fx!r} != {fixture!r}")
        for name, ts in prev.get("trials_by_config", {}).items():
            if name in trials:
                trials[name] = ts
        errors.update(prev.get("errors", {}))
        live = [c for c in live if c["name"] not in errors]
        first_round = prev.get("rounds_done", 0)
        for _ in range(first_round):
            rng.shuffle(list(live))
        log(f"resumed from {os.path.basename(resume_from)}: "
            f"{first_round} rounds done, {len(live)} live configs")
    else:
        for c in list(live):                   # warm-ups, in declaration order
            try:
                for _ in range(warmups):
                    run_once(c["cmd"], c["env"], timeout=timeout)
            except Exception as exc:
                errors[c["name"]] = f"{type(exc).__name__}: {exc}"
                log(f"{c['name']:26s} FAILED (warm-up): {type(exc).__name__}: {exc}")
                live.remove(c)

    for r in range(first_round, repeats):
        order = list(live)
        rng.shuffle(order)
        for c in order:
            try:
                trials[c["name"]].append(
                    run_once(c["cmd"], c["env"], timeout=timeout))
            except Exception as exc:
                errors[c["name"]] = f"{type(exc).__name__}: {exc}"
                log(f"{c['name']:26s} FAILED (round {r + 1}): "
                    f"{type(exc).__name__}: {exc}")
                live.remove(c)
        # Checkpoint EVERY round. The previous version held 30 rounds over 37
        # configurations in memory and wrote once at the end; a crash in the
        # summariser threw all of it away.
        if checkpoint:
            atomic_write_json(checkpoint, {
                "partial": True, "rounds_done": r + 1, "repeats": repeats,
                "seed": seed, "warmups": warmups, "fixture": fixture,
                "all_configs": [c["name"] for c in configs],
                "live_configs": [c["name"] for c in live],
                "errors": errors, "trials_by_config": trials,
            })
        log(f"  ... round {r + 1}/{repeats} done ({len(live)} live configs)")
    return trials, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=os.path.join(ROOT, "data/fixtures/item_full"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/bench"))
    ap.add_argument("--gpu", action="store_true",
                    help="add cuda + nccl configurations (GPU host only)")
    ap.add_argument("--timeout", type=float, default=900,
                    help="per-trial wall clock limit in seconds; the whole "
                         "process group is killed on expiry (0 disables)")
    ap.add_argument("--checkpoint",
                    help="path for the per-round checkpoint "
                         "(default: <out>/checkpoint_<stamp>.json)")
    ap.add_argument("--resume", help="continue from a checkpoint written by "
                                     "an interrupted run")
    ap.add_argument("--selftest", action="store_true",
                    help="run the timing-check self-test and exit")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the round-robin shuffle; recorded in the "
                         "output so a session can be replayed in the same order")
    ap.add_argument("--ranks", default="1,2,4,8",
                    help="comma-separated MPI rank counts; the default suits a "
                         "10-core laptop, a many-core host wants 1,2,4,8,16,32,64")
    ap.add_argument("--payloads", default="f64,i32,packed",
                    help="comma-separated AllReduce payload representations to "
                         "sweep for the nccl backend (contract §9)")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    os.makedirs(args.out, exist_ok=True)

    configs = [{"name": "serial", "cmd": [ENGINE, args.fixture, "--backend", "serial",
                                          "--validate"], "env": {}, "threads": 1}]
    for t in (1, 2, 4, 8, 16):
        for sched in ("static", "dynamic,1024"):
            configs.append({
                "name": f"openmp_t{t}_{sched.split(',')[0]}",
                "cmd": [ENGINE, args.fixture, "--backend", "openmp", "--validate"],
                "env": {"OMP_NUM_THREADS": str(t), "OMP_SCHEDULE": sched},
                "threads": t,
            })
    if os.path.exists(ENGINE_MPI):
        for r in [int(x) for x in args.ranks.split(",")]:
            configs.append({
                "name": f"mpi_r{r}",
                "cmd": ["mpirun", "-np", str(r), ENGINE_MPI, args.fixture, "--validate"],
                "env": {}, "threads": r,
            })
    if args.gpu:
        configs.append({"name": "cuda_1gpu",
                        "cmd": [ENGINE, args.fixture, "--backend", "cuda",
                                "--validate"], "env": {}, "threads": 1})
        for payload in args.payloads.split(","):
            for mode in ("sync", "async"):
                for g in (1, 2):
                    configs.append({
                        "name": f"nccl_{mode}_g{g}_{payload}",
                        "cmd": [ENGINE_NCCL, args.fixture, "--gpus", str(g),
                                "--mode", mode, "--payload", payload,
                                "--validate"],
                        "env": {}, "threads": g,
                    })
                    # async only: finalize on its own stream instead of the
                    # communication stream. Named with a _sep suffix so the
                    # pre-fix rows stay directly comparable.
                    if mode == "async":
                        configs.append({
                            "name": f"nccl_{mode}_g{g}_{payload}_sep",
                            "cmd": [ENGINE_NCCL, args.fixture, "--gpus", str(g),
                                    "--mode", mode, "--payload", payload,
                                    "--finalize-stream", "separate",
                                    "--validate"],
                            "env": {}, "threads": g,
                        })

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    print(f"round-robin order, seed={seed}, {args.repeats} rounds x "
          f"{len(configs)} configs after {args.warmups} warm-ups", flush=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt = args.checkpoint or os.path.join(args.out, f"checkpoint_{stamp}.json")

    def log(msg):
        print(msg, flush=True)          # unbuffered: a run this long is
                                        # watched from outside, and a silent
                                        # pipe is indistinguishable from a hang

    all_trials, errors = bench_round_robin(
        configs, args.warmups, args.repeats, seed, log=log,
        timeout=(args.timeout or None), checkpoint=ckpt,
        resume_from=args.resume, fixture=args.fixture)

    results = []
    for cfg in configs:
        trials = all_trials[cfg["name"]]
        if cfg["name"] in errors or not trials:
            results.append({"config": cfg["name"],
                            "threads_or_ranks": cfg["threads"],
                            "error": errors.get(cfg["name"], "no trials")})
            continue
        times = [time_of(t) for t in trials]
        # Quartiles need enough samples to mean anything; below 4 trials
        # report them as None rather than a number computed from 2 points.
        if len(times) >= 4:
            q1, _, q3 = statistics.quantiles(times, n=4)
            iqr, p25, p75 = q3 - q1, q1, q3
        else:
            iqr = p25 = p75 = None
        rec = {
            "config": cfg["name"],
            "threads_or_ranks": cfg["threads"],
            "median_s": statistics.median(times),
            "min_s": min(times),
            "max_s": max(times),
            "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
            "p25_s": p25,
            "p75_s": p75,
            "iqr_s": iqr,
            "n_trials": len(times),
            "max_abs_diff": max(t["max_abs_diff"] for t in trials),
            "tol_failures": max(t["tol_failures"] for t in trials),
            "timing_basis": trials[0].get("timing_basis", "legacy"),
            "trials": trials,
        }
        colds = [v for v in (cold_data_path_of(t) for t in trials) if v is not None]
        if colds:
            # In-process, and it excludes process launch, dynamic linking and
            # the CUDA driver's first touch -- and, since plan diagnostics are
            # opt-in, whatever those cost when they are switched on. It is a
            # data-path estimate, not a caller-visible latency.
            rec["data_path_median_s"] = statistics.median(colds)
            rec["cold_data_path_median_s"] = statistics.median(colds)
        walls = [t["t_process_wall_s"] for t in trials if "t_process_wall_s" in t]
        if walls:
            # The one a caller actually waits for.
            rec["process_wall_median_s"] = statistics.median(walls)
            rec["process_wall_min_s"] = min(walls)
        # Wrapped: this is reporting, and reporting must never be able to
        # destroy the measurement. It already did once.
        try:
            ident = [x for x in (timing_identity(t) for t in trials)
                     if x is not None]
            if ident:
                checked = [x for x in ident if x.get("checked")]
                rec["timing_identity_checked"] = bool(checked)
                rec["timing_identity_ok"] = all(x["ok"] for x in ident)
                if not checked and ident:
                    rec["timing_identity_skipped"] = ident[0].get("reason")
                if not rec["timing_identity_ok"]:
                    bad = next(x for x in ident if not x["ok"])
                    rec["timing_identity_detail"] = bad
                    print(timing_identity_warning(cfg["name"], bad), flush=True)
        except Exception as exc:                       # noqa: BLE001
            rec["timing_identity_error"] = repr(exc)
            print(f"  WARNING {cfg['name']}: timing check failed: {exc!r}",
                  flush=True)
        # The lane mapping belongs on the summary record, not only inside the
        # trials: a table that cannot see it labels the row from memory, which
        # is how the README kept saying "warp-per-pair" after the default
        # changed.
        for field in ("group", "pair_order", "hoist", "plan_order_basis"):
            if field in trials[0]:
                rec[field] = trials[0][field]
        for field in ("payload", "payload_bytes_per_pair", "allreduce_bytes"):
            if field in trials[0]:
                rec[field] = trials[0][field]
        # The async pipeline reports t_allreduce_s = 0: its collectives run
        # inside the timed pipeline and are not separately measurable. Leave
        # the field out rather than recording a 0 that reads as "free".
        if any(t.get("t_allreduce_s", 0.0) > 0.0 for t in trials):
            rec["median_allreduce_s"] = statistics.median(t["t_allreduce_s"] for t in trials)
            if "comm_fraction" in trials[0]:
                rec["comm_fraction"] = statistics.median(t["comm_fraction"] for t in trials)
            elif rec["median_s"] > 0:
                # A share of the iteration only where the stages are serial.
                # In async they overlap, so allreduce/pipeline is a ratio that
                # can legitimately exceed 1 and is not a fraction of anything;
                # it is named for what it is.
                key = ("allreduce_sum_over_pipeline"
                       if trials[0].get("mode") == "async" else "comm_fraction")
                rec[key] = rec["median_allreduce_s"] / rec["median_s"]
        results.append(rec)
        print(f"{cfg['name']:26s} median={rec['median_s']:.4f}s "
              f"stdev={rec['stdev_s']:.4f}s max_diff={rec['max_abs_diff']:.1e}")

    doc = {
        "schema": "pearson-bench-v1",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "fixture": args.fixture,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "order": "round-robin, reshuffled each round",
        "seed": seed,
        "environment": environment(),
        "results": results,
    }
    json_path = os.path.join(args.out, f"bench_{stamp}.json")
    atomic_write_json(json_path, doc)

    csv_path = os.path.join(args.out, f"bench_{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "threads_or_ranks", "n_trials", "median_s",
                    "min_s", "max_s", "stdev_s", "iqr_s", "p25_s", "p75_s",
                    "payload", "allreduce_bytes", "median_allreduce_s",
                    "comm_fraction", "max_abs_diff", "tol_failures", "error"])
        for r in results:
            if "error" in r:
                w.writerow([r["config"], r["threads_or_ranks"]] + [""] * 14
                           + [r["error"]])
                continue
            ar = r.get("median_allreduce_s")
            fmt = lambda v: f"{v:.6f}" if v is not None else ""
            w.writerow([r["config"], r["threads_or_ranks"], r.get("n_trials", ""),
                        f"{r['median_s']:.6f}",
                        f"{r['min_s']:.6f}", f"{r['max_s']:.6f}", f"{r['stdev_s']:.6f}",
                        fmt(r.get("iqr_s")), fmt(r.get("p25_s")), fmt(r.get("p75_s")),
                        r.get("payload", ""), r.get("allreduce_bytes", ""),
                        fmt(ar),
                        f"{r.get('comm_fraction', '')}", f"{r['max_abs_diff']:.3e}",
                        r["tol_failures"], ""])
    failed = [r["config"] for r in results if "error" in r]
    print(f"\nwrote {json_path}\nwrote {csv_path}")
    if failed:
        print(f"{len(failed)} config(s) FAILED and are recorded as such: "
              + ", ".join(failed))


if __name__ == "__main__":
    main()
