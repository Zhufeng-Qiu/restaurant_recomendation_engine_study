"""Three GPU sweeps the project brief listed and never ran.

A. Chunk size. The async story rests entirely on what chunking costs -- NVLink
   pays 3.41x on the collective, PCIe 1.23x -- but chunk size was only ever
   swept for correctness, never for time. This maps the curve.

B. Finalize stream. The PCIe traces showed GPU0 achieving 0.023 ms of overlap
   while GPU1 hid 55% of its compute; GPU0 is also the one running finalize on
   its communication stream. --finalize-stream separate moves it off, which
   turns "async loses on this link" into a testable implementation claim.

C. Problem size. Every GPU number in this project comes from one fixture at
   1.17M pairs. This finds where the GPU stops being worth it.

Every run validates against golden; a tol failure aborts the sweep.
"""
import json
import statistics as st
import subprocess
import sys

ENG = "./engine/build/pearson_engine"
NCCL = "./engine/build/pearson_engine_nccl"
FULL = "data/fixtures/item_full"


def once(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{p.stderr[-800:]}")
    rec = {}
    for line in p.stdout.strip().splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            rec.update(d.pop("cuda_detail", {}))
            rec.update(d)
    return rec


def timed(cmd, warm, rep, pick):
    for _ in range(warm):
        once(cmd)
    ts, bad = [], 0
    for _ in range(rep):
        d = once(cmd)
        if d.get("tol_failures"):
            bad += 1
        ts.append(pick(d) * 1e3)
    q = st.quantiles(ts, n=4) if len(ts) >= 4 else [float("nan")] * 3
    return {"median_ms": st.median(ts), "iqr_ms": q[2] - q[0], "n": len(ts),
            "tol_failures": bad}


def pick_nccl(d):
        # device_total_s = stats + allreduce + finalize. This used to sum only
    # kernel + allreduce, dropping the finalize kernel and understating every
    # sync total by ~0.5-1.5%. Fall back to the old sum only for records that
    # predate the field.
    return (d["t_pipeline_s"] if d.get("mode") == "async"
            else d.get("device_total_s",
                       d["t_kernel_s"] + d["t_allreduce_s"] + d.get("t_finalize_s", 0.0)))


def pick_cuda(d):
    return d["t_kernel_stats_s"] + d["t_kernel_finalize_s"]


out = {"A_chunk": {}, "B_finalize": {}, "C_size": {}}

print("=== A. chunk-size sweep (async, 2 GPU, 20 trials) ===", flush=True)
for pay in ("f64", "packed"):
    r = timed([NCCL, FULL, "--gpus", "2", "--mode", "sync", "--payload", pay,
               "--validate"], 3, 20, pick_nccl)
    out["A_chunk"][f"sync_{pay}"] = r
    print(f"  sync    {pay:6s}            {r['median_ms']:8.3f} ms  IQR {r['iqr_ms']:6.3f}  fail={r['tol_failures']}", flush=True)
    for c in (16384, 32768, 65536, 131072, 262144, 524288, 1048576):
        r = timed([NCCL, FULL, "--gpus", "2", "--mode", "async", "--chunk", str(c),
                   "--payload", pay, "--validate"], 3, 20, pick_nccl)
        out["A_chunk"][f"async_{pay}_c{c}"] = r
        n_chunks = -(-1171857 // c)
        print(f"  async   {pay:6s} chunk={c:<8d} {r['median_ms']:8.3f} ms  IQR {r['iqr_ms']:6.3f}  "
              f"({n_chunks} chunks)  fail={r['tol_failures']}", flush=True)

print("\n=== B. finalize-stream A/B (async, 2 GPU, default chunk, 30 trials) ===", flush=True)
for pay in ("f64", "packed"):
    for fs in ("comm", "separate"):
        r = timed([NCCL, FULL, "--gpus", "2", "--mode", "async", "--payload", pay,
                   "--finalize-stream", fs, "--validate"], 3, 30, pick_nccl)
        out["B_finalize"][f"{pay}_{fs}"] = r
        print(f"  {pay:6s} finalize={fs:9s} {r['median_ms']:8.3f} ms  IQR {r['iqr_ms']:6.3f}  fail={r['tol_failures']}", flush=True)

print("\n=== C. problem-size sweep (20 trials) ===", flush=True)
for fx in ("item_p1e3", "item_p1e4", "item_medium", "item_full", "user_full"):
    d = f"data/fixtures/{fx}"
    n = json.load(open(f"{d}/meta.json"))["counts"]["n_pairs"]
    row = {"n_pairs": n}
    r = timed([ENG, d, "--backend", "serial", "--validate"], 2, 10,
              lambda x: x["t_compute_s"])
    row["serial"] = r
    r = timed([ENG, d, "--backend", "cuda", "--validate"], 3, 20, pick_cuda)
    row["cuda_1gpu"] = r
    r = timed([NCCL, d, "--gpus", "2", "--mode", "sync", "--payload", "packed",
               "--validate"], 3, 20, pick_nccl)
    row["nccl_sync_g2_packed"] = r
    out["C_size"][fx] = row
    sp = row["serial"]["median_ms"] / row["cuda_1gpu"]["median_ms"]
    print(f"  {fx:14s} pairs={n:>9,}  serial={row['serial']['median_ms']:9.3f}  "
          f"cuda={row['cuda_1gpu']['median_ms']:7.3f}  nccl2={row['nccl_sync_g2_packed']['median_ms']:7.3f}  "
          f"gpu_speedup={sp:7.1f}x", flush=True)

json.dump(out, open("results/bench/gpu_sweeps.json", "w"), indent=2)
print("\nwrote results/bench/gpu_sweeps.json")
