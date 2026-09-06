"""Within-host P2P on/off A/B: same binary, same data, same GPUs, one variable.

The Sep 4 correction says the NVLink-vs-PCIe comparison is not a controlled
interconnect A/B, because the two hosts differed in CPU and in GPU speed. This
is the controlled version: NCCL_P2P_DISABLE=1 forces the collective through
host memory on the SAME machine, in the same session, so the transport is the
only thing that changes.

30 trials after 3 warm-ups, median + IQR, every trial validated against golden.
"""
import json
import os
import statistics as st
import subprocess

BIN = "./engine/build/pearson_engine_nccl"
FIX = "data/fixtures/item_full"
WARM, REP = 3, 30


def run(mode, pay, p2p):
    env = dict(os.environ)
    if not p2p:
        env["NCCL_P2P_DISABLE"] = "1"
    cmd = [BIN, FIX, "--gpus", "2", "--mode", mode, "--payload", pay, "--validate"]
    tot, ar, bad = [], [], 0
    for i in range(WARM + REP):
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        d = json.loads(p.stdout.strip().splitlines()[-1])
        if d["tol_failures"]:
            bad += 1
        if i < WARM:
            continue
        # device_total_s = stats + allreduce + finalize. This used to sum only
        # kernel + allreduce, dropping the finalize kernel and understating
        # every sync total by ~0.5-1.5%. The fallback is for records predating
        # the field.
        t = (d["t_pipeline_s"] if mode == "async"
             else d.get("device_total_s",
                        d["t_kernel_s"] + d["t_allreduce_s"]
                        + d.get("t_finalize_s", 0.0)))
        tot.append(t * 1e3)
        ar.append(d["t_allreduce_s"] * 1e3)
    q = st.quantiles(tot, n=4)
    return {
        "median_ms": st.median(tot),
        "p25_ms": q[0],
        "p75_ms": q[2],
        "iqr_ms": q[2] - q[0],
        "allreduce_ms": st.median(ar),
        "n_trials": len(tot),
        "tol_failures": bad,
    }


def main():
    out = {}
    for p2p in (True, False):
        for mode in ("sync", "async"):
            for pay in ("f64", "packed"):
                key = "{}_{}_p2p{}".format(mode, pay, "on" if p2p else "off")
                r = run(mode, pay, p2p)
                out[key] = r
                pct = r["iqr_ms"] / r["median_ms"] * 100
                print("  {:22s} median={:7.3f} ms  IQR={:6.3f} ({:5.2f}%)  "
                      "allreduce={:7.3f} ms  failures={}".format(
                          key, r["median_ms"], r["iqr_ms"], pct,
                          r["allreduce_ms"], r["tol_failures"]), flush=True)
    with open("results/bench/p2p_ab.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/bench/p2p_ab.json")


if __name__ == "__main__":
    main()
