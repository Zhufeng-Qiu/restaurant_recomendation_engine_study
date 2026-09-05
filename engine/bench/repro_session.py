"""Session-level replication of the compression effect (step 7).

Balanced crossover, not division inside a big round: 15 blocks f64->packed and
15 blocks packed->f64, order shuffled. Estimator is log(T_packed/T_f64) per
block, so the pairing cancels within-block drift; the CI is over the 30 blocks
and describes THIS session. Three sessions are three replication units -- the
90 blocks are not 90 independent samples.
"""
import json, math, os, platform, random, socket, statistics as st, subprocess, sys
NCCL="./engine/build/pearson_engine_nccl"; FX="data/fixtures/item_full"
DIFFS=[]                                   # per-trial validation residuals
def once(payload, mode="sync"):
    c=[NCCL,FX,"--gpus","2","--mode",mode,"--payload",payload,"--validate"]
    p=subprocess.run(c,capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stderr[-400:])
    d=json.loads(p.stdout.strip().splitlines()[-1])
    assert d["tol_failures"]==0
    DIFFS.append({"payload":payload,"mode":mode,"max_abs_diff":d["max_abs_diff"]})
    return d["device_total_s"] if mode=="sync" else d["t_pipeline_s"]
seed=int(sys.argv[1]); tag=sys.argv[2]
rng=random.Random(seed)
for pl in ("f64","packed"):
    for _ in range(3): once(pl)
blocks=[("f64","packed")]*15+[("packed","f64")]*15
rng.shuffle(blocks)
rows=[]
for a,b in blocks:
    ta=once(a); tb=once(b); d={a:ta,b:tb}
    rows.append({"first":a,"f64":d["f64"],"packed":d["packed"]})
lr=[math.log(r["packed"]/r["f64"]) for r in rows]
m=st.mean(lr); ci=1.96*st.stdev(lr)/math.sqrt(len(lr))
pt,lo,hi=(math.exp(m)-1)*100,(math.exp(m-ci)-1)*100,(math.exp(m+ci)-1)*100
# representative async config, for the noise claim
an=[once("packed","async") for _ in range(12)]
q=st.quantiles(an,n=4)
def sh(c):
    p=subprocess.run(c,shell=True,capture_output=True,text=True)
    return p.stdout.strip() if p.returncode==0 else None
def smi(q):
    v=sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader")
    return v.splitlines()[0].strip() if v else None
out={"session":tag,"seed":seed,"blocks":len(rows),
     "utc":__import__("datetime").datetime.utcnow().isoformat(timespec="seconds")+"Z",
     "host":socket.gethostname(),"platform":platform.platform(),
     "git_sha":sh("git rev-parse HEAD"),"git_dirty":bool(sh("git status --porcelain")),
     "image_digest":os.environ.get("BENCH_IMAGE_DIGEST"),
     "gpus":sh("nvidia-smi --query-gpu=name,uuid --format=csv,noheader"),
     "driver":smi("driver_version"),
     "cuda_runtime":sh("nvcc --version | tail -1"),
     "nccl_version":sh("python3 -c \"import ctypes;l=ctypes.CDLL('libnccl.so.2');"
                       "v=ctypes.c_int();l.ncclGetVersion(ctypes.byref(v));print(v.value)\""),
     "p2p_matrix":sh("nvidia-smi topo -p2p r | head -6"),
     "topo_matrix":sh("nvidia-smi topo -m | head -6"),
     # Exact command that produced this file, from the repository root.
     "cmd":f"python3 engine/bench/repro_session.py {seed} {tag}",
     # The order the 30 crossover blocks actually ran in, after the shuffle:
     # "f64" means f64 ran first in that block. The estimator is order-blind,
     # but without this the balance cannot be checked after the fact.
     "block_order":[r["first"] for r in rows],
     "validated_every_trial":True,
     "validation_max_abs_diff":max(d["max_abs_diff"] for d in DIFFS),
     "validation_per_trial":DIFFS,
     "estimator":"mean of log(T_packed/T_f64) over blocks, exponentiated",
     "ci_method":"1.96 x SE, normal approximation over the 30 block log-ratios",
     "ci_level":0.95,
     "compression_pct":pt,"ci_lo":lo,"ci_hi":hi,
     "median_f64_ms":st.median(r["f64"] for r in rows)*1e3,
     "median_packed_ms":st.median(r["packed"] for r in rows)*1e3,
     "async_packed_median_ms":st.median(an)*1e3,
     "async_packed_iqr_pct":(q[2]-q[0])/st.median(an)*100,
     "async_packed_s":an,                   # all 12 raw trials, not just summary
     "rows":rows}
os.makedirs("results/bench",exist_ok=True)
json.dump(out,open(f"results/bench/repro_{tag}.json","w"),indent=2)
print(f"  session={tag} seed={seed}")
print(f"  f64={out['median_f64_ms']:.4f} ms  packed={out['median_packed_ms']:.4f} ms")
print(f"  compression = {pt:+.2f}%   95% CI [{lo:+.2f}%, {hi:+.2f}%]")
print(f"  async packed: median={out['async_packed_median_ms']:.3f} ms  IQR={out['async_packed_iqr_pct']:.1f}%")
