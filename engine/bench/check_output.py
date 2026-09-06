"""Validate what a backend binary printed, strictly, and testably.

The ladder used to check only `tail -1`, so a cuda_detail line that printf had
turned into `"plan_metrics_computed":(null)` shipped undetected and dropped a
configuration out of the headline matrix an hour later. The replacement check
was written inline in shell and was still too weak: it accepted any output with
at least one JSON line and silently ignored every non-JSON line, so a binary
that failed to execute at all could pass.

The protocol is exact, because "at least one" is what let both defects through:

  cuda  -- exactly two JSON lines: a cuda_detail line, then a summary line
  nccl  -- exactly one JSON line
  any non-empty line that is not JSON is a failure

and the fields the harness later depends on must be present and of the right
kind, so a missing one fails here rather than after thirty rounds.

  check_output.py cuda|nccl < output
  check_output.py --selftest
"""

import json
import sys

# Fields the benchmark harness reads. A backend that stops emitting one of
# these does not fail until the summary, which is far too late.
COMMON = {"n_pairs": (int,), "tol_failures": (int,), "max_abs_diff": (int, float)}
CUDA_DETAIL = {"device_total_s": (int, float), "t_stats_s": (int, float),
               "t_plan_s": (int, float), "t_plan_metrics_s": (int, float),
               "group": (int,), "pair_order": (str,), "hoist": (bool,),
               "plan_metrics_computed": (bool,), "occupancy": (dict,)}
NCCL = dict(COMMON, **{"device_total_s": (int, float), "t_stats_s": (int, float),
                       "t_allreduce_s": (int, float), "t_plan_s": (int, float),
                       "group": (int,), "pair_order": (str,), "hoist": (bool,),
                       "plan_metrics_computed": (bool,),
                       "stage_windows_disjoint": (bool,),
                       "cold_data_path_s": (int, float), "occupancy": (dict,)})


def problems(kind, text):
    """Return a list of complaints; empty means the output is well formed."""
    out = []
    objs = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("{"):
            out.append(f"line {i} is not JSON: {line[:80]!r}")
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError as exc:
            out.append(f"line {i} is malformed JSON ({exc}): {line[:120]!r}")

    want = 2 if kind == "cuda" else 1
    if len(objs) != want:
        out.append(f"expected exactly {want} JSON line(s) for {kind}, got {len(objs)}")
        return out

    def check(obj, spec, where):
        for field, types in spec.items():
            if field not in obj:
                out.append(f"{where}: missing {field!r}")
            elif not isinstance(obj[field], types):
                out.append(f"{where}: {field!r} is {type(obj[field]).__name__}, "
                           f"want {'/'.join(t.__name__ for t in types)}")

    if kind == "cuda":
        if "cuda_detail" not in objs[0]:
            out.append("first cuda line has no cuda_detail object")
        else:
            check(objs[0]["cuda_detail"], CUDA_DETAIL, "cuda_detail")
        check(objs[1], COMMON, "cuda summary")
    else:
        check(objs[0], NCCL, "nccl summary")
    return out


def selftest():
    fails = 0

    def expect(ok, what):
        nonlocal fails
        if not ok:
            fails += 1
            print(f"  FAIL {what}")

    detail = {"cuda_detail": {"device_total_s": 1.0, "t_stats_s": 1.0,
                             "t_plan_s": 0.1, "t_plan_metrics_s": 0.0,
                             "group": 4, "pair_order": "source", "hoist": False,
                             "plan_metrics_computed": False,
                             "occupancy": {"registers_per_thread": 37}}}
    summary = {"n_pairs": 10, "tol_failures": 0, "max_abs_diff": 0.0}
    good = json.dumps(detail) + "\n" + json.dumps(summary)
    expect(problems("cuda", good) == [], "a well-formed cuda pair passes")

    # the exact defect that shipped: (null) makes the line unparseable
    expect(problems("cuda", good.replace("false", "(null)", 1)),
           "a (null) field is rejected")
    # a binary that could not execute prints nothing -- this used to pass
    expect(problems("cuda", ""), "empty output is rejected")
    expect(problems("cuda", "bash: ./pearson_engine: cannot execute\n"),
           "a non-JSON error line is rejected")
    # one line where two are required
    expect(problems("cuda", json.dumps(summary)), "a missing cuda_detail line is rejected")
    # a field the harness depends on, gone
    d2 = json.loads(json.dumps(detail))
    del d2["cuda_detail"]["occupancy"]
    expect(problems("cuda", json.dumps(d2) + "\n" + json.dumps(summary)),
           "a missing required field is rejected")
    # wrong type
    d3 = json.loads(json.dumps(detail))
    d3["cuda_detail"]["group"] = "4"
    expect(problems("cuda", json.dumps(d3) + "\n" + json.dumps(summary)),
           "a wrongly typed field is rejected")

    nccl = dict(summary, device_total_s=1.0, t_stats_s=1.0, t_allreduce_s=0.1,
                t_plan_s=0.1, group=4, pair_order="source", hoist=False,
                plan_metrics_computed=False, stage_windows_disjoint=True,
                cold_data_path_s=2.0, occupancy={})
    expect(problems("nccl", json.dumps(nccl)) == [], "a well-formed nccl line passes")
    expect(problems("nccl", json.dumps(nccl) + "\n" + json.dumps(nccl)),
           "two nccl lines are rejected")
    n2 = dict(nccl); del n2["stage_windows_disjoint"]
    expect(problems("nccl", json.dumps(n2)),
           "nccl without stage_windows_disjoint is rejected")

    print(f"check_output selftest: {'PASS' if fails == 0 else 'FAIL'}")
    return fails


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    if len(sys.argv) != 2 or sys.argv[1] not in ("cuda", "nccl"):
        raise SystemExit("usage: check_output.py cuda|nccl < output")
    bad = problems(sys.argv[1], sys.stdin.read())
    for b in bad:
        print(f"  {b}")
    raise SystemExit(1 if bad else 0)
