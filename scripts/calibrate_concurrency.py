#!/usr/bin/env python
"""Calibrate the cost model's 8-lane factor against a mixed-arm queue run (2026-10-02).

Input: a queue manifest + queue_times.tsv (tag, start, end, rc per process, 8 parallel workers via xargs -P 8).
Steady-state window = [first start, last start] (all 8 lanes busy until the queue empties). Work in the window =
sum over fits of model single-lane seconds x the fraction of the fit's process time inside the window.
Effective factor = work / window length; compare with the minimum per-arm factor the cost model uses.
Usage: python scripts/calibrate_concurrency.py --manifest M --times T --throughput docs/throughput_rtx3080_stock_backbone.csv
"""
import argparse
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--times", required=True)
    ap.add_argument("--throughput", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    T = pd.read_csv(a.throughput)
    C = pd.read_csv(a.throughput.replace(".csv", "_concurrency.csv"))
    M = pd.read_csv(a.manifest, sep="\t", comment="#").set_index("tag")
    Q = pd.read_csv(a.times, sep="\t", names=["tag", "start", "end", "rc"])
    assert (Q.rc == 0).all(), Q[Q.rc != 0]
    t0, tq, t1 = Q.start.min(), Q.start.max(), Q.end.max()
    work = 0.0; work_all = 0.0
    for q in Q.itertuples():
        r = M.loc[q.tag]
        ex = json.loads(r.extra)
        n = CM.TASK_N[r.task]
        steps = int(r.max_epochs) * math.ceil(n * float(r.train_size) / int(r.batch_size))
        lane_s = steps * CM.step_ms(T, CM.arm_label(r.arm, ex), r.task, int(r.n_critic), int(r.batch_size)) / 1e3 + 5.0
        inside = max(0.0, min(q.end, tq) - max(q.start, t0)) / (q.end - q.start)
        work += lane_s * inside; work_all += lane_s
    res = dict(fits=len(Q), makespan_s=round(t1 - t0, 1), window_s=round(tq - t0, 1),
               model_lane_s_total=round(work_all, 1), effective_factor_window=round(work / (tq - t0), 2),
               effective_factor_makespan=round(work_all / (t1 - t0), 2), model_factor_min=float(C.speedup.min()),
               per_arm_factors=dict(zip(C.arm, C.speedup)))
    res["wall_scale_vs_model"] = round(res["model_factor_min"] / res["effective_factor_window"], 3)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
