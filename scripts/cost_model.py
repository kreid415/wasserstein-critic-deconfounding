#!/usr/bin/env python
"""Wall-time model for a paper manifest, from MEASURED throughput (2026-10-02).

Step 1 (--measure BENCH_LATENT_DIR): read fit_seconds from benchmark latents written by
fit_paper_config.py (pairs of 3-epoch and 1-epoch fits of the same config) and write
docs/throughput_rtx3080.csv: ms per optimiser step = (t3 - t1) / (2 * steps_per_epoch).
Step 2 (--manifest M): cost every row = steps x ms/step(arm, K, n_critic, batch size), sum per
experiment, and convert to wall time with the measured 8-lane concurrency factor.

Measured on the local RTX 3080 (scvi-tools 1.4.2, torch 2.13, scIB backbone, batch 128,
OMP_NUM_THREADS=1 per lane). Concurrency: 8 simultaneous fits of the same config, aggregate
speed-up = 8 * t_single / t_concurrent.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

TASK_K = {"pancreas": 9, "lung": 16, "immune": 10, "immune_hum_mou": 23, "sim1": 6, "sim2": 16,
          "atac_small": 3, "atac_large": 11}
TASK_N = {"pancreas": 16382, "lung": 32472, "immune": 33506, "immune_hum_mou": 97861, "sim1": 12097,
          "sim2": 19318, "atac_small": 11270, "atac_large": 84813}


def measure(bench_dir, out_csv):
    rows = {}
    for f in glob.glob(os.path.join(bench_dir, "*.npz")):
        c = json.loads(str(np.load(f)["config"]))
        r = c["row"]
        ex = json.loads(r.get("extra", "{}"))
        arm = r["arm"] + (f"_warm{ex['bary_warm_iter']}" if ex.get("bary_warm_iter") else "")
        rows[r["tag"]] = dict(tag=r["tag"], task=r["task"], arm=arm, n_critic=int(r["n_critic"]),
                              batch_size=int(r["batch_size"]), epochs=int(r["max_epochs"]), n=c["n_cells"],
                              K=c["n_batches"], seconds=c["fit_seconds"], gpu=c["gpu"],
                              train_size=float(r["train_size"]), n_latent=int(r["n_latent"]), likelihood=r["likelihood"])
    out = []
    for tag, r in rows.items():
        if r["epochs"] != 3 or "conc8" in tag or (tag + "_e1") not in rows:
            continue
        t1 = rows[tag + "_e1"]["seconds"]
        spe = math.ceil(r["n"] * r["train_size"] / r["batch_size"])
        out.append(dict(r, seconds_1ep=t1, ms_per_step=round(1e3 * (r["seconds"] - t1) / (2 * spe), 2),
                        setup_s=round(t1 - (r["seconds"] - t1) / 2, 1)))
    conc = []
    for arm in ("pooled", "barycenter"):
        for pre in ("BENCH", "S"):
            tc = [r["seconds"] for t, r in rows.items() if f"{pre}_immune_{arm}_conc8" in t]
            ts = rows.get(f"{pre}_immune_{arm}", {}).get("seconds")
            if tc and ts:
                break
        if tc and ts:
            conc.append(dict(arm=arm, lanes=8, t_single=ts, t_concurrent=float(np.mean(tc)),
                             speedup=round(8 * ts / float(np.mean(tc)), 2)))
    df = pd.DataFrame(out).sort_values(["task", "arm", "batch_size"])
    df.to_csv(out_csv, index=False)
    pd.DataFrame(conc).to_csv(out_csv.replace(".csv", "_concurrency.csv"), index=False)
    return df, pd.DataFrame(conc)


def step_ms(T, arm, task, n_critic, batch_size):
    """ms/step from the measured table; barycenter linear in K; critics scale with n_critic."""
    K = TASK_K[task]
    m = T[(T.batch_size == 128)]

    def get(a, nc=None, tk=None):
        s = m[(m.arm == a) & ((m.n_critic == nc) if nc is not None else True) & ((m.task == tk) if tk else True)]
        return float(s.ms_per_step.mean()) if len(s) else None

    if arm.startswith("barycenter"):
        pts = m[m.arm == arm][["K", "ms_per_step"]].drop_duplicates()
        slope, icpt = np.polyfit(pts.K, pts.ms_per_step, 1) if len(pts) > 1 else (0.0, float(pts.ms_per_step.iloc[0]))
        ms = icpt + slope * K
    elif arm in ("scanvi",):
        ms = get("none") * 1.1
    else:
        base = arm
        ms = get(base, n_critic) or get(base)
        if ms is None and base.startswith("reference"):
            ms = get("reference")
        if ms is None:
            raise KeyError(f"no measurement for arm {arm} n_critic {n_critic}")
        if base in ("pooled", "reference", "reference_fixed", "pooled_sn") and get(base, n_critic) is None:
            # per-critic-step cost from the 5-step and 1-step pooled measurements
            c5, c1 = get("pooled", 5), get("pooled", 1)
            ms = c1 + (c5 - c1) * (n_critic - 1) / 4.0
    if batch_size != 128:
        b = T[(T.batch_size == batch_size) & (T.arm == arm.replace("_fixed", ""))]
        ref = get(arm.replace("_fixed", ""))
        factor = float(b.ms_per_step.mean()) / ref if len(b) and ref else 1.0
        ms *= factor
    return ms


def cost(manifest, T, speedup, score_s_per_cell=None):
    M = pd.read_csv(manifest, sep="\t", comment="#")
    hrs, shrs = [], []
    for r in M.itertuples():
        n = TASK_N[r.task]
        ex = json.loads(r.extra)
        if "subsample" in ex:
            n = ex["subsample"]["n_cells"]
        shrs.append(n * score_s_per_cell / 3600 if score_s_per_cell else np.nan)
        arm = r.arm + (f"_warm{ex['bary_warm_iter']}" if ex.get("bary_warm_iter") else "")
        steps = int(r.max_epochs) * math.ceil(n * float(r.train_size) / int(r.batch_size))
        ms = step_ms(T, arm, r.task, int(r.n_critic), int(r.batch_size))
        if r.arm == "scanvi":
            steps += min(10, max(2, round(int(r.max_epochs) / 3))) * math.ceil(n / int(r.batch_size))
        hrs.append(steps * ms / 3.6e6 + 5 / 3600)       # + ~5 s setup per fit
    M["lane_hours"] = hrs
    M["score_cpu_hours"] = shrs
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure")
    ap.add_argument("--throughput", default="docs/throughput_rtx3080.csv")
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--scoring", default="docs/scoring_time.csv",
                    help="measured single-thread scIB-native scoring time per config (task, n_cells, seconds)")
    a = ap.parse_args()
    if a.measure:
        T, C = measure(a.measure, a.throughput)
        print(T[["task", "arm", "n_critic", "batch_size", "K", "ms_per_step", "setup_s"]].to_string(index=False))
        print(C.to_string(index=False))
    if a.manifest:
        T = pd.read_csv(a.throughput)
        C = pd.read_csv(a.throughput.replace(".csv", "_concurrency.csv"))
        speedup = float(C.speedup.min())
        S = pd.read_csv(a.scoring) if os.path.exists(a.scoring) else None
        rate = float((S.seconds.sum() / S.n_cells.sum())) if S is not None else None   # LS slope through 0 ~ ratio
        M = cost(a.manifest, T, speedup, rate)
        g = M.groupby("experiment").agg(fits=("tag", "size"), lane_hours=("lane_hours", "sum"),
                                        score_cpu_hours=("score_cpu_hours", "sum"))
        g["local_wall_days_8lanes"] = g.lane_hours / speedup / 24
        g.loc["TOTAL"] = [g.fits.sum(), g.lane_hours.sum(), g.score_cpu_hours.sum(), g.lane_hours.sum() / speedup / 24]
        if rate:
            print(f"scoring: {1e3 * rate:.1f} ms per cell per config, single thread (from {a.scoring})")
        print(f"8-lane speed-up used: {speedup:.2f}x (minimum measured)")
        print(g.round(1).to_string())
        by_arm = M.groupby("arm").lane_hours.sum().sort_values(ascending=False)
        print("lane-hours by arm:", by_arm.round(0).to_dict())
        if a.out:
            g.round(2).to_csv(a.out)


if __name__ == "__main__":
    main()
