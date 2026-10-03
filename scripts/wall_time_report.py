#!/usr/bin/env python
"""Regenerate docs/wall_time_designs.csv and docs/WALL_TIME.md from the manifest builder and the
measured throughput (docs/throughput_rtx3080*.csv, docs/scoring_time.csv).

Designs: lambda design {pilot: A1 sets a 6-point grid per family, shared: one 10-point grid, no A1}
x unconditioned seeds {5, 3} x barycenter solver {10, 5, 3 cold fixed-point iterations per step}
(warm starts are excluded: they converge to a different fixed point, docs/barycenter_solver_check.csv).
Backbone (CONSTRAINTS.md SI-10): scvi-tools defaults ('stock'); --backbone scib for scIB's configuration.
Usage (repo root, scvi env): python scripts/wall_time_report.py [--backbone stock|scib]
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM  # noqa: E402

OUT = "docs"
TMP = os.environ.get("TMPDIR", "/tmp")


THROUGHPUT = {"stock": "throughput_rtx3080_stock_backbone", "scib": "throughput_rtx3080"}
# ms/step rows without a passing timing check: run1 of experiments/bench_missing_arms_step_cost (its NEW arms) failed
# PF-16 under CPU contention (lab notebook NB-20261002-07, -08). Empty the list once a re-measurement passes PF-16.
PROVISIONAL = {"stock": ["discriminator_r1", "discriminator_ref", "reference_stratified", "pooled_stratified",
                         "discriminator_iw", "reference_iw", "pooled_iw", "mmd_iw"], "scib": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=sorted(THROUGHPUT), default="stock")
    bk = ap.parse_args().backbone
    T = pd.read_csv(f"{OUT}/{THROUGHPUT[bk]}.csv")
    C = pd.read_csv(f"{OUT}/{THROUGHPUT[bk]}_concurrency.csv")
    S = pd.read_csv(f"{OUT}/scoring_time.csv")
    speed_min = float(C.speedup.min())
    cal_path = f"{OUT}/concurrency_calibration_{bk}.json"
    cal = json.load(open(cal_path)) if os.path.exists(cal_path) else None
    # primary: 8-lane factor measured on a mixed-arm queue (scripts/calibrate_concurrency.py); fallback: the
    # minimum factor over identical-arm 8-lane runs (conservative)
    speed = float(cal["effective_factor_window"]) if cal else speed_min
    rate = float(S.seconds.sum() / S.n_cells.sum())
    rows, per_exp = [], []
    for design, unc, bary in itertools.product(["pilot", "shared"], [5, 3], [10, 5, 3]):
        m = os.path.join(TMP, f"manifest_{bk}_{design}_{unc}_{bary}.tsv")
        cmd = [sys.executable, "scripts/build_paper_manifest.py", "--backbone", bk, "--design", design,
               "--uncond-seeds", str(unc), "--out", m] + (["--bary-iter", str(bary)] if bary != 10 else [])
        subprocess.run(cmd, check=True, capture_output=True)
        M = CM.cost(m, T, speed, rate)
        g = M.groupby("experiment").agg(fits=("tag", "size"), gpu_lane_hours=("lane_hours", "sum"),
                                        score_cpu_hours=("score_cpu_hours", "sum")).reset_index()
        g["local_wall_days_8lanes"] = g.gpu_lane_hours / speed / 24
        g.insert(0, "barycenter_iter", bary); g.insert(0, "uncond_seeds", unc); g.insert(0, "design", design)
        per_exp.append(g)
        crit = M.arm.str.replace("_sn", "", regex=False).isin(["reference", "reference_fixed", "pooled", "barycenter"])
        label = pd.Series([CM.arm_label(a, json.loads(e)) for a, e in zip(M.arm, M.extra)], index=M.index)
        prov = label.isin(PROVISIONAL[bk])      # costed label: X8 rows are <arm>_stratified, X3 rows <arm>_iw
        rows.append(dict(design=design, uncond_seeds=unc, barycenter_iter=bary, fits=len(M),
                         gpu_lane_hours=round(M.lane_hours.sum()), critic_share=round(M.lane_hours[crit].sum() / M.lane_hours.sum(), 2),
                         score_cpu_hours=round(M.score_cpu_hours.sum()),
                         local_wall_days_8lanes=round(M.lane_hours.sum() / speed / 24, 1),
                         local_wall_days_8lanes_conservative=round(M.lane_hours.sum() / speed_min / 24, 1),
                         provisional_fits=int(prov.sum()), provisional_lane_hours=round(M.lane_hours[prov].sum())))
    D = pd.DataFrame(rows)
    sfx = "" if bk == "stock" else f"_{bk}"
    D.to_csv(f"{OUT}/wall_time_designs{sfx}.csv", index=False)
    pd.concat(per_exp).round(2).to_csv(f"{OUT}/wall_time_by_experiment{sfx}.csv", index=False)
    t = T[T.batch_size == 128].groupby("arm").ms_per_step.agg(["min", "max"]).round(1)
    bdesc = {"stock": "scvi-tools default backbone (n_latent 10, 1 layer, ZINB, 90% train)",
             "scib": "scIB scVI backbone (n_latent 30, 2 layers, NB, all cells)"}[bk]
    md = [f"# Wall time for the Tier 1+2 rerun, {bk} backbone (regenerate: python scripts/wall_time_report.py --backbone {bk})", "",
          f"Measured on the local RTX 3080 (scvi-tools 1.4.2, {bdesc}, batch 128): ms per training step per arm "
          f"(docs/{THROUGHPUT[bk]}.csv); 8 concurrent lanes: "
          + (f"{speed:.2f}x aggregate on a mixed-arm queue of {cal['fits']} fits (docs/concurrency_calibration_{bk}.json); "
             f"conservative column uses the minimum identical-arm factor {speed_min:.2f}x " if cal else
             f"{speed:.2f}x aggregate (minimum identical-arm factor) ")
          + f"({', '.join(f'{r.arm} {r.speedup:.2f}x' for r in C.itertuples())}); scIB-native scoring "
          f"{1e3 * rate:.1f} ms per cell per config, single thread (docs/scoring_time.csv).", "",
          "| lambda design | uncond seeds | barycenter iterations | fits | GPU lane-h | critic share | scoring CPU-h | local wall (8 lanes) | conservative |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in D.itertuples():
        md.append(f"| {r.design} | {r.uncond_seeds} | {r.barycenter_iter} | {r.fits:,} | {r.gpu_lane_hours:,} | {r.critic_share:.0%} | "
                  f"{r.score_cpu_hours:,} | {r.local_wall_days_8lanes} d | {r.local_wall_days_8lanes_conservative} d |")
    md += ["", "ms/step (min-max over tasks): " + "; ".join(f"{a} {r['min']}-{r['max']}" for a, r in t.iterrows()), "",
           "Assumptions: fit cost = steps x measured ms/step (+5 s setup); steps = epochs x ceil(n x train_size / batch); "
           "the 8-lane factor measured on immune is applied to every arm and task; scoring runs on CPU cores not used by "
           "the fit lanes (its effect on fit throughput is not measured)."]
    if PROVISIONAL[bk]:
        missing = sorted(set(PROVISIONAL[bk]) - set(T.arm))
        if missing:
            raise ValueError(f"provisional arms {missing} have no row in docs/{THROUGHPUT[bk]}.csv")
        h = D.iloc[0]
        md += ["", f"Provisional: the ms/step of {len(PROVISIONAL[bk])} arms ({', '.join(PROVISIONAL[bk])}) are run1 medians of "
               "experiments/bench_missing_arms_step_cost, whose timing check PF-16 failed under CPU contention "
               "(lab notebook NB-20261002-07, -08). In the "
               f"{h.design}/{h.uncond_seeds}/{h.barycenter_iter} design they carry {h.provisional_fits:,} fits and "
               f"{h.provisional_lane_hours:,} of {h.gpu_lane_hours:,} GPU lane-h ({h.provisional_lane_hours / h.gpu_lane_hours:.0%}); "
               "columns provisional_fits and provisional_lane_hours of docs/wall_time_designs.csv give every design."]
    open(f"{OUT}/WALL_TIME{sfx}.md", "w").write("\n".join(md) + "\n")
    print(D.to_string(index=False))


if __name__ == "__main__":
    main()
