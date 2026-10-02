#!/usr/bin/env python
"""Regenerate docs/wall_time_designs.csv and docs/WALL_TIME.md from the manifest builder and the
measured throughput (docs/throughput_rtx3080*.csv, docs/scoring_time.csv).

Designs: lambda design {pilot: A1 sets a 6-point grid per family, shared: one 10-point grid, no A1}
x unconditioned seeds {5, 3} x barycenter target solver {cold: 10 fixed-point iterations,
warm: 3 iterations from the previous step's support}. Backbone = scIB's scVI configuration.
Usage (repo root, scvi env): python scripts/wall_time_report.py
"""
import itertools
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM  # noqa: E402

OUT = "docs"
TMP = os.environ.get("TMPDIR", "/tmp")


def main():
    T = pd.read_csv(f"{OUT}/throughput_rtx3080.csv")
    C = pd.read_csv(f"{OUT}/throughput_rtx3080_concurrency.csv")
    S = pd.read_csv(f"{OUT}/scoring_time.csv")
    speed = float(C.speedup.min())
    rate = float(S.seconds.sum() / S.n_cells.sum())
    rows, per_exp = [], []
    for design, unc, bary in itertools.product(["pilot", "shared"], [5, 3], ["warm", "cold"]):
        m = os.path.join(TMP, f"manifest_{design}_{unc}_{bary}.tsv")
        cmd = [sys.executable, "scripts/build_paper_manifest.py", "--backbone", "scib", "--design", design,
               "--uncond-seeds", str(unc), "--out", m] + (["--bary-warm-iter", "3"] if bary == "warm" else [])
        subprocess.run(cmd, check=True, capture_output=True)
        M = CM.cost(m, T, speed, rate)
        g = M.groupby("experiment").agg(fits=("tag", "size"), gpu_lane_hours=("lane_hours", "sum"),
                                        score_cpu_hours=("score_cpu_hours", "sum")).reset_index()
        g["local_wall_days_8lanes"] = g.gpu_lane_hours / speed / 24
        g.insert(0, "barycenter", bary); g.insert(0, "uncond_seeds", unc); g.insert(0, "design", design)
        per_exp.append(g)
        crit = M.arm.str.replace("_sn", "", regex=False).isin(["reference", "reference_fixed", "pooled", "barycenter"])
        rows.append(dict(design=design, uncond_seeds=unc, barycenter=bary, fits=len(M),
                         gpu_lane_hours=round(M.lane_hours.sum()), critic_share=round(M.lane_hours[crit].sum() / M.lane_hours.sum(), 2),
                         score_cpu_hours=round(M.score_cpu_hours.sum()),
                         local_wall_days_8lanes=round(M.lane_hours.sum() / speed / 24, 1)))
    D = pd.DataFrame(rows)
    D.to_csv(f"{OUT}/wall_time_designs.csv", index=False)
    pd.concat(per_exp).round(2).to_csv(f"{OUT}/wall_time_by_experiment.csv", index=False)
    t = T[T.batch_size == 128].groupby("arm").ms_per_step.agg(["min", "max"]).round(1)
    md = ["# Wall time for the Tier 1+2 rerun (regenerate: python scripts/wall_time_report.py)", "",
          f"Measured on the local RTX 3080 (scvi-tools 1.4.2, scIB scVI backbone, batch 128): ms per training step per arm "
          f"(docs/throughput_rtx3080.csv); 8 concurrent lanes give {speed:.2f}x aggregate (minimum of the measured "
          f"{', '.join(f'{r.arm} {r.speedup:.2f}x' for r in C.itertuples())}); scIB-native scoring "
          f"{1e3 * rate:.1f} ms per cell per config, single thread (docs/scoring_time.csv).", "",
          "| lambda design | uncond seeds | barycenter | fits | GPU lane-h | critic share | scoring CPU-h | local wall (8 lanes) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in D.itertuples():
        md.append(f"| {r.design} | {r.uncond_seeds} | {r.barycenter} | {r.fits:,} | {r.gpu_lane_hours:,} | {r.critic_share:.0%} | "
                  f"{r.score_cpu_hours:,} | {r.local_wall_days_8lanes} d |")
    md += ["", "ms/step (min-max over tasks): " + "; ".join(f"{a} {r['min']}-{r['max']}" for a, r in t.iterrows()), "",
           "Assumptions: fit cost = steps x measured ms/step (+5 s setup); steps = epochs x ceil(n x train_size / batch); "
           "the 8-lane factor measured on immune is applied to every arm and task; scoring runs on CPU cores not used by "
           "the fit lanes (its effect on fit throughput is not measured)."]
    open(f"{OUT}/WALL_TIME.md", "w").write("\n".join(md) + "\n")
    print(D.to_string(index=False))


if __name__ == "__main__":
    main()
