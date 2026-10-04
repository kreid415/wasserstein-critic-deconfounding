#!/usr/bin/env python
"""SI-41 report: X3 tag map and GPU cost, tagged manifest vs manifest v2 (code check CR-04, 2026-10-03).

Writes
  docs/x3_si41_tag_map.csv   one row per X3 row: old tag, new tag, task, arm, lam, seed, dose, iw, n_cells old / new
                             (rows matched by position; every field but tag / extra is equal, checked here)
  docs/x3_si41_cost.csv      X3 GPU lane-hours and scoring CPU-hours per task, old vs new (scripts/cost_model.py with
                             the stock-backbone throughput and scoring tables, as scripts/wall_time_report.py uses)
Usage: python scripts/x3_si41_report.py [--old manifests/paper_manifest_stock_pilot_u5_b10.tsv]
                                        [--new manifests/paper_manifest_stock_pilot_u5_b10_v2.tsv] [--out-dir docs]
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cost_model as CM  # noqa: E402


def read(p):
    return pd.read_csv(p, sep="\t", comment="#", dtype=str, keep_default_na=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10.tsv"))
    ap.add_argument("--new", default=os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v2.tsv"))
    ap.add_argument("--throughput", default=os.path.join(ROOT, "docs", "throughput_rtx3080_stock_backbone.csv"))
    ap.add_argument("--scoring", default=os.path.join(ROOT, "docs", "scoring_time.csv"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "docs"))
    a = ap.parse_args()
    O, N = read(a.old), read(a.new)
    if list(O.columns) != list(N.columns) or len(O) != len(N):
        raise ValueError("the two manifests differ in columns or row count")
    x3 = (O.experiment == "X3").to_numpy()
    if not (x3 == (N.experiment == "X3").to_numpy()).all():
        raise ValueError("X3 rows are not at the same positions")
    same = [c for c in O.columns if c not in ("tag", "extra")]
    diff = [c for c in same if not (O.loc[x3, c].to_numpy() == N.loc[x3, c].to_numpy()).all()]
    if diff:
        raise ValueError(f"X3 rows differ in {diff}")
    if not (O[~x3].to_numpy() == N[~x3].to_numpy()).all():
        raise ValueError("a non-X3 row differs")
    rows = []
    for (_, o), (_, n) in zip(O[x3].iterrows(), N[x3].iterrows()):
        eo, en = json.loads(o.extra), json.loads(n.extra)
        rows.append(dict(old_tag=o.tag, new_tag=n.tag, task=n.task, arm=n.arm, lam=n.lam, seed=n.seed,
                         dose=en["subsample"]["deplete_pct"], iw=en.get("iw", ""),
                         n_cells_old=eo["subsample"]["n_cells"], n_cells_new=en["subsample"]["n_cells"],
                         draw_new=en["subsample"]["draw"]))
    T = pd.DataFrame(rows)
    if not (T.old_tag != T.new_tag).all() or not T.new_tag.is_unique:
        raise ValueError("an X3 tag is unchanged or repeated")
    os.makedirs(a.out_dir, exist_ok=True)
    T.to_csv(os.path.join(a.out_dir, "x3_si41_tag_map.csv"), index=False)
    thr = pd.read_csv(a.throughput)
    conc = pd.read_csv(a.throughput.replace(".csv", "_concurrency.csv"))
    S = pd.read_csv(a.scoring)
    rate = float(S.seconds.sum() / S.n_cells.sum())
    out = []
    for label, path in (("old", a.old), ("new", a.new)):
        M = CM.cost(path, thr, float(conc.speedup.min()), rate)
        g = M[M.experiment == "X3"].groupby("task").agg(fits=("tag", "size"), gpu_lane_hours=("lane_hours", "sum"),
                                                         score_cpu_hours=("score_cpu_hours", "sum"))
        g.loc["X3 total"] = g.sum()
        out.append(g.add_suffix(f"_{label}"))
    C = pd.concat(out, axis=1)
    C["gpu_lane_hours_change"] = C.gpu_lane_hours_new - C.gpu_lane_hours_old
    C.round(2).to_csv(os.path.join(a.out_dir, "x3_si41_cost.csv"))
    print(C.round(1).to_string())
    print(f"[x3] {len(T)} X3 rows re-tagged -> {a.out_dir}/x3_si41_tag_map.csv, x3_si41_cost.csv")


if __name__ == "__main__":
    main()
