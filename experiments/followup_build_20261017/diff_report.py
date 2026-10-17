#!/usr/bin/env python
"""Diff of the follow-up manifest v4 against v3 (SI-93..SI-98) and its GPU cost (scripts/cost_model.py).

Reads only manifests (design rows, no scores). Fails (exit 1) unless every row that is not X3/X6/X7/X8/X12 is identical
in v3 and v4, tag included: X1, X13, A1-A3 and X15 must not move. Writes results/manifest_diff_v3_v4.csv: per
experiment x decoder (cond), rows in v3, rows in v4, rows added, rows removed (by tag), lane-hours v3 and v4.
Run from the repo root:  python experiments/followup_build_20261017/diff_report.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cost_model  # noqa: E402

V3 = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v3.tsv")
V4 = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v4.tsv")
OUT = os.path.join(ROOT, "experiments", "followup_build_20261017", "results", "manifest_diff_v3_v4.csv")
CHANGED = ["X3", "X6", "X7", "X8", "X12"]


def read(p):
    return pd.read_csv(p, sep="\t", comment="#", dtype=str, keep_default_na=False)


def main():
    m3, m4 = read(V3), read(V4)
    keep3, keep4 = m3[~m3.experiment.isin(CHANGED)], m4[~m4.experiment.isin(CHANGED)]
    if len(keep3) != len(keep4) or not keep3.reset_index(drop=True).equals(keep4.reset_index(drop=True)):
        raise SystemExit(f"rows outside {CHANGED} differ between v3 and v4 ({len(keep3)} vs {len(keep4)} rows)")
    if m4.tag.duplicated().any():
        raise SystemExit("duplicate tags in v4")
    T = pd.read_csv(os.path.join(ROOT, "docs", "throughput_rtx3080_stock_backbone.csv"))
    speed = float(pd.read_csv(os.path.join(ROOT, "docs", "throughput_rtx3080_stock_backbone_concurrency.csv")).speedup.min())
    c3, c4 = cost_model.cost(V3, T, speed), cost_model.cost(V4, T, speed)
    t3, t4 = set(m3.tag), set(m4.tag)
    rows = []
    for exp in sorted(set(m3.experiment) | set(m4.experiment), key=lambda e: (e not in CHANGED + ["X15"], e)):
        if exp not in CHANGED + ["X15"]:
            continue
        for cond in ["1", "0"]:
            a, b = m3[(m3.experiment == exp) & (m3.cond == cond)], m4[(m4.experiment == exp) & (m4.cond == cond)]
            rows.append(dict(experiment=exp, cond=int(cond), rows_v3=len(a), rows_v4=len(b),
                             added=len(set(b.tag) - t3), removed=len(set(a.tag) - t4),
                             lane_hours_v3=round(float(c3.lane_hours[(c3.experiment == exp) & (c3.cond.astype(str) == cond)].sum()), 1),
                             lane_hours_v4=round(float(c4.lane_hours[(c4.experiment == exp) & (c4.cond.astype(str) == cond)].sum()), 1)))
    d = pd.DataFrame(rows)
    d.loc[len(d)] = dict(experiment="TOTAL follow-ups", cond=-1, **{c: d[c].sum() for c in d.columns[2:]})
    d.to_csv(OUT, index=False)
    print(d.to_string(index=False))
    print(f"rows outside {CHANGED + ['X15']} identical: n={len(keep3)}; v3 total {len(m3)}, v4 total {len(m4)}")


if __name__ == "__main__":
    main()
