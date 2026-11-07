"""Res-3 numbers (Pass 23): how close the penalties' trade-off curves are at the matched level b* and how far apart at their
strongest settings, and the curve facts the paragraph names. Reads paper/figure_data/curves.csv (seed means per strength) and
paper/figure_data/ranking_pairs.csv (biology at b*), both copies of the sweep-analysis tables (SHA-256 in
paper/figure_data/MANIFEST.sha256). Writes paper/figure_data/res3_curve_numbers.json.
Run from the repository root:  python paper/tools/res3_curve_numbers.py"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/figure_data"
REAL = ["pancreas", "lung", "immune", "atac_small", "atac_large", "immune_hum_mou"]


def numbers(cv, rp):
    cv, rp = cv[cv.task.isin(REAL)], rp[rp.task.isin(REAL) & rp["rank"].notna()]
    assert cv.n_fail.sum() == 0
    fitted = cv[cv.lam > 0]
    strongest = fitted.sort_values("lam").groupby(["task", "cond", "arm"]).tail(1)
    spread = lambda s: s.max() - s.min()
    at_b = rp.groupby(["task", "cond"]).bio_at_bstar.agg(spread)
    at_s = strongest.groupby(["task", "cond"]).C.agg(spread).loc[at_b.index]  # same ranked pairs
    lowest = fitted.groupby(["task", "cond", "arm"]).C.min().reset_index()
    low_arm = lowest.loc[lowest.groupby(["task", "cond"]).C.idxmin()]
    bary_min = lowest[lowest.arm == "barycenter"].C
    lung = strongest[(strongest.task == "lung") & (strongest.cond == 1) & (strongest.arm == "barycenter")].iloc[0]
    plain = cv[(cv.task == "lung") & (cv.cond == 1) & (cv.lam == 0)].iloc[0]
    bmax = fitted[(fitted.task == "atac_large") & (fitted.cond == 1)].groupby("arm").B.max()
    panc = fitted[(fitted.task == "pancreas") & (fitted.cond == 0)]
    both_up = {a: int(((e.sort_values("lam").B.diff() > 0) & (e.sort_values("lam").C.diff() > 0)).sum())
               for a, e in cv[(cv.task == "pancreas") & (cv.cond == 0)].groupby("arm")}
    return {
        "n_ranked_pairs": int(len(at_b)),
        "median_spread_at_bstar": round(float(at_b.median()), 3),
        "median_spread_at_strongest": round(float(at_s.median()), 3),
        "n_pairs_wider_at_strongest": int((at_s > at_b).sum()),
        "n_real_pairs": int(len(low_arm)),
        "n_pairs_barycenter_lowest": int((low_arm.arm == "barycenter").sum()),
        "barycenter_lowest_min": round(float(bary_min.min()), 3), "barycenter_lowest_max": round(float(bary_min.max()), 3),
        "lung_bary_end_batch": round(float(lung.B), 3), "lung_bary_end_bio": round(float(lung.C), 3),
        "lung_plain_batch": round(float(plain.B), 3), "lung_plain_bio": round(float(plain.C), 3),
        "atac_large_bary_max_batch": round(float(bmax["barycenter"]), 3),
        "atac_large_others_max_batch_min": round(float(bmax.drop("barycenter").min()), 3),
        "atac_large_others_max_batch_max": round(float(bmax.drop("barycenter").max()), 3),
        "pancreas_uncond_penalties_both_up": sum(v > 0 for v in both_up.values()),
        "n_penalties": int(panc.arm.nunique()),
    }


if __name__ == "__main__":
    res = numbers(pd.read_csv(DATA / "curves.csv"), pd.read_csv(DATA / "ranking_pairs.csv"))
    (DATA / "res3_curve_numbers.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
