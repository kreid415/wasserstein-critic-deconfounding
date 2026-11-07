"""Collapse rate up to the highest common batch score (Pass 23; user choice: main text and Figure 5a use the common range,
Supplementary Table S8 keeps the rate over all strengths). A fit collapses when its biology score is at least 0.10 below plain
scVI's with the same task, decoder and seed (Supplementary Note 7); no sweep fit failed. A fit is in the common range when its
strength's mean batch score over the three seeds is at most b_common, the highest batch score every penalty reaches (the range
of Figure 4). Reads paper/figure_data/metric_scores.csv (per-fit metrics; biology = mean of the available biology metrics,
batch = mean of the five batch metrics, which reproduces curves.csv), curves.csv, paper_numbers.json (b_common) and
secondary.csv (all-strength counts, which the per-fit rebuild must reproduce) and ranking_pairs.csv (ranks at b*). Writes paper/figure_data/collapse_common_range.csv
and collapse_common_range.json. Run from the repository root:  python paper/tools/collapse_common_range.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/figure_data"
REAL = ["pancreas", "lung", "immune", "atac_small", "atac_large", "immune_hum_mou"]
BIO = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label", "isolated_label_F1", "isolated_label_silhouette", "cLISI",
       "cell_cycle_conservation", "trajectory"]
BATCH = ["ASW_label/batch", "PCR_batch", "graph_conn", "kBET", "iLISI"]
COLLAPSE_BIO, TOL = 0.10, 1e-9


def clopper_pearson(k, n):
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def fits(ms, cv, pn):
    """One row per sweep fit with its collapse call and whether its strength lies in the common batch range."""
    a1 = ms[ms.experiment == "A1"].assign(B=lambda d: d[BATCH].mean(1), C=lambda d: d[BIO].mean(1))
    mean = a1[a1.arm != "none"].groupby(["task", "cond", "arm", "lam"])[["B", "C"]].mean().reset_index()
    chk = mean.merge(cv[cv.lam > 0], on=["task", "cond", "arm", "lam"], suffixes=("", "_cv"), validate="1:1")
    assert len(chk) == (cv.lam > 0).sum() and np.allclose(chk[["B", "C"]], chk[["B_cv", "C_cv"]], atol=1e-12, rtol=0)
    plain = a1[a1.arm == "none"].set_index(["task", "cond", "seed"]).C
    assert plain.index.is_unique
    pen = a1[a1.arm != "none"].merge(mean[["task", "cond", "arm", "lam", "B"]].rename(columns={"B": "B_mean"}),
                                     on=["task", "cond", "arm", "lam"], validate="m:1")
    c0 = plain.loc[list(zip(pen.task, pen.cond, pen.seed))].to_numpy()
    bc = np.array([pn[f"pair.{t}.cond{c}.b_common"]["value"] for t, c in zip(pen.task, pen.cond)])
    return pen.assign(collapsed=pen.C <= c0 - COLLAPSE_BIO + TOL, common=pen.B_mean <= bc + 1e-12)


def table(f, sec):
    g = f.groupby(["task", "cond", "arm"])
    out = pd.DataFrame({"n_all": g.size(), "k_all": g.collapsed.sum(), "n_common": g.common.sum(),
                        "k_common": g.apply(lambda d: int((d.collapsed & d.common).sum()), include_groups=False)}).reset_index()
    chk = out.merge(sec[["task", "cond", "arm", "collapse_n", "collapsed"]], on=["task", "cond", "arm"], validate="1:1")
    assert len(chk) == len(sec) and (chk.n_all == chk.collapse_n).all() and (chk.k_all == chk.collapsed).all()
    assert (out.n_common > 0).all()
    ci = [clopper_pearson(k, n) for k, n in zip(out.k_common, out.n_common)]
    return out.assign(rate_common=out.k_common / out.n_common, lo_common=[a for a, _ in ci], hi_common=[b for _, b in ci])


def numbers(t, rp):
    r = lambda v: round(float(v), 3)
    real = t[t.task.isin(REAL)]
    out = {}
    for (c, a), d in real.groupby(["cond", "arm"]):
        m = d.loc[d.rate_common.idxmax()]
        top = sorted(d.task[np.isclose(d.rate_common, m.rate_common)])
        out[f"cond{c}.{a}"] = dict(median=r(d.rate_common.median()), max=r(m.rate_common), max_lo=r(m.lo_common),
                                   max_hi=r(m.hi_common), max_tasks=top, k_beyond=int((d.k_all - d.k_common).sum()),
                                   k_all=int(d.k_all.sum()))
    c1 = real[real.cond == 1].pivot(index="task", columns="arm", values="rate_common")
    out["cond1_mmd_below_discriminator"] = int((c1.mmd < c1.discriminator).sum())
    out["cond1_n_tasks"] = int(len(c1))
    rk = rp[(rp.cond == 1) & rp.task.isin(REAL) & rp["rank"].notna()].pivot(index="task", columns="arm", values="rank")
    out["cond1_mmd_ranked_above_discriminator"], out["cond1_n_ranked"] = int((rk.mmd < rk.discriminator).sum()), int(len(rk))
    out["n_common_min"], out["n_common_max"] = int(real.n_common.min()), int(real.n_common.max())
    return out


if __name__ == "__main__":
    f = fits(pd.read_csv(DATA / "metric_scores.csv"), pd.read_csv(DATA / "curves.csv"), json.load(open(DATA / "paper_numbers.json")))
    t = table(f, pd.read_csv(DATA / "secondary.csv"))
    t.to_csv(DATA / "collapse_common_range.csv", index=False, float_format="%.6f")
    res = numbers(t, pd.read_csv(DATA / "ranking_pairs.csv"))
    (DATA / "collapse_common_range.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
