"""Pareto frontier of each penalty (Pass 23, new main Figure 4; user choice, version A). Batch removal is scaled per task and
decoder as x = (B - b0) / (b_common - b0), so 0 is plain scVI, 0.5 is b* and 1 is the highest batch score every penalty reaches.
A penalty's frontier at x is the best biology, minus plain scVI's, among its fitted strengths with scaled batch at least x.
Its hypervolume up to b_common is the area under that frontier from x = 0 to 1, above 0.10 below plain scVI's biology: the
Supplementary Note 7 hypervolume with each point's batch score capped at b_common. The uncapped value (all strengths, the
definition before Pass 23) is kept as hypervolume_all and must reproduce figure_data/secondary.csv.
Reads paper/figure_data/curves.csv (seed means per strength; plain scVI is strength 0), paper_numbers.json (b0, b_common,
low-signal flag) and secondary.csv. Writes paper/figure_data/pareto_frontier.csv (per pair and penalty),
pareto_frontier_median.csv (median frontier over the ranked real tasks per decoder, x grid 0 to 1) and pareto_frontier.json
(the numbers the text cites). Run from the repository root:  python paper/tools/pareto_frontier.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/figure_data"
REAL = ["pancreas", "lung", "immune", "atac_small", "atac_large", "immune_hum_mou"]
COLLAPSE_BIO = 0.10  # reference point 0.10 below plain scVI's biology (Supplementary Note 7)
GRID = np.round(np.linspace(0, 1, 501), 6)


def hypervolume(x, y, cap=np.inf):
    """Area dominated by points (x, y) beyond the reference (0, -0.10); x capped at `cap` (union of rectangles)."""
    height, total = -COLLAPSE_BIO, 0.0
    for xi, yi in sorted(((min(a, cap), b) for a, b in zip(x, y) if a > 0 and b > -COLLAPSE_BIO), reverse=True):
        total += xi * max(0.0, yi - height)
        height = max(height, yi)
    return float(total)


def frontier(x, y, grid=GRID):
    """Best y among points with x at least each grid value (NaN where no point reaches)."""
    return np.array([y[x >= g - 1e-12].max() if (x >= g - 1e-12).any() else np.nan for g in grid])


def scaled(cv, pn):
    """Fitted strengths with scaled batch x and biology minus plain scVI y, for the pairs that are not low-signal."""
    rows = []
    for (t, c), d in cv.groupby(["task", "cond"]):
        key = f"pair.{t}.cond{c}"
        if pn[f"{key}.low_signal"]["value"]:
            continue
        b0, bc = pn[f"{key}.b0"]["value"], pn[f"{key}.b_common"]["value"]
        c0 = d[d.lam == 0].C.iloc[0]
        f = d[d.lam > 0]
        rows.append(f.assign(x=(f.B - b0) / (bc - b0), y=f.C - c0))
    return pd.concat(rows)


def tables(cv, pn, sec):
    assert cv.n_fail.sum() == 0
    pts = scaled(cv, pn)
    per = pd.DataFrame([dict(task=t, cond=c, arm=a, hypervolume_to_common=hypervolume(d.x, d.y, 1.0),
                             hypervolume_all=hypervolume(d.x, d.y), reach=float(d.x.max()))
                        for (t, c, a), d in pts.groupby(["task", "cond", "arm"])])
    chk = per.merge(sec[["task", "cond", "arm", "hypervolume"]], on=["task", "cond", "arm"], validate="1:1")
    assert len(chk) == sec.hypervolume.notna().sum() and np.allclose(chk.hypervolume_all, chk.hypervolume, atol=1e-12, rtol=0)
    real = pts[pts.task.isin(REAL)]
    med = []
    for (c, a), d in real.groupby(["cond", "arm"]):
        m = np.array([frontier(e.x.to_numpy(), e.y.to_numpy()) for _, e in d.groupby("task")])
        assert not np.isnan(m).any()  # every penalty reaches x = 1 in every pair (b_common is common to all)
        med.append(pd.DataFrame(dict(cond=c, arm=a, x=GRID, median=np.median(m, 0), n_tasks=len(m))))
    return per, pd.concat(med, ignore_index=True)


def numbers(per, med):
    r = lambda v: round(float(v), 3)
    real = per[per.task.isin(REAL)]
    out = {"n_tasks": {str(c): int(d.task.nunique()) for c, d in real.groupby("cond")}}
    for c, d in real.groupby("cond"):
        for a, e in d.groupby("arm"):
            m = med[(med.cond == c) & (med.arm == a)].set_index("x")["median"]
            out[f"cond{c}.{a}"] = dict(hv_min=r(e.hypervolume_to_common.min()), hv_max=r(e.hypervolume_to_common.max()),
                                       hv_median=r(e.hypervolume_to_common.median()),
                                       hv_all_median=r(e.hypervolume_all.median()),
                                       front_at_bstar=r(m.loc[0.5]), front_at_common=r(m.loc[1.0]),
                                       front_min=r(m.min()), reach_max=r(e.reach.max()))
    out["n_real_pairs"] = int(real.groupby(["task", "cond"]).ngroups)
    out["n_pairs_barycenter_reach_1"] = int((np.isclose(real[real.arm == "barycenter"].reach, 1.0)).sum())
    beyond = real[~np.isclose(real.reach, 1.0)].groupby("arm").size()
    out["n_pairs_reach_beyond_1"] = {a: int(beyond.get(a, 0)) for a in sorted(real.arm.unique())}
    out["max_abs_front_up_to_bstar"] = r(med[med.x <= 0.5]["median"].abs().max())
    hv = {f"cond{c}.{a}": out[f"cond{c}.{a}"]["hv_median"] for c in (0, 1) for a in sorted(real.arm.unique())}
    rest = [v for k, v in hv.items() if k != "cond1.discriminator"]
    out["hv_median_others_min"], out["hv_median_others_max"] = min(rest), max(rest)
    drop = {c: {a: -out[f"cond{c}.{a}"]["front_at_common"] for a in sorted(real.arm.unique())} for c in (0, 1)}
    assert all(v > 0 for c in drop for v in drop[c].values())
    others = [v for a, v in drop[1].items() if a != "discriminator"]
    assert drop[1]["discriminator"] > max(others)
    out["text"] = {"drop_at_common_cond1_discriminator": drop[1]["discriminator"], "drop_at_common_cond1_others_min": min(others),
                   "drop_at_common_cond1_others_max": max(others), "drop_at_common_cond0_min": min(drop[0].values()),
                   "drop_at_common_cond0_max": max(drop[0].values())}  # the drops below plain scVI that Res-3b cites
    return out


if __name__ == "__main__":
    per, med = tables(pd.read_csv(DATA / "curves.csv"), json.load(open(DATA / "paper_numbers.json")),
                      pd.read_csv(DATA / "secondary.csv"))
    per.to_csv(DATA / "pareto_frontier.csv", index=False, float_format="%.6f")
    med.to_csv(DATA / "pareto_frontier_median.csv", index=False, float_format="%.6f")
    res = numbers(per, med)
    (DATA / "pareto_frontier.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
