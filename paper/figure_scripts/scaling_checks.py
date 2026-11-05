"""Scaling checks for Supplementary Figure S5 (exploratory; PAPER_PLAN section 5, user decision SI-82; no counterpart in the sweep analysis code).

The ranking of the paper reads biology at b* from unscaled scIB metrics (batch score B = mean of the five batch metrics, biology score C =
mean of the task's biology metrics). Here the six penalties are re-ranked after a change of how the metrics are scaled or combined, on the
same fits, with the same reading rules (spec sections 2 and 3: seed means over the three sweep seeds, plain scVI as the first point,
b* = midpoint between plain scVI's B and the highest B all six penalties reach, linear interpolation at the first crossing, ties within
1e-9 share the average rank). Variants:
  minmax       each metric rescaled to 0-1 over one comparison set per task (every scored fit of the task: both decoders, all penalty
               strengths, plain scVI and the standard tools), as scIB does
  lomo_<fam>   as minmax, with the fits of one method family left out of the set that fixes the minimum and maximum (13 families)
  zscore       each metric standardised over the same comparison set
  no_silhouette  unscaled, without the three silhouette metrics (batch ASW, cell-type ASW, isolated-label silhouette)
The variants change B and C and therefore b*, the readings and the ranks; low-signal pairs stay excluded as in the primary ranking.
primary_check reproduces the sweep analysis' readings and ranks from metric_scores.csv (positive control, asserted).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

DATA = Path(__file__).resolve().parents[1] / "figure_data"
PEN = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]
BATCH = ["PCR_batch", "ASW_label/batch", "graph_conn", "kBET", "iLISI"]
BIO = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label", "isolated_label_F1", "isolated_label_silhouette", "cLISI",
       "cell_cycle_conservation", "trajectory"]
SILHOUETTE = ["ASW_label/batch", "ASW_label", "isolated_label_silhouette"]
TOL = 1e-9
FAMILIES = PEN + ["none", "scvi_adv", "scanvi", "sysvi", "harmony", "scanorama", "pca"]


def metric_columns(df):
    """Per task, the biology metrics that exist for it (cell cycle on four RNA tasks, trajectory on the two immune tasks)."""
    return {t: [m for m in BIO if g[m].notna().all()] for t, g in df.groupby("task")}


def scaled(df, variant, left_out=None):
    """df with columns B and C (batch and biology score) under `variant`."""
    df = df.copy()
    bio = metric_columns(df)
    out_b, out_c = pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    for t, g in df.groupby("task"):
        bm, cm = list(BATCH), list(bio[t])
        if variant == "no_silhouette":
            bm, cm = [m for m in bm if m not in SILHOUETTE], [m for m in cm if m not in SILHOUETTE]
        ref = g[g.arm != left_out] if left_out else g          # the comparison set that fixes the scale
        vals = g[bm + cm].copy()
        if variant in ("minmax", "lomo"):
            lo, hi = ref[bm + cm].min(), ref[bm + cm].max()
            vals = (vals - lo) / (hi - lo)
        elif variant == "zscore":
            vals = (vals - ref[bm + cm].mean()) / ref[bm + cm].std(ddof=0)
        out_b[g.index], out_c[g.index] = vals[bm].mean(axis=1), vals[cm].mean(axis=1)
    df["B"], df["C"] = out_b, out_c
    return df


def curves(df, task, cond):
    """{arm: [(lam, seed-mean B, seed-mean C)]} for the six penalties, and the plain-scVI anchor (b0, c0)."""
    g = df[(df.task == task) & (df.cond == cond) & (df.experiment == "A1")]
    zero = g[g.arm == "none"]
    assert len(zero) == 3, (task, cond, len(zero))
    b0, c0 = zero.B.mean(), zero.C.mean()
    res = {}
    for arm in PEN:
        p = g[g.arm == arm].groupby("lam")[["B", "C"]].mean().sort_index()
        res[arm] = [(0.0, b0, c0)] + [(lam, r.B, r.C) for lam, r in p.iterrows()]
    return res, b0


def reading(curve, bstar):
    for i, (_, b1, c1) in enumerate(curve):
        if abs(b1 - bstar) <= TOL:
            return c1
        if i + 1 < len(curve):
            _, b2, c2 = curve[i + 1]
            if abs(b2 - bstar) > TOL and (b1 - bstar) * (b2 - bstar) < 0:
                return c1 + (bstar - b1) * (c2 - c1) / (b2 - b1)
    return None


def ranks(readings):
    names = sorted(readings, key=lambda k: (readings[k] is None, -(readings[k] or 0.0)))
    tied = lambda a, b: (a is None and b is None) or (a is not None and b is not None and a - b <= TOL)
    out, i = {}, 0
    while i < len(names):
        j = i + 1
        while j < len(names) and tied(readings[names[i]], readings[names[j]]):
            j += 1
        out.update(dict.fromkeys(names[i:j], (i + 1 + j) / 2))
        i = j
    return out


def read_pair(df, task, cond):
    cv, b0 = curves(df, task, cond)
    b_common = min(max(b for _, b, _ in c) for c in cv.values())      # highest B that every penalty reaches
    bstar = b0 + 0.5 * (b_common - b0)
    rd = {a: reading(c, bstar) for a, c in cv.items()}
    return dict(bstar=bstar, bio=rd, rank=ranks(rd))


def run(ranked_pairs, variants):
    """ranked_pairs: [(task, cond)]; variants: {name: (variant, left_out)}. Returns a long table of readings and ranks."""
    ms = pd.read_csv(DATA / "metric_scores.csv")
    rows = []
    for name, (var, lo) in variants.items():
        d = scaled(ms, var, lo)
        for task, cond in ranked_pairs:
            r = read_pair(d, task, cond)
            for a in PEN:
                rows.append(dict(variant=name, task=task, cond=cond, arm=a, bstar=r["bstar"], bio=r["bio"][a], rank=r["rank"][a]))
    return pd.DataFrame(rows)


def primary_check(ranked):
    """The raw variant must reproduce ranking_pairs.csv (b*, biology at b*, rank) for every ranked pair."""
    rp = pd.read_csv(DATA / "ranking_pairs.csv")
    got = run(sorted({(t, c) for t, c in zip(ranked.task, ranked.cond)}), {"raw": ("raw", None)})
    m = got.merge(rp, on=["task", "cond", "arm"], suffixes=("", "_ref"), validate="one_to_one")
    assert len(m) == len(rp) == 78
    assert np.allclose(m.bstar, m.bstar_ref, atol=1e-9) and np.allclose(m.bio, m.bio_at_bstar, atol=1e-9) and np.allclose(m["rank"], m.rank_ref, atol=1e-9)
    return got


def kendall(a, b):
    return float(kendalltau(a, b, variant="b").statistic)
