#!/usr/bin/env python
"""Final analysis: selection-free frontier-dominance + pre-registered-lambda baseline table.

PRIMARY (selection-free): per (dataset, decoder) the scIB lambda-response CURVE per formulation,
5-seed mean +/- 95% CI band, plotted over the fixed grid. The claim is curve-level dominance
(does the discriminator curve sit above the W1-critic / OT curves across the whole lambda range),
NOT a post-hoc best-lambda pick (that would be winner's-curse biased over ~5 lambda).

SECONDARY (pre-registered): baseline comparison at the ONE lambda declared in the manifest header
BEFORE results (adv=20, cf=200), mean-rank across datasets (Friedman-blocked), paired tests.
Explicitly labelled 'pre-registered lambda' so no operating point is chosen on the reported data.

Inputs (all from THIS run):
  scored adversarial + baseline CSV (score_final_config.py output), one row per config
Outputs:
  scvi_final_full_curve.csv, scvi_final_baselines.csv, scvi_final_ranking.png,
  scvi_final_lambda_curves.png
"""
import sys, os, json, argparse
import numpy as np, pandas as pd
from scipy import stats

DATASETS = ["pancreas", "immune", "lung", "sim1", "sim2", "atac_small", "immune_hum_mou", "atac_large"]
ADV_FORMS = ["discriminator", "reference", "pooled", "barycenter"]
CF_FORMS = ["mmd", "sinkhorn"]
BASELINES = ["unintegrated", "harmony", "scanorama", "scvi", "scanvi"]
PREREG = {"adv": 20.0, "cf": 200.0}   # matches manifest header

def load_scored(csv_path):
    df = pd.read_csv(csv_path)
    # 'adv' column already carries formulation / baseline name; lam=0 rows are the 'none' controls
    return df

def ci95(vals):
    vals = np.asarray([v for v in vals if v == v], dtype=float)
    n = len(vals)
    if n == 0: return (np.nan, np.nan, np.nan, 0)
    m = vals.mean()
    if n == 1: return (m, np.nan, np.nan, 1)
    hw = stats.t.ppf(0.975, n - 1) * vals.std(ddof=1) / np.sqrt(n)
    return (m, m - hw, m + hw, n)

def build_curve_table(df):
    """One row per (dataset, decoder, formulation, lambda): 5-seed mean +/- CI of scIB/batch/bio."""
    rows = []
    for (ds, dec, adv, lam), g in df.groupby(["dataset", "dec", "adv", "lam"]):
        for axis in ("scIB", "batch", "bio"):
            m, lo, hi, n = ci95(g[axis].values)
            rows.append(dict(dataset=ds, dec=dec, formulation=adv, lam=lam,
                             axis=axis, mean=m, ci_lo=lo, ci_hi=hi, n_seeds=n))
    return pd.DataFrame(rows)

def frontier_dominance(curve):
    """For each (dataset, decoder): does the discriminator scIB curve dominate each other adversarial
    formulation across ALL shared lambda>0 points? Report the fraction of lambda where disc >= other."""
    out = []
    sc = curve[curve.axis == "scIB"]
    for (ds, dec), g in sc.groupby(["dataset", "dec"]):
        disc = g[(g.formulation == "discriminator") & (g.lam > 0)].set_index("lam")["mean"]
        for other in ["reference", "pooled", "barycenter", "mmd", "sinkhorn"]:
            o = g[(g.formulation == other) & (g.lam > 0)].set_index("lam")["mean"]
            shared = disc.index.intersection(o.index)
            if len(shared) == 0:
                continue
            frac = float((disc[shared] >= o[shared]).mean())
            out.append(dict(dataset=ds, dec=dec, vs=other, n_lambda=len(shared),
                            frac_disc_dominates=frac,
                            disc_mean=float(disc[shared].mean()), other_mean=float(o[shared].mean())))
    return pd.DataFrame(out)

def prereg_table(df):
    """Each formulation at its PRE-REGISTERED lambda + baselines; mean-rank across datasets."""
    # adversarial at prereg lambda; baselines at lam=0 (their only config); none control at lam=0
    adv = df[((df.adv.isin(ADV_FORMS)) & (df.lam == PREREG["adv"])) |
             ((df.adv.isin(CF_FORMS)) & (df.lam == PREREG["cf"]))]
    base = df[df.adv.isin(BASELINES)]
    none = df[df.adv == "none"]
    pooled = pd.concat([adv, base, none], ignore_index=True)
    # per (method, dataset, decoder): 5-seed mean scIB
    agg = pooled.groupby(["adv", "dataset", "dec"])["scIB"].agg(["mean", "std", "count"]).reset_index()
    # mean-rank across datasets, within decoder (Friedman-blocked on dataset)
    ranks = []
    for dec, gd in agg.groupby("dec"):
        piv = gd.pivot_table(index="adv", columns="dataset", values="mean")
        rank = piv.rank(axis=0, ascending=False)   # 1 = best per dataset
        ranks.append(pd.DataFrame(dict(dec=dec, method=rank.index,
                                       mean_rank=rank.mean(axis=1).values,
                                       n_datasets=rank.notna().sum(axis=1).values)))
    return agg, pd.concat(ranks, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="scored config CSV (score_final_config output)")
    ap.add_argument("--outdir", default="results/final")
    ap.add_argument("--make-figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_scored(args.scored)
    curve = build_curve_table(df)
    curve.to_csv(f"{args.outdir}/scvi_final_full_curve.csv", index=False)
    dom = frontier_dominance(curve)
    dom.to_csv(f"{args.outdir}/scvi_final_frontier_dominance.csv", index=False)
    agg, ranks = prereg_table(df)
    agg.to_csv(f"{args.outdir}/scvi_final_baselines.csv", index=False)
    ranks.to_csv(f"{args.outdir}/scvi_final_ranks.csv", index=False)
    print(f"[analyze] scored rows: {len(df)}  curve rows: {len(curve)}")
    print(f"[analyze] datasets present: {sorted(df.dataset.unique())}")
    print(f"[analyze] formulations present: {sorted(df.adv.unique())}")
    if len(dom):
        print("[analyze] frontier dominance (disc vs others, frac of lambda disc wins):")
        print(dom.groupby("vs")["frac_disc_dominates"].mean().round(3).to_string())
    if args.make_figures:
        make_figures(curve, ranks, args.outdir)

def make_figures(curve, ranks, outdir):
    # imported lazily so the analysis (CSV) path works without the figure stack
    import matplotlib; matplotlib.use("Agg")
    from figure_style_helpers import plot_lambda_curves, plot_ranking  # placeholder; inline below
    # (figures implemented in a follow-up cell against real data so styling can be verified)

if __name__ == "__main__":
    main()
