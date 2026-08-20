#!/usr/bin/env python
"""Dataset-stratified method comparison: composite, integration (batch), biology.

Aggregates the LEAN validation wave (barycenter, 3 seeds) against E3 baselines
(harmony/scanorama/PCA), Seurat (6-dataset CCA), producing the paper's honest
comparison tables + figures.

WHY STRATIFIED, NOT POOLED: datasets differ enormously in baseline difficulty
(sim2 floor ~0.31 vs pancreas ~0.50), so a mean over datasets is dominated by
which datasets are in the panel, and a between-dataset CI measures dataset
heterogeneity, not method uncertainty. We compare methods WITHIN each dataset
(where the barycenter's 3 seeds give a real CI) and aggregate by mean RANK
(Friedman-blocked on dataset), never by pooling raw scores across datasets.

scIB categories are Luecken et al. only:
  batch (SB) = ilisi asw_batch graph_conn pcr kbet         (all +1)
  bio   (SO) = clisi(-1) ari nmi asw_celltype isolated_asw isolated_f1
  composite  = 0.4*batch + 0.6*bio
cell_cycle excluded (E3 baseline harness does not compute it); PAGA/Trust/Cont
are geometry columns reported separately, never in the composite.

Inputs (run from repo root):
  results/wave/*_XV_*.csv                 barycenter LEAN wave (3 seeds)
  <E3 wave csv or artifact>               harmony/scanorama/unintegrated(=PCA)
  results/seurat_<ds>/seurat_metrics.json Seurat per dataset
Writes: stratified_composite.csv, stratified_batch.csv, stratified_bio.csv,
        stratified_ranks.csv, composite_ci.json and the matching PNGs.
"""
import sys, os, json, glob, argparse
import numpy as np
import pandas as pd
from scipy import stats

SB = {"ilisi": 1, "asw_batch": 1, "graph_conn": 1, "pcr": 1, "kbet": 1}
SO = {"clisi": -1, "ari": 1, "nmi": 1, "asw_celltype": 1, "isolated_asw": 1, "isolated_f1": 1}
DS6 = ["pancreas", "immune", "lung", "sim1", "sim2", "atac_small"]
SEUR_DIRS = {"pancreas": "seurat_in", "immune": "seurat_immune", "lung": "seurat_lung",
             "sim1": "seurat_sim1", "sim2": "seurat_sim2", "atac_small": "seurat_atac_small"}
METHODS = ["bary LDVAE_unc", "bary LDVAE", "LDVAE no-adv", "PCA", "Seurat", "harmony", "scanorama"]


def catrow(r, spec):
    v = [r[m] * (1 if s > 0 else -1) for m, s in spec.items() if m in r and pd.notna(r[m])]
    return np.mean(v) if v else np.nan


def add_cats(df):
    df = df.copy()
    df["batch"] = df.apply(lambda r: catrow(r, SB), axis=1)
    df["bio"] = df.apply(lambda r: catrow(r, SO), axis=1)
    df["scIB"] = 0.4 * df["batch"] + 0.6 * df["bio"]
    return df


def load_all(args):
    wave = add_cats(pd.concat([pd.read_csv(f) for f in glob.glob(args.wave_glob)],
                              ignore_index=True))
    e3 = add_cats(pd.read_csv(args.e3_csv))
    e3 = e3[e3.experiment == "E3"] if "experiment" in e3.columns else e3
    seur = add_cats(pd.DataFrame([
        {**json.load(open(os.path.join(args.results, dn, "seurat_metrics.json"))), "dataset": ds}
        for ds, dn in SEUR_DIRS.items()
        if os.path.exists(os.path.join(args.results, dn, "seurat_metrics.json"))
    ]))
    return wave, e3, seur


def series(m, axis, wave, e3, seur):
    """(mean-per-dataset, seed-sd-per-dataset) for one method on one axis."""
    if m == "bary LDVAE_unc":
        g = wave[(wave.backbone == "LDVAE_uncond") & (wave.d_coef == 0.2)]
        return g.groupby("dataset")[axis].mean(), g.groupby("dataset")[axis].std()
    if m == "bary LDVAE":
        g = wave[(wave.backbone == "LDVAE") & (wave.d_coef == 0.2)]
        return g.groupby("dataset")[axis].mean(), g.groupby("dataset")[axis].std()
    if m == "LDVAE no-adv":
        g = wave[(wave.backbone == "LDVAE") & (wave.d_coef == 0.0)]
        return g.groupby("dataset")[axis].mean(), g.groupby("dataset")[axis].std()
    if m == "Seurat":
        return seur.set_index("dataset")[axis], None
    key = {"PCA": "unintegrated", "harmony": "harmony", "scanorama": "scanorama"}[m]
    return e3[e3.method == key].set_index("dataset")[axis], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave-glob", default="results/wave/*_XV_*.csv")
    ap.add_argument("--e3-csv", required=True,
                    help="CSV with E3 baseline rows (harmony/scanorama/unintegrated)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--make-figures", action="store_true")
    args = ap.parse_args()
    wave, e3, seur = load_all(args)

    mats = {}
    for axis in ["scIB", "batch", "bio"]:
        M = pd.DataFrame({m: series(m, axis, wave, e3, seur)[0].reindex(DS6) for m in METHODS}).T
        name = {"scIB": "composite"}.get(axis, axis)
        M.round(4).to_csv(f"stratified_{name}.csv")
        mats[axis] = M

    # rank aggregate + Friedman on the composite
    Mc = mats["scIB"]
    ranks = Mc.rank(axis=0, ascending=False)
    ranks["mean_rank"] = ranks.mean(axis=1)
    ranks.round(3).sort_values("mean_rank").to_csv("stratified_ranks.csv")
    fr = stats.friedmanchisquare(*[Mc[c].values for c in Mc.columns])
    print("mean rank (1=best):")
    print(ranks["mean_rank"].sort_values().round(2).to_string())
    print(f"Friedman chi2={fr.statistic:.2f} p={fr.pvalue:.4f}")

    # between-dataset 95% CI (reported WITH the caveat that it measures dataset
    # heterogeneity; the stratified rank is the primary analysis)
    tcrit = stats.t.ppf(0.975, len(DS6) - 1)
    ci = {}
    for m in METHODS:
        v = Mc.loc[m].values.astype(float)
        ci[m] = [float(v.mean()), float(tcrit * v.std(ddof=1) / np.sqrt(len(DS6)))]
    json.dump(ci, open("composite_ci.json", "w"), indent=2)
    print("wrote stratified_{composite,batch,bio}.csv, stratified_ranks.csv, composite_ci.json")

    if args.make_figures:
        import matplotlib
        matplotlib.use("Agg")
        try:
            from figure_style import apply_figure_style
            apply_figure_style()
        except Exception:
            pass
        import matplotlib.pyplot as plt
        cols = {"bary LDVAE_unc": "#c0392b", "bary LDVAE": "#e67e22", "Seurat": "#8e44ad",
                "harmony": "#27ae60", "scanorama": "#16a085", "PCA": "#7f8c8d",
                "LDVAE no-adv": "#bdc3c7"}
        titles = {"scIB": "scIB composite", "batch": "Integration (batch mixing)",
                  "bio": "Biology conservation"}
        for axis in ["scIB", "batch", "bio"]:
            M = mats[axis]
            fig, axes = plt.subplots(2, 3, figsize=(11, 6))
            for ax, ds in zip(axes.flat, DS6):
                order = M[ds].sort_values(ascending=False)
                yy = np.arange(len(order))[::-1]
                for yi, (m, val) in zip(yy, order.items()):
                    ax.plot(val, yi, "o", color=cols[m], ms=5)
                ax.set_yticks(yy); ax.set_yticklabels(order.index, fontsize=6.2)
                ax.set_title(ds, loc="left", fontsize=8, fontweight="bold")
                ax.margins(x=0.18); ax.tick_params(axis="x", labelsize=6)
            fig.suptitle(f"{titles[axis]} — stratified by dataset (higher = better)",
                         fontsize=9.5, y=1.0)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            fig.savefig(f"stratified_{titles[axis].split()[0].lower()}.png", dpi=200)
            plt.close(fig)
        print("wrote stratified figures")


if __name__ == "__main__":
    main()
