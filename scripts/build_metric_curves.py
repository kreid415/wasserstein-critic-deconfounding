"""Per-individual-metric lambda-response curves, one figure per decoder.

Rows = individual scIB component metrics (batch block then bio block); cols = datasets.
Adversarial formulations drawn as lambda-response lines; external baselines (no lambda) drawn
as horizontal reference lines spanning each panel. kbet and cell_cycle are omitted (not computed
in this scoring pass — all-NaN). cLISI is a lower-is-better metric (marked with down arrow).
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.lines import Line2D

BATCH = ["ilisi", "asw_batch", "graph_conn", "pcr"]          # kbet dropped (all-NaN)
BIO   = ["clisi", "ari", "nmi", "asw_celltype", "isolated_asw", "isolated_f1"]  # cell_cycle dropped
METRICS = BATCH + BIO
MLABEL = {"ilisi":"iLISI ↑","asw_batch":"batch-ASW ↑","graph_conn":"graph-conn ↑","pcr":"PCR ↑",
          "clisi":"cLISI ↓","ari":"ARI ↑","nmi":"NMI ↑","asw_celltype":"celltype-ASW ↑",
          "isolated_asw":"iso-ASW ↑","isolated_f1":"iso-F1 ↑"}
FORMS = ["discriminator","reference","pooled","barycenter","mmd","sinkhorn","none"]
FCOL = {"discriminator":"#d62728","reference":"#1f77b4","pooled":"#2ca02c","barycenter":"#9467bd",
        "mmd":"#ff7f0e","sinkhorn":"#17becf","none":"#7f7f7f"}
FLAB = {"discriminator":"discriminator (JS)","reference":"reference (W)","pooled":"pooled (W)",
        "barycenter":"barycenter (W)","mmd":"MMD","sinkhorn":"Sinkhorn","none":"none (λ=0)"}
BASE = ["scvi","scanvi","harmony","scanorama","unintegrated"]
BCOL = {"scvi":"#111111","scanvi":"#7b3294","harmony":"#8c8c8c","scanorama":"#bcbc44","unintegrated":"#c0c0c0"}
BLAB = {"scvi":"scVI","scanvi":"scANVI (labels)","harmony":"Harmony","scanorama":"Scanorama","unintegrated":"unintegrated"}
DS_ORDER = ["pancreas","immune","lung","sim1","sim2","atac_small","immune_hum_mou","atac_large"]


def build(sc, dec, outfile, apply_style):
    apply_style(sizes=(8, 7, 6))
    dsets = [d for d in DS_ORDER if d in set(sc.dataset)]
    d = sc[sc.dec == dec]
    dbase = sc[sc.adv.isin(BASE)]                     # baselines are dec-agnostic
    nrow, ncol = len(METRICS), len(dsets)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.15 * ncol, 1.5 * nrow),
                             squeeze=False, sharex=True)
    for i, m in enumerate(METRICS):
        # shared y per metric-row for cross-dataset comparability
        rowvals = pd.concat([d[d.adv.isin(FORMS)][m], dbase[m]]).dropna()
        ylo, yhi = (rowvals.min(), rowvals.max()) if len(rowvals) else (0, 1)
        pad = 0.06 * (yhi - ylo + 1e-9)
        for j, ds in enumerate(dsets):
            ax = axes[i][j]
            g = d[d.dataset == ds]
            # formulation lambda-curves
            for form in FORMS:
                gf = g[g.adv == form]
                agg = gf.groupby("lam")[m].mean().dropna()
                if agg.empty:
                    continue
                if form == "none":                    # single point at lam0
                    ax.scatter([0], [agg.iloc[0]], s=10, color=FCOL[form], zorder=3)
                else:
                    ax.plot(agg.index.values, agg.values, "-o", ms=2.4, lw=1.0,
                            color=FCOL[form], zorder=3)
            # baselines as horizontal lines (mean over seeds, this dataset)
            gb = dbase[dbase.dataset == ds]
            for b in BASE:
                v = gb[gb.adv == b][m].mean()
                if np.isfinite(v):
                    ax.axhline(v, ls="--", lw=0.8, color=BCOL[b], alpha=0.9, zorder=2)
            ax.set_ylim(ylo - pad, yhi + pad)
            ax.set_xscale("symlog")
            ax.grid(alpha=0.2, lw=0.4)
            if i == 0:
                ax.set_title(ds, fontsize=7)
            if j == 0:
                ax.set_ylabel(MLABEL[m], fontsize=6.5)
            else:
                ax.tick_params(labelleft=False)
            if i == nrow - 1:
                ax.set_xlabel(r"$\lambda$", fontsize=7)
            ax.tick_params(labelsize=5)
    # row-group separators: batch block vs bio block
    fig.canvas.draw()
    y_sep = (axes[len(BATCH)-1][0].get_position().y0 + axes[len(BATCH)][0].get_position().y1) / 2
    fig.add_artist(Line2D([0.02, 0.99], [y_sep, y_sep], color="0.5", lw=0.8, ls=":"))
    fig.text(0.005, (axes[0][0].get_position().y1 + axes[len(BATCH)-1][0].get_position().y0)/2,
             "BATCH removal", rotation=90, va="center", ha="left", fontsize=7, color="0.35", weight="bold")
    fig.text(0.005, (axes[len(BATCH)][0].get_position().y1 + axes[-1][0].get_position().y0)/2,
             "BIO conservation", rotation=90, va="center", ha="left", fontsize=7, color="0.35", weight="bold")
    # legend
    fh = [Line2D([0],[0], color=FCOL[f], lw=1.4, marker="o", ms=3, label=FLAB[f]) for f in FORMS]
    bh = [Line2D([0],[0], color=BCOL[b], lw=1.2, ls="--", label=BLAB[b]) for b in BASE]
    fig.legend(handles=fh+bh, loc="upper center", ncol=6, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f"Per-metric λ-response — {dec} decoder   (curves = adversarial arms; "
                 f"dashed = baselines; kBET & cell-cycle omitted, not computed)",
                 y=1.025, fontsize=8.5)
    fig.tight_layout(rect=(0.03, 0, 1, 0.985))
    fig.savefig(outfile, dpi=170, bbox_inches="tight")
    return fig


def _plain_style(sizes=(8, 7, 6)):
    """Fallback styler when the figure-style skill kernel is not loaded (standalone/cluster run)."""
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": sizes[0], "axes.titlesize": sizes[0], "legend.fontsize": sizes[1],
                         "xtick.labelsize": sizes[2], "ytick.labelsize": sizes[2]})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Per-metric lambda-response curves with baselines as hlines")
    ap.add_argument("--scored", required=True, help="scored_final.csv (one row per config, all metric cols)")
    ap.add_argument("--decoders", nargs="+", default=["lin", "nl"])
    ap.add_argument("--outdir", default="results/final")
    args = ap.parse_args()
    try:
        from kernel import apply_figure_style as _style   # figure-style skill, if available
    except Exception:
        _style = _plain_style
    import os
    sc = pd.read_csv(args.scored)
    os.makedirs(args.outdir, exist_ok=True)
    for dec in args.decoders:
        out = os.path.join(args.outdir, f"metric_curves_{dec}.png")
        build(sc, dec, out, _style)
        print(f"[metric-curves] wrote {out}")

