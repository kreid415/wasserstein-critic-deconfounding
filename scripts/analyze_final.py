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
    df["diverged"] = False
    return df

def load_divergences(driver_logs, manifest):
    """Return diverged (NaN) configs as explicit rows so the curve marks them, not drops them.

    A config that appears in a driver log as [FAIL] with a NaN encoder output is a genuine
    training divergence (high-lambda adversarial instability), NOT a missing computation. We
    record it with scIB=NaN and diverged=True so the frontier curve shows the arm ENDING at the
    last stable lambda rather than silently omitting the point.

    driver_logs may be a single path OR a glob pattern OR a list of paths/globs. A resumed sweep
    spreads its [FAIL] lines across several allocation logs (each Slurm job writes its own
    slurm-<id>.out under a separate job dir), so ALL of them must be read and the tags de-duped —
    reading only one log silently drops the divergences recorded by the other allocations, which
    would erase the discriminator-divergence finding from the frontier curves.
    """
    import glob, re
    if isinstance(driver_logs, str):
        driver_logs = [driver_logs]
    paths = []
    for pat in driver_logs:
        if not pat:
            continue
        hits = glob.glob(pat)
        paths.extend(hits if hits else ([pat] if os.path.exists(pat) else []))
    if not paths:
        return pd.DataFrame()
    fails = set()
    # the runner logs:  [FAIL] gpu<N> <tag> rc=<code>
    # so the tag is the token that looks like a config tag (contains '_XZ_'), NOT simply the
    # first token after [FAIL] (which is the 'gpu0' lane id). Pick the _XZ_ token robustly.
    for p in paths:
        for line in open(p):
            if "[FAIL]" not in line:
                continue
            after = line.split("[FAIL]", 1)[1].strip()
            tok = next((t for t in after.split() if "_XZ_" in t), None)
            if tok:
                fails.add(tok)
    if not fails:
        return pd.DataFrame()
    # parse tag: <dataset>_XZ_<dec>_uncond_<adv>_lam<lam>_s<seed>
    rows = []
    for tag in sorted(fails):
        m = re.match(r"(.+)_XZ_(lin|nl)_uncond_([a-z]+)_lam(\d+)_s(\d+)", tag)
        if not m:
            continue
        ds, dec, adv, lam, seed = m.groups()
        rows.append(dict(tag=tag, dataset=ds, adv=adv, lam=float(lam), dec=dec,
                         seed=int(seed), batch=np.nan, bio=np.nan, scIB=np.nan, diverged=True))
    return pd.DataFrame(rows)

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
        n_div = int(g["diverged"].sum()) if "diverged" in g else 0
        for axis in ("scIB", "batch", "bio"):
            m, lo, hi, n = ci95(g[axis].values)
            rows.append(dict(dataset=ds, dec=dec, formulation=adv, lam=lam,
                             axis=axis, mean=m, ci_lo=lo, ci_hi=hi, n_seeds=n,
                             n_diverged=n_div))
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
    """Each formulation at its PRE-REGISTERED lambda + baselines; mean-rank across datasets.

    The external baselines (harmony/scvi/scanvi/scanorama/unintegrated) are decoder-AGNOSTIC — they
    do not use the scVI decoder at all, so they carry dec='base'. To rank them HEAD-TO-HEAD against
    the adversarial arms (which are split into lin/nl decoder groups), we replicate each baseline
    into BOTH decoder groups. Otherwise baselines would rank only among themselves (a meaningless
    5-method ranking) and the reader could not see where scVI/scANVI sit relative to the critics.
    """
    # adversarial at prereg lambda; baselines (only config); none control (lam=0)
    adv = df[((df.adv.isin(ADV_FORMS)) & (df.lam == PREREG["adv"])) |
             ((df.adv.isin(CF_FORMS)) & (df.lam == PREREG["cf"]))]
    base = df[df.adv.isin(BASELINES)].copy()
    none = df[df.adv == "none"].copy()
    # decoder groups the adversarial arms actually span
    adv_decs = [d for d in adv["dec"].unique() if d not in ("base",)]
    if not adv_decs:
        adv_decs = ["lin", "nl"]
    # replicate decoder-agnostic rows (baselines + none control) into each adversarial decoder group
    dec_agnostic = pd.concat([base, none], ignore_index=True)
    replicated = []
    for dec in adv_decs:
        r = dec_agnostic.copy(); r["dec"] = dec
        replicated.append(r)
    pooled = pd.concat([adv] + replicated, ignore_index=True)
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
    ap.add_argument("--driver-log", nargs="+",
                    default=["logs/wave/_driver_final.log",
                             os.path.expanduser("~/.claude-science-scratch/.claude-science/jobs/*/slurm-*.out")],
                    help="sweep driver log(s) — [FAIL] lines flag diverged (NaN) configs. "
                         "Accepts multiple paths/globs; a resumed sweep spreads FAILs across "
                         "several allocation logs, so ALL must be read.")
    ap.add_argument("--outdir", default="results/final")
    ap.add_argument("--make-figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_scored(args.scored)
    div = load_divergences(args.driver_log, None)
    if len(div):
        # keep diverged rows only if they were NOT later rescored (a retry that succeeded)
        div = div[~div["tag"].isin(set(df["tag"]))]
        df = pd.concat([df, div], ignore_index=True)
        print(f"[analyze] diverged (NaN) configs recorded: {len(div)} "
              f"({sorted(div['dataset'].unique())} at lam {sorted(div['lam'].unique())})")
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
    """Two publication figures, both from the selection-free curve table (no best-lambda pick).

    FIG 1 scvi_final_lambda_curves.png — the PRIMARY result. Per (dataset, decoder) panel, the
    scIB lambda-response curve of each formulation: 5-seed mean line + 95% CI band over the fixed
    grid. A diverged arm (all seeds NaN at a lambda) is drawn ENDING at its last stable lambda with
    an open marker at the divergence point, so the discriminator instability is visible, not hidden.

    FIG 2 scvi_final_ranking.png — the mean-rank across datasets per formulation (lower = better),
    one bar group per decoder. Summarises the curve dominance as a single ordering.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FORMS = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn", "none"]
    COL = {"discriminator": "#d62728", "reference": "#1f77b4", "pooled": "#2ca02c",
           "barycenter": "#9467bd", "mmd": "#ff7f0e", "sinkhorn": "#17becf", "none": "#7f7f7f"}
    sc = curve[curve.axis == "scIB"].copy()
    dsets = [d for d in DATASETS if d in set(sc.dataset)]
    decs = sorted(set(sc.dec))
    if dsets and decs:
        nrow, ncol = len(decs), len(dsets)
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 3.0 * nrow),
                                 squeeze=False, sharey="row")
        for i, dec in enumerate(decs):
            for j, ds in enumerate(dsets):
                ax = axes[i][j]
                g = sc[(sc.dataset == ds) & (sc.dec == dec)]
                for form in FORMS:
                    gf = g[g.formulation == form].sort_values("lam")
                    if gf.empty:
                        continue
                    stable = gf[gf["mean"].notna()]
                    if stable.empty:
                        continue
                    x, y = stable["lam"].values, stable["mean"].values
                    # label on every panel; the legend builder dedups by label
                    ax.plot(x, y, "-o", ms=3, lw=1.3, color=COL.get(form, "k"), label=form)
                    lo, hi = stable["ci_lo"].values, stable["ci_hi"].values
                    if (~np.isnan(lo)).any():
                        ax.fill_between(x, lo, hi, color=COL.get(form, "k"), alpha=0.15, lw=0)
                    # mark a divergence: a lambda where this arm has seeds but a NaN mean
                    div = gf[(gf["mean"].isna()) & (gf["n_diverged"] > 0)]
                    if not div.empty and not stable.empty:
                        ax.plot([stable["lam"].values[-1]], [stable["mean"].values[-1]],
                                marker="x", ms=8, mew=2, color=COL.get(form, "k"))
                if i == nrow - 1:
                    ax.set_xlabel(r"$\lambda$")
                if j == 0:
                    ax.set_ylabel(f"{dec}\nscIB")
                if i == 0:
                    ax.set_title(ds, fontsize=9)
                ax.set_xscale("symlog")
                ax.grid(alpha=0.25, lw=0.5)
        # collect legend entries across ALL panels (a single panel may not plot every arm),
        # dedup by label, and guard against an empty legend
        hl = {}
        for row in axes:
            for ax in row:
                for h, l in zip(*ax.get_legend_handles_labels()):
                    hl.setdefault(l, h)
        if hl:
            fig.legend(list(hl.values()), list(hl.keys()), loc="upper center",
                       ncol=len(hl), fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(f"{outdir}/scvi_final_lambda_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if len(ranks):
        decs2 = sorted(set(ranks.dec))
        fig, axes = plt.subplots(1, len(decs2), figsize=(4.2 * len(decs2), 3.4),
                                 squeeze=False, sharey=True)
        for k, dec in enumerate(decs2):
            ax = axes[0][k]
            g = ranks[ranks.dec == dec].sort_values("mean_rank")
            ax.barh(g["method"], g["mean_rank"],
                    color=[COL.get(m, "#555") for m in g["method"]])
            ax.invert_yaxis()
            ax.set_xlabel("mean rank across datasets (1 = best)")
            ax.set_title(f"decoder: {dec}", fontsize=9)
            ax.grid(axis="x", alpha=0.25, lw=0.5)
        fig.tight_layout()
        fig.savefig(f"{outdir}/scvi_final_ranking.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    main()
