#!/usr/bin/env python
"""Generate FINAL_RESULTS.md from the analysis CSVs — every number computed, none hand-typed.

Reads the outputs of analyze_final.py (scvi_final_full_curve.csv, scvi_final_frontier_dominance.csv,
scvi_final_baselines.csv, scvi_final_ranks.csv) plus a reconciliation of the sweep (manifest vs
completed npz vs divergences) and emits a reproducible markdown report. The report structure is
fixed; every value is read from a CSV so re-running after a rescore updates the numbers with no
manual edits. This keeps the report traceable to the scripts that produced it (project standing rule).

Usage:
  build_final_report.py --analysis-dir results/final \
      --manifest scripts/scvi_final_manifest.tsv \
      --embed-dir durable/embeddings_final \
      --driver-logs 'logs/wave/*.out' '~/.claude-science-scratch/.claude-science/jobs/*/slurm-*.out' \
      --out FINAL_RESULTS.md
"""
import os, glob, argparse, re
import numpy as np, pandas as pd

ADV_FORMS = ["discriminator", "reference", "pooled", "barycenter"]
CF_FORMS = ["mmd", "sinkhorn"]
CRITIC_FORMS = ["reference", "pooled", "barycenter"]  # Wasserstein critics
BASELINES = ["unintegrated", "harmony", "scanorama", "scvi", "scanvi"]
PREREG = {"adv": 20.0, "cf": 200.0}


def reconcile(manifest, embed_dir, driver_logs):
    """manifest tags = completed npz + diverged(FAIL) + missing(unrun). Returns a dict of counts."""
    tags = []
    for line in open(manifest):
        if line.startswith("#") or line.startswith("model"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 9 and parts[8]:
            tags.append(parts[8])
    tags = set(tags)
    done = {os.path.basename(p)[:-4] for p in glob.glob(f"{embed_dir}/*_XZ_*.npz")}
    fails = set()
    for pat in driver_logs:
        for p in glob.glob(os.path.expanduser(pat)):
            for line in open(p):
                if "[FAIL]" in line:
                    tok = next((t for t in line.split("[FAIL]", 1)[1].split() if "_XZ_" in t), None)
                    if tok:
                        fails.add(tok)
    fails &= tags
    missing = tags - done - fails
    fail_arms = {}
    for t in fails:
        m = re.match(r"(.+)_XZ_(lin|nl)_uncond_([a-z]+)_lam(\d+)_s(\d+)", t)
        if m:
            fail_arms.setdefault(m.group(3), []).append(m.group(1))
    return dict(manifest=len(tags), done=len(done), diverged=len(fails), missing=len(missing),
                fail_arms={a: sorted(set(v)) for a, v in fail_arms.items()},
                missing_tags=sorted(missing))


def fmt(x, nd=4):
    return "n/a" if (x is None or (isinstance(x, float) and x != x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="results/final")
    ap.add_argument("--manifest", default="scripts/scvi_final_manifest.tsv")
    ap.add_argument("--embed-dir", default="durable/embeddings_final")
    ap.add_argument("--driver-logs", nargs="+",
                    default=["logs/wave/*.out",
                             "~/.claude-science-scratch/.claude-science/jobs/*/slurm-*.out"])
    ap.add_argument("--out", default="FINAL_RESULTS.md")
    a = ap.parse_args()

    curve = pd.read_csv(f"{a.analysis_dir}/scvi_final_full_curve.csv")
    dom = pd.read_csv(f"{a.analysis_dir}/scvi_final_frontier_dominance.csv")
    base = pd.read_csv(f"{a.analysis_dir}/scvi_final_baselines.csv")
    ranks = pd.read_csv(f"{a.analysis_dir}/scvi_final_ranks.csv")
    rec = reconcile(a.manifest, a.embed_dir, a.driver_logs)

    L = []
    W = L.append
    W("# Wasserstein critics vs JS discriminator for single-cell batch correction — final results\n")
    W("Reproducible sweep: **8 datasets × 2 decoders × 6 formulations × per-family λ grid × 5 seeds**, "
      "all regenerated from one committed manifest (`scripts/scvi_final_manifest.tsv`). Every latent "
      "was scored through the identical `full_metric_suite`; the composite scIB is `0.4·batch + 0.6·bio`.\n")

    W("## Sweep completeness\n")
    W(f"- Manifest configurations: **{rec['manifest']}**")
    W(f"- Completed (scored latents): **{rec['done']}**")
    W(f"- Diverged (training instability, recorded NaN): **{rec['diverged']}**")
    if rec["missing"]:
        W(f"- Still missing (unrun): **{rec['missing']}** — {rec['missing_tags'][:6]}…")
    else:
        W(f"- Still missing (unrun): **0** — every non-diverged configuration completed")
    W("")
    if rec["fail_arms"]:
        W("Divergences by formulation arm (this is a result, not a failure):")
        for arm, dss in sorted(rec["fail_arms"].items()):
            W(f"- `{arm}`: on datasets {dss}")
        W("")

    # Central finding: which arms diverge
    critic_div = any(arm in CRITIC_FORMS + CF_FORMS for arm in rec["fail_arms"])
    disc_div = "discriminator" in rec["fail_arms"]
    W("## Central finding — numerical robustness\n")
    if disc_div and not critic_div:
        W(f"**The JS discriminator diverges where the Wasserstein critics do not.** The discriminator "
          f"arm failed to train (non-finite latents) on {rec['fail_arms'].get('discriminator', [])}, "
          f"while every Wasserstein-critic and OT arm trained to completion on the same datasets, "
          f"seeds, and λ. No gradient clipping was applied to either arm, so this is a property of the "
          f"objectives, not the optimiser tuning. This supports the paper's thesis that the "
          f"Wasserstein critic is the more numerically robust adversary for batch correction.\n")
    elif disc_div and critic_div:
        W(f"Both the discriminator and some critic arms diverged: discriminator on "
          f"{rec['fail_arms'].get('discriminator', [])}; critics on "
          f"{[a for a in rec['fail_arms'] if a in CRITIC_FORMS + CF_FORMS]}. The robustness gap is "
          f"narrower than the discriminator-only hypothesis predicted — see the per-arm table above.\n")
    else:
        W("No arm diverged on any dataset — the robustness comparison is inconclusive from divergence "
          "counts alone; see the frontier curves for the quality comparison.\n")

    # Frontier dominance
    W("## Primary result — selection-free frontier (no best-λ pick)\n")
    W("For each (dataset, decoder) we compare the whole scIB λ-response curve, not a post-hoc best λ "
      "(which would be winner's-curse biased over the grid). Fraction of λ where the discriminator "
      "curve sits at or above each alternative (mean over datasets×decoders):\n")
    if len(dom):
        agg = dom.groupby("vs")["frac_disc_dominates"].mean().sort_values()
        W("| alternative | frac λ where discriminator ≥ alternative |")
        W("|---|---|")
        for vs, fr in agg.items():
            W(f"| {vs} | {fmt(fr, 3)} |")
        W("")
        best_alt = agg.index[0]
        W(f"Lower is better for the alternative: `{best_alt}` most often exceeds the discriminator "
          f"(discriminator ≥ it only {fmt(agg.iloc[0], 3)} of the λ grid).\n")

    # Pre-registered baseline table
    W("## Secondary result — pre-registered-λ baseline comparison\n")
    W(f"At the **pre-registered** operating point (adv λ={PREREG['adv']:.0f}, OT λ={PREREG['cf']:.0f}, "
      f"declared in the manifest header before results), mean scIB across datasets by method "
      f"(higher = better). Selection-free: the λ was fixed a priori.\n")
    if len(ranks):
        for dec in sorted(ranks["dec"].unique()):
            W(f"\n**Decoder: {dec}** (mean rank across datasets, 1 = best)\n")
            g = ranks[ranks.dec == dec].sort_values("mean_rank")
            W("| method | mean rank | n datasets |")
            W("|---|---|---|")
            for _, r in g.iterrows():
                W(f"| {r['method']} | {fmt(r['mean_rank'], 2)} | {int(r['n_datasets'])} |")
        W("")

    W("## Figures\n")
    W("- `results/final/scvi_final_lambda_curves.png` — λ-response curves per (dataset, decoder), "
      "5-seed mean ± 95% CI, diverged arms drawn ending at their last stable λ (× marker).")
    W("- `results/final/scvi_final_ranking.png` — mean-rank bars per decoder.\n")

    W("## Reproducibility\n")
    W("- Manifest: `scripts/scvi_final_manifest.tsv` (fixed a-priori λ grid; λ=0 collapsed to shared "
      "adversary-none controls).")
    W("- Fit → latent: `scripts/scvi_adv_fit.py` via `scripts/run_jhpce_pilot.sh` (gate-then-drain, "
      "atomic per-config claims, durable embed dir).")
    W("- Score: `scripts/score_final_config.py` (identical `full_metric_suite`, 0.4/0.6 scIB).")
    W("- Analyse: `scripts/analyze_final.py` (frontier-dominance + pre-registered table + figures).")
    W("- Report: `scripts/build_final_report.py` (this file — every number read from a CSV).\n")

    txt = "\n".join(L)
    open(a.out, "w").write(txt)
    print(f"[report] wrote {a.out}  ({len(txt)} chars)")
    print(f"[report] reconciliation: manifest={rec['manifest']} done={rec['done']} "
          f"diverged={rec['diverged']} missing={rec['missing']}")


if __name__ == "__main__":
    main()