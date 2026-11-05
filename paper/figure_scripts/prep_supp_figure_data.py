#!/usr/bin/env python
"""Derived inputs of Supplementary Figures S2-S5, written to paper/figure_data (run on the workstation; reads the sweep analysis).

  sensitivity_ranks.csv   Figure S2: per task, decoder, penalty and analysis variant (primary, per_run, no_trajectory, drop_failed),
                          the rank used by that variant, from analysis_sweep.json (sensitivity block; analysis commit 8595a16)
  metric_scores.csv       Figures S4 and S5: every scored fit's individual scIB metrics, read from the 3,480 score files listed in
                          the sweep analysis' inputs/scores_list.txt (the files the sweep analysis itself scored)
Usage: python paper/figure_scripts/prep_supp_figure_data.py [--sweep DIR] [--only sensitivity|metrics]
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figure_data"
SWEEP = "/home/kendall/experiment_data/wasserstein-critic-deconfounding/sweep_analysis_final_20261105"
PEN = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]
VARIANTS = {"primary": None, "per_run": "per_run", "no_trajectory": "no_trajectory", "drop_failed": "drop_failed"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sensitivity(sweep):
    a = json.load(open(Path(sweep) / "analysis_sweep.json"))
    rows = []
    for name, key in VARIANTS.items():
        blk = a["primary"] if key is None else a["sensitivity"][key]
        for cell, c in blk["cells"].items():
            for arm in PEN:
                p = c["penalties"].get(arm)
                if p is None:
                    continue
                rows.append(dict(variant=name, task=c["task"], cond=c["cond"], arm=arm, low_signal=bool(c["low_signal"]),
                                 rank=p.get("rank")))
    df = pd.DataFrame(rows)
    assert set(df.variant) == set(VARIANTS) and len(df) == 4 * 96, (set(df.variant), len(df))
    return df, sha(Path(sweep) / "analysis_sweep.json")


BATCH = ["PCR_batch", "ASW_label/batch", "graph_conn", "kBET", "iLISI"]
BIO = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label", "isolated_label_F1", "isolated_label_silhouette", "cLISI",
       "cell_cycle_conservation", "trajectory"]


def metrics(sweep):
    """All scored fits of the sweep analysis with their individual metrics (same files, same tags as the analysis)."""
    sw = Path(sweep) / "inputs"
    files = [ln.strip() for ln in open(sw / "scores_list.txt") if ln.strip()]
    assert len(files) == 3480 and len(set(files)) == 3480, len(files)
    sc = pd.concat([pd.read_csv(f, usecols=lambda c: c in ["tag", "latent_sha256"] + BATCH + BIO) for f in files], ignore_index=True)
    assert sc.tag.is_unique and len(sc) == 3480
    man = pd.read_csv(sw / "manifest.tsv", sep="\t", comment="#")
    keep = man[["tag", "experiment", "task", "arm", "lam", "n_latent", "cond", "seed"]]
    out = keep.merge(sc, on="tag", how="inner", validate="one_to_one")
    assert len(out) == 3480, len(out)
    out["setting"] = out.arm + "|" + out.lam.map(lambda v: f"{v:g}") + "|" + out.n_latent.astype(str)
    return out, sha(sw / "scores_list.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--only", choices=["sensitivity", "metrics"])
    a = ap.parse_args()
    if a.only in (None, "sensitivity"):
        df, h = sensitivity(a.sweep)
        df.to_csv(DATA / "sensitivity_ranks.csv", index=False)
        print("sensitivity_ranks.csv", len(df), "from analysis_sweep.json sha256", h)
    if a.only in (None, "metrics"):
        df, h = metrics(a.sweep)
        df.to_csv(DATA / "metric_scores.csv", index=False)
        print("metric_scores.csv", len(df), "scores_list.txt sha256", h)
