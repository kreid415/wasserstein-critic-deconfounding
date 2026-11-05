#!/usr/bin/env python
"""Per-cell-type F1 of label transfer at the matched strength, for Supplementary Figure S3 (written to paper/figure_data/label_transfer_types.csv).

The sweep analysis stores only the macro F1 of each label-transfer fit (figure_data/label_transfer.csv). This script repeats the
same classifier on the same saved embeddings and keeps the F1 of every cell type: cell types present in one batch only are dropped,
each batch is held out in turn, its cells are predicted from the other batches by a 15-nearest-neighbour classifier (Euclidean, equal
weights), and a type's F1 is averaged over the held-out batches that contain it. The macro F1 recomputed from these values must equal the
stored one for every row (asserted, 1e-9); a mismatch stops the run.
Run on the workstation with the scvi-api environment: python paper/figure_scripts/prep_label_transfer_types.py [--sweep DIR] [--jobs N]
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figure_data"
SWEEP = "/home/kendall/experiment_data/wasserstein-critic-deconfounding/sweep_analysis_final_20261105"
K = 15


def per_type_f1(path, expect_sha):
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert h == expect_sha, f"{path}: latent differs from the one that was scored"
    with np.load(path) as d:
        Z, ct, bt = d["z"], d["celltype"], d["batch"]
    dropped = {t for t in set(ct) if len(set(bt[ct == t])) == 1}
    keep = ~np.isin(ct, sorted(dropped))
    Z, ct, bt = Z[keep], ct[keep], bt[keep]
    by_type, macro = {}, []
    for b in sorted(set(bt)):
        test = bt == b
        pred = KNeighborsClassifier(n_neighbors=K).fit(Z[~test], ct[~test]).predict(Z[test])
        labels = sorted(set(ct[test]))
        f = f1_score(ct[test], pred, labels=labels, average=None, zero_division=0)
        macro.append(float(np.mean(f)))
        for t, v in zip(labels, f):
            by_type.setdefault(str(t), []).append(float(v))
    return {t: float(np.mean(v)) for t, v in by_type.items()}, float(np.mean(macro))


def job(a):
    return a[0], per_type_f1(a[1], a[2])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    ms = pd.read_csv(DATA / "metric_scores.csv")
    lt = pd.read_csv(DATA / "label_transfer.csv")
    lat = pd.read_csv(Path(a.sweep) / "inputs" / "latents.tsv", sep="\t", header=None, names=["tag", "path"], comment="#")
    assert lat.tag.is_unique
    lat = lat.set_index("tag").join(ms.set_index("tag").latent_sha256.rename("sha"), how="inner")   # sha256 of the latent each score row was made from
    work, spec = [], []
    for r in lt.itertuples():
        g = ms[(ms.task == r.task) & (ms.cond == r.cond) & (ms.arm == r.arm) & np.isclose(ms.lam, 0.0 if pd.isna(r.lam) else r.lam)]   # plain scVI and the scvi-tools adversary are unmatched (strength NaN in label_transfer.csv; 0 in the manifest)
        assert len(g) == r.n_runs, (r.task, r.cond, r.arm, r.lam, len(g), r.n_runs)
        for t in g.tag:
            work.append((t, lat.loc[t, "path"], lat.loc[t, "sha"]))
        spec.append((r, list(g.tag)))
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(job, work, chunksize=2))
    rows = []
    for r, tags in spec:
        macro = np.mean([res[t][1] for t in tags])
        assert abs(macro - r.f1_mean) < 1e-9, (r.task, r.cond, r.arm, r.lam, macro, r.f1_mean)
        types = sorted(set().union(*[set(res[t][0]) for t in tags]))
        for ty in types:
            v = [res[t][0][ty] for t in tags if ty in res[t][0]]
            rows.append(dict(cond=r.cond, task=r.task, arm=r.arm, lam=r.lam, celltype=ty, f1_mean=float(np.mean(v)), n_runs=len(v),
                             batch_score=r.batch_score, distance=r.distance))
    out = pd.DataFrame(rows)
    out.loc[out.arm.isin(["none", "scvi_adv"]), "lam"] = np.nan
    out.to_csv(DATA / "label_transfer_types.csv", index=False)
    print("label_transfer_types.csv", len(out), "rows; macro F1 recomputed equal to label_transfer.csv for", len(spec), "rows")
