#!/usr/bin/env python
"""Regression (iii) of the code-check fix CR-01: the scorer's PCR reference for a subsampled latent is the
all-feature PCA of the subset's cells (scIB's full-data convention), not the tagged scorer's HVG PCA.

  make-latent  --prepped P --manifest M --tag T --out Z.npz
      cells of manifest row T (an X3 row: fit_paper_config.subsample, draw nested_v1) and z = the prepped file's
      uncorrected HVG PCA, first n_latent columns (an uncorrected embedding: PCR_batch is far from 1)
  check        --prepped P --npz Z.npz --score S.csv [--out table.csv]
      recomputes, on the same cells, PCR_batch with the all-feature reference (no precomputed PCA) and with the
      tagged scorer's reference (sc.pp.pca(n_comps=50, mask_var='highly_variable')); exit 0 iff the score row's
      PCR_batch equals the all-feature value exactly.
Env: wcd-kbet (scib 1.1.7), KMP_AFFINITY=disabled, 1 thread.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def make_latent(a):
    import anndata as ad
    import fit_paper_config as fpc
    r = fpc.load_row(a.manifest, a.tag)
    spec = json.loads(r["extra"])["subsample"]
    pre = ad.read_h5ad(a.prepped)
    sub = fpc.subsample(pre, spec, int(r["seed"]))
    if "X_pca" not in sub.obsm:
        raise KeyError(f"{a.prepped} has no obsm X_pca")
    z = np.asarray(sub.obsm["X_pca"][:, :int(r["n_latent"])], dtype=np.float32)
    np.savez_compressed(a.out, z=z, obs_names=sub.obs_names.to_numpy(dtype="U128"),
                        batch=sub.obs["batch"].astype(str).to_numpy(dtype="U64"),
                        celltype=sub.obs["celltype"].astype(str).to_numpy(dtype="U64"),
                        config=json.dumps(dict(row=r, note="regression latent: uncorrected HVG PCA of the X3 cells")))
    print(f"[latent] {a.tag}: {sub.n_obs} of {pre.n_obs} cells, z {z.shape} -> {a.out}")


def check(a):
    import scanpy as sc
    import scib
    pre = sc.read_h5ad(a.prepped)
    if "pca" in pre.uns:
        raise ValueError(f"{a.prepped} carries uns['pca']; the production prepped files do not")
    d = np.load(a.npz, allow_pickle=False)
    names = d["obs_names"].astype(str)
    idx = pd.Index(pre.obs_names).get_indexer(names)
    if (idx < 0).any() or len(np.unique(idx)) != len(idx):
        raise ValueError("latent cells missing from or repeated in the prepped file")
    sub = pre[idx].copy()
    bk = pre.uns.get("batch_key", "batch")
    integ = sub.copy()
    integ.obsm["X_emb"] = d["z"].astype(np.float32)
    allgene = float(scib.metrics.pcr_comparison(sub.copy(), integ, covariate=bk, embed="X_emb"))
    hv = sub.copy()
    sc.pp.pca(hv, n_comps=50, mask_var="highly_variable")       # the tagged scorer's subset reference
    hvg = float(scib.metrics.pcr_comparison(hv, integ, covariate=bk, embed="X_emb"))
    before_all = float(scib.metrics.pcr(sub.copy(), covariate=bk, recompute_pca=True, n_comps=50, linreg_method="numpy"))
    before_hvg = float(scib.metrics.pcr(hv, covariate=bk, n_comps=50, linreg_method="numpy"))
    s = pd.read_csv(a.score)
    if len(s) != 1:
        raise ValueError(f"{a.score}: {len(s)} rows")
    got = float(s["PCR_batch"].iloc[0])
    T = pd.DataFrame([dict(tag=s["tag"].iloc[0], n_cells=int(sub.n_obs), n_cells_prepped=int(pre.n_obs),
                           n_features=int(sub.n_vars), n_hvg=int(sub.var["highly_variable"].sum()),
                           pcr_before_all_features=before_all, pcr_before_hvg=before_hvg,
                           PCR_batch_scorer=got, PCR_batch_all_feature_reference=allgene, PCR_batch_hvg_reference=hvg,
                           scorer_equals_all_feature=bool(got == allgene), scorer_equals_hvg=bool(got == hvg))])
    if a.out:
        T.to_csv(a.out, index=False)
    print(T.T.to_string(header=False))
    sys.exit(0 if got == allgene else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-latent")
    m.add_argument("--prepped", required=True)
    m.add_argument("--manifest", required=True)
    m.add_argument("--tag", required=True)
    m.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--prepped", required=True)
    c.add_argument("--npz", required=True)
    c.add_argument("--score", required=True)
    c.add_argument("--out")
    a = ap.parse_args()
    make_latent(a) if a.cmd == "make-latent" else check(a)


if __name__ == "__main__":
    main()
