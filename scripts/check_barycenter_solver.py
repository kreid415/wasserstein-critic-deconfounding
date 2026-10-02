#!/usr/bin/env python
"""How many fixed-point iterations does the per-step barycenter target need, and is a warm start equivalent?

Latent: stock scvi-tools SCVI (defaults) trained 15 epochs on prepped_scib/immune__scib.h5ad (scIB key, 10
batches), seed 0. 41 minibatches of 128 cells (fixed permutation, seed 0); the first 6 are burn-in for the warm
start. For every minibatch:
  converged      = 50 cold iterations (init = the minibatch cells)
  gap(variant)   = W2^2(variant support, converged support), same minibatch
  noise          = W2^2(converged support of minibatch t-1, of minibatch t)   (sampling difference)
  rng_gap        = W2^2(converged with two different torch seeds)
  obj_ratio      = sum_k w_k W2^2(P_k, support) / same for converged
Variants: cold 1/2/3/5/7/10 iterations; warm 3/5/10/20 iterations (init = previous step's support).
Writes docs/barycenter_solver_check.csv (per variant) and docs/barycenter_solver_check_raw.csv (per minibatch).
Usage (repo root, scvi env): WCD_SRC=src python scripts/check_barycenter_solver.py --prepped DIR
"""
import argparse
import importlib.util
import os
import time

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepped", required=True)
    ap.add_argument("--out", default="docs/barycenter_solver_check.csv")
    a = ap.parse_args()
    import anndata as ad
    import ot
    import scvi
    spec = importlib.util.spec_from_file_location("bary", os.path.join(os.environ.get("WCD_SRC", "src"), "wcd_vae", "wcd", "barycenter.py"))
    bary = importlib.util.module_from_spec(spec); spec.loader.exec_module(bary)

    adata = ad.read_h5ad(os.path.join(a.prepped, "immune__scib.h5ad"))
    adata = adata[:, adata.var["highly_variable"].values].copy()
    adata.X = adata.layers["counts"]
    scvi.settings.seed = 0
    scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    m = scvi.model.SCVI(adata)
    m.train(max_epochs=15, batch_size=128, early_stopping=False, enable_progress_bar=False)
    Z = torch.tensor(m.get_latent_representation(), dtype=torch.float64)
    B = torch.tensor(adata.obs["batch"].cat.codes.values, dtype=torch.long)
    perm = np.random.default_rng(0).permutation(len(Z))

    def w2(x, y):
        return float(ot.emd2(np.full(len(x), 1 / len(x)), np.full(len(y), 1 / len(y)), ot.dist(x.numpy(), y.numpy())))

    variants = [("cold", k) for k in (1, 2, 3, 5, 7, 10)] + [("warm", k) for k in (3, 5, 10, 20)]
    prev = {v: None for v in variants}
    prev_conv, rec = None, []
    for step in range(41):
        idx = torch.as_tensor(perm[step * 128:(step + 1) * 128]); z, b = Z[idx], B[idx]
        torch.manual_seed(1000 + step); conv = bary.batch_barycenter_support(z, b, n_iter=50)
        torch.manual_seed(2000 + step); conv2 = bary.batch_barycenter_support(z, b, n_iter=50)
        obj_conv = bary.barycenter_objective(conv, z, b)
        for v in variants:
            kind, it = v
            t = time.perf_counter()
            s = bary.batch_barycenter_support(z, b, n_iter=it, init=(prev[v] if kind == "warm" else None))
            ms = 1e3 * (time.perf_counter() - t)
            prev[v] = s
            if step >= 6:
                rec.append(dict(step=step, start=kind, n_iter=it, gap=w2(s, conv), noise=w2(prev_conv, conv),
                                rng_gap=w2(conv2, conv), obj_ratio=bary.barycenter_objective(s, z, b) / obj_conv, ms=ms))
        prev_conv = conv
    R = pd.DataFrame(rec)
    R.to_csv(a.out.replace(".csv", "_raw.csv"), index=False)
    S = R.groupby(["start", "n_iter"]).agg(minibatches=("step", "size"), median_gap=("gap", "median"), max_gap=("gap", "max"),
                                           median_noise=("noise", "median"), median_obj_ratio=("obj_ratio", "median"),
                                           max_obj_ratio=("obj_ratio", "max"), max_rng_gap=("rng_gap", "max"),
                                           ms_per_call=("ms", "mean")).reset_index()
    S["gap_over_noise"] = S.median_gap / S.median_noise
    S.round(5).to_csv(a.out, index=False)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
