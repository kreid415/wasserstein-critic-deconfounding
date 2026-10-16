"""PCR_batch reference ('before integration') as scib 1.1.7 computes it from the prepped files.
For full-data fits score_scib_native.py passes the prepped AnnData unchanged; it has obsm['X_pca'] but no
uns['pca'], so scib.metrics.pcr() takes its 'recompute PCA' branch on adata.X (ALL genes). For subsampled fits
(X3, X8) the scorer first runs sc.pp.pca(pre, n_comps=50, mask_var='highly_variable'), which writes uns['pca'],
so scib uses the HVG PCA. This script computes both references on the same full-data cells (read-only inputs)."""
import json, os, sys, time
import numpy as np, pandas as pd, scanpy as sc, scib
PREPPED, OUT = sys.argv[1], sys.argv[2]
tasks = sys.argv[3:]
rows = []
for t in tasks:
    pre = sc.read_h5ad(os.path.join(PREPPED, f"{t}__scib.h5ad"))
    assert "pca" not in pre.uns
    t0 = time.time()
    before_all = scib.metrics.pcr(pre, covariate="batch", recompute_pca=False, linreg_method="numpy")   # = scorer path
    t1 = time.time()
    p2 = pre.copy()
    sc.pp.pca(p2, n_comps=50, mask_var="highly_variable")           # = scorer path for subsampled fits
    before_hvg = scib.metrics.pcr(p2, covariate="batch", recompute_pca=False, linreg_method="numpy")
    # an uncorrected 10-d embedding (first 10 HVG PCs) scored against each reference
    emb = p2.obsm["X_pca"][:, :10].copy()
    after = scib.metrics.pc_regression(emb, pre.obs["batch"], n_comps=50, linreg_method="numpy")
    rows.append(dict(task=t, n_genes_X=int(pre.n_vars), n_hvg=int(pre.var["highly_variable"].sum()),
                     pcr_before_allgenes=float(before_all), pcr_before_hvg=float(before_hvg),
                     ratio_all_over_hvg=float(before_all / before_hvg),
                     pcr_after_hvgpca10=float(after),
                     PCR_batch_vs_allgenes=float(max(0.0, (before_all - after) / before_all)),
                     PCR_batch_vs_hvg=float(max(0.0, (before_hvg - after) / before_hvg)),
                     seconds_allgenes=round(t1 - t0, 1)))
    print(json.dumps(rows[-1]), flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
