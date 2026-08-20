import os, numpy as np, scanpy as sc
DS = os.environ.get("SCVI_DS", "pancreas")
adata = sc.read_h5ad(f"results/scvi_single/{DS}_prepped.h5ad")
batch_key = adata.uns["batch_key"]; celltype_key = adata.uns["celltype_key"]
print(f"[{DS}] n_obs={adata.n_obs} n_batches={adata.obs[batch_key].nunique()} "
      f"celltypes={adata.obs[celltype_key].nunique()} | counts layer: {'counts' in adata.layers}", flush=True)
import scvi, torch
scvi.settings.seed = 0
a = adata.copy()
a.X = a.layers["counts"].copy()          # scVI needs raw counts
scvi.model.SCVI.setup_anndata(a, batch_key=batch_key)   # BATCH-CONDITIONED
m = scvi.model.SCVI(a, n_latent=30)
m.train(max_epochs=200, early_stopping=True, enable_progress_bar=False)
Z = m.get_latent_representation()
print(f"[{DS}] scVI trained: latent {Z.shape} | cuda {torch.cuda.is_available()}", flush=True)
# PCA for the metric suite's X_pca slot (same as save_embedding stores)
import scanpy as sc
tmp = sc.AnnData(Z); sc.pp.pca(tmp, n_comps=min(50, Z.shape[1]-1))
out = f"results/scvi_single/{DS}_scvi.npz"
np.savez_compressed(out,
    z=Z.astype(np.float32),
    batch=a.obs[batch_key].astype(str).to_numpy(dtype="U64"),
    celltype=a.obs[celltype_key].astype(str).to_numpy(dtype="U64"),
    X_pca=tmp.obsm["X_pca"].astype(np.float32))
print(f"[{DS}] wrote {out}", flush=True)
