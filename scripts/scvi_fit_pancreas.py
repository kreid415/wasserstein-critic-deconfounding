import os, numpy as np, scanpy as sc
DS = os.environ.get("SCVI_DS", "pancreas")
SEED = int(os.environ.get("SCVI_SEED", "0"))
# MODEL selects the scvi-tools model: SCVI (nonlinear decoder) or LinearSCVI (linear
# decoder = the interpretable LDVAE architecture the paper is about; off-the-shelf
# reference for the adversarial barycenter and an implementation control on our own LDVAE).
MODEL = os.environ.get("SCVI_MODEL", "SCVI")
TAG = {"SCVI": "scvi", "LinearSCVI": "linearscvi"}[MODEL]
adata = sc.read_h5ad(f"results/scvi_single/{DS}_prepped.h5ad")
batch_key = adata.uns["batch_key"]; celltype_key = adata.uns["celltype_key"]
print(f"[{DS} {MODEL} s{SEED}] n_obs={adata.n_obs} n_batches={adata.obs[batch_key].nunique()} "
      f"celltypes={adata.obs[celltype_key].nunique()} | counts layer: {'counts' in adata.layers}", flush=True)
import scvi, torch
scvi.settings.seed = SEED
a = adata.copy()
a.X = a.layers["counts"].copy()          # scVI/LinearSCVI need raw counts
Model = getattr(scvi.model, MODEL)
Model.setup_anndata(a, batch_key=batch_key)   # BATCH-CONDITIONED
m = Model(a, n_latent=30)
# batch_size / max_epochs env-configurable so the scVI reference can be regenerated at a matched
# batch (defaults preserve the original run: 128, 200 epochs, ES on). The wcd LinearSCVI profile
# must use the SAME batch for the reproduction gate to stay a like-for-like comparison.
_BATCH = int(os.environ.get("SCVI_BATCH", "128"))
_EPOCHS = int(os.environ.get("SCVI_MAX_EPOCHS", "200"))
_ES = os.environ.get("SCVI_ES", "1") == "1"
print(f"[{DS} {MODEL} s{SEED}] train: batch={_BATCH} max_epochs={_EPOCHS} early_stopping={_ES}", flush=True)
m.train(max_epochs=_EPOCHS, batch_size=_BATCH, early_stopping=_ES, enable_progress_bar=False)
Z = m.get_latent_representation()
print(f"[{DS} {MODEL} s{SEED}] trained: latent {Z.shape} | cuda {torch.cuda.is_available()}", flush=True)
# PCA for the metric suite's X_pca slot (same as save_embedding stores)
import scanpy as sc
tmp = sc.AnnData(Z); sc.pp.pca(tmp, n_comps=min(50, Z.shape[1]-1))
# seed 0 keeps the original bare filename so the pre-existing pancreas scVI latent is unchanged
_suf = "" if SEED == 0 else f"_s{SEED}"
out = f"results/scvi_single/{DS}_{TAG}{_suf}.npz"
np.savez_compressed(out,
    z=Z.astype(np.float32),
    batch=a.obs[batch_key].astype(str).to_numpy(dtype="U64"),
    celltype=a.obs[celltype_key].astype(str).to_numpy(dtype="U64"),
    X_pca=tmp.obsm["X_pca"].astype(np.float32))
print(f"[{DS} {MODEL} s{SEED}] wrote {out}", flush=True)
