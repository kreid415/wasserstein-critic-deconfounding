#!/usr/bin/env python
"""Stock scVI exactly as a typical user runs it: SCVI(adata) with all defaults, train() with all
defaults, batch-conditioned on the counts layer. Reference line for the counts pilot.
Env: SCVI_DS, SEED, OUT (npz path)."""
import os, time, numpy as np, scanpy as sc, scvi

DS, SEED, OUT = os.environ["SCVI_DS"], int(os.environ["SEED"]), os.environ["OUT"]
adata = sc.read_h5ad(f"results/scvi_single/{DS}_prepped.h5ad")
a = adata.copy()
a.X = a.layers["counts"].copy()
scvi.settings.seed = SEED
scvi.model.SCVI.setup_anndata(a, batch_key=adata.uns["batch_key"])
m = scvi.model.SCVI(a)                       # defaults: n_latent=10, ZINB, 1 hidden layer of 128
t = time.time()
m.train(enable_progress_bar=False)           # defaults: heuristic max_epochs, batch 128, no ES
Z = m.get_latent_representation()
tmp = OUT + ".tmp.npz"
np.savez(tmp, z=Z.astype(np.float32), batch=adata.obs[adata.uns["batch_key"]].astype(str).values,
         celltype=adata.obs[adata.uns["celltype_key"]].astype(str).values)
os.replace(tmp, OUT)
print(f"[stock scVI {DS} s{SEED}] {m.history['elbo_train'].shape[0]} epochs, latent {Z.shape}, "
      f"{time.time()-t:.0f}s -> {OUT}", flush=True)
