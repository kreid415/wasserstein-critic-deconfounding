"""Fit ONE adversarial-LinearSCVI config and save the latent npz (scored later in wcd-kbet).

Env: SCVI_DS, ADV (none|discriminator|reference|barycenter), DCOEF, DISC_ITER, COND (1/0),
     SEED, MAXEP, BATCH, OUT (npz path), WCD_SRC.
"""
import os
import sys
import time

import numpy as np
import scanpy as sc

sys.path.insert(0, os.path.dirname(__file__))
from scvi_adversarial_plan import fit_adversarial_linearscvi  # noqa: E402

DS = os.environ.get("SCVI_DS", "immune")
ADV = os.environ.get("ADV", "none")
DCOEF = float(os.environ.get("DCOEF", "0"))
DISC_ITER = int(os.environ.get("DISC_ITER", "10"))
COND = os.environ.get("COND", "1") == "1"
SEED = int(os.environ.get("SEED", "0"))
MAXEP = int(os.environ.get("MAXEP", "239"))
BATCH = int(os.environ.get("BATCH", "512"))
OUT = os.environ["OUT"]

adata = sc.read_h5ad(f"results/scvi_single/{DS}_prepped.h5ad")
bk = adata.uns["batch_key"]
ck = adata.uns["celltype_key"]

t = time.time()
print(f"[{DS} adv={ADV} λ={DCOEF} cond={COND} s{SEED}] fitting {MAXEP}ep batch={BATCH} "
      f"disc_iter={DISC_ITER}...", flush=True)
Z = fit_adversarial_linearscvi(
    adata, bk, adversary=ADV, d_coef=DCOEF, disc_iter=DISC_ITER,
    reference_batch=0, n_latent=30, max_epochs=MAXEP, batch_size=BATCH,
    seed=SEED, conditioned=COND,
)
dt = time.time() - t

import scanpy as sc2
tmp = sc2.AnnData(Z)
sc2.pp.pca(tmp, n_comps=min(30, Z.shape[1] - 1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(
    OUT,
    z=Z.astype(np.float32),
    batch=adata.obs[bk].astype(str).to_numpy(dtype="U64"),
    celltype=adata.obs[ck].astype(str).to_numpy(dtype="U64"),
    X_pca=tmp.obsm["X_pca"].astype(np.float32),
)
print(f"[{DS} adv={ADV} λ={DCOEF} cond={COND} s{SEED}] wrote {OUT} in {dt:.0f}s (latent {Z.shape})",
      flush=True)
