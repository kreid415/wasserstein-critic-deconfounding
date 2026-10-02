"""Fit ONE adversarial-LinearSCVI config and save the latent npz (scored later in wcd-kbet).

Env: SCVI_DS, ADV (none|discriminator|reference|pooled|barycenter|{ref,pooled,bary}_sn|mmd|sinkhorn),
     DCOEF, DISC_ITER, COND (1/0), SCVI_MAX_KL (optional KL scale),
     SEED, MAXEP (int, or "auto" = scvi-tools default epoch heuristic), BATCH, NLAT (latent dim,
     default 30; scvi-tools default is 10), OUT (npz path), WCD_SRC.
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
ZSTD = ADV.endswith("_zs")          # arm suffix _zs = standardise z before the adversary
if ZSTD:
    ADV = ADV[:-3]
DCOEF = float(os.environ.get("DCOEF", "0"))
DISC_ITER = int(os.environ.get("DISC_ITER", "10"))
COND = os.environ.get("COND", "1") == "1"
SEED = int(os.environ.get("SEED", "0"))
_mx = os.environ.get("MAXEP", "239")
MAXEP = None if _mx == "auto" else int(_mx)   # None -> scvi heuristic min(400, 20000/n*400)
NLAT = int(os.environ.get("NLAT", "30"))
BATCH = int(os.environ.get("BATCH", "512"))
MODEL = os.environ.get("SCVI_MODEL", "LinearSCVI")  # LinearSCVI (linear dec) | SCVI (nonlinear dec)
MKL = os.environ.get("SCVI_MAX_KL")                  # scvi max_kl_weight (KL scale); None => scvi default 1.0
OUT = os.environ["OUT"]

adata = sc.read_h5ad(f"results/scvi_single/{DS}_prepped.h5ad")
bk = adata.uns["batch_key"]
ck = adata.uns["celltype_key"]
# reference batch: chosen at PREP time by the repo's entropy rule (select_reference_batch: most even
# cell-type coverage) and stored in uns, replacing the old alphabetical index 0. Refuse to guess.
if "reference_batch" not in adata.uns:
    raise KeyError(f"{DS}_prepped.h5ad has no uns['reference_batch']; re-run prep (load_task returns it)")
REF = str(adata.uns["reference_batch"])

t = time.time()
print(f"[{DS} adv={ADV} λ={DCOEF} cond={COND} s{SEED}] fitting {MAXEP}ep batch={BATCH} "
      f"disc_iter={DISC_ITER} nlat={NLAT} zstd={ZSTD} ref={REF}...", flush=True)
Z = fit_adversarial_linearscvi(
    adata, bk, adversary=ADV, d_coef=DCOEF, disc_iter=DISC_ITER,
    reference_batch=REF, zstd=ZSTD, n_latent=NLAT, max_epochs=MAXEP, batch_size=BATCH,
    seed=SEED, conditioned=COND, model_name=MODEL,
    max_kl_weight=(float(MKL) if MKL is not None else None),
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
