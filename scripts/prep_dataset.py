"""Prep ONE dataset via the harness load_task -> scoring-ready h5ad with counts + keys.
Writes to durable storage AND the harness-expected path. Env: PREP_DS, WCD_DATA (raw dir)."""
import os, sys, scanpy as sc
sys.path.insert(0, "src")
from wcd_vae.wcd.experiment import load_task
DS=os.environ["PREP_DS"]; DATA=os.environ["WCD_DATA"]; DUR=os.environ["DUR"]
adata, bk, ck, ref = load_task(DS, data_root=DATA)
adata.obs["batch"]=adata.obs[bk].astype(str).astype("category")
adata.obs["celltype"]=adata.obs[ck].astype(str).astype("category")
# the fit runner + gate read the KEY NAMES from uns and index obs[bk]; point them at the
# standardized columns so every dataset presents an identical (batch, celltype) interface.
adata.uns["batch_key"]="batch"
adata.uns["celltype_key"]="celltype"
out_dur=f"{DUR}/prepped_final/{DS}_prepped.h5ad"
out_loc=f"results/scvi_single/{DS}_prepped.h5ad"
adata.write_h5ad(out_dur)
os.makedirs("results/scvi_single", exist_ok=True)
import shutil; shutil.copy(out_dur, out_loc)
print(f"[prep] {DS}: n_obs={adata.n_obs} n_batches={adata.obs['batch'].nunique()} "
      f"celltypes={adata.obs['celltype'].nunique()} ref={ref} counts={'counts' in adata.layers} -> {out_dur}", flush=True)
