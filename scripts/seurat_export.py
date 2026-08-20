import sys, json, os, numpy as np, scipy.io, scipy.sparse as sp, pandas as pd
sys.path.insert(0,"src")
from wcd_vae.wcd.experiment import load_task
DS=os.environ["SEURAT_DS"]; OUT=os.environ["SEURAT_DIR"]; os.makedirs(OUT,exist_ok=True)
reg=json.load(open("configs/dataset_registry.json"))
adata,bk,ck,_=load_task(DS,data_root=os.environ["WCD_DATA"],registry=reg)
cnt=adata.layers["counts"]; cnt=cnt if sp.issparse(cnt) else sp.csr_matrix(cnt)
scipy.io.mmwrite(f"{OUT}/counts.mtx", cnt.T.tocoo())   # genes x cells
pd.DataFrame({"cell":np.arange(adata.n_obs),"batch":adata.obs[bk].astype(str).to_numpy(),
              "celltype":adata.obs[ck].astype(str).to_numpy()}).to_csv(f"{OUT}/cells.csv",index=False)
pd.DataFrame({"gene":[f"g{i}" for i in range(adata.shape[1])]}).to_csv(f"{OUT}/genes.csv",index=False)
print(f"[{DS}] exported {adata.n_obs} cells x {adata.shape[1]} genes, {adata.obs[bk].nunique()} batches", flush=True)
