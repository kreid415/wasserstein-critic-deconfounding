import sys, json, os, pandas as pd
sys.path.insert(0,"src")
from wcd_vae.wcd.experiment import load_task, full_metric_suite
DS=os.environ["SEURAT_DS"]; D=os.environ["SEURAT_DIR"]
reg=json.load(open("configs/dataset_registry.json"))
adata,bk,ck,_=load_task(DS,data_root=os.environ["WCD_DATA"],registry=reg)
emb=pd.read_csv(f"{D}/seurat_emb.csv",index_col=0).to_numpy().astype("float32")
assert emb.shape[0]==adata.n_obs,(emb.shape,adata.n_obs)
adata.obsm["X_emb"]=emb
m=full_metric_suite(adata,bk,ck,embed_key="X_emb")
json.dump({k:(float(v) if v==v else None) for k,v in m.items()}, open(f"{D}/seurat_metrics.json","w"), indent=2)
print(f"[{DS}] scIB metrics written", flush=True)
