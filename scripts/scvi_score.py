import os, sys, numpy as np, scanpy as sc
sys.path.insert(0, "src")
from wcd_vae.wcd.experiment import full_metric_suite
DS = os.environ.get("SCVI_DS", "pancreas")
d = np.load(f"results/scvi_single/{DS}_scvi.npz", allow_pickle=True)
ad = sc.AnnData(np.zeros((len(d["batch"]),1), dtype=np.float32))
ad.obsm["X_emb"] = d["z"].astype(np.float32)
import pandas as pd
ad.obs["batch"] = pd.Categorical(d["batch"].astype(str))
ad.obs["celltype"] = pd.Categorical(d["celltype"].astype(str))
m = full_metric_suite(ad, "batch", "celltype", embed_key="X_emb")
import json
json.dump({k: (float(v) if v==v else None) for k,v in m.items()},
          open(f"results/scvi_single/{DS}_scvi_metrics.json","w"), indent=2)
print(f"[{DS}] metrics:", {k: round(v,4) for k,v in m.items() if v==v}, flush=True)
