"""Per-cell-type fraction of lung cells removed by prep_data's filter_cells(min_genes=300) (read-only source)."""
import h5py, numpy as np, pandas as pd, json
D = "/home/kendall/experiment_data/wasserstein-critic-deconfounding/data_final/"
with h5py.File(D + "Lung_atlas_public.h5ad", "r") as h:
    o = h["obs"]; g = o["cell_type"]
    ct = g["categories"][:].astype(str)[g["codes"][:]] if isinstance(g, h5py.Group) else np.array(
        [c.decode() if isinstance(c, bytes) else str(c) for c in o["__categories"]["cell_type"][:]])[g[:]]
    ng = np.diff(h["layers"]["counts"]["indptr"][:])
df = pd.DataFrame({"ct": ct, "keep": ng >= 300})
s = df.groupby("ct").agg(n=("keep", "size"), dropped=("keep", lambda x: int((~x).sum())))
s["drop_frac"] = (s.dropped / s.n).round(3); s = s.sort_values("drop_frac", ascending=False)
json.dump({"lung": dict(total_dropped=int(s.dropped.sum()), n=int(s.n.sum()), top=s.head(6).reset_index().to_dict("records"))},
          open("checks/min_genes_drop_lung.json", "w"), indent=1)
print(int(s.dropped.sum()), int(s.n.sum())); print(s.head(6).to_string())
