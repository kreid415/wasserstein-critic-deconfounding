"""Read-only audit of source h5ads: (a) which cells/cell types the RNA prep's min_genes=300 filter drops
(lung, immune_hum_mou, compared with the scIB task sizes); (b) where immune_hum_mou's non-integer counts
live (which obs columns could select them for rounding)."""
import h5py, numpy as np, pandas as pd, json, sys
D = "/home/kendall/experiment_data/wasserstein-critic-deconfounding/data_final/"
def obs_col(h, k):
    g = h["obs"][k]
    if isinstance(g, h5py.Group):
        cats = g["categories"][:].astype(str); return cats[g["codes"][:]]
    return g[:].astype(str)
def csr_nnz_per_row(g):
    indptr = g["indptr"][:]; return np.diff(indptr)
out = {}
for ds, f, ck, bk in [("lung","Lung_atlas_public.h5ad","cell_type","protocol"),
                      ("immune_hum_mou","Immune_ALL_hum_mou.h5ad","final_annotation","species"),
                      ("immune","Immune_ALL_human.h5ad","final_annotation","chemistry"),
                      ("pancreas","human_pancreas_norm_complexBatch.h5ad","celltype","tech")]:
    with h5py.File(D+f, "r") as h:
        obs_keys = list(h["obs"].keys())
        g = h["layers"]["counts"] if "counts" in h["layers"] else h["X"]
        enc = g.attrs.get("encoding-type", "dense")
        ct = obs_col(h, ck); b = obs_col(h, bk)
        if isinstance(g, h5py.Group) and "indptr" in g:
            if enc == "csr_matrix":
                ngenes = csr_nnz_per_row(g)
                data = g["data"]; indptr = g["indptr"][:]
                # non-integer per cell (stream in chunks)
                bad = np.zeros(len(indptr)-1, bool)
                step = 20_000_000
                nd = data.shape[0]
                rowid = None
                for s in range(0, nd, step):
                    x = data[s:s+step]
                    nz = np.flatnonzero(np.abs(x - np.round(x)) > 1e-6) + s
                    if nz.size:
                        bad[np.searchsorted(indptr, nz, side="right") - 1] = True
            else:
                raise RuntimeError(f"{ds}: counts encoding {enc}")
        else:
            A = g[:]; ngenes = (A > 0).sum(1); bad = (np.abs(A - np.round(A)) > 1e-6).any(1)
    keep = ngenes >= 300
    df = pd.DataFrame({"ct": ct, "b": b, "keep": keep, "bad": bad})
    drop_by_ct = (1 - df.groupby("ct")["keep"].mean()).sort_values(ascending=False)
    rec = dict(n_cells=int(len(df)), n_kept_min_genes300=int(keep.sum()),
               top_dropped_celltypes={k: round(float(v), 3) for k, v in drop_by_ct.head(6).items()},
               n_ct_fully_dropped=int((drop_by_ct >= 0.999).sum()),
               nonint_cells=int(bad.sum()), nonint_by_batchkey=df[df.bad].b.value_counts().to_dict())
    for alt in ("batch", "chemistry", "study", "tech", "protocol"):
        if alt in obs_keys and alt != bk:
            a = obs_col(h5py.File(D+f, "r"), alt)
            rec[f"nonint_by_{alt}"] = pd.Series(a[bad]).value_counts().head(8).to_dict()
            rec[f"n_levels_{alt}"] = int(len(np.unique(a)))
    out[ds] = rec
    print(ds, json.dumps(rec), flush=True)
json.dump(out, open("checks/source_audit.json", "w"), indent=1)
