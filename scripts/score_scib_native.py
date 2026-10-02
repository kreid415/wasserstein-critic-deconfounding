#!/usr/bin/env python
"""Score ONE latent with scIB's own entry point, scib.metrics.metrics(), against the unintegrated
prepped data. Replaces the project's hand-assembled composite (score_final_config.py), which
added raw scib.me.pcr with the wrong sign (code audit 2026-10-01, C1).

What runs (scIB, Luecken et al. 2022, Supplementary Table 2, embedding outputs):
  batch: PCR (pcr_comparison vs unintegrated PCA), batch ASW, graph iLISI, graph connectivity, kBET
  bio:   NMI, ARI (optimal-resolution clustering), cell-type ASW, isolated-label F1 and ASW,
         graph cLISI, cell-cycle conservation (where the organism's cell-cycle genes are present)
  not applicable to embeddings: HVG conservation; trajectory needs pseudotime annotations we lack.
This script writes RAW metric values only. The scIB overall score (per-metric min-max scaling
within a dataset across all runs, then 0.4*batch + 0.6*bio) is computed at analysis time by
scib_overall() below, because the scaling depends on the full set of runs.

Env: PREPPED (prep_scib_task.py output: X = scIB's normalised values, obsm X_pca, obs batch/celltype,
     uns organism / modality), NPZ (latent: z, batch, celltype, obs_names), TAG, META (json), OUT_CSV.
Per-task settings follow scIB (code review N13): cell-cycle conservation only for RNA tasks with a
real organism (scIB did not compute it for ATAC, and simulations have no cell-cycle genes);
trajectory conservation for the immune tasks (obs dpt_pseudotime, scIB's trajectory label); the
organism comes from the prepped file, never from a default. Subsampled fits (X3, X8) are scored
against the same cells of the unintegrated data (matched by obs_names).
"""
import os, json, time, numpy as np, pandas as pd, scanpy as sc, scib

BATCH_METRICS = ["PCR_batch", "ASW_label/batch", "iLISI", "graph_conn", "kBET"]
BIO_METRICS = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label", "isolated_label_F1",
               "isolated_label_silhouette", "cLISI", "cell_cycle_conservation"]


def scib_overall(df):
    """scIB overall score: min-max scale every metric within each dataset across ALL rows in df,
    average within batch / bio (NaN metrics skipped, as in scIB), then 0.4*batch + 0.6*bio."""
    out = df.copy()
    for m in BATCH_METRICS + BIO_METRICS:
        if m not in out:
            continue
        g = out.groupby("dataset")[m]
        lo, hi = g.transform("min"), g.transform("max")
        out[m + "_scaled"] = (out[m] - lo) / (hi - lo).replace(0, np.nan)
    b = [m + "_scaled" for m in BATCH_METRICS if m + "_scaled" in out]
    o = [m + "_scaled" for m in BIO_METRICS if m + "_scaled" in out]
    out["batch_score"] = out[b].mean(axis=1, skipna=True)
    out["bio_score"] = out[o].mean(axis=1, skipna=True)
    out["overall"] = 0.4 * out["batch_score"] + 0.6 * out["bio_score"]
    return out


def main():
    t0 = time.time()
    pre = sc.read_h5ad(os.environ["PREPPED"])
    bk, ck = pre.uns.get("batch_key", "batch"), pre.uns.get("celltype_key", "celltype")
    d = np.load(os.environ["NPZ"], allow_pickle=True)
    if "obs_names" in d and len(d["obs_names"]) != pre.n_obs:
        pre = pre[pd.Index(pre.obs_names).get_indexer(d["obs_names"].astype(str))].copy()
        sc.pp.pca(pre, n_comps=50, mask_var="highly_variable")   # PCR reference on the same cells
    assert (d["batch"].astype(str) == pre.obs[bk].astype(str).values).all(), "cell order mismatch"
    integ = pre.copy()
    integ.obsm["X_emb"] = d["z"].astype(np.float32)
    sc.pp.neighbors(integ, use_rep="X_emb")          # scIB pipeline: kNN graph on the embedding
    org = str(pre.uns["organism"])
    cell_cycle = pre.uns.get("modality") == "rna" and org in ("human", "mouse")
    trajectory = "dpt_pseudotime" in pre.obs and pre.obs["dpt_pseudotime"].notna().any()
    res = scib.metrics.metrics(
        pre, integ, batch_key=bk, label_key=ck, embed="X_emb", type_="embed",
        ari_=True, nmi_=True, silhouette_=True, pcr_=True, isolated_labels_f1_=True,
        isolated_labels_asw_=True, graph_conn_=True, kBET_=True, ilisi_=True, clisi_=True,
        cell_cycle_=cell_cycle, organism=(org if cell_cycle else "human"),
        hvg_score_=False, trajectory_=trajectory,
    )
    row = dict(tag=os.environ["TAG"], **json.loads(os.environ.get("META", "{}")))
    row.update({k: float(v) for k, v in res.iloc[:, 0].items()})
    row["score_seconds"] = round(time.time() - t0, 1)
    tmp = os.environ["OUT_CSV"] + ".tmp"
    pd.DataFrame([row]).to_csv(tmp, index=False)
    os.replace(tmp, os.environ["OUT_CSV"])
    print(f"[scored] {row['tag']} in {row['score_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
