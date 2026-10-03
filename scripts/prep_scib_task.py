#!/usr/bin/env python
"""Prepare one scIB integration task (Luecken et al. 2022) for the benchmark. 2026-10-02.

Tasks, files and keys are scIB's: batch/label keys from scIB's reproduce_paper.yaml
(theislab/scib-pipeline configs) for the RNA tasks; the ATAC gene-activity tasks use the
figshare files' batchname_all / final_cell_label. Cells are kept exactly as distributed (no extra
QC filtering). Feature space follows scIB's 'hvg' setting for RNA (scib.pp.hvg_batch, 2,000
genes, flavor cell_ranger, n_bins 20, on scIB's normalised X) and 'full_feature' for ATAC
(scIB scored the gene-activity tasks on full features only, data/metrics_atac.csv).

Two count modes (--counts):
  scib   the 'counts' layer exactly as scIB distributes it. Some batches are not counts; their
         status is recorded per batch in uns['count_status'] (scIB itself fed these to scVI).
  valid  integer counts only: exact repairs where the source defines them, batches without
         recoverable counts dropped. Repairs, with the primary source checked (2026-10-02):
           pancreas celseq, celseq2  UMI-collision correction x' = -K ln(1 - x/K) inverted,
                                     x = K (1 - exp(-x'/K)), K = 256 (>= 99.99% exact integers)
           pancreas fluidigmc1       RSEM expected counts (GEO GSE86469 file
                                     '...RSEM.raw.expected.counts.csv.gz') -> rounded
           pancreas smarter          RPKM only (GEO GSE81608 holds only '..._rpkm.txt.gz') -> dropped
           immune(_hum_mou) Villani  TPM (GEO GSE94820 matrix: every cell sums to 1e6) -> dropped
           lung 10x batches          normalised, not counts; HCA's processed h5ads for the same cells
                                     are log1p(CP10k) of non-integer values and GEO GSE130148 counts
                                     cover only the Drop-seq cells -> dropped (lung 'valid' = the 4 Drop-seq batches B1-B4, 9,701 of 32,472 cells)

Output h5ad (all genes kept so the scorer sees the full unintegrated data):
  X                     scIB's normalised values (as distributed)
  layers['counts']      counts per the mode
  var['highly_variable'] training features (scIB HVGs for RNA; all features for ATAC)
  obs batch, celltype   standardised copies of the scIB keys (originals kept)
  obsm['X_pca']         50-PC PCA of X on the training features (PCR reference for scoring)
  uns                   task, mode, keys, count_status, repairs, hvg settings, source md5

Usage: python scripts/prep_scib_task.py --task immune --counts scib --raw-dir DIR --out-dir DIR
"""
import argparse
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import scipy.sparse as sp

TASKS = {
    "pancreas":       dict(file="human_pancreas_norm_complexBatch.h5ad", batch="tech", label="celltype",
                           modality="rna", organism="human",
                           repairs={"celseq": "invert_umi_256", "celseq2": "invert_umi_256",
                                    "fluidigmc1": "round_expected", "smarter": "drop_rpkm"}),
    "lung":           dict(file="Lung_atlas_public.h5ad", batch="batch", label="cell_type",
                           modality="rna", organism="human",
                           repairs={b: "drop_normalised" for b in
                                    ["1", "2", "3", "4", "5", "6", "A1", "A2", "A3", "A4", "A5", "A6"]}),
    "immune":         dict(file="Immune_ALL_human.h5ad", batch="batch", label="final_annotation",
                           modality="rna", organism="human", repairs={"Villani": "drop_tpm"}),
    "immune_hum_mou": dict(file="Immune_ALL_hum_mou.h5ad", batch="batch", label="final_annotation",
                           modality="rna", organism="human", repairs={"Villani": "drop_tpm"}),
    "sim1":           dict(file="sim1_1_norm.h5ad", batch="Batch", label="Group", modality="rna",
                           organism="none", repairs={}),
    "sim2":           dict(file="sim2_norm.h5ad", batch="SubBatch", label="Group", modality="rna",
                           organism="none", repairs={}),
    "atac_small":     dict(file="small_atac_gene_activity.h5ad", batch="batchname_all",
                           label="final_cell_label", modality="atac", organism="mouse", repairs={}),
    "atac_large":     dict(file="large_atac_gene_activity.h5ad", batch="batchname_all",
                           label="final_cell_label", modality="atac", organism="mouse", repairs={}),
}
INT_TOL = 1e-3


def _md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _frac_int(v):
    return float(np.mean(np.abs(v - np.round(v)) < INT_TOL)) if v.size else 1.0


def count_status(C, batches):
    out = {}
    for b in pd.unique(batches):
        d = C[np.where(batches == b)[0]].data.astype(np.float64)
        out[str(b)] = dict(cells=int((batches == b).sum()), frac_integer_values=round(_frac_int(d), 5),
                           min_nonzero=float(d.min()) if d.size else None, max=float(d.max()) if d.size else None)
    return out


def apply_repairs(C, batches, repairs):
    """Return (C_repaired, keep_mask, log). C: csr float64."""
    C = C.tocsr(copy=True).astype(np.float64)
    keep = np.ones(C.shape[0], dtype=bool)
    log = {}
    for b, rule in repairs.items():
        rows = np.where(batches == b)[0]
        if rows.size == 0:
            raise ValueError(f"repair rule for batch {b!r} but no such batch")
        if rule.startswith("drop_"):
            keep[rows] = False
            log[b] = dict(rule=rule, cells=int(rows.size))
            continue
        for i in rows:
            s, e = C.indptr[i], C.indptr[i + 1]
            v = C.data[s:e]
            if rule == "invert_umi_256":
                v = 256.0 * (1.0 - np.exp(-v / 256.0))
            elif rule != "round_expected":
                raise ValueError(rule)
            C.data[s:e] = v
        d = C[rows].data
        fi = _frac_int(d)
        if rule == "invert_umi_256" and fi < 0.999:
            raise ValueError(f"{b}: UMI-collision inversion gives only {fi:.4f} integer values")
        for i in rows:
            s, e = C.indptr[i], C.indptr[i + 1]
            C.data[s:e] = np.rint(C.data[s:e])
        log[b] = dict(rule=rule, cells=int(rows.size), frac_integer_before_rounding=round(fi, 5))
    C.eliminate_zeros()
    return C, keep, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--counts", required=True, choices=["scib", "valid"])
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-hvg", type=int, default=2000)
    args = ap.parse_args()
    import anndata as ad
    import scanpy as sc
    import scib

    t = TASKS[args.task]
    src = os.path.join(args.raw_dir, t["file"])
    a = ad.read_h5ad(src)
    batches = a.obs[t["batch"]].astype(str).values
    if "counts" not in a.layers:
        raise ValueError(f"{src} has no counts layer")
    C = a.layers["counts"]
    C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
    status = count_status(C, batches)
    repairs_log = {}
    if args.counts == "valid":
        C, keep, repairs_log = apply_repairs(C, batches, t["repairs"])
        a = a[keep].copy()
        C = C[keep]
        batches = batches[keep]
        bad = {b: s for b, s in count_status(C, batches).items() if s["frac_integer_values"] < 1.0}
        if bad:
            raise ValueError(f"non-integer counts remain after repair: {bad}")
    a.layers["counts"] = C.astype(np.float32)
    a.obs["batch"] = pd.Categorical(batches)
    a.obs["celltype"] = pd.Categorical(a.obs[t["label"]].astype(str).values)

    # training features: scIB HVGs (on scIB's normalised X) for RNA, all features for ATAC
    if t["modality"] == "rna":
        Xn = a.copy()
        Xn.X = Xn.X.astype(np.float32)
        hvg = scib.pp.hvg_batch(Xn, batch_key="batch", target_genes=args.n_hvg, flavor="cell_ranger",
                                n_bins=20, adataOut=False)
        a.var["highly_variable"] = a.var_names.isin(hvg)
        hvg_info = dict(method="scib.pp.hvg_batch", target_genes=args.n_hvg, flavor="cell_ranger", n_bins=20,
                        n_selected=int(a.var["highly_variable"].sum()))
    else:
        a.var["highly_variable"] = True
        hvg_info = dict(method="all features (scIB full_feature for ATAC gene activity)",
                        n_selected=int(a.n_vars))
    sub = a[:, a.var["highly_variable"].values]
    Xs = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)
    a.obsm["X_pca"] = sc.pp.pca(Xs.astype(np.float32), n_comps=50, random_state=0)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    a.uns.update(dict(
        task=args.task, counts_mode=args.counts, batch_key="batch", celltype_key="celltype",
        scib_batch_key=t["batch"], scib_label_key=t["label"], modality=t["modality"], organism=t["organism"],
        count_status_as_distributed=json.dumps(status), repairs=json.dumps(repairs_log),
        hvg=json.dumps(hvg_info), source_file=t["file"], source_md5=_md5(src), prep_code_sha=sha,
    ))
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.task}__{args.counts}.h5ad")
    a.write_h5ad(out, compression="gzip")
    nonint = {b: s["frac_integer_values"] for b, s in status.items() if s["frac_integer_values"] < 1.0}
    print(json.dumps(dict(task=args.task, mode=args.counts, cells=int(a.n_obs), batches=int(a.obs["batch"].nunique()),
                          labels=int(a.obs["celltype"].nunique()), features=hvg_info["n_selected"],
                          non_integer_batches_as_distributed=nonint, repairs=repairs_log, out=out)), flush=True)


if __name__ == "__main__":
    main()
