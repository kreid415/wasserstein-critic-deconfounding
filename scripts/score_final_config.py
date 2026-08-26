#!/usr/bin/env python
"""Score ONE final-sweep latent npz through full_metric_suite -> one CSV row.

Identical scoring path for adversarial AND baseline configs (both write z+batch+celltype npz;
both score here). Resume-safe: skips if the row already exists in the output CSV.

scIB convention (Luecken et al., matching scripts/compare_methods_stratified.py — the paper analysis):
  batch (SB) = mean(ilisi, asw_batch, graph_conn, pcr, kbet)                  all +1
  bio   (SO) = mean(clisi(-1), ari, nmi, asw_celltype, isolated_asw, isolated_f1)
  scIB       = 0.4*batch + 0.6*bio
RAW per-category mean (NOT cross-candidate min-max scaled): gives an ABSOLUTE score comparable
across lambda within a dataset, which the frontier-dominance analysis requires. (The scaled
_scib_overall in hyperparameter.py is for SELECTION only and is deliberately not used here.)

Env: NPZ (latent path), TAG, DATASET, ADV, LAM, DEC, SEED, OUT_CSV.
"""
import os, sys, csv, numpy as np, pandas as pd, scanpy as sc
sys.path.insert(0, "src")
from wcd_vae.wcd.experiment import full_metric_suite

SB = {"ilisi": 1, "asw_batch": 1, "graph_conn": 1, "pcr": 1, "kbet": 1}
SO = {"clisi": -1, "ari": 1, "nmi": 1, "asw_celltype": 1, "isolated_asw": 1, "isolated_f1": 1}

def catmean(m, spec):
    v = [m[k] * (1 if s > 0 else -1) for k, s in spec.items() if k in m and m[k] == m[k]]
    return float(np.mean(v)) if v else float("nan")

def main():
    npz = os.environ["NPZ"]; tag = os.environ["TAG"]; out_csv = os.environ["OUT_CSV"]
    # resume: skip if tag already scored
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv, usecols=["tag"])["tag"]) if os.path.getsize(out_csv) else set()
        if tag in done:
            print(f"[skip] {tag} already scored"); return
    d = np.load(npz, allow_pickle=True)
    ad = sc.AnnData(np.zeros((len(d["batch"]), 1), dtype=np.float32))
    ad.obsm["X_emb"] = d["z"].astype(np.float32)
    ad.obs["batch"] = pd.Categorical(d["batch"].astype(str))
    ad.obs["celltype"] = pd.Categorical(d["celltype"].astype(str))
    m = full_metric_suite(ad, "batch", "celltype", embed_key="X_emb")
    batch = catmean(m, SB); bio = catmean(m, SO)
    scib = 0.4 * batch + 0.6 * bio if (batch == batch and bio == bio) else float("nan")
    row = dict(tag=tag, dataset=os.environ["DATASET"], adv=os.environ["ADV"],
               lam=float(os.environ["LAM"]), dec=os.environ["DEC"], seed=int(os.environ["SEED"]),
               batch=batch, bio=bio, scIB=scib,
               **{k: (float(v) if v == v else "") for k, v in m.items()})
    hdr = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    # append one row; header written once. (concurrent appends: each scorer writes ONE line,
    # and CSV single-line appends under the OS append flag are atomic for lines < PIPE_BUF)
    with open(out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if hdr: w.writeheader()
        w.writerow(row)
    print(f"[scored] {tag}: scIB={scib:.4f} batch={batch:.4f} bio={bio:.4f}", flush=True)

if __name__ == "__main__":
    main()
