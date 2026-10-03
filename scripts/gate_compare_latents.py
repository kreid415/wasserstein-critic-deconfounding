#!/usr/bin/env python
"""Cross-host gate G3, latent part: agreement of the same config fitted on two hosts (same seed => same init and
minibatch order). Per config: per-dimension Pearson r between the two latents (latent dims are matched by index:
identical init), max |dz|, relative Frobenius difference, and the k-NN overlap (mean over cells of
|N_k^A(i) & N_k^B(i)| / k, exact Euclidean k-NN without the cell itself).
Reference scale (same file): the k-NN overlap and per-dim |r| between two DIFFERENT seeds of the same (arm, cond) on the
local host, i.e. how different two legitimate replicates are.

Usage: python scripts/gate_compare_latents.py --local DIR --remote DIR --order queue_order.txt --manifest M.tsv
                                              --k 15 --out-csv latent_agreement.csv [--out-json summary.json]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def load(d, tag):
    f = os.path.join(d, f"{tag}.npz")
    z = np.load(f)
    out = dict(z=np.asarray(z["z"], dtype=np.float64), obs=z["obs_names"].astype(str), batch=z["batch"].astype(str),
               cfg=json.loads(str(z["config"])))
    if not np.isfinite(out["z"]).all():
        raise ValueError(f"non-finite latent {f}")
    return out


def knn(z, k):
    _, idx = cKDTree(z).query(z, k=k + 1)
    return idx[:, 1:]                       # drop self (distance 0; duplicates would be resolved arbitrarily)


def knn_overlap(ia, ib):
    k = ia.shape[1]
    return np.array([len(np.intersect1d(a, b, assume_unique=True)) for a, b in zip(ia, ib)]) / k


def per_dim_r(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    if (den == 0).any():
        raise ValueError("constant latent dimension")
    return (a * b).sum(0) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--order", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json")
    a = ap.parse_args()
    tags = [l.strip() for l in open(a.order) if l.strip()]
    M = pd.read_csv(a.manifest, sep="\t", dtype=str, keep_default_na=False).set_index("tag")
    rows, knn_local = [], {}
    for t in tags:
        L, R = load(a.local, t), load(a.remote, t)
        if not (np.array_equal(L["obs"], R["obs"]) and np.array_equal(L["batch"], R["batch"])):
            raise ValueError(f"{t}: cell order differs between hosts")
        if L["cfg"]["row"] != R["cfg"]["row"]:
            raise ValueError(f"{t}: manifest rows differ between hosts")
        r = per_dim_r(L["z"], R["z"])
        il, ir = knn(L["z"], a.k), knn(R["z"], a.k)
        knn_local[t] = il
        ov = knn_overlap(il, ir)
        d = R["z"] - L["z"]
        rows.append(dict(tag=t, arm=M.loc[t, "arm"], cond=int(M.loc[t, "cond"]), seed=int(M.loc[t, "seed"]),
                         bitwise_equal=bool(np.array_equal(L["z"], R["z"])),
                         r_min=float(r.min()), r_median=float(np.median(r)), r_mean=float(r.mean()),
                         max_abs_dz=float(np.abs(d).max()), rel_frobenius=float(np.linalg.norm(d) / np.linalg.norm(L["z"])),
                         knn_overlap_mean=float(ov.mean()), knn_overlap_p05=float(np.quantile(ov, 0.05)),
                         gpu_local=L["cfg"]["gpu"], gpu_remote=R["cfg"]["gpu"], torch_local=L["cfg"]["torch"],
                         torch_remote=R["cfg"]["torch"], fit_s_local=L["cfg"]["fit_seconds"], fit_s_remote=R["cfg"]["fit_seconds"]))
    D = pd.DataFrame(rows)
    # replicate scale: consecutive local seeds within (arm, cond)
    ref = []
    for (arm, cond), g in D.groupby(["arm", "cond"]):
        ts = list(g.sort_values("seed").tag)
        for t1, t2 in zip(ts[:-1], ts[1:]):
            z1, z2 = load(a.local, t1)["z"], load(a.local, t2)["z"]
            ref.append(dict(arm=arm, cond=cond, pair=f"{t1}~{t2}",
                            knn_overlap_mean=float(knn_overlap(knn_local[t1], knn_local[t2]).mean()),
                            abs_r_median=float(np.median(np.abs(per_dim_r(z1, z2))))))
    Rf = pd.DataFrame(ref)
    D.to_csv(a.out_csv, index=False)
    Rf.to_csv(a.out_csv.replace(".csv", "_seed_reference.csv"), index=False)
    summ = dict(n=len(D), k=a.k, bitwise_equal=int(D.bitwise_equal.sum()),
                r_min_over_configs=float(D.r_min.min()), r_median_over_configs=float(D.r_median.median()),
                knn_overlap_mean_over_configs=float(D.knn_overlap_mean.mean()),
                knn_overlap_min_over_configs=float(D.knn_overlap_mean.min()),
                by_arm=D.groupby("arm").agg(r_min=("r_min", "min"), knn=("knn_overlap_mean", "mean")).round(4).to_dict("index"),
                seed_reference_knn_overlap_mean=float(Rf.knn_overlap_mean.mean()) if len(Rf) else None,
                seed_reference_abs_r_median=float(Rf.abs_r_median.median()) if len(Rf) else None)
    if a.out_json:
        json.dump(summ, open(a.out_json, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
