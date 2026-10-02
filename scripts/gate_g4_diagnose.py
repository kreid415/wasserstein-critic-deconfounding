#!/usr/bin/env python
"""Diagnose cross-host scoring differences (G4): rebuild the scorer's kNN graph (scripts/score_scib_native.py:
sc.pp.neighbors(use_rep='X_emb') with scanpy defaults) for one latent and save it, so two hosts' graphs can be compared
entry by entry. Run once per environment variant (numba reads NUMBA_CPU_NAME at import, so one process per variant).

Usage: PREPPED=... NPZ=... OUT=graph_<variant>.npz python scripts/gate_g4_diagnose.py
       python scripts/gate_g4_diagnose.py --compare A.npz B.npz
"""
import hashlib
import json
import os
import sys

import numpy as np


def build():
    import scanpy as sc
    import numba
    pre = sc.read_h5ad(os.environ["PREPPED"])
    d = np.load(os.environ["NPZ"], allow_pickle=True)
    assert (d["batch"].astype(str) == pre.obs[pre.uns.get("batch_key", "batch")].astype(str).values).all()
    integ = pre.copy()
    integ.obsm["X_emb"] = d["z"].astype(np.float32)
    sc.pp.neighbors(integ, use_rep="X_emb")
    D, C = integ.obsp["distances"].tocsr(), integ.obsp["connectivities"].tocsr()
    D.sort_indices(); C.sort_indices()
    meta = dict(numba=numba.__version__, numba_cpu_name=os.environ.get("NUMBA_CPU_NAME", "host"),
                llvm_cpu=numba.config.CPU_NAME or "host", host=os.uname().nodename,
                d_sha=hashlib.sha1(D.data.tobytes() + D.indices.tobytes() + D.indptr.tobytes()).hexdigest(),
                c_sha=hashlib.sha1(C.data.tobytes() + C.indices.tobytes() + C.indptr.tobytes()).hexdigest())
    np.savez_compressed(os.environ["OUT"], d_data=D.data, d_indices=D.indices, d_indptr=D.indptr,
                        c_data=C.data, c_indices=C.indices, c_indptr=C.indptr, meta=json.dumps(meta))
    print(json.dumps(meta))


def compare(fa, fb):
    a, b = np.load(fa), np.load(fb)
    ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    n = len(a["d_indptr"]) - 1
    same_rows = 0
    for i in range(n):
        ra = a["d_indices"][a["d_indptr"][i]:a["d_indptr"][i + 1]]
        rb = b["d_indices"][b["d_indptr"][i]:b["d_indptr"][i + 1]]
        same_rows += np.array_equal(ra, rb)
    same_struct = np.array_equal(a["d_indices"], b["d_indices"]) and np.array_equal(a["d_indptr"], b["d_indptr"])
    out = dict(a=ma, b=mb, cells=n, knn_rows_identical=int(same_rows), knn_identical=bool(same_struct),
               dist_max_abs_diff=float(np.abs(a["d_data"] - b["d_data"]).max()) if same_struct else None,
               conn_identical=bool(ma["c_sha"] == mb["c_sha"]), dist_identical=bool(ma["d_sha"] == mb["d_sha"]))
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        build()
