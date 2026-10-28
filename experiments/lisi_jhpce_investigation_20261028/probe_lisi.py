"""LISI-only probe: scores graph iLISI / cLISI of stored latents through the SAME scib 1.1.7 code path as
scripts/score_scib_native.py (sc.pp.neighbors on X_emb, then scib.metrics.ilisi_graph / clisi_graph with
type_='embed', subsample=50.0, scale=True, n_cores=1, the arguments scib.metrics.metrics passes) and records what
the compiled knn_graph helper wrote, so that run-to-run differences can be traced to a step.
Usage: probe_lisi.py OUT.json LATENT.npz [LATENT.npz ...]   (env: the thread settings of run_stage.thread_env(1))
Nothing here changes a score: it calls the scib functions unchanged and only observes the helper's output files."""
import hashlib, json, os, subprocess, sys, tempfile
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, scib
from scib.metrics import lisi as L

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    if os.environ.get(k) != "1":
        raise SystemExit(f"{k} must be 1 (run_stage.thread_env(1)); got {os.environ.get(k)!r}")

HELPER = {}          # filled by the observing wrapper around subprocess.run
_real_run = subprocess.run
def _observed_run(args, *a, **kw):
    r = _real_run(args, *a, **kw)
    prefix = args[2]
    ind = open(prefix + "_indices_0.txt", "rb").read()
    dis = open(prefix + "_distances_0.txt", "rb").read()
    HELPER.update(returncode=r.returncode, argv_tail=args[3:], mtx_sha=hashlib.sha256(open(args[1], "rb").read()).hexdigest(),
                  indices_sha=hashlib.sha256(ind).hexdigest(), distances_sha=hashlib.sha256(dis).hexdigest(),
                  rows_kept=ind.count(b"\n"), first_rows=[int(x.split(b",")[0]) for x in ind.split(b"\n")[:8] if x])
    if r.returncode != 0:
        raise RuntimeError(f"knn_graph helper exit {r.returncode}")   # scib ignores a non-zero exit; the probe must not
    return r
L.subprocess.run = _observed_run

def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def one(path):
    d = np.load(path, allow_pickle=True)
    a = ad.AnnData(X=np.zeros((len(d["z"]), 1), dtype=np.float32), obs=pd.DataFrame(
        {"batch": d["batch"].astype(str), "celltype": d["celltype"].astype(str)}, index=d["obs_names"].astype(str)))
    a.obsm["X_emb"] = d["z"].astype(np.float32)
    out = {"latent": os.path.basename(path), "n_cells": a.n_obs}
    sc.pp.neighbors(a, use_rep="X_emb")                      # the scorer's own graph (score_scib_native.py line 207)
    out["graph0_conn_sha"] = sha(a.obsp["connectivities"].data) + ":" + sha(a.obsp["connectivities"].indices)
    # the graph each LISI call rebuilds, then the LISI value; HELPER holds the helper's files of that call
    for name, fn, key in (("ilisi", L.ilisi_graph, "batch"), ("clisi", L.clisi_graph, "celltype")):
        HELPER.clear()
        kw = dict(batch_key=key) if name == "ilisi" else dict(label_key=key)
        recomputed = L.recompute_knn(a, "embed", "X_emb")
        out[f"{name}_graph_conn_sha"] = sha(recomputed.obsp["connectivities"].data) + ":" + sha(recomputed.obsp["connectivities"].indices)
        HELPER.clear()
        v = fn(a, type_="embed", use_rep="X_emb", subsample=0.5 * 100, scale=True, n_cores=1, **kw)
        out[name] = float(v); out[f"{name}_hex"] = float(v).hex(); out[f"{name}_helper"] = dict(HELPER)
    return out

if __name__ == "__main__":
    res = {"host": os.uname().nodename, "scib": scib.__version__, "env": {k: os.environ.get(k) for k in
           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_THREADING_LAYER", "LD_PRELOAD", "NUMBA_CPU_NAME")},
           "rows": [one(p) for p in sys.argv[2:]]}
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps([{k: r[k] for k in ("latent", "ilisi", "clisi")} for r in res["rows"]]))
