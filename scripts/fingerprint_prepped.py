#!/usr/bin/env python
"""Content fingerprint of prep_scib_task.py outputs, to verify that two machines hold the same task inputs.

Exact (sha1) parts: shape, obs_names, var_names, layers['counts'] and X (CSR with sorted indices: data, indices,
indptr bytes), var['highly_variable'], obs batch and celltype, the task-defining uns fields (UNS_KEYS).
Toleranced part: obsm['X_pca'] (sign and BLAS dependent): per-component mean |x| and std, compared with a relative
tolerance (default 1e-3) because PCA is not bitwise reproducible across BLAS builds.

Usage:
  python scripts/fingerprint_prepped.py --dir PREPPED_DIR --out fp.json            # write
  python scripts/fingerprint_prepped.py --dir PREPPED_DIR --compare docs/prepped_fingerprints_scib.json
Exit code 1 on any mismatch (compare mode). Files matched: *__scib.h5ad (override with --glob).
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import scipy.sparse as sp


# uns fields that define the task (prep_code_sha is excluded: the same inputs may come from different commits)
UNS_KEYS = ("task", "counts_mode", "batch_key", "celltype_key", "scib_batch_key", "scib_label_key", "modality", "organism",
            "source_file", "source_md5", "hvg", "repairs", "count_status_as_distributed")


def _sha(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _csr_sha(M):
    M = M.tocsr() if sp.issparse(M) else sp.csr_matrix(M)
    M = M.copy(); M.sort_indices()
    return _sha(M.data.astype(np.float64), M.indices.astype(np.int64), M.indptr.astype(np.int64))


def _str_sha(values):
    return hashlib.sha1("\x1f".join(map(str, values)).encode()).hexdigest()


def fingerprint(path):
    import anndata as ad
    a = ad.read_h5ad(path)
    fp = dict(file=os.path.basename(path), shape=list(a.shape), obs_names=_str_sha(a.obs_names), var_names=_str_sha(a.var_names),
              counts=_csr_sha(a.layers["counts"]), X=_csr_sha(a.X),
              hvg=_sha(a.var["highly_variable"].values.astype(np.int8)), n_hvg=int(a.var["highly_variable"].sum()),
              batch=_str_sha(a.obs["batch"].astype(str)), celltype=_str_sha(a.obs["celltype"].astype(str)),
              uns={k: (a.uns[k] if isinstance(a.uns.get(k), (str, int, float)) else json.dumps(_plain(a.uns.get(k)), sort_keys=True))
                   for k in UNS_KEYS if k in a.uns})
    P = np.asarray(a.obsm["X_pca"], dtype=np.float64)
    fp["X_pca"] = dict(shape=list(P.shape), abs_mean=np.abs(P).mean(0).round(10).tolist(), std=P.std(0).round(10).tolist())
    return fp


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v) for v in list(x)]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x if isinstance(x, (str, int, float, bool, type(None))) else str(x)


def compare(a, b, rtol):
    problems = []
    for k in ("shape", "obs_names", "var_names", "counts", "X", "hvg", "n_hvg", "batch", "celltype", "uns"):
        if a.get(k) != b.get(k):
            problems.append(k)
    pa, pb = a["X_pca"], b["X_pca"]
    if pa["shape"] != pb["shape"]:
        problems.append("X_pca.shape")
    else:
        for s in ("abs_mean", "std"):
            x, y = np.array(pa[s]), np.array(pb[s])
            if not np.allclose(x, y, rtol=rtol, atol=0):
                problems.append(f"X_pca.{s} (max rel diff {float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12))):.2e})")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--glob", default="*__scib.h5ad")
    ap.add_argument("--out")
    ap.add_argument("--compare")
    ap.add_argument("--rtol", type=float, default=1e-3)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, a.glob)))
    if not files:
        sys.exit(f"no files match {a.glob} in {a.dir}")
    fps = {os.path.basename(f): fingerprint(f) for f in files}
    if a.out:
        json.dump(fps, open(a.out, "w"), indent=1, sort_keys=True)
        print(f"wrote {len(fps)} fingerprints -> {a.out}")
    if a.compare:
        ref = json.load(open(a.compare))
        bad = 0
        for name in sorted(set(ref) | set(fps)):
            if name not in fps or name not in ref:
                print(f"{name}: MISSING on {'this machine' if name not in fps else 'reference'}"); bad += 1; continue
            p = compare(fps[name], ref[name], a.rtol)
            print(f"{name}: {'OK' if not p else 'MISMATCH ' + ', '.join(p)}")
            bad += bool(p)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
