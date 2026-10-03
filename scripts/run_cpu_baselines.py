#!/usr/bin/env python
"""X13 CPU baselines: PCA, Harmony and Scanorama on scIB's normalised X of one prepped task.

Follows scIB's own integration functions (scib 1.1.7, env wcd-kbet; source read 2026-10-02):
  scib.integration.harmony(adata, batch):  sc.tl.pca(adata) (scanpy defaults, 50 PCs), then
      X_emb = harmony.harmonize(adata.obsm["X_pca"], adata.obs, batch_key=batch)   (harmony-pytorch 0.1.7, the
      version scib's docstring names; all harmonize arguments at their defaults)
  scib.integration.scanorama(adata, batch, **kwargs):  split by batch (category order),
      scanorama.correct_scanpy(split, return_dimred=True, **kwargs), concatenate (index_unique=None),
      X_emb = obsm["X_scanorama"]   (scanorama 1.7.4; dimred default 100, seed 0)
  PCA: sc.tl.pca, scIB's unintegrated embedding.
Settings (docs/paper_experiment_matrix.csv X13; CONSTRAINTS.md SI-04):
  primary  10 dimensions, the backbone's n_latent: PCA n_comps=10; Harmony on 10 PCs; Scanorama dimred=10
  default  the tools' own defaults: PCA 50 (scanpy N_PCS), Harmony on 50 PCs, Scanorama dimred 100
At the default setting Harmony and Scanorama are scib's functions called as they are. At 10 dimensions, Scanorama
is scib's function with dimred=10 (its kwargs reach scanorama.correct); scib's Harmony wrapper has no n_comps
argument, so the runner runs its two statements with sc.tl.pca(n_comps=10) (a test checks that these statements
reproduce scib.integration.harmony exactly at the default setting).
Training features: var["highly_variable"] of the prepped file, subset before the method (scIB pipeline).
One run per task and setting (PCA random_state 0, harmonize random_state 0, scanorama seed 0). Measured
2026-10-02 on the local machine (NB-20261002-10): PCA reruns are identical at any thread count. Harmony is not
reproducible run to run: two runs of this script on atac_small with the same recorded settings gave identical d10
latents but d50 latents differing by max|dz| 0.114 (NMI 0.024 apart). On a toy input, harmonize reruns differed by
up to 5.7e-06 inside a long-running process and with MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE, and by 0 in a fresh
process with default settings (4 calls). Whether X13 repeats Harmony or pins its threads is an open decision.

Output: OUT_DIR/latents/X13_<task>_<method>_d<dims>.npz with z, obs_names, batch, celltype, config, history:
the format of scripts/fit_paper_config.py, scored unchanged by scripts/score_scib_native.py.
Usage (env wcd-kbet): python scripts/run_cpu_baselines.py --task atac_small --prepped-dir <dir> --out-dir <dir>
"""
import argparse
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time

import numpy as np

METHODS = ("pca", "harmony", "scanorama")
SETTINGS = ("primary", "default")
PRIMARY_DIMS = 10                     # = backbone n_latent (BACKBONES['stock'] in scripts/build_paper_manifest.py)
TOOL_DEFAULT_DIMS = {"pca": 50, "harmony": 50, "scanorama": 100}


def harmony_scib_statements(adata, batch, n_comps):
    """The two statements of scib.integration.harmony, with the PCA size exposed (n_comps=None = scanpy default)."""
    import scanpy as sc
    from harmony import harmonize
    if n_comps is None:
        sc.tl.pca(adata)
    else:
        sc.tl.pca(adata, n_comps=n_comps)
    adata.obsm["X_emb"] = harmonize(adata.obsm["X_pca"], adata.obs, batch_key=batch)
    return adata


def embed(a, method, setting, batch="batch"):
    """Return (z [n_cells, d] in a's cell order, tool kwargs actually passed)."""
    import scanpy as sc
    import scib
    x = a.copy()
    if method == "pca":
        kw = {} if setting == "default" else {"n_comps": PRIMARY_DIMS}
        sc.tl.pca(x, **kw)
        return np.asarray(x.obsm["X_pca"]), kw
    if method == "harmony":
        if setting == "default":
            out = scib.integration.harmony(x, batch)
            return np.asarray(out.obsm["X_emb"]), {}
        out = harmony_scib_statements(x, batch, PRIMARY_DIMS)
        return np.asarray(out.obsm["X_emb"]), {"n_comps": PRIMARY_DIMS}
    if method == "scanorama":
        kw = {} if setting == "default" else {"dimred": PRIMARY_DIMS}
        out = scib.integration.scanorama(x, batch, **kw)
        if not out.obs_names.is_unique or set(out.obs_names) != set(a.obs_names):
            raise ValueError("scanorama output cells do not match the input cells")
        return np.asarray(out[a.obs_names].obsm["X_emb"]), kw
    raise ValueError(f"unknown method {method!r}; valid: {METHODS}")


def provenance(root):
    sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip() != ""
    vers = {p: md.version(p) for p in ("scib", "scanpy", "anndata", "harmony-pytorch", "scanorama", "numpy",
                                        "scikit-learn", "torch")}
    threads = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    return dict(git_sha=sha, git_dirty=dirty, versions=vers, threads=threads, cpu_count=os.cpu_count())


def tag_for(task, method, dims):
    return f"X13_{task}_{method}_d{dims}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--prepped-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    ap.add_argument("--skip-existing", action="store_true", help="skip outputs that already exist (default: refuse)")
    a = ap.parse_args()
    import anndata as ad
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prov = provenance(root)
    adata = ad.read_h5ad(os.path.join(a.prepped_dir, f"{a.task}__scib.h5ad"))
    for col in ("batch", "celltype"):
        if col not in adata.obs:
            raise KeyError(f"{a.task}: obs lacks {col!r}")
    if "highly_variable" not in adata.var:
        raise KeyError(f"{a.task}: var lacks 'highly_variable'")
    hvg = adata.var["highly_variable"].values.astype(bool)
    x = adata[:, hvg].copy()
    x.obs["batch"] = x.obs["batch"].astype(str).astype("category")
    os.makedirs(os.path.join(a.out_dir, "latents"), exist_ok=True)
    expected = []
    for method in a.methods:
        for setting in a.settings:
            dims = PRIMARY_DIMS if setting == "primary" else TOOL_DEFAULT_DIMS[method]
            tag = tag_for(a.task, method, dims)
            npz = os.path.join(a.out_dir, "latents", f"{tag}.npz")
            expected.append(f"latents/{tag}.npz")
            if os.path.exists(npz):
                if a.skip_existing:
                    print(f"[skip] {tag} exists", flush=True)
                    continue
                raise FileExistsError(f"{npz} exists; pass --skip-existing to keep it")
            t0 = time.time()
            z, kw = embed(x, method, setting)
            secs = time.time() - t0
            if z.shape != (x.n_obs, dims) or not np.isfinite(z).all():
                raise ValueError(f"{tag}: latent shape {z.shape} (expected {(x.n_obs, dims)}) or non-finite values")
            cfg = dict(row=dict(tag=tag, experiment="X13", task=a.task, arm=method, setting=setting, n_dims=dims,
                                counts="scib", features="highly_variable"),
                       tool_kwargs=kw, n_cells=int(x.n_obs), n_hvg=int(hvg.sum()), n_batches=int(x.obs["batch"].nunique()),
                       fit_seconds=round(secs, 1), **prov)
            tmp = npz + ".tmp.npz"
            np.savez_compressed(tmp, z=z.astype(np.float32), obs_names=x.obs_names.to_numpy(dtype="U128"),
                                batch=x.obs["batch"].astype(str).to_numpy(dtype="U64"),
                                celltype=x.obs["celltype"].astype(str).to_numpy(dtype="U64"),
                                config=json.dumps(cfg), history=json.dumps({}))
            os.replace(tmp, npz)
            print(f"[x13] {tag} {secs:.1f}s z{z.shape}", flush=True)
    missing = [p for p in expected if not os.path.exists(os.path.join(a.out_dir, p))]
    if missing:
        print(f"[x13] missing outputs: {missing}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
