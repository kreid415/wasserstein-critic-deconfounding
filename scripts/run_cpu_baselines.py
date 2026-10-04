#!/usr/bin/env python
"""X13 CPU baselines: PCA, Harmony and Scanorama on scIB's normalised X of one prepped task, one manifest row each.

Rows: the X13 block of scripts/build_paper_manifest.py (CPU_BASELINES; CONSTRAINTS.md SI-34..SI-36), arm in
{harmony, scanorama, pca}; the knob value is the row's 'lam', the embedding size its 'n_latent', seed 0 (each
configuration runs once). scripts/fit_paper_config.py refuses these arms.
  harmony    theta in {0, 0.5, 1, 2, 4, 8} at 10 PCs, plus theta 2 at 50 PCs (tool default)        (SI-34)
  scanorama  knn in {5, 10, 20, 40, 80, 160} at dimred 10, plus knn 20 at dimred 100 (tool default) (SI-35)
  pca        10 PCs, plus 50 PCs (tool default); no strength knob: the uncorrected anchor            (SI-36)
Knob sources: Korsunsky et al. 2019 (Harmony, Methods Eq. 3-4 and 5.4: theta = weight of the penalty on batch-
cluster dependence, theta 0 = no penalty, default 2); Hie et al. 2019 (Scanorama, Methods: knn nearest
neighbours for mutual matching, default 20). docs/SPECS_missing_arms.md section 8.

Follows scIB's own integration functions (scib 1.1.7, env wcd-kbet; source read 2026-10-02/03):
  scib.integration.harmony(adata, batch):  sc.tl.pca(adata) (scanpy defaults, 50 PCs), then
      X_emb = harmony.harmonize(adata.obsm["X_pca"], adata.obs, batch_key=batch)   (harmony-pytorch 0.1.7, all
      other harmonize arguments at their defaults). The wrapper does not forward kwargs, so the runner runs these
      two statements itself with n_comps and theta exposed (harmony_scib_statements; a test checks that they
      reproduce scib.integration.harmony at the default setting).
  scib.integration.scanorama(adata, batch, **kwargs):  split by batch, scanorama.correct_scanpy(split,
      return_dimred=True, **kwargs), concatenate, X_emb = obsm["X_scanorama"]; the runner passes dimred and knn
      (scanorama 1.7.4 defaults 100 and 20; seed 0).
  PCA: sc.tl.pca(n_comps), scIB's unintegrated embedding.
Training features: var["highly_variable"] of the prepped file, subset before the method (scIB pipeline).

Reproducibility (measured 2026-10-02/03, NB-20261002-10): PCA and Scanorama reruns are identical. Harmony was not
reproducible run to run in a shared process (atac_small d50: max|dz| 0.114 between two runs), so every Harmony fit
runs in a FRESH process with every thread pool pinned (--threads, default 1: OMP/MKL/OpenBLAS/NumExpr threads and
harmonize(n_jobs)); tests/x13 checks that two such fits on atac_small at 50 PCs are bit-identical.

Output: OUT_DIR/latents/<tag>.npz with z, obs_names, batch, celltype, config, history: the format of
scripts/fit_paper_config.py, scored unchanged by scripts/score_scib_native.py.
Usage (env wcd-kbet):
  python scripts/run_cpu_baselines.py --manifest M --prepped-dir D --out-dir O [--task T | --tag TAG ...]
"""
import argparse
import importlib.metadata as md
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

CPU_ARMS = {"harmony": "theta", "scanorama": "knn", "pca": None}   # arm: knob (manifest 'lam')
# scVI-backbone columns do not apply to CPU rows: they must hold these placeholders (cpu_row in the builder), so a
# row that sets one of them is refused instead of being silently ignored
NOT_APPLICABLE = {"counts": "scib", "n_critic": 0, "adv_input": "na", "zstd": 0, "cond": 0, "decoder": "na",
                  "n_layers": 0, "n_hidden": 0, "likelihood": "na", "batch_size": 0, "max_epochs": 0,
                  "train_size": 0, "reference": "auto"}


def _matches(value, expected):
    if isinstance(expected, str):
        return value == expected
    try:
        return float(value) == float(expected)
    except ValueError:
        return False
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def harmony_scib_statements(adata, batch, n_comps, theta=2.0, n_jobs=-1):
    """The two statements of scib.integration.harmony, with the PCA size (None = scanpy default), theta and
    harmonize's thread count exposed (theta 2.0 and n_jobs -1 are harmonize's defaults)."""
    import scanpy as sc
    from harmony import harmonize
    if n_comps is None:
        sc.tl.pca(adata)
    else:
        sc.tl.pca(adata, n_comps=n_comps)
    adata.obsm["X_emb"] = harmonize(adata.obsm["X_pca"], adata.obs, batch_key=batch, theta=theta, n_jobs=n_jobs)
    return adata


def harmony_fresh_process(a, theta, n_comps, threads=1, batch="batch"):
    """Run harmony_scib_statements in a new Python process with every thread pool pinned to `threads`.
    Returns (z in a's cell order, kwargs passed). The input is handed over as an h5ad file (exact float32 copy)."""
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    with tempfile.TemporaryDirectory(prefix="x13_harmony_") as tmp:
        src, out = os.path.join(tmp, "in.h5ad"), os.path.join(tmp, "z.npy")
        a.write_h5ad(src)
        env = dict(os.environ, KMP_AFFINITY="disabled", **{k: str(threads) for k in THREAD_VARS})
        cmd = [sys.executable, os.path.abspath(__file__), "--harmony-worker", src, out, "--theta", repr(float(theta)),
               "--n-comps", str(int(n_comps)), "--threads", str(int(threads)), "--batch", batch]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"harmony worker failed (exit {r.returncode}):\n{r.stderr[-3000:]}")
        z = np.load(out)
    return z, {"n_comps": int(n_comps), "theta": float(theta), "n_jobs": int(threads), "fresh_process": True}


def _harmony_worker(src, out, theta, n_comps, threads, batch):
    import anndata as ad
    import torch
    torch.set_num_threads(threads)
    a = ad.read_h5ad(src)
    z = np.asarray(harmony_scib_statements(a, batch, n_comps, theta=theta, n_jobs=threads).obsm["X_emb"])
    np.save(out, z)


def embed(a, arm, dims, value, threads=1, batch="batch"):
    """Return (z [n_cells, dims] in a's cell order, tool kwargs actually passed) for one X13 CPU row."""
    import scanpy as sc
    import scib
    if arm not in CPU_ARMS:
        raise ValueError(f"unknown CPU baseline {arm!r}; valid: {sorted(CPU_ARMS)}")
    if arm == "pca":
        if float(value) != 0:
            raise ValueError(f"pca has no strength knob; its row must carry lam 0, got {value!r}")
        x = a.copy()
        sc.tl.pca(x, n_comps=int(dims))
        return np.asarray(x.obsm["X_pca"]), {"n_comps": int(dims)}
    if arm == "harmony":
        if float(value) < 0:
            raise ValueError(f"harmony theta must be >= 0, got {value!r}")
        return harmony_fresh_process(a, float(value), int(dims), threads=threads, batch=batch)
    knn = float(value)
    if knn != int(knn) or knn < 1:
        raise ValueError(f"scanorama knn must be a positive integer, got {value!r}")
    kw = {"dimred": int(dims), "knn": int(knn)}
    out = scib.integration.scanorama(a.copy(), batch, **kw)
    if not out.obs_names.is_unique or set(out.obs_names) != set(a.obs_names):
        raise ValueError("scanorama output cells do not match the input cells")
    return np.asarray(out[a.obs_names].obsm["X_emb"]), kw


def provenance(root):
    sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip() != ""
    vers = {p: md.version(p) for p in ("scib", "scanpy", "anndata", "harmony-pytorch", "scanorama", "numpy",
                                        "scikit-learn", "torch")}
    threads = {k: os.environ.get(k) for k in THREAD_VARS}
    from host_info import cpu_info      # scripts/host_info.py (device fields pinned by the lead 2026-10-03, CR-02)
    return dict(git_sha=sha, git_dirty=dirty, versions=vers, threads=threads, cpu_count=os.cpu_count(),
                device="cpu", cpu_model=cpu_info()[0])


def select_rows(manifest, task=None, tags=None):
    """X13 CPU rows of the manifest (all, one task's, or the named tags); refuses unknown or non-CPU tags."""
    import pandas as pd
    m = pd.read_csv(manifest, sep="\t", comment="#", dtype=str, keep_default_na=False)
    cpu = m[(m.experiment == "X13") & m.arm.isin(list(CPU_ARMS))]
    if tags:
        unknown = sorted(set(tags) - set(m.tag))
        if unknown:
            raise KeyError(f"tags not in the manifest: {unknown}")
        not_cpu = sorted(set(tags) - set(cpu.tag))
        if not_cpu:
            raise ValueError(f"not X13 CPU rows (run them with scripts/fit_paper_config.py): {not_cpu}")
        cpu = cpu[cpu.tag.isin(tags)]
    if task:
        cpu = cpu[cpu.task == task]
    if cpu.empty:
        raise ValueError(f"no X13 CPU rows selected (task={task}, tags={tags})")
    for r in cpu.itertuples():
        extra = json.loads(r.extra)
        if extra != ({"knob": CPU_ARMS[r.arm]} if CPU_ARMS[r.arm] else {}) or int(r.seed) != 0:
            raise ValueError(f"{r.tag}: expected extra knob {CPU_ARMS[r.arm]!r} and seed 0, got {extra} and seed {r.seed}")
        bad = {k: getattr(r, k) for k, v in NOT_APPLICABLE.items() if not _matches(getattr(r, k), v)}
        if bad:
            raise ValueError(f"{r.tag}: scVI-backbone fields set on a CPU row (they would be ignored): {bad}")
    return cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--prepped-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--task", default=None)
    ap.add_argument("--tag", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=1, help="pinned thread count of each Harmony process (default 1)")
    ap.add_argument("--skip-existing", action="store_true", help="skip outputs that already exist (default: refuse)")
    ap.add_argument("--harmony-worker", nargs=2, metavar=("IN_H5AD", "OUT_NPY"), help=argparse.SUPPRESS)
    ap.add_argument("--theta", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--n-comps", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--batch", default="batch", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.harmony_worker:
        _harmony_worker(*a.harmony_worker, a.theta, a.n_comps, a.threads, a.batch)
        return
    for k in ("manifest", "prepped_dir", "out_dir"):
        if getattr(a, k) is None:
            ap.error(f"--{k.replace('_', '-')} is required")
    import anndata as ad
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prov = provenance(root)
    rows = select_rows(a.manifest, a.task, a.tag)
    os.makedirs(os.path.join(a.out_dir, "latents"), exist_ok=True)
    expected = []
    for task, rows_t in rows.groupby("task", sort=True):
        adata = ad.read_h5ad(os.path.join(a.prepped_dir, f"{task}__scib.h5ad"))
        for col in ("batch", "celltype"):
            if col not in adata.obs:
                raise KeyError(f"{task}: obs lacks {col!r}")
        if "highly_variable" not in adata.var:
            raise KeyError(f"{task}: var lacks 'highly_variable'")
        hvg = adata.var["highly_variable"].values.astype(bool)
        x = adata[:, hvg].copy()
        x.obs["batch"] = x.obs["batch"].astype(str).astype("category")
        del adata
        for r in rows_t.itertuples():
            npz = os.path.join(a.out_dir, "latents", f"{r.tag}.npz")
            expected.append(npz)
            if os.path.exists(npz):
                if a.skip_existing:
                    print(f"[skip] {r.tag} exists", flush=True)
                    continue
                raise FileExistsError(f"{npz} exists; pass --skip-existing to keep it")
            dims = int(r.n_latent)
            t0 = time.time()
            z, kw = embed(x, r.arm, dims, r.lam, threads=a.threads)
            secs = time.time() - t0
            if z.shape != (x.n_obs, dims) or not np.isfinite(z).all():
                raise ValueError(f"{r.tag}: latent shape {z.shape} (expected {(x.n_obs, dims)}) or non-finite values")
            cfg = dict(row=r._asdict(), tool_kwargs=kw, knob=CPU_ARMS[r.arm], knob_value=float(r.lam),
                       features="highly_variable", n_cells=int(x.n_obs), n_hvg=int(hvg.sum()),
                       n_batches=int(x.obs["batch"].nunique()), fit_seconds=round(secs, 1), **prov)
            cfg["row"].pop("Index", None)
            tmp = npz + ".tmp.npz"
            np.savez_compressed(tmp, z=z.astype(np.float32), obs_names=x.obs_names.to_numpy(dtype="U128"),
                                batch=x.obs["batch"].astype(str).to_numpy(dtype="U64"),
                                celltype=x.obs["celltype"].astype(str).to_numpy(dtype="U64"),
                                config=json.dumps(cfg), history=json.dumps({}))
            os.replace(tmp, npz)
            print(f"[x13] {r.tag} {secs:.1f}s z{z.shape}", flush=True)
    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        print(f"[x13] missing outputs: {missing}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
