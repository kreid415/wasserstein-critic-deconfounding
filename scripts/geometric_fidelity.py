#!/usr/bin/env python
"""Trustworthiness & continuity (geometric fidelity) for every method.

Convention matches SteinOBrienLab/disentanglement-benchmarking
``calculate_geometric_fidelity`` VERBATIM: high_d = ambient expression matrix
(``load_task`` X, the VAE input space), low_d = the latent; a single GLOBAL
5000-cell subsample drawn with ``np.random.default_rng(42)``; k=15 with the
``k < n/2`` guard sklearn.manifold.trustworthiness requires. Continuity is the
dual call trustworthiness(low_d, high_d).

WHY NOT per-batch-against-PCA: an earlier version computed per-batch T&C against
each batch's own PCA. That is NOT the harness convention and gives different
numbers; the harness scores the latent against the full ambient space, globally.

Reads embeddings (all carry keys z / batch / celltype / X_pca):
  VAE latents   $WCD_EMBED_OUT/<ds>/*.npz          (barycenter wave)
  baselines     $WCD_BASE_EMB/<ds>/{harmony,scanorama,unintegrated}.npz
  scVI          <results>/scvi_single/pancreas_scvi.npz
  Seurat        <results>/seurat_<ds>/seurat_emb.csv  (ambient X via load_task)

Writes tc_barycenter.csv (one row per VAE latent) and tc_baselines.csv
(one row per method x dataset). Run from the repo root.
"""
import sys, os, json, glob, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from wcd_vae.wcd.experiment import load_task  # noqa: E402
from sklearn.manifold import trustworthiness  # noqa: E402

GEOMETRIC_FIDELITY_K = 15
METRICS_RANDOM_SEED = 42
KNN_MIN_CELLS = 3
DS6 = ["pancreas", "immune", "lung", "sim1", "sim2", "atac_small"]


def geometric_fidelity(high_d, low_d, max_cells=5000,
                       k=GEOMETRIC_FIDELITY_K, seed=METRICS_RANDOM_SEED):
    """Verbatim port of disentanglement-benchmarking calculate_geometric_fidelity."""
    rng = np.random.default_rng(seed)
    if hasattr(high_d, "toarray"):
        high_d = high_d.toarray()
    if hasattr(low_d, "toarray"):
        low_d = low_d.toarray()
    high_d = np.asarray(high_d)
    low_d = np.asarray(low_d)
    if high_d.shape[0] > max_cells:
        idx = rng.choice(high_d.shape[0], max_cells, replace=False)
        high_d, low_d = high_d[idx], low_d[idx]
    n = int(high_d.shape[0])
    k_max = int(np.ceil(n / 2.0)) - 1
    if n < KNN_MIN_CELLS or k_max < 1:
        return np.nan, np.nan
    ke = int(min(int(k), k_max))
    trust = float(trustworthiness(high_d, low_d, n_neighbors=ke))
    cont = float(trustworthiness(low_d, high_d, n_neighbors=ke))
    return trust, cont


def _backbone(fn):
    return "LDVAE_uncond" if "LDVAE_uncond" in fn else ("LDVAE" if "LDVAE" in fn else "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-out", default=os.environ.get("WCD_EMBED_OUT"),
                    help="VAE latent root (per-dataset subdirs of .npz)")
    ap.add_argument("--base-emb", default=os.environ.get("WCD_BASE_EMB"),
                    help="baseline latent root (per-dataset harmony/scanorama/unintegrated.npz)")
    ap.add_argument("--data-root", default=os.environ.get("WCD_DATA"))
    ap.add_argument("--registry", default="configs/dataset_registry.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--datasets", nargs="*", default=DS6)
    args = ap.parse_args()

    reg = json.load(open(args.registry))
    Xamb = {}
    for ds in args.datasets:
        a, bk, ck, _ = load_task(ds, data_root=args.data_root, registry=reg)
        Xamb[ds] = np.asarray(a.X.todense()) if hasattr(a.X, "todense") else np.asarray(a.X)
        print(f"[{ds}] ambient X {Xamb[ds].shape}", flush=True)

    # --- VAE latents ---
    brows = []
    if args.embed_out:
        for f in sorted(glob.glob(os.path.join(args.embed_out, "**", "*.npz"), recursive=True)):
            ds = os.path.basename(os.path.dirname(f))
            if ds not in Xamb:
                continue
            fn = os.path.basename(f)
            z = np.load(f)
            lam = 0.2 if "lam0p2" in fn else (0.0 if "lam0p0" in fn else np.nan)
            try:
                seed = int(fn.split("_s")[1][0])
            except Exception:
                seed = -1
            T, Cn = geometric_fidelity(Xamb[ds], z["z"])
            brows.append(dict(dataset=ds, backbone=_backbone(fn), lam=lam,
                              seed=seed, trust=T, cont=Cn, file=fn))
        pd.DataFrame(brows).to_csv("tc_barycenter.csv", index=False)
        print(f"wrote tc_barycenter.csv ({len(brows)} latents)")

    # --- baselines + scVI + Seurat ---
    xrows = []
    if args.base_emb:
        for m in ["harmony", "scanorama", "unintegrated"]:
            for ds in args.datasets:
                f = os.path.join(args.base_emb, ds, f"{m}.npz")
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                T, Cn = geometric_fidelity(Xamb[ds], z["z"])
                xrows.append(dict(method=m, dataset=ds, trust=T, cont=Cn))
    scvi = os.path.join(args.results, "scvi_single", "pancreas_scvi.npz")
    if os.path.exists(scvi) and "pancreas" in Xamb:
        z = np.load(scvi)
        T, Cn = geometric_fidelity(Xamb["pancreas"], z["z"])
        xrows.append(dict(method="scVI", dataset="pancreas", trust=T, cont=Cn))
    sdirs = {"pancreas": "seurat_in", "immune": "seurat_immune", "lung": "seurat_lung",
             "sim1": "seurat_sim1", "sim2": "seurat_sim2", "atac_small": "seurat_atac_small"}
    for ds, dn in sdirs.items():
        p = os.path.join(args.results, dn, "seurat_emb.csv")
        if not os.path.exists(p) or ds not in Xamb:
            continue
        emb = pd.read_csv(p, index_col=0).to_numpy().astype("float32")
        if emb.shape[0] != Xamb[ds].shape[0]:
            print(f"  WARN seurat {ds}: {emb.shape} vs X {Xamb[ds].shape}")
            continue
        T, Cn = geometric_fidelity(Xamb[ds], emb)
        xrows.append(dict(method="Seurat", dataset=ds, trust=T, cont=Cn))
    pd.DataFrame(xrows).to_csv("tc_baselines.csv", index=False)
    print(f"wrote tc_baselines.csv ({len(xrows)} rows)")


if __name__ == "__main__":
    main()
