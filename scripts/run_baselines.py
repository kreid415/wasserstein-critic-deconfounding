"""E3 — external integration baselines (R2.4, R3.1).

# WHY: R2.4/R3.1 ask how the adversarial methods compare to the mainstream integration
#      tools. To place them we run the standard scIB baselines on the SAME preprocessed
#      tasks and score them with the SAME metric suite (harness.full_metric_suite), so
#      the numbers sit in one table with the critic/discriminator results.
# HOW: unintegrated (PCA), Harmony, scVI, scANVI (label-aware), Scanorama. Each produces
#      an embedding written to adata.obsm["X_emb"]; the shared suite scores it.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import scanpy as sc

from wcd_vae.wcd.experiment import full_metric_suite, load_task, save_embedding


def emb_unintegrated(adata, batch_key, celltype_key):
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=50)
    adata.obsm["X_emb"] = adata.obsm["X_pca"][:, :50]


def emb_harmony(adata, batch_key, celltype_key):
    # WHY: call harmonypy directly + assign with an explicit shape check, rather than
    #      sc.external.pp.harmony_integrate whose obsm write is version-fragile
    #      (observed a (n_pcs,)-shaped write instead of (n_cells, n_pcs)).
    import harmonypy

    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=50)
    pca = np.asarray(adata.obsm["X_pca"][:, :50])
    ho = harmonypy.run_harmony(pca, adata.obs, [batch_key])
    z = np.asarray(ho.Z_corr)
    # orient to (n_cells, n_pcs) regardless of harmonypy's return convention.
    if z.shape[0] != adata.n_obs and z.shape[1] == adata.n_obs:
        z = z.T
    assert z.shape[0] == adata.n_obs, f"harmony emb shape {z.shape} != n_obs {adata.n_obs}"
    adata.obsm["X_emb"] = z


def emb_scanorama(adata, batch_key, celltype_key):
    # WHY: run scanorama per-batch then reassemble in original cell order; the scanpy
    #      wrapper had the same fragile-obsm-write issue as harmony.
    import scanorama

    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=50)
    order = np.argsort(adata.obs[batch_key].astype(str).values, kind="stable")
    inv = np.argsort(order)
    ad_sorted = adata[order].copy()
    batches = ad_sorted.obs[batch_key].astype(str).values
    splits, mats = [], []
    for b in pd.unique(batches):
        m = batches == b
        splits.append(np.where(m)[0])
        mats.append(np.asarray(ad_sorted.obsm["X_pca"][m, :50]))
    integrated = scanorama.assemble(mats)  # list of corrected per-batch matrices
    z_sorted = np.zeros((ad_sorted.n_obs, integrated[0].shape[1]), dtype=np.float32)
    for idx, mat in zip(splits, integrated):
        z_sorted[idx] = np.asarray(mat)
    adata.obsm["X_emb"] = z_sorted[inv]  # back to original order
    assert adata.obsm["X_emb"].shape[0] == adata.n_obs


def _fit_scvi(adata, batch_key, n_latent=30):
    """Fit an SCVI model on raw counts; return (model, setup_adata)."""
    import scvi

    a = adata.copy()
    a.X = a.layers["counts"].copy()  # scVI needs raw counts
    scvi.model.SCVI.setup_anndata(a, batch_key=batch_key)
    m = scvi.model.SCVI(a, n_latent=n_latent)
    m.train(max_epochs=200, early_stopping=True, enable_progress_bar=False)
    return m, a


def emb_scvi(adata, batch_key, celltype_key):
    m, _a = _fit_scvi(adata, batch_key)
    adata.obsm["X_emb"] = m.get_latent_representation()


def emb_scanvi(adata, batch_key, celltype_key):
    import scvi

    m, a = _fit_scvi(adata, batch_key)  # warm-start SCVI
    a.obs[celltype_key] = adata.obs[celltype_key].values
    scanvi = scvi.model.SCANVI.from_scvi_model(m, unlabeled_category="Unknown", labels_key=celltype_key)
    scanvi.train(max_epochs=100, enable_progress_bar=False)
    adata.obsm["X_emb"] = scanvi.get_latent_representation()


METHODS = {
    "unintegrated": emb_unintegrated,
    "harmony": emb_harmony,
    "scvi": emb_scvi,
    "scanvi": emb_scanvi,
    "scanorama": emb_scanorama,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-count", type=int, default=None)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--embed-out", default=None,
                    help="directory to persist baseline latents (.npz per method); a "
                         "DATASET-SPECIFIC subdirectory is created under it")
    ap.add_argument("--no-embed-ok", action="store_true",
                    help="acknowledge running WITHOUT --embed-out (smoke runs only)")
    args = ap.parse_args()

    if not args.embed_out and not args.no_embed_ok:
        import warnings
        warnings.warn(
            "\n*** RUNNING WITHOUT --embed-out: baseline latents will NOT be persisted. ***\n"
            "    Pass --embed-out <dir>, or --no-embed-ok for a smoke run.\n",
            stacklevel=2,
        )

    with open(args.registry) as fh:
        registry = json.load(fh)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    adata, batch_key, celltype_key, _largest = load_task(
        args.dataset, batch_count=args.batch_count, balance=args.balance,
        data_root=args.data_root, registry=registry,
    )
    print(f"[{args.dataset}] n_obs={adata.n_obs} n_batches={adata.obs[batch_key].nunique()}")

    rows = []
    for name in args.methods:
        try:
            ad = adata.copy()
            METHODS[name](ad, batch_key, celltype_key)
            if args.embed_out:
                # E3 varies exactly one field -- the baseline method -- so the method
                # name is a sufficient tag. Persisted so a new embedding-derived metric
                # does not require re-running scVI/scANVI/Harmony/Scanorama either.
                save_embedding(ad, os.path.join(args.embed_out, args.dataset), name,
                               batch_key, celltype_key, embed_key="X_emb")
            metrics = full_metric_suite(ad, batch_key, celltype_key, embed_key="X_emb")
            row = {"experiment": "E3", "dataset": args.dataset, "method": name,
                   "balanced": args.balance, "n_batches": int(adata.obs[batch_key].nunique()),
                   **metrics}
            rows.append(row)
            print(f"  {name:14s} | iLISI={metrics['ilisi']:.3f} cLISI={metrics['clisi']:.3f} "
                  f"ASWb={metrics['asw_batch']:.3f} ARI={metrics['ari']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
        except Exception as e:
            print(f"  {name:14s} FAILED {type(e).__name__}: {e}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote {len(rows)} rows -> {args.out}")
    _ = np


if __name__ == "__main__":
    main()
