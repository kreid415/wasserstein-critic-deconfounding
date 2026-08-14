"""E6 / Theme F — quantify batch support overlap (R1.major.2).

# WHY: The Wasserstein critic is motivated by DISJOINT support between batches, where
#      the JS discriminator's gradients vanish. Reviewer R1.major.2 asks whether the
#      benchmark datasets actually exhibit that setting. This script measures support
#      overlap directly in PCA space, with the cross-species immune task (expected
#      disjoint) and the simulations (known-overlapping ground truth) as controls.
# HOW: Three complementary, model-free overlap measures per dataset x {balanced,
#      unbalanced}, computed on the shared PCA embedding used for clustering:
#        (1) batch-classifier accuracy  -- a logistic/RF classifier's ability to predict
#            batch from expression; HIGH => separable => low overlap.
#        (2) kNN cross-batch fraction   -- mean fraction of each cell's k nearest
#            neighbours drawn from a DIFFERENT batch; LOW => low overlap.
#        (3) linear MMD^2 between batch pairs -- distributional distance; HIGH => low overlap.
#      All three are reported so the premise is triangulated, not asserted.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors

from wcd_vae.wcd.data import prep_data

warnings.filterwarnings("ignore")


def knn_cross_batch_fraction(x, batch_codes, k=30):
    # WHY: neighbourhood-level overlap; HOW: fraction of kNN from a different batch, averaged.
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    _, idx = nn.kneighbors(x)
    idx = idx[:, 1:]  # drop self
    same = (batch_codes[idx] == batch_codes[:, None]).mean(axis=1)
    return float((1.0 - same).mean())


def linear_mmd2(xa, xb):
    # WHY: distributional distance between two batches; HOW: squared MMD with linear kernel
    #      = ||mean_a - mean_b||^2 (closed form, no kernel matrix needed).
    ma, mb = xa.mean(axis=0), xb.mean(axis=0)
    return float(np.sum((ma - mb) ** 2))


def batch_classifier_accuracy(x, batch_codes, seed=0, max_cells=20000):
    # WHY: global separability; HOW: 3-fold CV balanced accuracy of an RF batch predictor.
    #      Chance = 1/n_batches; >> chance => batches separable => low support overlap.
    #      Subsample to max_cells (uniform random, proportional to batch sizes in
    #      expectation) so large atlases stay tractable without changing the estimand.
    rng = np.random.default_rng(seed)
    if len(batch_codes) > max_cells:
        idx = rng.choice(len(batch_codes), size=max_cells, replace=False)
        x, batch_codes = x[idx], batch_codes[idx]
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=seed)
    pred = cross_val_predict(clf, x, batch_codes, cv=3)
    return float(balanced_accuracy_score(batch_codes, pred))


def analyze(adata, batch_key, n_pcs=50, k=30):
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=n_pcs)
    x = adata.obsm["X_pca"][:, :n_pcs]
    codes = adata.obs[batch_key].astype("category").cat.codes.values
    n_batches = len(np.unique(codes))
    chance = 1.0 / n_batches

    acc = batch_classifier_accuracy(x, codes)
    xbf = knn_cross_batch_fraction(x, codes, k=k)

    # pairwise MMD^2 (mean over batch pairs)
    uniq = np.unique(codes)
    mmds = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            mmds.append(linear_mmd2(x[codes == uniq[i]], x[codes == uniq[j]]))
    mmd_mean = float(np.mean(mmds)) if mmds else np.nan

    # normalised classifier accuracy above chance in [0,1]: (acc-chance)/(1-chance)
    sep = (acc - chance) / (1 - chance) if n_batches > 1 else 0.0
    return {
        "n_batches": int(n_batches),
        "chance_acc": chance,
        "batch_clf_bal_acc": acc,
        "separability_norm": float(sep),  # 0 = at chance (full overlap), 1 = perfectly separable (disjoint)
        "knn_cross_batch_frac": xbf,      # 1 = fully mixed, ~0 = disjoint
        "pairwise_mmd2_mean": mmd_mean,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of registry keys")
    ap.add_argument("--k", type=int, default=30)
    # WHY --data-root EXISTS HERE TOO: this was the ONLY harness of the nine that could not
    # be pointed at a data directory. It resolved entry["path"], an ABSOLUTE path baked into
    # the registry (/home/kendall/data_gsteino1/wcd_data/...), while every other harness
    # joins --data-root to entry["file"]. On any machine where the data lives elsewhere,
    # all 12 configs failed with FileNotFoundError and the shard wrote a 1-byte CSV with no
    # header -- which then raises EmptyDataError in every downstream reader. Defaults to
    # $WCD_DATA so the launcher's existing export is honoured.
    ap.add_argument("--data-root", default=os.environ.get("WCD_DATA"),
                    help="directory holding the .h5ad files; joined to registry entry['file']. "
                         "Falls back to registry entry['path'] when unset.")
    args = ap.parse_args()

    with open(args.registry) as fh:
        registry = json.load(fh)
    keys = args.datasets or list(registry)

    rows = []
    for name in keys:
        entry = registry[name]
        for balance in (False, True):
            # simulations have exactly-shared groups; balancing is a no-op but harmless.
            try:
                path = (os.path.join(args.data_root, entry["file"])
                        if args.data_root and entry.get("file")
                        else entry["path"])
                adata, _largest = prep_data(
                    os.path.expandvars(os.path.expanduser(path)),
                    batch_key=entry["batch_key"],
                    celltype_key=entry["celltype_key"],
                    batch_count=entry.get("n_batches", 2),
                    balance=balance,
                    modality=entry.get("prep", "rna"),
                    cluster=False,  # E6 only needs PCA, not the triplet-loss Leiden labels
                )
                res = analyze(adata, entry["batch_key"], k=args.k)
                res.update({
                    "dataset": name, "balanced": balance, "modality": entry.get("prep", "rna"),
                    "batch_key": entry["batch_key"], "n_obs": int(adata.n_obs), "role": entry.get("role", ""),
                })
                rows.append(res)
                print(f"[{name} balance={balance}] sep_norm={res['separability_norm']:.3f} "
                      f"knn_xbatch={res['knn_cross_batch_frac']:.3f} mmd2={res['pairwise_mmd2_mean']:.2f}")
            except Exception as e:
                print(f"[{name} balance={balance}] FAILED: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
