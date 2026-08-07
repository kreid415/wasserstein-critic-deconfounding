from numba import njit
import numpy as np
from sklearn.neighbors import NearestNeighbors


# NOTE (K. Reid, revision): fastmath removed. Under fastmath=True the compiler assumes
# no inf/nan, which breaks the `beta == np.inf` guards in the perplexity binary search
# below; on recent numba builds this silently collapses every LISI score to the
# n_neighbors fallback (a constant), corrupting iLISI/cLISI. fastmath=False restores
# the correct adaptive-bandwidth computation with negligible speed cost.
@njit(cache=True)
def compute_simpson_numba(indices, distances, batch_codes, n_batches, perplexity=30):
    """
    Numba-accelerated LISI computation.
    Replaces the slow Python loop and binary search.
    """
    n_cells = indices.shape[0]
    n_neighbors = indices.shape[1]
    lisi_scores = np.zeros(n_cells)

    # Pre-compute target entropy
    target_entropy = np.log2(perplexity)

    for i in range(n_cells):
        # Get distances for this cell (excluding self at index 0)
        # Input distances should be squared distances for Gaussian kernel
        dists = distances[i, 1:]
        batches = batch_codes[indices[i, 1:]]

        # Binary search for beta = 1 / (2 * sigma^2)
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        # Binary search for 50 iterations (standard t-SNE optimization)
        for _ in range(50):
            # Compute Gaussian kernel
            p = np.exp(-dists * beta)
            sum_p = np.sum(p)

            if sum_p == 0:
                sum_p = 1e-10

            # Entropy calculation
            h = np.log2(sum_p) + beta * np.sum(dists * p) / sum_p / np.log(2)

            diff = h - target_entropy

            if np.abs(diff) < 1e-5:
                break

            if diff > 0:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        # Final weights
        weights = np.exp(-dists * beta)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights /= weights_sum

        # Batch probabilities
        batch_probs = np.zeros(n_batches)
        for j in range(len(weights)):
            b = batches[j]
            batch_probs[b] += weights[j]

        # Inverse Simpson Index
        simpson = np.sum(batch_probs**2)
        if simpson > 0:
            lisi_scores[i] = 1.0 / simpson
        else:
            lisi_scores[i] = n_neighbors  # Fallback if numerical instability

    return lisi_scores


def compute_lisi(x, metadata, label_colname, perplexity=30):
    """
    Compute Local Inverse Simpson Index (LISI) using optimized Numba backend.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The embedded data matrix
    metadata : pandas.DataFrame
        Metadata containing batch/label information
    label_colname : str
        Column name in metadata containing the batch labels
    perplexity : int, default=30
        Perplexity parameter for Gaussian kernel

    Returns:
    --------
    lisi_scores : array-like
        LISI score for each cell
    """
    n_cells = x.shape[0]

    # 1. Prepare Batches (Integer encoding)
    if label_colname not in metadata:
        raise ValueError(f"Column {label_colname} not found in metadata")

    # Convert to category codes for Numba
    batch_codes = metadata[label_colname].astype("category").cat.codes.values
    n_batches = len(np.unique(batch_codes))

    # 2. Nearest Neighbors
    # k must be > perplexity. 3*perplexity is a standard heuristic.
    k = min(int(perplexity * 3), n_cells - 1)

    print(f"Computing {k} nearest neighbors for {n_cells} cells...")
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(x)
    distances, indices = nbrs.kneighbors(x)

    # 3. Compute LISI (Numba accelerated)
    print(f"Computing LISI scores for {label_colname} (Optimized)...")
    # Pass squared distances because Gaussian kernel is exp(-d^2 / 2sigma^2)
    # NearestNeighbors returns Euclidean distance (d), so we pass d^2
    lisi_scores = compute_simpson_numba(indices, distances**2, batch_codes, n_batches, perplexity)

    return lisi_scores


def ilisi_graph(
    adata, batch_key, type="embed", use_rep="X_pca", perplexity=30, subset_indices=None
):
    """
    Compute integration Local Inverse Simpson Index (iLISI) for an AnnData object.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    batch_key : str
        Key in adata.obs containing batch information
    type : str, default="embed"
        Type of data to use ("embed" for embeddings)
    use_rep : str, default="X_pca"
        Key in adata.obsm for the embedding to use
    perplexity : int, default=30
        Perplexity parameter for neighborhood definition
    subset_indices : array-like, optional
        Indices of cells to include in the computation

    Returns:
    --------
    float
        Normalized mean iLISI score across all cells (0-1 range)
    """
    if type == "embed":
        print("Using embed")
        if use_rep not in adata.obsm:
            raise ValueError(f"Embedding {use_rep} not found in adata.obsm")
        x = adata.obsm[use_rep]
    else:
        x = adata.X

    if batch_key not in adata.obs:
        raise ValueError(f"Batch key {batch_key} not found in adata.obs")

    # Get number of unique batches for normalization
    n_batches = len(adata.obs[batch_key].unique())

    # Compute LISI scores
    print("Computing LISI")
    lisi_scores = compute_lisi(x, adata.obs, batch_key, perplexity)

    # Normalize by number of batches (perfect mixing = 1.0, no mixing = 1/n_batches)
    # Avoid division by zero if n_batches == 1
    if n_batches > 1:
        normalized_scores = (lisi_scores - 1) / (n_batches - 1)
    else:
        normalized_scores = lisi_scores - 1  # Should be 0

    # If indices are provided, only return the mean for those cells
    if subset_indices is not None:
        return np.mean(normalized_scores[subset_indices])

    # Return mean normalized iLISI score
    return np.mean(normalized_scores)


def clisi_graph(
    adata, label_key, type="embed", use_rep="X_pca", perplexity=30, subset_indices=None
):
    """
    Compute cell-type Local Inverse Simpson Index (cLISI) for an AnnData object.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    label_key : str
        Key in adata.obs containing cell type information
    type : str, default="embed"
        Type of data to use ("embed" for embeddings)
    use_rep : str, default="X_pca"
        Key in adata.obsm for the embedding to use
    perplexity : int, default=30
        Perplexity parameter for neighborhood definition

    Returns:
    --------
    float
        Normalized mean cLISI score across all cells (0-1 range)
    """
    if type == "embed":
        print("Using embed")
        if use_rep not in adata.obsm:
            raise ValueError(f"Embedding {use_rep} not found in adata.obsm")
        x = adata.obsm[use_rep]
    else:
        x = adata.X

    if label_key not in adata.obs:
        raise ValueError(f"Label key {label_key} not found in adata.obs")

    # Get number of unique cell types for normalization
    n_celltypes = len(adata.obs[label_key].unique())

    print("Computing LISI")
    # Compute LISI scores
    lisi_scores = compute_lisi(x, adata.obs, label_key, perplexity)

    # Normalize by number of cell types (perfect mixing = 1.0, no mixing = 1/n_celltypes)
    if n_celltypes > 1:
        normalized_scores = (lisi_scores - 1) / (n_celltypes - 1)
    else:
        normalized_scores = lisi_scores - 1

    if subset_indices is not None:
        return np.mean(normalized_scores[subset_indices])

    # Return mean normalized cLISI score
    return np.mean(normalized_scores)


# ---------------------------------------------------------------------------
# Probe-based conservation / mixing metrics — AUTHORED (K. Reid).
#
# WHY: scIB's ARI/NMI/ASW_celltype all require the latent to hold cell types in
#      COMPACT, geometrically separable clusters. A latent can encode cell type
#      perfectly well in a linearly-decodable but diffuse way and still score ~0 on
#      all three (observed: native NB_uncond on pancreas scores ARI 0.036 /
#      asw_celltype 0.497 while a linear probe reads cell type at 0.724 vs a 0.333
#      majority baseline). Probing measures the information that is present rather
#      than the geometry a clustering algorithm happens to recover.
# HOW: cross-validated classifiers on the embedding. Reported ABOVE the majority-class
#      baseline so the number is interpretable when classes are imbalanced.
#      The batch probe is the mirror image: it should be near ZERO after successful
#      integration, and it doubles as a guard against a silently inert adversary
#      (a batch probe far above baseline at high lambda_adv means no mixing happened).
# ---------------------------------------------------------------------------


def _probe_accuracy(embedding, labels, kind="knn", n_splits=3, max_cells=6000, seed=0):
    """Cross-validated probe accuracy and the majority-class baseline.

    Returns ``(accuracy, majority_baseline, lift)`` where ``lift = accuracy - majority``.
    ``lift`` near 0 means the embedding carries no information about ``labels``.
    """
    from collections import Counter

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    x = np.asarray(embedding)
    y = np.asarray(labels).astype(str)

    # Subsample for tractability on atlas-scale data; stratification keeps rare types.
    rng = np.random.default_rng(seed)
    if len(x) > max_cells:
        idx = rng.choice(len(x), size=max_cells, replace=False)
        x, y = x[idx], y[idx]

    # Drop classes too small to appear in every CV fold.
    counts = Counter(y)
    keep = np.array([counts[v] >= n_splits for v in y])
    if keep.sum() < n_splits or len(set(y[keep])) < 2:
        return float("nan"), float("nan"), float("nan")
    x, y = x[keep], y[keep]

    clf = (
        KNeighborsClassifier(n_neighbors=15)
        if kind == "knn"
        else LogisticRegression(max_iter=1000)
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc = float(cross_val_score(clf, x, y, cv=cv, scoring="accuracy").mean())
    majority = float(max(Counter(y).values()) / len(y))
    return acc, majority, acc - majority


def probe_metrics(adata, label_key, batch_key, embed_key="X_latent", seed=0):
    """Probe-based conservation (cell type) and residual-batch-signal metrics.

    Conservation: ``knn_label_acc`` / ``linear_label_acc`` with their ``*_lift`` over the
    majority-class baseline — higher is better.
    Residual batch: ``knn_batch_acc`` / ``linear_batch_acc`` and lifts — LOWER is better;
    a large positive ``linear_batch_lift`` means batch is still linearly decodable, i.e.
    integration did not remove it.
    """
    z = adata.obsm[embed_key]
    out = {}
    for kind, tag in (("knn", "knn"), ("linear", "linear")):
        acc, maj, lift = _probe_accuracy(z, adata.obs[label_key], kind=kind, seed=seed)
        out[f"{tag}_label_acc"] = acc
        out[f"{tag}_label_majority"] = maj
        out[f"{tag}_label_lift"] = lift
        acc_b, maj_b, lift_b = _probe_accuracy(z, adata.obs[batch_key], kind=kind, seed=seed)
        out[f"{tag}_batch_acc"] = acc_b
        out[f"{tag}_batch_majority"] = maj_b
        out[f"{tag}_batch_lift"] = lift_b
    return out


def gpu_silhouette_samples(x=None, labels=None, *, X=None, metric="euclidean",  # noqa: N803
                           chunk=2048, device=None, **_ignored):
    """Per-sample silhouette width, computed on the GPU when one is available.

    Drop-in replacement for ``sklearn.metrics.silhouette_samples`` with Euclidean
    metric. Falls back to sklearn when CUDA is absent, so callers need no branch.

    # WHY: silhouette-derived metrics are the largest GPU-addressable block in the
    #   metric suite -- measured 35% of the warm suite on pancreas (asw_celltype
    #   6.26s + isolated_asw 6.92s + asw_batch 1.59s of 41.9s) and 40% of a whole
    #   config on atac_large (353.05s of ~875s). The share GROWS with n, so this
    #   pays most on exactly the datasets that dominate the programme. Meanwhile the
    #   GPU sits near 20% utilisation because the suite is otherwise CPU-bound.
    # HOW: silhouette needs, per point, the mean distance to its own cluster (a) and
    #   the minimum mean distance to any other cluster (b). Both follow from
    #   per-cluster distance SUMS, so we never materialise the full n x n matrix:
    #   rows are processed in chunks and reduced straight into a (chunk, n_clusters)
    #   accumulator via index_add_. Peak VRAM is therefore O(chunk * n), measured at
    #   0.69 GB on pancreas and 3.25 GB on atac_large (84,813 cells) -- inside an 8 GB card.
    # PRECISION: float64 deliberately, not float32. float32 is ~94x but disagrees with
    #   sklearn by up to 7.6e-05 per sample; float64 still gives 9.5x on BOTH pancreas
    #   (5.43s -> 0.57s) and atac_large (143.20s -> 15.08s) while agreeing to 6.4e-09
    #   and 1.3e-11 respectively -- machine precision, so reported numbers do not
    #   depend on which backend ran. Singleton clusters return exactly 0.0, matching
    #   sklearn's convention (verified directly).
    """
    import numpy as np

    # scib calls these two ways: silhouette_samples(embed, labels) positionally, and
    # silhouette_score(X=embed, labels=..., metric=...) by keyword with a CAPITAL X
    # (sklearn's own parameter name). Accept both or the patch raises at the call site.
    if x is None:
        x = X
    if x is None or labels is None:
        raise TypeError("gpu_silhouette_samples requires the data matrix and labels")

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep of the package
        torch = None

    # The GPU kernel implements EUCLIDEAN distance only (torch.cdist p=2). Any other
    # metric must fall through to sklearn rather than silently returning euclidean
    # numbers under a different metric's name.
    if torch is None or not torch.cuda.is_available() or metric != "euclidean":
        from sklearn.metrics import silhouette_samples

        return silhouette_samples(x, labels, metric=metric)

    if device is None:
        device = "cuda"
    codes = np.unique(np.asarray(labels), return_inverse=True)[1]
    n_clusters = int(codes.max()) + 1
    if n_clusters < 2:
        raise ValueError("silhouette requires at least 2 clusters")

    xt = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=device)
    ct = torch.as_tensor(codes, dtype=torch.long, device=device)
    counts = torch.bincount(ct, minlength=n_clusters).to(torch.float64)
    out = torch.empty(xt.shape[0], dtype=torch.float64, device=device)

    for start in range(0, xt.shape[0], chunk):
        stop = min(start + chunk, xt.shape[0])
        dists = torch.cdist(xt[start:stop], xt)
        sums = torch.zeros(stop - start, n_clusters, dtype=torch.float64, device=device)
        sums.index_add_(1, ct, dists)
        own = ct[start:stop]
        # a: own cluster excludes the point itself, hence count-1 (0 for singletons)
        a = sums.gather(1, own[:, None]).squeeze(1) / (counts[own] - 1).clamp(min=1)
        mean_other = sums / counts[None, :].clamp(min=1)
        mean_other.scatter_(1, own[:, None], float("inf"))
        b = mean_other.min(dim=1).values
        sil = (b - a) / torch.maximum(a, b).clamp(min=1e-12)
        # sklearn defines the silhouette of a singleton cluster as 0
        out[start:stop] = torch.where(counts[own] > 1, sil, torch.zeros_like(sil))

    return out.cpu().numpy()


def gpu_silhouette_score(x=None, labels=None, *, X=None, metric="euclidean", **kwargs):  # noqa: N803
    """Mean silhouette width; drop-in for ``sklearn.metrics.silhouette_score``."""
    import numpy as np

    if x is None:
        x = X
    return float(np.mean(gpu_silhouette_samples(x, labels, metric=metric, **kwargs)))
