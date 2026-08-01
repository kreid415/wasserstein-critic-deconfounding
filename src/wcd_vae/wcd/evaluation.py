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
