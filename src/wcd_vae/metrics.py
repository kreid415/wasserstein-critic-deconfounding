from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import torch
from torchmetrics import Metric


class BatchEntropy(Metric):
    def __init__(self, k: int = 30):
        super().__init__()
        self.k = k
        self.add_state("total_entropy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, embeddings: torch.Tensor, batch_labels: torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
        batch_labels_np = batch_labels.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(embeddings_np)
        indices = nbrs.kneighbors(embeddings_np, return_distance=False)[:, 1:]

        entropies = []
        for neighbors in indices:
            neigh_labels = batch_labels_np[neighbors]
            _, counts = np.unique(neigh_labels, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        self.total_entropy += torch.tensor(entropies).sum()
        self.total_count += len(entropies)

    def compute(self):
        return self.total_entropy / self.total_count


class LISI(Metric):
    def __init__(self, k: int = 30):
        super().__init__()
        self.k = k
        self.add_state("total_lisi", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
        labels_np = labels.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(embeddings_np)
        indices = nbrs.kneighbors(embeddings_np, return_distance=False)[:, 1:]

        lisi_scores = []
        for neighbors in indices:
            neigh_labels = labels_np[neighbors]
            _, counts = np.unique(neigh_labels, return_counts=True)
            probs = counts / counts.sum()
            simpson_index = 1.0 / np.sum(probs**2)
            lisi_scores.append(simpson_index)
        self.total_lisi += torch.tensor(lisi_scores).sum()
        self.total_count += len(lisi_scores)

    def compute(self):
        return self.total_lisi / self.total_count


class SilhouetteScore(Metric):
    def __init__(self):
        super().__init__()
        self.embeddings = []
        self.labels = []

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings.append(embeddings.cpu())
        self.labels.append(labels.cpu())

    def compute(self):
        embeddings = torch.cat(self.embeddings, dim=0).numpy()
        labels = torch.cat(self.labels, dim=0)

        # Handle one-hot encoded labels
        if labels.ndim == 2 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)

        labels = labels.numpy()

        return silhouette_score(embeddings, labels)


class NormalizedMutualInfo(Metric):
    def __init__(self, n_clusters: Optional[int] = None):
        super().__init__()
        self.embeddings = []
        self.true_labels = []
        self.n_clusters = n_clusters  # If None, will infer from labels

    def update(self, embeddings: torch.Tensor, true_labels: torch.Tensor):
        self.embeddings.append(embeddings.cpu())

        # Ensure true_labels is 1D (convert one-hot or 2D to class indices)
        if true_labels.ndim == 2:
            true_labels = torch.argmax(true_labels, dim=1)

        self.true_labels.append(true_labels.cpu())

    def compute(self):
        embeddings = torch.cat(self.embeddings, dim=0).numpy()
        true_labels = torch.cat(self.true_labels, dim=0).numpy()

        # Compute number of clusters if not provided
        n_clusters = self.n_clusters or len(np.unique(true_labels))

        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        pred_labels = kmeans.fit_predict(embeddings)

        return normalized_mutual_info_score(true_labels, pred_labels)


def compute_metrics(
    embeddings: torch.Tensor,
    batch_labels: torch.Tensor,
    cell_type_labels: torch.Tensor,
    k_entropy: int = 50,
    k_lisi: int = 90,
) -> dict:
    """
    Compute key integration and conservation metrics from embeddings.

    Args:
        embeddings (torch.Tensor): Embedding matrix of shape (N, D)
        batch_labels (torch.Tensor): Batch labels for each cell (N,)
        cell_type_labels (torch.Tensor): Cell type labels for each cell (N,)
        k_entropy (int): Neighborhood size for BatchEntropy
        k_lisi (int): Neighborhood size for LISI

    Returns:
        dict: Dictionary with metric names and computed values
    """
    be = BatchEntropy(k=k_entropy)
    ilisi = LISI(k=k_lisi)
    clisi = LISI(k=k_lisi)
    sil = SilhouetteScore()
    nmi = NormalizedMutualInfo()

    # Update metrics
    be.update(embeddings, batch_labels)
    ilisi.update(embeddings, batch_labels)
    clisi.update(embeddings, cell_type_labels)
    sil.update(embeddings, cell_type_labels)
    nmi.update(cell_type_labels, batch_labels)

    # Return results
    return {
        "batch_entropy": be.compute().item(),
        "ilisi_batch": ilisi.compute().item(),
        "clisi_celltype": clisi.compute().item(),
        "silhouette_score": sil.compute(),
        "normalized_mutual_info": nmi.compute(),
    }


def compute_lisi(x, metadata, label_colname, perplexity=30):
    """
    Compute Local Inverse Simpson Index (LISI) for batch mixing evaluation.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The embedded data matrix
    metadata : pandas.DataFrame
        Metadata containing batch/label information
    label_colname : str
        Column name in metadata containing the batch labels
    perplexity : int, default=30
        Perplexity parameter for Gaussian kernel (similar to t-SNE)

    Returns:
    --------
    lisi_scores : array-like
        LISI score for each cell
    """
    n_cells = x.shape[0]

    # Get batch labels
    batch_labels = metadata[label_colname].values
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    # Create mapping from batch to index
    batch_to_idx = {batch: idx for idx, batch in enumerate(unique_batches)}
    batch_indices = np.array([batch_to_idx[batch] for batch in batch_labels])

    # Find k-nearest neighbors (k should be larger than perplexity)
    k = min(90, n_cells - 1)  # Use 90 neighbors or n_cells-1 if smaller
    print(f"Computing {k} nearest neighbors for {n_cells} cells...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(x)
    distances, indices = nbrs.kneighbors(x)

    lisi_scores = np.zeros(n_cells)

    # Add progress bar for LISI computation
    print(f"Computing LISI scores for {label_colname}...")
    for i in range(n_cells):
        # Get neighbors and distances for current cell
        neighbor_indices = indices[i, 1:]  # Exclude self (index 0)
        neighbor_distances = distances[i, 1:]

        # Compute Gaussian kernel weights with adaptive bandwidth
        # Find bandwidth that gives desired perplexity
        sigma = find_sigma(neighbor_distances, perplexity)
        weights = np.exp(-(neighbor_distances**2) / (2 * sigma**2))
        weights = weights / np.sum(weights)  # Normalize

        # Get batch labels of neighbors
        neighbor_batches = batch_indices[neighbor_indices]

        # Compute probability of each batch in neighborhood
        batch_probs = np.zeros(n_batches)
        for j, batch_idx in enumerate(neighbor_batches):
            batch_probs[batch_idx] += weights[j]

        # Avoid division by zero
        batch_probs = batch_probs + 1e-12

        # Compute Simpson diversity (inverse Simpson index)
        simpson_index = np.sum(batch_probs**2)
        lisi_scores[i] = 1.0 / simpson_index

    return lisi_scores


def find_sigma(distances, target_perplexity, tol=1e-5, max_iter=50):
    """
    Find the Gaussian kernel bandwidth (sigma) that achieves target perplexity.
    Uses binary search similar to t-SNE implementation.
    """

    def perplexity_fn(sigma):
        if sigma <= 0:
            return 0
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        weights = weights / np.sum(weights)
        # Avoid log(0)
        weights = np.maximum(weights, 1e-12)
        h = -np.sum(weights * np.log2(weights))
        return 2**h

    # Binary search for sigma
    sigma_min, sigma_max = 1e-20, 1000.0

    for _ in range(max_iter):
        sigma = (sigma_min + sigma_max) / 2.0
        perp = perplexity_fn(sigma)

        if abs(perp - target_perplexity) < tol:
            break

        if perp > target_perplexity:
            sigma_max = sigma
        else:
            sigma_min = sigma

    return sigma


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
    normalized_scores = (lisi_scores - 1) / (n_batches - 1)

    # --- START NEW CODE ---
    # If indices are provided, only return the mean for those cells
    if subset_indices is not None:
        return np.mean(normalized_scores[subset_indices])
    # --- END NEW CODE ---

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
    normalized_scores = (lisi_scores - 1) / (n_celltypes - 1)

    # --- START NEW CODE ---
    if subset_indices is not None:
        return np.mean(normalized_scores[subset_indices])
    # --- END NEW CODE ---

    # Return mean normalized cLISI score
    return np.mean(normalized_scores)
