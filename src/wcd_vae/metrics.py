from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import torch
from torchmetrics import Metric


class BatchEntropy(Metric):
    def __init__(self, k: int = 50):
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
    def __init__(self, k: int = 90):
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
