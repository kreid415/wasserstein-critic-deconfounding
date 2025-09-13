import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score as sk_silhouette
import torch

import math

from wcd_vae.metrics import (
    LISI,
    BatchEntropy,
    NormalizedMutualInfo,
    SilhouetteScore,
    compute_metrics,
)


def test_batch_entropy():
    # Two batches clearly separated
    X = np.vstack(
        [
            np.random.normal(loc=0.0, scale=1.0, size=(10, 2)),
            np.random.normal(loc=5.0, scale=1.0, size=(10, 2)),
        ]
    )
    labels = np.array([0] * 10 + [1] * 10)

    embeddings = torch.tensor(X, dtype=torch.float32)
    batch_labels = torch.tensor(labels, dtype=torch.int64)

    metric = BatchEntropy(k=5)
    metric.update(embeddings, batch_labels)
    result = metric.compute().item()

    # Use a small epsilon to tolerate floating-point imprecision
    assert result >= -1e-6, f"Entropy is unexpectedly negative: {result}"
    assert result <= math.log(2), f"Entropy exceeds theoretical upper bound: {result}"


def test_lisi_behavior():
    # Well-separated clusters with distinct labels
    X, y = make_blobs(n_samples=30, centers=3, n_features=5, random_state=42)
    embeddings = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.int64)

    metric = LISI(k=5)
    metric.update(embeddings, labels)
    result = metric.compute().item()

    assert result >= 1.0
    assert result <= 3.0  # Max LISI equals number of unique labels


def test_silhouette_score():
    # Two very distinct clusters
    X, y = make_blobs(n_samples=40, centers=2, cluster_std=0.5, n_features=3, random_state=42)
    embeddings = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.int64)

    metric = SilhouetteScore()
    metric.update(embeddings, labels)
    score = metric.compute()

    expected_score = sk_silhouette(X, y)
    assert np.isclose(score, expected_score, atol=1e-4)


def test_nmi_score():
    # Two clusters and correct labels
    X, y = make_blobs(n_samples=50, centers=2, n_features=3, random_state=42)
    embeddings = torch.tensor(X, dtype=torch.float32)
    true_labels = torch.tensor(y, dtype=torch.int64)

    metric = NormalizedMutualInfo()
    metric.update(embeddings, true_labels)
    score = metric.compute()

    assert 0.0 <= score <= 1.0
    assert score > 0.9  # Clusters should match well


def test_nmi_one_hot_labels():
    # Test with one-hot labels
    X, y = make_blobs(n_samples=40, centers=2, n_features=3, random_state=42)
    embeddings = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    y_onehot = torch.nn.functional.one_hot(y_tensor, num_classes=2).float()

    metric = NormalizedMutualInfo()
    metric.update(embeddings, y_onehot)
    score = metric.compute()

    assert 0.0 <= score <= 1.0


def test_silhouette_one_hot_labels():
    # One-hot encoded labels
    X, y = make_blobs(n_samples=30, centers=2, n_features=3, random_state=42)
    embeddings = torch.tensor(X, dtype=torch.float32)
    labels = torch.nn.functional.one_hot(torch.tensor(y), num_classes=2).float()

    metric = SilhouetteScore()
    metric.update(embeddings, labels)
    result = metric.compute()

    assert 0.0 <= result <= 1.0


def test_batch_entropy_low_when_batches_separated():
    X = np.vstack(
        [
            np.random.normal(loc=0.0, scale=0.5, size=(20, 2)),
            np.random.normal(loc=5.0, scale=0.5, size=(20, 2)),
        ]
    )
    labels = np.array([0] * 20 + [1] * 20)
    embeddings = torch.tensor(X, dtype=torch.float32)
    batch_labels = torch.tensor(labels, dtype=torch.int64)

    metric = BatchEntropy(k=5)
    metric.update(embeddings, batch_labels)
    entropy_low = metric.compute().item()

    # Mix batches at same location for high entropy
    X_mixed = np.random.normal(loc=0.0, scale=1.0, size=(40, 2))
    labels_mixed = np.array([0, 1] * 20)
    embeddings_mixed = torch.tensor(X_mixed, dtype=torch.float32)
    batch_labels_mixed = torch.tensor(labels_mixed, dtype=torch.int64)

    metric_high = BatchEntropy(k=5)
    metric_high.update(embeddings_mixed, batch_labels_mixed)
    entropy_high = metric_high.compute().item()

    assert entropy_low < entropy_high, "Batch entropy should be lower when batches are separated"


def test_lisi_high_when_mixed_labels_low_when_homogeneous():
    # Create two spatial clusters
    X = np.vstack([np.random.normal(0, 0.1, size=(15, 2)), np.random.normal(5, 0.1, size=(15, 2))])
    embeddings = torch.tensor(X, dtype=torch.float32)

    # Homogeneous labeling: each cluster has a single label
    labels_homogeneous = np.array([0] * 15 + [1] * 15)
    labels_homogeneous = torch.tensor(labels_homogeneous, dtype=torch.int64)

    metric_low = LISI(k=5)
    metric_low.update(embeddings, labels_homogeneous)
    lisi_low = metric_low.compute().item()

    # Mixed labeling: random across space
    labels_mixed = np.random.choice([0, 1], size=30)
    labels_mixed = torch.tensor(labels_mixed, dtype=torch.int64)

    metric_high = LISI(k=5)
    metric_high.update(embeddings, labels_mixed)
    lisi_high = metric_high.compute().item()

    print(f"LISI (homogeneous): {lisi_low:.4f}, LISI (mixed): {lisi_high:.4f}")
    assert lisi_low < lisi_high, (
        f"LISI should be higher for mixed labels. Got low={lisi_low}, high={lisi_high}"
    )


def test_silhouette_high_for_separated_clusters_low_for_overlap():
    # High silhouette (2 well-separated clusters)
    X_sep, y_sep = make_blobs(n_samples=40, centers=2, cluster_std=0.3, random_state=42)
    embeddings_sep = torch.tensor(X_sep, dtype=torch.float32)
    labels_sep = torch.tensor(y_sep, dtype=torch.int64)

    metric_high = SilhouetteScore()
    metric_high.update(embeddings_sep, labels_sep)
    score_high = metric_high.compute()

    # Low silhouette (overlapping clusters)
    X_ovlp, y_ovlp = make_blobs(n_samples=40, centers=2, cluster_std=2.5, random_state=42)
    embeddings_ovlp = torch.tensor(X_ovlp, dtype=torch.float32)
    labels_ovlp = torch.tensor(y_ovlp, dtype=torch.int64)

    metric_low = SilhouetteScore()
    metric_low.update(embeddings_ovlp, labels_ovlp)
    score_low = metric_low.compute()

    assert score_low < score_high, "Silhouette score should be lower for overlapping clusters"


def test_nmi_high_when_clusters_match_labels_low_when_random():
    X, y = make_blobs(n_samples=50, centers=3, n_features=3, random_state=42)
    embeddings = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.int64)

    metric_high = NormalizedMutualInfo()
    metric_high.update(embeddings, labels)
    nmi_high = metric_high.compute()

    # Shuffle labels to create mismatch
    rng = np.random.default_rng(seed=42)
    shuffled_labels = torch.tensor(rng.permutation(y), dtype=torch.int64)

    metric_low = NormalizedMutualInfo()
    metric_low.update(embeddings, shuffled_labels)
    nmi_low = metric_low.compute()

    assert nmi_low < nmi_high, "NMI should be lower when clustering doesn't match true labels"
