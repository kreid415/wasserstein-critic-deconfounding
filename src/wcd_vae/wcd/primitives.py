"""Core numerical primitives for the adversarial-deconfounding VAEs -- AUTHORED (K. Reid).

Independent implementations of the standard building blocks the models need:
negative-binomial log-likelihood, the Gaussian reparameterization sample, multi-class
cross-entropy, deterministic seeding, weight initialization, and an inference dataloader.
These are textbook quantities; the code here is written from their mathematical
definitions and shares no implementation with any third-party package.
"""
import random

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def nb_log_likelihood(x, mu, theta, eps=1e-8):
    """Per-element log-likelihood of counts ``x`` under NB(mean=mu, inverse-dispersion=theta).

    Uses the mean/inverse-dispersion parameterization. Writing p = mu / (mu + theta),
    the NB log-pmf is

        log Gamma(x + theta) - log Gamma(theta) - log Gamma(x + 1)
        + theta * log(theta / (theta + mu)) + x * log(mu / (theta + mu)).

    We compute the shared denominator log(theta + mu) once for numerical stability.
    """
    log_denom = torch.log(theta + mu + eps)
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - log_denom)
        + x * (torch.log(mu + eps) - log_denom)
    )


def gaussian_sample(mu, var, eps=1e-8):
    """Draw a reparameterized sample z = mu + sqrt(var) * epsilon, epsilon ~ N(0, I)."""
    std = torch.sqrt(var + eps)
    return mu + std * torch.randn_like(std)


def gaussian_kl_to_standard_normal(mu, var, eps=1e-8):
    """Closed-form KL( N(mu, var) || N(0, I) ), summed over the latent dimension."""
    return 0.5 * torch.sum(var + mu.pow(2) - 1.0 - torch.log(var + eps), dim=1)


class MultiClassCrossEntropy(nn.Module):
    """Standard V-way cross-entropy over batch logits (log-softmax + NLL)."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, reduction=self.reduction)


def seed_everything(seed):
    """Seed Python, NumPy, and torch (incl. CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_batchnorm_weights(module):
    """Initialize BatchNorm affine parameters (weight ~ N(1, 0.02), bias = 0)."""
    if module.__class__.__name__.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def inference_dataloader(adata, batch_size=2048):
    """Yield (expression, original_index) batches over adata.X in fixed order."""
    x = adata.X
    x = x.toarray() if isinstance(x, scipy.sparse.spmatrix) else np.asarray(x)
    x = torch.as_tensor(x, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x, torch.arange(x.shape[0]))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
