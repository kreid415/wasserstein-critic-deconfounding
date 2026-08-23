"""Critic-free alignment divergences (MMD, Sinkhorn) and the spectral-norm critic option.

These are the "other formulations" implemented for future experiments alongside the pooled critic:
- alignment.py provides critic-FREE divergences (no adversary network) between each batch and the
  pool, testing whether the IPM geometry helps once the adversarial estimator is removed;
- Discriminator(spectral_norm=True) enforces the critic's Lipschitz constraint per-layer instead
  of via the sampled gradient penalty.
"""
import numpy as np
import pytest
import torch

from wcd_vae.wcd.alignment import (
    CRITIC_FREE_LOSSES,
    mmd_batch_pool,
    sinkhorn_batch_pool,
)
from wcd_vae.wcd.adversarial import Discriminator


def _two_batches(sep, n=120, d=16, seed=0):
    """Two Gaussian batches separated by `sep` along axis 0. sep=0 => identical distributions."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(2 * n, d, generator=g)
    z[n:, 0] += sep
    b = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    return z, b


@pytest.mark.parametrize("fn", [mmd_batch_pool, sinkhorn_batch_pool])
def test_alignment_nonnegative_and_finite(fn):
    z, b = _two_batches(sep=3.0)
    v = fn(z, b)
    assert torch.isfinite(v), f"{fn.__name__}: non-finite"
    assert v.item() >= -1e-6, f"{fn.__name__}: negative divergence {v.item()}"


@pytest.mark.parametrize("fn", [mmd_batch_pool, sinkhorn_batch_pool])
def test_alignment_larger_when_batches_more_separated(fn):
    """The divergence must INCREASE with batch separation (more separable = worse mixing)."""
    z0, b0 = _two_batches(sep=0.0)
    z1, b1 = _two_batches(sep=5.0)
    lo = fn(z0, b0).item()
    hi = fn(z1, b1).item()
    assert hi > lo, f"{fn.__name__}: not monotone in separation (sep0={lo:.4f}, sep5={hi:.4f})"


@pytest.mark.parametrize("fn", [mmd_batch_pool, sinkhorn_batch_pool])
def test_alignment_differentiable_wrt_z(fn):
    """The generator minimises the divergence, so it must be differentiable in z."""
    z, b = _two_batches(sep=3.0)
    z = z.clone().requires_grad_(True)
    v = fn(z, b)
    v.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0, f"{fn.__name__}: zero gradient"


def test_registry_names():
    assert set(CRITIC_FREE_LOSSES) == {"mmd", "sinkhorn"}


def test_spectral_norm_critic_builds_and_forwards():
    """Discriminator(spectral_norm=True) must build a valid critic and return (loss, gp)."""
    z, b = _two_batches(sep=2.0, d=16)
    for formu, ref in (("reference", 0), ("pooled", None), ("barycenter", None)):
        head = Discriminator(n_input=16, domain_number=2, critic=True,
                             reference_batch=ref, formulation=formu, spectral_norm=True)
        out = head(z, b, reference_batch=ref)
        assert isinstance(out, tuple) and torch.isfinite(out[0]).all()
        # spectral_norm registers a parametrization on each fc layer
        assert any("parametrizations" in n for n, _ in head.named_modules()), \
            f"{formu}: spectral_norm not applied"


def test_spectral_norm_constrains_layer_norm_to_one():
    """Spectral normalisation bounds each fc layer's largest singular value to 1 (that IS the
    Lipschitz constraint). Run a forward to trigger the power-iteration update, then check each
    normalised weight's spectral norm sits at ~1, while the un-normalised head's does not."""
    z = torch.randn(64, 16)
    sn = Discriminator(n_input=16, domain_number=2, critic=True,
                       reference_batch=0, formulation="reference", spectral_norm=True)
    plain = Discriminator(n_input=16, domain_number=2, critic=True,
                          reference_batch=0, formulation="reference", spectral_norm=False)
    # several forwards so the power iteration converges
    with torch.no_grad():
        for _ in range(10):
            sn(z, None)
    for layer in (sn.fc1, sn.fc2, sn.fc3):
        s = torch.linalg.matrix_norm(layer.weight, ord=2).item()
        assert abs(s - 1.0) < 0.15, f"spectral-norm layer singular value {s:.3f} not near 1"
    # the plain head is NOT so constrained (at least one layer's spectral norm strays from 1)
    plain_svs = [torch.linalg.matrix_norm(l.weight, ord=2).item() for l in (plain.fc1, plain.fc2, plain.fc3)]
    assert any(abs(s - 1.0) >= 0.15 for s in plain_svs), f"plain head unexpectedly ~1-Lipschitz: {plain_svs}"
