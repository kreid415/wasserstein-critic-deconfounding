"""Checks for the 2026-10-02 critic-free losses and the barycenter target.

Sinkhorn: non-negative, zero for identical batches, close to the exact W1 (POT network simplex)
for small blur, batched result equals one problem at a time, finite gradients; value and gradient
equal to a converged reference (two Diracs exactly; standardised and narrow-far geometries against
the final review's float64 reference), and an unconverged solve or a bad p raises.
MMD: equals the explicit per-batch formula, zero for identical batches.
Barycenter: reproduces a single measure, sits at the midpoint of two translated clouds, and
its W2 objective beats using either input cloud as the target.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

ot = pytest.importorskip("ot")

from wcd_vae.wcd.alignment import (
    _batch_masks,
    _log_w,
    _ot_eps,
    _ot_eps_sym,
    _sqdist,
    mmd_batch_others,
    mmd_reference,
    sinkhorn_batch_others,
    sinkhorn_divergence_weighted,
)
from wcd_vae.wcd.barycenter import barycenter_objective, batch_barycenter_support


def _two_clouds(n0=40, n1=30, d=5, shift=2.0, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n0, d, generator=g, dtype=dtype)
    x1 = torch.randn(n1, d, generator=g, dtype=dtype) + shift
    z = torch.cat([x0, x1])
    b = torch.cat([torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long)])
    return z, b


def test_sinkhorn_nonnegative_and_zero_for_identical_batches():
    z, b = _two_clouds()
    assert float(sinkhorn_batch_others(z, b)) > 0
    x = z[:30]
    zz = torch.cat([x, x])
    bb = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)])
    assert abs(float(sinkhorn_batch_others(zz, bb))) < 1e-6


def test_sinkhorn_approximates_exact_w1_for_small_blur():
    z, b = _two_clouds(shift=1.5)
    x0, x1 = z[b == 0].numpy(), z[b == 1].numpy()
    w1 = ot.emd2(np.full(len(x0), 1 / len(x0)), np.full(len(x1), 1 / len(x1)), ot.dist(x0, x1, metric="euclidean"))
    s = float(sinkhorn_batch_others(z, b, blur_frac=0.002, scaling=0.95))
    assert abs(s - w1) / w1 < 0.03, (s, w1)


def test_sinkhorn_batched_equals_one_problem_at_a_time():
    g = torch.Generator().manual_seed(3)
    z = torch.randn(60, 4, generator=g, dtype=torch.float64)
    b = torch.randint(0, 4, (60,), generator=g)
    A, B, K = _batch_masks(b, z)
    C = torch.sqrt(_sqdist(z) + 1e-12)
    blur = 0.05 * math.sqrt(8.0)
    # converged far below SINKHORN_TOL: a batched solve iterates until its slowest problem converges
    batched = sinkhorn_divergence_weighted(C, A, B, blur, tol=1e-10)
    single = torch.stack([sinkhorn_divergence_weighted(C, A[k:k + 1], B[k:k + 1], blur, tol=1e-10)[0]
                          for k in range(K)])
    assert torch.allclose(batched, single, atol=1e-8, rtol=0)


@pytest.mark.parametrize("p", [1, 2])
def test_sinkhorn_two_diracs_exact(p):
    """Regression (final review I031): the tagged code's gradient was twice the envelope gradient (+-2 at p=1)."""
    z = torch.tensor([[0.0], [2.0]], requires_grad=True)
    v = sinkhorn_batch_others(z, torch.tensor([0, 1]), p=p)
    (gz,) = torch.autograd.grad(v, z)
    # cross cost |x - y| = 2 or |x - y|^2 / 2 = 2; each self-transport costs sqrt(1e-12) at p=1 and 0 at p=2
    assert v.dtype == torch.float32 and abs(float(v) - (2.0 - (1e-6 if p == 1 else 0.0))) < 1e-6
    assert torch.allclose(gz, torch.tensor([[-1.0], [1.0]]) * p, atol=1e-6)


def _review_reference():
    """The final review's converged float64 reference (review/esc2/sinkhorn_astra, copied unchanged)."""
    path = Path(__file__).resolve().parents[2] / "experiments" / "sinkhorn_fix_20261013" / "sinkhorn_reference.py"
    spec = importlib.util.spec_from_file_location("sinkhorn_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _standardised():
    """A3-like: four overlapping batches of unequal size, standardised per dimension as the plan's zstd does."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(128, 10, generator=g, dtype=torch.float64)
    b = torch.multinomial(torch.tensor([0.4, 0.3, 0.2, 0.1]), 128, replacement=True, generator=g)
    z[:, :3] += 0.3 * b[:, None].double()
    return z, b, True


def _narrow_far():
    """R1-extension-like: a latent of spread 0.01 at distance ~16 from the origin."""
    g = torch.Generator().manual_seed(0)
    z = 5.0 + 0.01 * torch.randn(128, 10, generator=g, dtype=torch.float64)
    b = torch.randint(0, 3, (128,), generator=g)
    z[b == 1, 0] += 0.01
    return z, b, False


@pytest.mark.parametrize("case", [_standardised, _narrow_far])
# test-reason: catches an unconverged final eps (I032) and the float32 distance cancellation; the Diracs cannot
def test_sinkhorn_matches_converged_reference(case):
    """Value and gradient (w.r.t. the raw float32 input, through the standardisation) within 1e-5 and 1e-3 of the
    converged reference. The tagged code: gradient error 1.45 / 1.13; the gradient fix without the convergence
    loop: 4.4e-2 (standardised); without the stable distances: 0.33 (narrow-far)."""
    ref = _review_reference()
    z64, b, zstd = case()

    def adv(z):
        return (z - z.mean(0, keepdim=True)) / z.std(0, unbiased=False, keepdim=True).clamp_min(1e-4) if zstd else z

    mu = z64.float().requires_grad_(True)
    v = sinkhorn_batch_others(adv(mu), b)
    (g,) = torch.autograd.grad(v, mu)
    m64 = z64.clone().requires_grad_(True)
    a64 = adv(m64)
    v_ref, _ = ref.reference_cost(ref.cost_matrix(a64), b, 0.05 * math.sqrt(2 * a64.shape[1]), tol=1e-10)
    (g_ref,) = torch.autograd.grad(v_ref, m64)
    assert abs(float(v) - float(v_ref)) < 1e-5 * abs(float(v_ref))
    assert float((g.double() - g_ref).norm() / g_ref.norm()) < 1e-3


def _small_problem():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(8, 2, generator=g, dtype=torch.float64)
    A, B, _ = _batch_masks(torch.tensor([0, 1] * 4), z)
    return torch.sqrt(_sqdist(z) + 1e-12), _log_w(A), _log_w(B)


@pytest.mark.parametrize("call, exc, match", [
    (lambda: sinkhorn_batch_others(torch.randn(8, 2), torch.tensor([0, 1] * 4), p=3), ValueError, "p must be 1 or 2"),
    (lambda: _ot_eps(*_small_problem(), [0.1], 0.0, 3), RuntimeError, "after 3 iterations"),
    (lambda: _ot_eps_sym(*_small_problem()[:2], [0.1], 0.0, 3), RuntimeError, "after 3 iterations"),
])
def test_sinkhorn_refuses_bad_p_and_unconverged_solve(call, exc, match):
    with pytest.raises(exc, match=match):
        call()


@pytest.mark.parametrize("fn", [mmd_batch_others, sinkhorn_batch_others])
def test_gradients_finite_including_duplicate_points(fn):
    z, b = _two_clouds(dtype=torch.float32)
    z = torch.cat([z, z[:3]]).clone().requires_grad_(True)       # exact duplicates -> zero distances
    b = torch.cat([b, b[:3]])
    loss = fn(z, b)
    (gz,) = torch.autograd.grad(loss, z)
    assert torch.isfinite(gz).all() and gz.abs().sum() > 0


def test_mmd_matches_explicit_formula():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(50, 3, generator=g, dtype=torch.float64)
    b = torch.randint(0, 3, (50,), generator=g)
    d = z.shape[1]

    def k(x, y):
        d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
        return sum(torch.exp(-d2 / (2 * d * m * m)) for m in (0.25, 0.5, 1.0, 2.0, 4.0)) / 5

    terms = []
    for c in torch.unique(b):
        x, y = z[b == c], z[b != c]
        terms.append(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())
    assert torch.allclose(mmd_batch_others(z, b), torch.stack(terms).mean(), atol=1e-10, rtol=0)


def test_mmd_zero_for_identical_batches():
    x = torch.randn(25, 4, dtype=torch.float64)
    zz = torch.cat([x, x])
    bb = torch.cat([torch.zeros(25, dtype=torch.long), torch.ones(25, dtype=torch.long)])
    assert abs(float(mmd_batch_others(zz, bb))) < 1e-12


def test_barycenter_of_one_measure_is_that_measure():
    torch.manual_seed(0)
    x = torch.randn(30, 3, dtype=torch.float64)
    b = torch.zeros(30, dtype=torch.long)
    s = batch_barycenter_support(x, b, n_support=30, n_iter=20)
    assert barycenter_objective(s, x, b) < 1e-10


def test_barycenter_of_two_translated_clouds_is_the_midpoint():
    torch.manual_seed(0)
    z, b = _two_clouds(n0=60, n1=60, d=3, shift=4.0)
    s = batch_barycenter_support(z, b, n_support=60, n_iter=30)
    mid = 0.5 * (z[b == 0].mean(0) + z[b == 1].mean(0))
    assert torch.allclose(s.mean(0), mid, atol=0.05), (s.mean(0), mid)
    obj = barycenter_objective(s, z, b)
    for k in (0, 1):
        assert obj < barycenter_objective(z[b == k], z, b)


def test_barycenter_support_is_detached_and_reproducible():
    z, b = _two_clouds()
    z = z.clone().requires_grad_(True)
    torch.manual_seed(5); s1 = batch_barycenter_support(z, b)
    torch.manual_seed(5); s2 = batch_barycenter_support(z, b)
    assert not s1.requires_grad and torch.equal(s1, s2) and s1.shape == z.shape


def test_mmd_reference_matches_explicit_formula():
    g = torch.Generator().manual_seed(2)
    z = torch.randn(60, 3, generator=g, dtype=torch.float64)
    b = torch.randint(0, 4, (60,), generator=g)
    d = z.shape[1]

    def k(x, y):
        d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
        return sum(torch.exp(-d2 / (2 * d * m * m)) for m in (0.25, 0.5, 1.0, 2.0, 4.0)) / 5

    r = z[b == 1]
    terms = [k(z[b == c], z[b == c]).mean() + k(r, r).mean() - 2 * k(z[b == c], r).mean()
             for c in torch.unique(b) if int(c) != 1]
    assert torch.allclose(mmd_reference(z, b, 1), torch.stack(terms).mean(), atol=1e-10, rtol=0)


def test_barycenter_custom_init_stays_in_hull_and_near_cold_objective():
    """From any init, one fixed-point step puts every atom inside the convex hull of the current
    minibatch, and 20 iterations from a bad init reach within 2% of the cold-start objective.
    (This does NOT make a warm start equivalent: it can converge to a different support; see the
    docstring of batch_barycenter_support.)"""
    torch.manual_seed(1)
    z, b = _two_clouds(n0=64, n1=64, d=4, shift=3.0)
    cold = batch_barycenter_support(z, b, n_support=64, n_iter=50)
    far = torch.full((64, 4), 100.0, dtype=torch.float64)          # deliberately bad init
    warm = batch_barycenter_support(z, b, n_support=64, n_iter=20, init=far)
    assert warm.abs().max() < z.abs().max() + 1e-9
    assert barycenter_objective(warm, z, b) <= barycenter_objective(cold, z, b) * 1.02
