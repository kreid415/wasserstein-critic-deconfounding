"""Unit tests for the X6 / X7 / X8 / X3 pieces (docs/SPECS_missing_arms.md, signed off 2026-10-02).

R1 (SI-22): gradcheck; R1 = 0 for a constant head; linear two-batch head gives (gamma/2)||w1 - w0||^2 exactly;
            three batches match a finite-difference evaluation; K = 2 log-odds are the binary logit.
reference JS (SI-23): gradcheck; log 4 at zero logits; log 4 - 2 JSD at the Bayes logit of two Gaussians;
            equals an explicit per-head loop; inactive heads; generator gradient reaches reference cells.
importance weights (SI-25): for the discriminator CE / fool loss, the reference and pooled critic and MMD:
            unit weights give the unweighted loss, integer weights equal duplicated cells, zero-weight cells
            get no gradient, gradcheck.
stratified sampler (SI-24): exact per-batch counts, step count, distinct indices, once per cycle,
            reproducibility, refusal below quota.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from wcd_vae.wcd.alignment import mmd_batch_others
from wcd_vae.wcd.critic import ReferenceWassersteinLoss
from wcd_vae.wcd.discriminator_losses import (
    fool_loss,
    one_vs_rest_logodds,
    r1_penalty,
    reference_js_losses,
    weighted_ce,
)
from wcd_vae.wcd.sampling import StratifiedBatchSampler

D64 = torch.float64


def _data(n=60, d=4, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g, dtype=D64)
    b = torch.arange(n) % k
    return z, b[torch.randperm(n, generator=g)]


def _tanh_head(params):
    w1, b1, w2, b2 = params
    return lambda x: torch.tanh(x @ w1 + b1) @ w2 + b2


def _head_params(d=4, h=6, k=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(d, h, generator=g, dtype=D64), torch.randn(h, generator=g, dtype=D64),
            torch.randn(h, k, generator=g, dtype=D64), torch.randn(k, generator=g, dtype=D64)]


# ------------------------------------------------------------------ R1 (X6)
def test_one_vs_rest_logodds_two_batches_is_the_binary_logit():
    z, b = _data(k=2)
    logits = torch.randn(z.shape[0], 2, dtype=D64)
    d = one_vs_rest_logodds(logits, b)
    expect = torch.where(b == 1, logits[:, 1] - logits[:, 0], logits[:, 0] - logits[:, 1])
    assert torch.allclose(d, expect, rtol=0, atol=1e-14)


def test_r1_is_zero_for_a_constant_head():
    z, b = _data()
    bias = torch.randn(3, dtype=D64)
    pen, _ = r1_penalty(lambda x: (x @ torch.zeros(4, 3, dtype=D64)) + bias, z, b, gamma=10.0)
    assert float(pen) == 0.0


def test_r1_linear_two_batch_head_known_answer():
    z, b = _data(k=2)
    w = torch.randn(4, 2, dtype=D64)
    c = torch.randn(2, dtype=D64)
    pen, _ = r1_penalty(lambda x: x @ w + c, z, b, gamma=10.0)
    expect = 0.5 * 10.0 * float((w[:, 1] - w[:, 0]).pow(2).sum())
    assert math.isclose(float(pen), expect, rel_tol=1e-12)


def test_r1_three_batches_matches_finite_differences():
    z, b = _data(n=12)
    f = _tanh_head(_head_params())
    pen, _ = r1_penalty(f, z, b, gamma=10.0)
    eps, tot = 1e-6, 0.0
    for i in range(z.shape[0]):
        g = torch.zeros(z.shape[1], dtype=D64)
        for j in range(z.shape[1]):
            e = torch.zeros(z.shape[1], dtype=D64)
            e[j] = eps
            up = one_vs_rest_logodds(f((z[i] + e)[None]), b[i:i + 1])
            dn = one_vs_rest_logodds(f((z[i] - e)[None]), b[i:i + 1])
            g[j] = (up - dn) / (2 * eps)
        tot += float(g.pow(2).sum())
    assert math.isclose(float(pen), 5.0 * tot / z.shape[0], rel_tol=1e-6)


def test_r1_gradcheck_wrt_input_and_head_parameters():
    z, b = _data(n=10)
    params = [p.clone().requires_grad_(True) for p in _head_params()]
    zz = z.clone().requires_grad_(True)

    def fn(zi, *ps):
        return r1_penalty(_tanh_head(ps), zi, b, gamma=10.0, detach_input=False)[0]

    assert torch.autograd.gradcheck(fn, (zz, *params), eps=1e-6, atol=1e-6)


def test_r1_refuses_non_positive_gamma():
    z, b = _data()
    with pytest.raises(ValueError, match="gamma"):
        r1_penalty(lambda x: x @ torch.ones(4, 3, dtype=D64), z, b, gamma=0.0)


# ------------------------------------------------------------------ reference JS (X7)
def _refjs_loop(logits, b, r):
    """Explicit per-head computation of SPECS section 2."""
    ld, lg = [], []
    for k in range(logits.shape[1]):
        if k == r or not bool((b == k).any()) or not bool((b == r).any()):
            continue
        lk, lr = logits[b == k, k], logits[b == r, k]
        ld.append(F.softplus(-lk).mean() + F.softplus(lr).mean())
        lg.append(F.softplus(lk).mean() + F.softplus(-lr).mean())
    return torch.stack(ld).mean(), torch.stack(lg).mean()


def test_reference_js_log4_at_zero_logits():
    _, b = _data(k=4)
    l_d, l_g = reference_js_losses(torch.zeros(b.numel(), 4, dtype=D64), b, 1)
    assert math.isclose(float(l_d), math.log(4.0), rel_tol=1e-14)
    assert math.isclose(float(l_g), math.log(4.0), rel_tol=1e-14)


def test_reference_js_equals_explicit_per_head_loop_with_unequal_sides():
    g = torch.Generator().manual_seed(4)
    b = torch.tensor([0] * 7 + [1] * 30 + [2] * 3 + [3] * 12)
    logits = torch.randn(b.numel(), 4, generator=g, dtype=D64)
    for r in range(4):
        got = reference_js_losses(logits, b, r)
        exp = _refjs_loop(logits, b, r)
        assert torch.allclose(got[0], exp[0], rtol=0, atol=1e-13) and torch.allclose(got[1], exp[1], rtol=0, atol=1e-13)


def test_reference_js_at_the_bayes_logit_is_log4_minus_two_jsd():
    mu, n = 1.5, 400_000
    g = torch.Generator().manual_seed(7)
    x_k = torch.randn(n, generator=g, dtype=D64) + mu
    x_r = torch.randn(n, generator=g, dtype=D64)
    x = torch.cat([x_r, x_k])
    b = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    logits = torch.stack([torch.zeros_like(x), mu * x - mu * mu / 2], 1)      # log N(x; mu,1) - log N(x; 0,1)
    l_d, _ = reference_js_losses(logits, b, reference_batch=0)
    grid = np.linspace(-12, 14, 200_001)
    p = np.exp(-(grid - mu) ** 2 / 2) / math.sqrt(2 * math.pi)
    q = np.exp(-grid ** 2 / 2) / math.sqrt(2 * math.pi)
    m = 0.5 * (p + q)
    jsd = 0.5 * np.trapz(p * np.log(p / m), grid) + 0.5 * np.trapz(q * np.log(q / m), grid)
    assert abs(float(l_d) - (math.log(4.0) - 2 * jsd)) < 5e-3, (float(l_d), math.log(4.0) - 2 * jsd)


def test_reference_js_inactive_heads_and_missing_reference():
    b = torch.tensor([0, 0, 2, 2, 2])
    logits = torch.randn(5, 4, dtype=D64)
    got = reference_js_losses(logits, b, 0)
    exp = _refjs_loop(logits, b, 0)                     # batches 1 and 3 absent: only head 2 is active
    assert torch.allclose(got[0], exp[0]) and torch.allclose(got[1], exp[1])
    l_d, l_g = reference_js_losses(logits, b, 1)        # reference absent
    assert float(l_d) == 0.0 and float(l_g) == 0.0


def test_reference_js_generator_gradient_reaches_reference_cells():
    z, b = _data(k=3)
    w = torch.randn(4, 3, dtype=D64)
    zz = z.clone().requires_grad_(True)
    _, l_g = reference_js_losses(zz @ w, b, 0)
    (gz,) = torch.autograd.grad(l_g, zz)
    assert gz[b == 0].abs().sum() > 0 and gz[b != 0].abs().sum() > 0


def test_reference_js_gradcheck():
    _, b = _data(n=20, k=3)
    logits = torch.randn(20, 3, dtype=D64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda l: torch.stack(reference_js_losses(l, b, 2)), (logits,))


# ------------------------------------------------------------------ importance weights (X3)
def _dup(z, b, w_int):
    rep = torch.repeat_interleave(torch.arange(z.shape[0]), w_int)
    return z[rep], b[rep]


def test_weighted_ce_and_fool_loss_reduce_and_duplicate():
    z, b = _data(n=30)
    logits = torch.randn(30, 3, dtype=D64)
    ones = torch.ones(30, dtype=D64)
    assert torch.allclose(weighted_ce(logits, b, ones), F.cross_entropy(logits, b), rtol=0, atol=1e-14)
    assert torch.allclose(fool_loss(logits, b, ones), fool_loss(logits, b), rtol=0, atol=1e-14)
    w_int = torch.randint(0, 4, (30,), generator=torch.Generator().manual_seed(2))
    w_int[0] = 1
    ld, bd = _dup(logits, b, w_int)
    assert torch.allclose(weighted_ce(logits, b, w_int.to(D64)), F.cross_entropy(ld, bd), rtol=0, atol=1e-13)
    assert torch.allclose(fool_loss(logits, b, w_int.to(D64)), fool_loss(ld, bd), rtol=0, atol=1e-13)


@pytest.mark.parametrize("bad", [torch.tensor(-1.0), torch.tensor(float("nan")), "zeros"])
def test_weights_are_validated(bad):
    _, b = _data(n=6)
    logits = torch.randn(6, 3, dtype=D64)
    w = torch.zeros(6, dtype=D64) if isinstance(bad, str) else torch.ones(6, dtype=D64)
    if not isinstance(bad, str):
        w[0] = bad
    with pytest.raises(ValueError):
        weighted_ce(logits, b, w)


@pytest.mark.parametrize("form", ["reference", "pooled"])
def test_weighted_critic_unit_weights_and_duplication(form):
    z, b = _data(n=40, k=4)
    out = torch.randn(40, 4, dtype=D64)
    loss = ReferenceWassersteinLoss(reference_class=1, formulation=form)
    ones = torch.ones(40, dtype=D64)
    ref = 1 if form == "reference" else None
    assert torch.allclose(loss(out, b, ref, weights=ones), loss(out, b, ref), rtol=0, atol=1e-14)
    if form == "pooled":       # pooled: identical arithmetic, bit for bit
        assert torch.equal(loss(out, b, ref, weights=ones), loss(out, b, ref))
    w_int = torch.randint(1, 4, (40,), generator=torch.Generator().manual_seed(3))
    od, bd = _dup(out, b, w_int)
    assert torch.allclose(loss(out, b, ref, weights=w_int.to(D64)), loss(od, bd, ref), rtol=0, atol=1e-13)


@pytest.mark.parametrize("form", ["reference", "pooled"])
def test_weighted_critic_zero_weight_cells_get_no_gradient_and_gradcheck(form):
    z, b = _data(n=24, k=3)
    out = torch.randn(24, 3, dtype=D64, requires_grad=True)
    w = torch.rand(24, dtype=D64, generator=torch.Generator().manual_seed(5)) + 0.1
    w[:5] = 0.0
    loss = ReferenceWassersteinLoss(reference_class=0, formulation=form)
    ref = 0 if form == "reference" else None
    (g,) = torch.autograd.grad(loss(out, b, ref, weights=w), out)
    assert torch.all(g[:5] == 0) and g[5:].abs().sum() > 0
    assert torch.autograd.gradcheck(lambda o: loss(o, b, ref, weights=w), (out,))


def test_weighted_critic_refuses_barycenter():
    z, b = _data(n=12)
    with pytest.raises(NotImplementedError):
        ReferenceWassersteinLoss(formulation="barycenter")(torch.randn(12, 3), b, None, torch.randn(5, 3),
                                                           weights=torch.ones(12))


def test_weighted_mmd_unit_weights_duplication_zero_weights_gradcheck():
    z, b = _data(n=30, d=3, k=3)
    ones = torch.ones(30, dtype=D64)
    assert torch.equal(mmd_batch_others(z, b, weights=ones), mmd_batch_others(z, b))
    w_int = torch.randint(1, 4, (30,), generator=torch.Generator().manual_seed(6))
    zd, bd = _dup(z, b, w_int)
    # duplicated cells add zero-distance kernel pairs exactly as the weighted V-statistic does
    assert torch.allclose(mmd_batch_others(z, b, weights=w_int.to(D64)), mmd_batch_others(zd, bd), rtol=0, atol=1e-13)
    w = torch.rand(30, dtype=D64, generator=torch.Generator().manual_seed(8)) + 0.1
    w[:4] = 0.0
    zz = z.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(mmd_batch_others(zz, b, weights=w), zz)
    assert torch.all(g[:4] == 0) and g[4:].abs().sum() > 0
    zz = z[:12].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: mmd_batch_others(x, b[:12], weights=w[:12] + 0.1), (zz,))


# ------------------------------------------------------------------ stratified sampler (X8)
def _groups(sizes, seed=0):
    g = np.concatenate([np.full(s, k) for k, s in enumerate(sizes)])
    return np.random.default_rng(seed).permutation(g)


@pytest.mark.parametrize("V", [2, 4, 8, 16])
def test_sampler_exact_equal_counts_when_v_divides_128(V):
    groups = _groups([170 + 13 * k for k in range(V)])
    torch.manual_seed(0)
    s = StratifiedBatchSampler(groups, 128)
    assert len(s) == math.ceil(len(groups) / 128)
    batches = list(s)
    assert len(batches) == len(s)
    for mb in batches:
        assert len(mb) == 128 and len(set(mb)) == 128
        cnt = np.bincount(groups[mb], minlength=V)
        assert (cnt == 128 // V).all(), cnt


@pytest.mark.parametrize("V", [3, 10])
def test_sampler_remainder_counts_differ_by_at_most_one_and_balance_out(V):
    groups = _groups([300] * V)
    torch.manual_seed(1)
    s = StratifiedBatchSampler(groups, 128)
    tot = np.zeros(V)
    n_mb = 0
    for _ in range(20):
        for mb in s:
            cnt = np.bincount(groups[mb], minlength=V)
            assert len(mb) == 128 and cnt.max() - cnt.min() <= 1 and cnt.min() == 128 // V
            tot += cnt
            n_mb += 1
    assert np.abs(tot / n_mb - 128 / V).max() < 0.1, tot / n_mb


def test_sampler_each_cell_once_per_cycle():
    groups = _groups([64, 200])
    torch.manual_seed(2)
    s = StratifiedBatchSampler(groups, 128)
    seq = {0: [], 1: []}
    for _ in range(3):
        for mb in s:
            for i in mb:
                seq[int(groups[i])].append(i)
    members0 = set(np.flatnonzero(groups == 0).tolist())
    # batch 0 has 64 cells and 64 slots per minibatch: every minibatch is one full cycle
    for c in range(len(seq[0]) // 64):
        assert set(seq[0][64 * c:64 * (c + 1)]) == members0
    members1 = sorted(np.flatnonzero(groups == 1).tolist())
    assert len(seq[1]) == 3 * 3 * 64                     # 3 epochs x 3 steps x 64 slots
    for c in range(len(seq[1]) // 200):                  # cycles continue across epochs
        assert sorted(seq[1][200 * c:200 * (c + 1)]) == members1


def test_sampler_is_reproducible_from_the_torch_seed():
    groups = _groups([150, 150, 150, 150])
    torch.manual_seed(5)
    a = [mb for mb in StratifiedBatchSampler(groups, 128)]
    torch.manual_seed(5)
    b = [mb for mb in StratifiedBatchSampler(groups, 128)]
    torch.manual_seed(6)
    c = [mb for mb in StratifiedBatchSampler(groups, 128)]
    assert a == b and a != c


def test_sampler_refuses_batches_below_quota_and_too_many_batches():
    with pytest.raises(ValueError, match="quota"):
        StratifiedBatchSampler(_groups([200, 200, 31]), 128)     # quota 43 for V = 3
    with pytest.raises(ValueError, match="batch size"):
        StratifiedBatchSampler(_groups([5] * 9), 8)
