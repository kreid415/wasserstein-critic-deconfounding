"""The vectorised critic loss and gradient penalty (wcd.critic) must equal a per-head loop.

The per-head loop below is the pre-2026-10-02 implementation (git 9bdff38, critic.py), with one
intended change: the pooled target is the OTHER batches (code review N4) instead of the whole
minibatch. The reference and barycenter formulations are compared to the old code unchanged.
Random draws in the gradient penalty are injected so both versions see the same partners and
interpolation weights.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from wcd_vae.wcd.critic import ReferenceWassersteinLoss, multi_class_gradient_penalty


class _Head(nn.Module):
    def __init__(self, d, v):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(d, 32), nn.Linear(32, 32), nn.Linear(32, v)
        self.domain_number = v

    def forward(self, x, batch_ids=None):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def _loop_loss(output, batch_ids, formulation, ref=None, target_output=None):
    v = output.shape[1]
    tot, k_used = 0.0, 0
    for k in range(v):
        if formulation == "reference" and k == ref:
            continue
        mk = batch_ids == k
        if mk.sum() == 0:
            continue
        if formulation == "reference":
            t = output[batch_ids == ref][:, k]
        elif formulation == "pooled":
            t = output[~mk][:, k]
            if t.numel() == 0:
                continue
        else:
            t = target_output[:, k]
        tot = tot + (output[mk, k].mean() - t.mean())
        k_used += 1
    return -tot / k_used


def _loop_gp(critic, z, batch_ids, formulation, partner_of_row, eps_of_row, ref=None, target=None, lam=10.0):
    v = critic.domain_number
    tot, k_used = 0.0, 0
    pool = target if formulation == "barycenter" else z
    for k in range(v):
        if formulation == "reference" and k == ref:
            continue
        rows = (batch_ids == k).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        if formulation == "pooled" and (batch_ids != k).sum() == 0:
            continue
        zh = eps_of_row[rows] * z[rows] + (1 - eps_of_row[rows]) * pool[partner_of_row[rows]]
        zh.requires_grad_(True)
        g = torch.autograd.grad(critic(zh)[:, k].sum(), zh, create_graph=True)[0]
        tot = tot + ((g.norm(2, dim=1) - 1) ** 2).mean()
        k_used += 1
    return lam * tot / k_used


def _setup(seed, n=64, d=6, v=5, drop=None):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g, dtype=torch.float64)
    b = torch.randint(0, v, (n,), generator=g)
    if drop is not None:
        b[b == drop] = (drop + 1) % v          # one batch absent from the minibatch
    torch.manual_seed(seed)
    head = _Head(d, v).double()
    return z, b, head


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("formulation", ["reference", "pooled", "barycenter"])
@pytest.mark.parametrize("drop", [None, 3])
def test_loss_matches_per_head_loop(seed, formulation, drop):
    z, b, head = _setup(seed, drop=drop)
    out = head(z)
    tgt = head(torch.randn(40, z.shape[1], dtype=torch.float64)) if formulation == "barycenter" else None
    ref = 1 if formulation == "reference" else None
    fast = ReferenceWassersteinLoss(formulation=formulation)(out, b, reference_batch=ref, target_output=tgt)
    slow = _loop_loss(out, b, formulation, ref=ref, target_output=tgt)
    assert torch.allclose(fast, slow, atol=1e-12, rtol=0), (float(fast), float(slow))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("formulation", ["reference", "pooled", "barycenter"])
@pytest.mark.parametrize("drop", [None, 3])
def test_gradient_penalty_matches_per_head_loop(seed, formulation, drop):
    z, b, head = _setup(seed, drop=drop)
    ref = 1 if formulation == "reference" else None
    target = torch.randn(40, z.shape[1], dtype=torch.float64) if formulation == "barycenter" else None
    n = z.shape[0]
    g = torch.Generator().manual_seed(100 + seed)
    # one partner and one eps per row, drawn from that row's target set
    partner = torch.empty(n, dtype=torch.long)
    for i in range(n):
        if formulation == "reference":
            cand = (b == ref).nonzero(as_tuple=True)[0]
        elif formulation == "pooled":
            cand = (b != b[i]).nonzero(as_tuple=True)[0]
        else:
            cand = torch.arange(target.shape[0])
        partner[i] = cand[torch.randint(0, cand.numel(), (1,), generator=g)]
    eps = torch.rand(n, 1, generator=g, dtype=torch.float64)
    rows = (b != ref).nonzero(as_tuple=True)[0] if formulation == "reference" else torch.arange(n)
    fast = multi_class_gradient_penalty(head, z, b, reference_batch=ref, formulation=formulation,
                                        target_samples=target, draws=(partner[rows], eps[rows]))
    slow = _loop_gp(head, z, b, formulation, partner, eps, ref=ref, target=target)
    assert torch.allclose(fast, slow, atol=1e-10, rtol=0), (float(fast), float(slow))
    # and the gradients the critic optimiser sees are identical
    # (allow_unused: the last-layer bias never enters a gradient penalty)
    ps = list(head.parameters())
    gf = torch.autograd.grad(fast, ps, retain_graph=True, allow_unused=True)
    gs = torch.autograd.grad(slow, ps, allow_unused=True)
    for p_, a, c in zip(ps, gf, gs):
        a = torch.zeros_like(p_) if a is None else a
        c = torch.zeros_like(p_) if c is None else c
        assert torch.allclose(a, c, atol=1e-9, rtol=0)


def test_reference_formulation_equals_old_code_exactly():
    """Unchanged formulation: the vectorised loss equals the committed old implementation."""
    import importlib.util, pathlib, subprocess
    root = pathlib.Path(__file__).resolve().parents[2]
    try:
        src = subprocess.run(["git", "show", "9bdff38:src/wcd_vae/wcd/critic.py"], cwd=root,
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip("old critic.py not available from git")
    p = root / ".pytest_old_critic.py"
    p.write_text(src)
    try:
        spec = importlib.util.spec_from_file_location("old_critic", p)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
    finally:
        p.unlink()
    for seed in range(3):
        z, b, head = _setup(seed)
        out = head(z)
        new = ReferenceWassersteinLoss(formulation="reference")(out, b, reference_batch=2)
        ref_old = old.ReferenceWassersteinLoss(formulation="reference")(out, b, reference_batch=2)
        assert torch.allclose(new, ref_old, atol=1e-12, rtol=0)


def test_pooled_two_batches_is_symmetric_pairwise():
    """N4 fix: with two batches, each pooled head's target is the other batch."""
    z, b, head = _setup(0, v=2)
    out = head(z)
    pooled = ReferenceWassersteinLoss(formulation="pooled")(out, b)
    m0, m1 = b == 0, b == 1
    expect = -0.5 * ((out[m0, 0].mean() - out[m1, 0].mean()) + (out[m1, 1].mean() - out[m0, 1].mean()))
    assert torch.allclose(pooled, expect, atol=1e-12, rtol=0)
