"""Critic-free batch-alignment divergences — AUTHORED (K. Reid).

# WHY: The adversarial critic/discriminator confounds two things: the integral-probability-metric
#      GEOMETRY (Wasserstein / MMD) and the ADVERSARIAL TRAINING (an inner-loop network estimating
#      it). These losses keep the geometry and drop the adversary — they are differentiable,
#      closed-form (or fixed-iteration) divergences between each batch and the global pool, added
#      DIRECTLY to the generator objective. That makes them the sharpest test of the project's
#      standing finding ("the adversary is not the lever"): if a critic-FREE OT/MMD loss also fails
#      to beat the discriminator, the deficiency is the objective family, not the adversarial
#      estimator.
#
# TARGET = the global pool (every cell in the minibatch), mirroring the "pooled" critic so the
#      critic-free vs adversarial comparison is like-for-like. Each returns a NON-NEGATIVE
#      divergence (0 iff every batch matches the pool); higher = batches more separable = worse
#      mixing. The generator MINIMISES it (the plan negates it into the fool-objective sign).
#
# COST: O(n^2) in the minibatch (pairwise kernel / cost matrix). At batch_size 512 that is a 512x512
#      matrix per batch group -- cheap. Do NOT call on the full dataset.
"""

import torch


# -------------------------------------------------------------------------------------------------
# MMD -- multi-kernel RBF maximum mean discrepancy (kernel IPM), each batch vs the pool.
# -------------------------------------------------------------------------------------------------
def _pdist2(a, b):
    """Squared Euclidean distances, [n, m]. Clamped at 0 for numerical safety."""
    return torch.cdist(a, b, p=2).pow(2).clamp_min(0.0)


def _median_bandwidth(d2):
    """Median-heuristic RBF bandwidth from a squared-distance matrix (detached scalar)."""
    with torch.no_grad():
        m = d2[d2 > 0]
        med = m.median() if m.numel() else d2.new_tensor(1.0)
    return med.clamp_min(1e-6)


def mmd_batch_pool(z, batch_index, scales=(0.5, 1.0, 2.0, 4.0)):
    """Mean over batches of MMD^2(batch, pool) using a sum of RBF kernels at median-heuristic
    bandwidth times ``scales``. Non-negative; 0 iff each batch is distributed like the pool.

    MMD^2(X, Y) = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]   (biased estimator; batches are small).
    """
    z = z if z.dim() == 2 else z.reshape(z.shape[0], -1)
    n = z.shape[0]
    if n < 4:
        return z.new_zeros(())
    dpp = _pdist2(z, z)                       # pool-pool, reused for every batch
    base_bw = _median_bandwidth(dpp)
    gammas = [1.0 / (2.0 * base_bw * s) for s in scales]

    def _mmd2(idx):
        x = z[idx]
        nx = x.shape[0]
        if nx < 2:
            return z.new_zeros(())
        dxx = _pdist2(x, x)
        dxy = _pdist2(x, z)                   # batch vs the whole pool
        val = z.new_zeros(())
        for g in gammas:
            kxx = torch.exp(-g * dxx)
            kyy = torch.exp(-g * dpp)
            kxy = torch.exp(-g * dxy)
            val = val + kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
        return val / len(gammas)

    ubs = torch.unique(batch_index)
    terms = [_mmd2((batch_index == b).nonzero(as_tuple=True)[0]) for b in ubs]
    terms = [t for t in terms if torch.isfinite(t)]
    return torch.stack(terms).mean() if terms else z.new_zeros(())


# -------------------------------------------------------------------------------------------------
# Sinkhorn divergence -- debiased entropic optimal transport, each batch vs the pool.
# -------------------------------------------------------------------------------------------------
def _sinkhorn_cost(x, y, eps, n_iter):
    """Entropic OT cost OT_eps(x, y) with uniform marginals, squared-Euclidean ground cost,
    ``n_iter`` Sinkhorn iterations in log space. Returns a scalar (the transport cost)."""
    C = _pdist2(x, y)                         # [n, m] ground cost
    n, m = C.shape
    log_a = x.new_full((n,), -torch.log(torch.tensor(float(n), device=x.device)))
    log_b = y.new_full((m,), -torch.log(torch.tensor(float(m), device=y.device)))
    f = torch.zeros(n, device=x.device)
    g = torch.zeros(m, device=y.device)
    Ce = C / eps
    for _ in range(n_iter):
        # f_i = -eps * logsumexp_j( log_b_j + g_j/eps - C_ij/eps )   (and symmetric for g)
        f = -eps * torch.logsumexp(log_b[None, :] + (g[None, :] / eps) - Ce, dim=1)
        g = -eps * torch.logsumexp(log_a[:, None] + (f[:, None] / eps) - Ce, dim=0)
    # transport cost = <P, C> recovered from the dual potentials
    log_P = log_a[:, None] + log_b[None, :] + (f[:, None] + g[None, :]) / eps - Ce
    P = torch.exp(log_P)
    return (P * C).sum()


def sinkhorn_batch_pool(z, batch_index, eps=0.1, n_iter=50):
    """Mean over batches of the DEBIASED Sinkhorn divergence S(batch, pool), where
    S(a, b) = OT_eps(a, b) - 1/2 OT_eps(a, a) - 1/2 OT_eps(b, b).
    Debiasing makes S >= 0 with S = 0 iff a == b (a proper divergence, unlike raw entropic OT)."""
    z = z if z.dim() == 2 else z.reshape(z.shape[0], -1)
    n = z.shape[0]
    if n < 4:
        return z.new_zeros(())
    # normalise scale so eps is comparable across datasets (cost ~ O(1))
    with torch.no_grad():
        scale = _pdist2(z, z).mean().clamp_min(1e-6).sqrt()
    zz = z / scale
    ott_pool = _sinkhorn_cost(zz, zz, eps, n_iter)   # OT_eps(pool, pool), reused
    ubs = torch.unique(batch_index)
    terms = []
    for b in ubs:
        x = zz[(batch_index == b).nonzero(as_tuple=True)[0]]
        if x.shape[0] < 2:
            continue
        s = _sinkhorn_cost(x, zz, eps, n_iter) - 0.5 * _sinkhorn_cost(x, x, eps, n_iter) - 0.5 * ott_pool
        terms.append(s)
    terms = [t for t in terms if torch.isfinite(t)]
    return torch.stack(terms).mean().clamp_min(0.0) if terms else z.new_zeros(())


# Registry of critic-free alignment divergences, by the `adversary` name the plan accepts.
CRITIC_FREE_LOSSES = {
    "mmd": mmd_batch_pool,
    "sinkhorn": sinkhorn_batch_pool,
}
