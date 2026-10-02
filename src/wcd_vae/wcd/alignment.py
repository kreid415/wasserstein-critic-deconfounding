"""Critic-free batch-alignment divergences -- AUTHORED (K. Reid). Rewritten 2026-10-02.

Both losses compare each batch with the OTHER batches in the minibatch and average over the
batches present (equal weight per batch, the same target and weighting as the pooled critic):

    L(z) = mean_k D(P_k, P_{-k})

  mmd       : D = MMD^2 with a fixed multi-scale RBF kernel (V-statistic).
  sinkhorn  : D = debiased Sinkhorn divergence S_eps (Genevay et al. 2018; Feydy et al. 2019)
              with Euclidean ground cost (p=1), so for small blur it approximates W1, the
              distance the Wasserstein critics estimate.

Changes from the previous version (code review H2/N16):
  * Target was the whole minibatch (it contained the batch itself), so each term was a
    (1 - pi_k)-shrunk comparison; now the other batches, as in the pooled critic.
  * Sinkhorn used a squared-Euclidean cost (W2^2, not W1) -> Euclidean cost (p=1 default).
  * Sinkhorn returned the linear transport cost <P, C> inside the debiasing formula and clamped
    the result at 0 (which also zeroes the gradient); now the regularised dual value
    OT_eps = <a, f> + <b, g>, for which the debiased divergence is non-negative.
  * Both losses rescaled z by a statistic computed under no_grad at every step. The loss then
    never registered a uniform contraction of the latent while every step's gradient still
    favoured it. Scales are now FIXED: kernel bandwidths and the Sinkhorn blur are set relative
    to the N(0, I_d) prior scale (sqrt(d) and sqrt(2d)). Latent-scale shortcuts are handled the
    same way for every arm by the optional standardisation of the adversary input (plan zstd).
  * Sinkhorn unrolled 15 differentiable iterations; now eps-annealed iterations run without
    gradient and one final extrapolation step carries the gradient (the envelope-theorem
    gradient at convergence, as in geomloss).
  * Vectorised over batches (one kernel / cost matrix per step instead of one per batch).
"""
import math

import torch

BW_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)      # RBF sigma = sqrt(d) * multiplier
SINKHORN_BLUR_FRAC = 0.05                          # blur = 0.05 * sqrt(2 d)
SINKHORN_SCALING = 0.8                             # eps annealing ratio per iteration (per blur)


def _sqdist(z):
    """Pairwise squared Euclidean distances via the expansion |x|^2 + |y|^2 - 2<x, y>.
    (torch.cdist's backward is undefined at zero distance, i.e. on the diagonal.)"""
    sq = (z * z).sum(1)
    return (sq[:, None] + sq[None, :] - 2.0 * z @ z.T).clamp_min(0.0)


def _batch_masks(batch_index, z):
    """(A, B, keep): row-normalised weights of batch k (A) and of all other cells (B), [K, n]."""
    ubs = torch.unique(batch_index)
    onehot = (batch_index[None, :] == ubs[:, None]).to(z.dtype)          # [K, n]
    n_k = onehot.sum(1)
    n_other = batch_index.numel() - n_k
    keep = (n_k > 0) & (n_other > 0)
    A = onehot / n_k.clamp_min(1)[:, None]
    B = (1.0 - onehot) / n_other.clamp_min(1)[:, None]
    return A[keep], B[keep], int(keep.sum())


def mmd_batch_others(z, batch_index, multipliers=BW_MULTIPLIERS):
    """mean_k MMD^2(P_k, P_{-k}), RBF kernel averaged over sigma = sqrt(d) * multipliers."""
    z = z if z.dim() == 2 else z.reshape(z.shape[0], -1)
    A, B, K = _batch_masks(batch_index, z)
    if K == 0:
        return z.new_zeros(())
    d2 = _sqdist(z)
    d = z.shape[1]
    kmat = sum(torch.exp(-d2 / (2.0 * d * m * m)) for m in multipliers) / len(multipliers)
    AK, BK = A @ kmat, B @ kmat           # plain matmuls (a 3-operand einsum dispatches to a
    aka = (AK * A).sum(1)                 # Triton kernel in recent torch builds)
    bkb = (BK * B).sum(1)
    akb = (AK * B).sum(1)
    return (aka + bkb - 2.0 * akb).mean()


def mmd_reference(z, batch_index, reference_batch, multipliers=BW_MULTIPLIERS):
    """mean over non-reference batches k of MMD^2(P_k, P_ref) (same kernel as mmd_batch_others).
    Reference-anchored alignment without a Wasserstein critic (X7 control)."""
    z = z if z.dim() == 2 else z.reshape(z.shape[0], -1)
    ref = batch_index == int(reference_batch)
    others = torch.unique(batch_index[~ref])
    if not bool(ref.any()) or others.numel() == 0:
        return z.new_zeros(())
    onehot = (batch_index[None, :] == others[:, None]).to(z.dtype)
    A = onehot / onehot.sum(1, keepdim=True)
    B = (ref.to(z.dtype) / ref.sum())[None, :].expand_as(A)
    d2 = _sqdist(z)
    d = z.shape[1]
    kmat = sum(torch.exp(-d2 / (2.0 * d * m * m)) for m in multipliers) / len(multipliers)
    AK, BK = A @ kmat, B @ kmat
    return ((AK * A).sum(1) + (BK * B).sum(1) - 2.0 * (AK * B).sum(1)).mean()


def _log_w(W):
    return torch.where(W > 0, W.log(), torch.full_like(W, -math.inf))


def _softmin(eps, C, log_w, h):
    """-eps * logsumexp_j( log_w[k, j] + (h[k, j] - C[i, j]) / eps )  ->  [K, n]."""
    return -eps * torch.logsumexp(log_w[:, None, :] + (h[:, None, :] - C[None, :, :]) / eps, dim=2)


def _eps_schedule(diameter, blur, p, scaling):
    eps_list = [diameter ** p]
    while eps_list[-1] > blur ** p * (1.0 / scaling ** p):
        eps_list.append(eps_list[-1] * scaling ** p)
    eps_list.append(blur ** p)
    return eps_list


def _ot_eps(C, la, lb, eps_list):
    """Regularised OT value OT_eps(a, b) for K problems sharing the cost C. Gradient flows through
    C only via the final extrapolation step."""
    with torch.no_grad():
        Cd = C.detach()
        f = torch.zeros_like(la)
        g = torch.zeros_like(lb)
        for eps in eps_list:
            f_new = _softmin(eps, Cd, lb, g)
            g_new = _softmin(eps, Cd.T, la, f)
            f, g = 0.5 * (f + f_new), 0.5 * (g + g_new)
    eps = eps_list[-1]
    f_fin = _softmin(eps, C, lb, g)
    g_fin = _softmin(eps, C.T, la, f)
    a, b = la.exp(), lb.exp()
    return (a * f_fin).sum(1) + (b * g_fin).sum(1)


def _ot_eps_sym(C, la, eps_list):
    """OT_eps(a, a) by symmetric iterations (single potential)."""
    with torch.no_grad():
        Cd = C.detach()
        f = torch.zeros_like(la)
        for eps in eps_list:
            f = 0.5 * (f + _softmin(eps, Cd, la, f))
    f_fin = _softmin(eps_list[-1], C, la, f.detach())
    return 2.0 * (la.exp() * f_fin).sum(1)


def sinkhorn_divergence_weighted(C, A, B, blur, p=1, scaling=SINKHORN_SCALING):
    """Debiased S_eps(a_k, b_k) for each row k of the weight matrices A, B (shared cost C)."""
    diameter = float(C.detach().max().clamp_min(blur)) ** (1.0 / p)
    eps_list = _eps_schedule(diameter, blur, p, scaling)
    la, lb = _log_w(A), _log_w(B)
    return _ot_eps(C, la, lb, eps_list) - 0.5 * _ot_eps_sym(C, la, eps_list) - 0.5 * _ot_eps_sym(C, lb, eps_list)


def sinkhorn_batch_others(z, batch_index, p=1, blur_frac=SINKHORN_BLUR_FRAC, scaling=SINKHORN_SCALING):
    """mean_k S_eps(P_k, P_{-k}) with ground cost |x - y| (p=1) or |x - y|^2 / 2 (p=2)."""
    z = z if z.dim() == 2 else z.reshape(z.shape[0], -1)
    A, B, K = _batch_masks(batch_index, z)
    if K == 0:
        return z.new_zeros(())
    d2 = _sqdist(z)
    if p == 1:
        # |x - y| is not differentiable at 0 (the diagonal); +1e-12 inside the root fixes that
        C = torch.sqrt(d2 + 1e-12)
    elif p == 2:
        C = 0.5 * d2
    else:
        raise ValueError("p must be 1 or 2")
    blur = blur_frac * math.sqrt(2.0 * z.shape[1])
    return sinkhorn_divergence_weighted(C, A, B, blur, p=p, scaling=scaling).mean()


# Registry of critic-free alignment divergences, by the `adversary` name the plan accepts.
CRITIC_FREE_LOSSES = {
    "mmd": mmd_batch_others,
    "sinkhorn": sinkhorn_batch_others,
    "mmd_ref": mmd_reference,          # needs reference_batch
}
NEEDS_REFERENCE = {"mmd_ref"}

# Backwards-compatible names (the old functions compared each batch with the whole minibatch).
mmd_batch_pool = mmd_batch_others
sinkhorn_batch_pool = sinkhorn_batch_others
