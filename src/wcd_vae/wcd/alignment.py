"""Critic-free batch-alignment divergences -- AUTHORED (K. Reid). Rewritten 2026-10-02.

Both losses compare each batch with the OTHER batches in the minibatch and average over the
batches present (equal weight per batch, the same target and weighting as the pooled critic). A minibatch
with no pair to compare (one batch present; for mmd_ref, no reference or no other batch) contributes 0, as a
zero that carries the graph so the plan's torch.autograd.grad works (final review I033):

    L(z) = mean_k D(P_k, P_{-k})

  mmd       : D = MMD^2 with a fixed multi-scale RBF kernel (V-statistic).
  sinkhorn  : D = debiased Sinkhorn divergence S_eps (Genevay et al. 2018; Feydy et al. 2019)
              with Euclidean ground cost (p=1) at eps = blur = 0.05 sqrt(2d). S_eps tends to W1 (the
              distance the Wasserstein critics estimate) only as the blur goes to 0; at this blur it is
              well below W1 (review of the Sinkhorn fix, F7), so the loss is not a W1 estimate.

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

Changes of 2026-10-13 to Sinkhorn only (final review I031/I032, user decision SI-51; MMD unchanged):
  * Both potentials' final transforms back-propagated through both arguments of the shared cost, i.e.
    twice the envelope gradient. The cost now carries gradient through its row point only (geomloss's
    cost(x, y.detach())), so the gradient is the envelope-theorem gradient once.
  * The final eps got one damped update, so the potentials were not converged. Updates now continue at
    the final eps until every plan's L1 marginal error is below SINKHORN_TOL; reaching
    SINKHORN_MAX_ITER raises. A NaN error (what a NaN or infinite latent gives) stops the loop at once:
    the loss is then NaN, which the training guard records as the fit outcome 'diverged' (review F1).
  * The solve runs in SINKHORN_DTYPE (float64) on distances from direct coordinate differences (the float32
    expansion |x|^2 + |y|^2 - 2<x, y> cancels for a narrow latent far from the origin).
"""
import math

import torch

BW_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)      # RBF sigma = sqrt(d) * multiplier
SINKHORN_BLUR_FRAC = 0.05                          # blur = 0.05 * sqrt(2 d)
SINKHORN_SCALING = 0.8                             # eps annealing ratio per iteration (per blur)
# Solve accuracy (experiments/sinkhorn_fix_20261013): at tol 1e-4 the gradient is within 7.5e-4 (relative) of a
# converged float64 reference on 3,076 minibatches of the 234 pilot fits. Iterations at the final eps have a heavy
# tail on standardised inputs (max 5,273 there, 9,789 on minibatches of the pre-fix A3 latents: review F2); a cap
# hit raises, so the cap sits ~50x above that tail (user decision SI-59).
SINKHORN_TOL = 1e-4                                # L1 marginal error of every plan at the final eps
SINKHORN_MAX_ITER = 500000                         # iterations at the final eps before raising
SINKHORN_DTYPE = torch.float64                     # solve precision; float32 was no faster on the GPU (SI-58)


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


def _batch_masks_weighted(batch_index, z, weights):
    """As _batch_masks, with cell weights: row k of A holds w_i / sum_{i in k} w_i on batch k, row k of B holds
    w_i / sum_{i not in k} w_i on the other cells (self-normalised importance weights; X3, SI-25). With all
    weights 1 the arithmetic is that of _batch_masks."""
    w = weights.to(z.dtype)
    if w.dim() != 1 or w.shape[0] != batch_index.numel():
        raise ValueError(f"weights must have shape [{batch_index.numel()}], got {tuple(w.shape)}")
    if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError("MMD weights must be finite and non-negative")
    ubs = torch.unique(batch_index)
    onehot = (batch_index[None, :] == ubs[:, None]).to(z.dtype)          # [K, n]
    a_w = onehot * w[None, :]
    b_w = (1.0 - onehot) * w[None, :]
    s_a, s_b = a_w.sum(1), b_w.sum(1)
    keep = (s_a > 0) & (s_b > 0)
    A = a_w / torch.where(s_a > 0, s_a, torch.ones_like(s_a))[:, None]
    B = b_w / torch.where(s_b > 0, s_b, torch.ones_like(s_b))[:, None]
    return A[keep], B[keep], int(keep.sum())


def mmd_batch_others(z, batch_index, multipliers=BW_MULTIPLIERS, weights=None):
    """mean_k MMD^2(P_k, P_{-k}), RBF kernel averaged over sigma = sqrt(d) * multipliers.
    weights (X3 importance weights): every empirical measure becomes the self-normalised weighted measure."""
    z = z if z.dim() == 2 else z.reshape(z.shape[0], -1)
    if weights is None:
        A, B, K = _batch_masks(batch_index, z)
    else:
        A, B, K = _batch_masks_weighted(batch_index, z, weights)
    if K == 0:
        return (z * 0).sum()        # no pair of batches to compare: 0, keeping the graph (I033)
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
        return (z * 0).sum()        # no pair of batches to compare: 0, keeping the graph (I033)
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


def _marginal_err(log_w, h_old, h_new, eps):
    """Largest L1 marginal error over the K plans, on the side whose potential moves from h_old to h_new
    (that marginal of the plan is w * exp((h_old - h_new) / eps)); the other side is exact after its update."""
    return (torch.exp(log_w + (h_old - h_new) / eps) - log_w.exp()).abs().sum(1).max()


def _not_converged(err, tol, max_iter):
    raise RuntimeError(f"Sinkhorn: L1 marginal error {float(err):.3g} > tol {tol:g} after {max_iter} "
                       "iterations at the final eps")


def _ot_eps(C, la, lb, eps_list, tol, max_iter):
    """Regularised OT value OT_eps(a, b) for K problems sharing the cost C. eps-scaling warm start
    (symmetrised updates, as geomloss), then Sinkhorn updates at the final eps until converged. Gradient
    flows through C only via one final transform per potential; C carries gradient through its row point
    only, so this is the envelope-theorem gradient once."""
    with torch.no_grad():
        Cd = C.detach()
        f = torch.zeros_like(la)
        g = torch.zeros_like(lb)
        for eps in eps_list:
            f_new = _softmin(eps, Cd, lb, g)
            g_new = _softmin(eps, Cd.T, la, f)
            f, g = 0.5 * (f + f_new), 0.5 * (g + g_new)
        eps = eps_list[-1]
        for _ in range(max_iter):
            f = _softmin(eps, Cd, lb, g)
            g_new = _softmin(eps, Cd.T, la, f)
            err, g = _marginal_err(lb, g, g_new, eps), g_new
            if not err >= tol:     # converged, or NaN (NaN/inf latent): stop now, the NaN loss is 'diverged' (F1)
                break
        else:
            _not_converged(err, tol, max_iter)
    f_fin = _softmin(eps, C, lb, g)
    g_fin = _softmin(eps, C, la, f)            # C, not C.T: the g-side point is C's differentiable row
    a, b = la.exp(), lb.exp()
    return (a * f_fin).sum(1) + (b * g_fin).sum(1)


def _ot_eps_sym(C, la, eps_list, tol, max_iter):
    """OT_eps(a, a) by symmetric (damped) iterations, converged at the final eps as in _ot_eps."""
    with torch.no_grad():
        Cd = C.detach()
        f = torch.zeros_like(la)
        for eps in eps_list:
            f = 0.5 * (f + _softmin(eps, Cd, la, f))
        eps = eps_list[-1]
        for _ in range(max_iter):
            t = _softmin(eps, Cd, la, f)
            err, f = _marginal_err(la, f, t, eps), 0.5 * (f + t)
            if not err >= tol:     # as in _ot_eps
                break
        else:
            _not_converged(err, tol, max_iter)
    f_fin = _softmin(eps, C, la, f)
    return 2.0 * (la.exp() * f_fin).sum(1)


def sinkhorn_divergence_weighted(C, A, B, blur, p=1, scaling=SINKHORN_SCALING, tol=SINKHORN_TOL,
                                 max_iter=SINKHORN_MAX_ITER):
    """Debiased S_eps(a_k, b_k) for each row k of the weight matrices A, B (shared cost C, gradient
    through its row point only: see _cost). A cost differentiable in both points would double-count the
    gradient (review F8). p = 2 (eps = blur^2) is not used by the arm and is tested on two Diracs only."""
    diameter = float(C.detach().max().clamp_min(blur)) ** (1.0 / p)
    eps_list = _eps_schedule(diameter, blur, p, scaling)
    la, lb = _log_w(A), _log_w(B)
    return (_ot_eps(C, la, lb, eps_list, tol, max_iter) - 0.5 * _ot_eps_sym(C, la, eps_list, tol, max_iter)
            - 0.5 * _ot_eps_sym(C, lb, eps_list, tol, max_iter))


def _cost(z, p):
    """Ground cost c(z_i, z_j) from direct coordinate differences (exactly symmetric, no cancellation), with
    gradient through z_i only (geomloss's cost(x, y.detach())): each final transform then carries its own
    half of the envelope gradient instead of all of it."""
    d2 = (z[:, None, :] - z.detach()[None, :, :]).square().sum(-1)
    if p == 1:
        # |x - y| is not differentiable at 0 (the diagonal); +1e-12 inside the root fixes that
        return torch.sqrt(d2 + 1e-12)
    if p == 2:
        return 0.5 * d2
    raise ValueError("p must be 1 or 2")


def sinkhorn_batch_others(z, batch_index, p=1, blur_frac=SINKHORN_BLUR_FRAC, scaling=SINKHORN_SCALING):
    """mean_k S_eps(P_k, P_{-k}) with ground cost |x - y| (p=1) or |x - y|^2 / 2 (p=2), solved in SINKHORN_DTYPE."""
    z = z if z.dim() == 2 else z.reshape(z.shape[0], -1)
    zd = z.to(SINKHORN_DTYPE)
    A, B, K = _batch_masks(batch_index, zd)
    if K == 0:
        return (z * 0).sum()        # no pair of batches to compare: 0, keeping the graph (I033)
    blur = blur_frac * math.sqrt(2.0 * z.shape[1])
    return sinkhorn_divergence_weighted(_cost(zd, p), A, B, blur, p=p, scaling=scaling).mean().to(z.dtype)


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
