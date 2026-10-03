"""JS-discriminator losses for the X6 / X7 / X3 arms -- AUTHORED (K. Reid), 2026-10-02.

Specifications and sources: docs/SPECS_missing_arms.md (signed off 2026-10-02; CONSTRAINTS.md SI-22, SI-23,
SI-25). All functions take the logits l(z) in R^K of the existing MLP head (wcd.adversarial.Discriminator with
batch_ids=None) or a callable producing them.

  r1_penalty            X6: R1 penalty of Mescheder, Geiger & Nowozin 2018 (ICML, arXiv:1801.04406, Eq. 9)
                        for a K-way classifier, applied to each cell's own-batch one-vs-rest log-odds
                        D_k = l_k - log sum_{j != k} exp(l_j)  (= the paper's logit D_psi when K = 2).
  reference_js_losses   X7: per-batch binary heads 'batch k vs reference r' (output k), class-balanced
                        cross-entropy (Goodfellow et al. 2014, Eq. 1; optimum log 4 - 2 JSD, Theorem 1) and the
                        non-saturating label-flipped generator loss (Goodfellow et al. 2014 Sec. 3; ADDA Eq. 7).
  weighted_ce, fool_loss  X3: the discriminator's cross-entropy and scvi-tools' fool loss with importance
                        weights, as self-normalised weighted means (Tachet des Combes et al. 2020, Eq. 7 with
                        the weights normalised to sum to one over the minibatch).
"""
import torch
import torch.nn.functional as F  # noqa: N812


def one_vs_rest_logodds(logits, idx):
    """D_{idx_i}(z_i) = l_{idx_i} - logsumexp_{j != idx_i} l_j, one value per row. Needs K >= 2."""
    if logits.dim() != 2 or logits.shape[1] < 2:
        raise ValueError(f"one-vs-rest log-odds need logits [n, K>=2], got {tuple(logits.shape)}")
    own = logits.gather(1, idx[:, None]).squeeze(1)
    mask = F.one_hot(idx, logits.shape[1]).bool()
    return own - torch.logsumexp(logits.masked_fill(mask, float("-inf")), dim=1)


def r1_penalty(logits_fn, z, batch_ids, gamma, detach_input=True):
    """(gamma / 2) * (1/n) sum_i ||grad_z D_{b_i}(z_i)||^2   (SPECS section 1, SI-22).

    logits_fn maps [n, d] -> [n, K] row-wise (no layer may couple rows, e.g. batch norm: the gradient of the
    summed log-odds is then the per-row gradient). The penalty is evaluated at z (the detached minibatch
    adversary inputs when detach_input=True) and keeps the graph (create_graph=True) so that it can be
    backpropagated into the head parameters. Returns (penalty, logits at z) so the caller's cross-entropy can
    reuse the same forward pass.
    """
    gamma = float(gamma)
    if not gamma > 0:
        raise ValueError(f"R1 needs gamma > 0, got {gamma}")
    zr = z.detach().requires_grad_(True) if detach_input else z
    logits = logits_fn(zr)
    d = one_vs_rest_logodds(logits, batch_ids)
    (g,) = torch.autograd.grad(d.sum(), zr, create_graph=True)
    return 0.5 * gamma * g.pow(2).sum(1).mean(), logits


def reference_js_losses(logits, batch_ids, reference_batch):
    """(L_D, L_G) of the reference-anchored JS discriminator (SPECS section 2, SI-23).

    Output k (k != r) is the logit of a binary discriminator 'batch k (label 1) vs reference r (label 0)';
    output r is unused. A head is active when its batch is in the minibatch and the reference is too.
      L_D = mean_k [ mean_{i in k} softplus(-l_k(z_i)) + mean_{j in r} softplus(l_k(z_j)) ]
      L_G = mean_k [ mean_{i in k} softplus(l_k(z_i))  + mean_{j in r} softplus(-l_k(z_j)) ]
    Per-side means: each head has the equal-prior objective of Goodfellow et al. (2014) Eq. 1, whose optimum
    is log 4 - 2 JSD(P_k || P_r). Reference cells are not detached (counterpart of the critic 'reference').
    """
    n, k = logits.shape
    r = int(reference_batch)
    if not 0 <= r < k:
        raise ValueError(f"reference batch {r} outside 0..{k - 1}")
    is_ref = batch_ids == r
    if not bool(is_ref.any()):
        z0 = logits.sum() * 0.0
        return z0, z0
    counts = torch.bincount(batch_ids, minlength=k)
    active = counts > 0
    active[r] = False
    if not bool(active.any()):
        z0 = logits.sum() * 0.0
        return z0, z0
    own = logits.gather(1, batch_ids[:, None]).squeeze(1)
    src = ~is_ref
    denom = counts.clamp_min(1).to(logits.dtype)
    src_d = logits.new_zeros(k).index_add_(0, batch_ids[src], F.softplus(-own[src])) / denom
    src_g = logits.new_zeros(k).index_add_(0, batch_ids[src], F.softplus(own[src])) / denom
    ref_logits = logits[is_ref]
    tgt_d = F.softplus(ref_logits).mean(0)
    tgt_g = F.softplus(-ref_logits).mean(0)
    return (src_d + tgt_d)[active].mean(), (src_g + tgt_g)[active].mean()


def _check_weights(w, n):
    if w.dim() != 1 or w.shape[0] != n:
        raise ValueError(f"weights must have shape [{n}], got {tuple(w.shape)}")
    if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError("weights must be finite and non-negative")
    if not float(w.sum()) > 0:
        raise ValueError("all weights in the minibatch are zero")


def weighted_ce(logits, batch_ids, weights):
    """sum_i w_i CE_i / sum_i w_i (self-normalised importance-weighted cross-entropy; SI-25)."""
    _check_weights(weights, logits.shape[0])
    ce = F.cross_entropy(logits, batch_ids, reduction="none")
    w = weights.to(ce.dtype)
    return (w * ce).sum() / w.sum()


def fool_loss(logits, batch_ids, weights=None):
    """scvi-tools fool loss: cross-entropy between the prediction and uniform-over-other-batches.
    weights=None: mean over cells (the plan's existing loss); otherwise the self-normalised weighted mean."""
    k = logits.shape[1]
    logp = F.log_softmax(logits, dim=1)
    other = (~F.one_hot(batch_ids, k).bool()).to(logp.dtype) / (k - 1)
    per_cell = -(logp * other).sum(dim=1)
    if weights is None:
        return per_cell.mean()
    _check_weights(weights, logits.shape[0])
    w = weights.to(per_cell.dtype)
    return (w * per_cell).sum() / w.sum()
