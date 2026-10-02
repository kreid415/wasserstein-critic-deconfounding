"""Wasserstein-barycenter target for the barycenter critic -- AUTHORED (K. Reid), 2026-10-02.

The barycenter formulation aligns every batch to one shared, batch-neutral target: the
free-support Wasserstein-2 barycenter of the batch distributions (Agueh & Carlier 2011),

    B* = argmin_B  sum_k w_k W2^2(P_k, B),

computed for the current minibatch with the fixed-point algorithm of Cuturi & Doucet (2014,
Alg. 2 with theta = 1; Alvarez-Esteban et al. 2016) as implemented in POT
(ot.lp.free_support_barycenter, exact OT plans). The critic heads then estimate W1(P_k, B*) and
the encoder moves each P_k towards B*. B* is recomputed every generator step from the detached
latents and is never optimised by gradient: at alignment (all P_k equal) it coincides with them.

Why W2 for the target although the critic estimates W1: the W2 barycenter is unique and
well defined for empirical measures; W1 "barycenters" are generally not unique (for two
measures every point of a W1 geodesic minimises the sum of distances).

Replaces the previous design (code review N3): 64 learnable anchors initialised at ~0 and moved
by the generator optimiser, which gave a near-origin fixed point at small lambda and a
contracting quantiser at large lambda.

Choices (all recorded with each fit):
  weights   'equal' (w_k = 1/K over the batches present; batch-neutral, matching the critic's
            equal weighting of heads) or 'size' (w_k = n_k / n).
  n_support barycenter support size; default = minibatch size (at least as many atoms as cells).
  n_iter    fixed-point iterations per step (default 10; init = a random subset of the minibatch).
"""
import numpy as np
import torch

try:
    import ot  # POT
except ImportError as e:  # pragma: no cover
    ot = None
    _OT_ERR = e


def batch_barycenter_support(z, batch_ids, n_support=None, n_iter=10, weights="equal", stop_thr=1e-7, init=None):
    """Support points [m, d] of the free-support W2 barycenter of the per-batch empirical measures
    in this minibatch (uniform weights 1/m on the support). Returned detached, on z's device.

    Default initialisation (init=None, n_support = minibatch size): the support starts at the
    minibatch cells themselves, so the solve is (up to ties) a deterministic function of the minibatch.
    init: optional [m, d] starting support, e.g. the previous step's barycenter ("warm start").
    The problem is non-convex in the support: a warm start converges to a DIFFERENT fixed point
    (measured 2026-10-02 on a scVI latent of immune, 35 minibatches: W2^2 to the cold solution was
    26-29% of the minibatch-to-minibatch sampling difference after 3-20 iterations, vs 3.2% / 1.4% /
    0.14% for 3 / 5 / 10 cold iterations). The benchmark therefore uses cold starts only; `init` is
    kept for diagnostics (docs/barycenter_solver_check.csv; scripts/check_barycenter_solver.py)."""
    if ot is None:
        raise ImportError(f"POT is required for the barycenter critic: {_OT_ERR}")
    zd = z.detach()
    n = zd.shape[0]
    m = int(n_support or n)
    x = zd.cpu().double().numpy()
    b = batch_ids.detach().cpu().numpy()
    present = np.unique(b)
    locs = [x[b == k] for k in present]
    wts = [np.full(len(l), 1.0 / len(l)) for l in locs]
    if weights == "equal":
        lam = np.full(len(present), 1.0 / len(present))
    elif weights == "size":
        lam = np.array([len(l) for l in locs], dtype=float) / n
    else:
        raise ValueError("weights must be 'equal' or 'size'")
    # random initial support drawn from the pooled minibatch (torch RNG -> seeded by scvi.settings.seed)
    if init is not None and tuple(init.shape) == (m, x.shape[1]):
        X0 = init.detach().cpu().double().numpy().copy()
    else:
        idx = torch.randint(0, n, (m,)).numpy() if m > n else torch.randperm(n)[:m].numpy()
        X0 = x[idx].copy()
    X = ot.lp.free_support_barycenter(locs, wts, X0, b=np.full(m, 1.0 / m), weights=lam,
                                      numItermax=int(n_iter), stopThr=stop_thr)
    return torch.as_tensor(np.asarray(X), dtype=zd.dtype, device=zd.device)


def barycenter_objective(support, z, batch_ids, weights="equal"):
    """sum_k w_k W2^2(P_k, B) for a candidate support B (uniform weights); used by the tests."""
    x = z.detach().cpu().double().numpy()
    s = support.detach().cpu().double().numpy()
    b = batch_ids.detach().cpu().numpy()
    present = np.unique(b)
    tot = 0.0
    for k in present:
        xk = x[b == k]
        lam = 1.0 / len(present) if weights == "equal" else len(xk) / len(x)
        M = ot.dist(xk, s)                                   # squared Euclidean
        tot += lam * ot.emd2(np.full(len(xk), 1.0 / len(xk)), np.full(len(s), 1.0 / len(s)), M)
    return float(tot)
