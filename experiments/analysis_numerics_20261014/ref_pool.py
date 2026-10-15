# Provenance: written by the independent re-check of 2026-10-14 (evidence archive, artifact version
# 4fc04d2a-1c2e-481c-bb7e-5157820f755f, scripts/ref_pool.py); copied here without change to the code.
"""Independent reference for the pooled model of docs/ANALYSIS_SPEC.md section 4 (written for this re-check; it never
imports scripts/analysis_x1.py).

Model: d_t ~ N(theta_t, s_t^2), theta_t = Delta + u_f(t) + v_t, Delta ~ N(0, delta^2), u_f ~ N(0, tf^2),
v_t ~ N(0, td^2), tf and td half-normal(scale) (family structure) or one spread td (independent / pairs).

Differences from the code under test, on purpose:
  * algebra: closed form for the block structure (diagonal D = s^2 + td^2 plus one rank-one term per family, then the
    rank-one Delta term by Sherman-Morrison), no n x n solve or inverse, weighted-deviation form for d'A^-1 d;
  * quadrature: composite Gauss-Legendre in u = log(tau) over a wider range ([1e-18, 12 prior scales] by default),
    not the trapezoid rule on 201 points;
  * arithmetic: numpy longdouble (80 bit) for the batteries, mpmath at 50 digits for the headline cases.
"""
import math
import numpy as np
from scipy import special, optimize

LD = np.longdouble


def gl_u(lo, hi, panel_w=1.0, g=8):
    a, b = math.log(lo), math.log(hi)
    K = max(1, int(math.ceil((b - a) / panel_w)))
    edges = np.linspace(a, b, K + 1)
    x, w = np.polynomial.legendre.leggauss(g)
    half = (edges[1:, None] - edges[:-1, None]) / 2
    u = ((edges[:-1, None] + edges[1:, None]) / 2 + half * x).ravel()
    return u, (half * w).ravel()


def node_ld(d, s, fam, xf, xd, delta2):
    """Block algebra in longdouble. xf, xd: arrays (m,) of tau_f^2, tau_d^2 (xf = 0 for one spread).
    fam: int array (n,) of family indices. Returns loglik (up to a constant), posterior mean and variance of Delta."""
    d, s = np.asarray(d, LD), np.asarray(s, LD)
    n = len(d)
    F = int(fam.max()) + 1
    m = len(xd)
    D = s[None, :] ** 2 + xd[:, None]                     # (m, n)
    a = np.zeros(m, LD); b = np.zeros(m, LD); q = np.zeros(m, LD); logdet = np.log(D).sum(axis=1)
    for f in range(F):
        idx = np.where(fam == f)[0]
        Df, df = D[:, idx], d[idx][None, :]
        W = (1 / Df).sum(axis=1)
        Y = (df / Df).sum(axis=1)
        mf = Y / W
        g = 1 + xf * W
        dev = ((df - mf[:, None]) ** 2 / Df).sum(axis=1)
        a += W / g
        b += Y / g
        q += dev + mf ** 2 * W / g
        logdet += np.log(g)
    prec = 1 / delta2 + a
    quad = q - b ** 2 / prec
    logdetV = logdet + np.log(delta2 * prec)
    return -(logdetV + quad) / 2, b / prec, 1 / prec


def reference(d, s, fam_labels, structure, scale, margin, lo=1e-18, span=12.0, panel_w=1.0, g=8, want_het=True):
    """structure: 'family' (two spreads), 'independent' or 'pairs' (one spread; for 'pairs' pass the already averaged d, s)."""
    d = np.asarray(d, float); s = np.asarray(s, float)
    two = structure == "family"
    labs = sorted(set(fam_labels))
    fam = np.array([labs.index(x) for x in fam_labels]) if two else np.zeros(len(d), int)
    u, wu = gl_u(lo, span * scale, panel_w, g)
    tau = np.exp(u.astype(LD)); lwu = np.log(wu.astype(LD)) + u.astype(LD) - tau ** 2 / (2 * LD(scale) ** 2)
    delta2 = LD(scale) ** 2
    if two:
        TF, TD = np.meshgrid(np.arange(len(u)), np.arange(len(u)), indexing="ij")
        TF, TD = TF.ravel(), TD.ravel()
        xf, xd = tau[TF] ** 2, tau[TD] ** 2
        lw0 = lwu[TF] + lwu[TD]
    else:
        xf, xd = np.zeros(len(u), LD), tau ** 2
        lw0 = lwu.copy()
    ll, mean, var = node_ld(d, s, fam, xf, xd, delta2)
    lw = lw0 + ll
    lw = lw - lw.max()
    w = np.exp(lw); w = w / w.sum()
    sd = np.sqrt(var)
    wf, mf_, sdf = w.astype(float), mean.astype(float), sd.astype(float)
    # probabilities: weights in longdouble, normal cdf evaluated in double from the standardised distances
    def pw(x):
        return float((w * special.ndtr(((LD(x) - mean) / sd).astype(float)).astype(LD)).sum())
    p_above = float((w * special.ndtr(((mean - LD(margin)) / sd).astype(float)).astype(LD)).sum())
    p_below = float((w * special.ndtr(((-LD(margin) - mean) / sd).astype(float)).astype(LD)).sum())
    p_within = float((w * (special.ndtr(((LD(margin) - mean) / sd).astype(float)) - special.ndtr(((-LD(margin) - mean) / sd).astype(float))).astype(LD)).sum())
    keep = wf > 1e-18
    wk, mk, sk = wf[keep] / wf[keep].sum(), mf_[keep], sdf[keep]
    lo_x, hi_x = float(np.min(mk - 12 * sk)), float(np.max(mk + 12 * sk))
    cdf = lambda x: float(np.sum(wk * special.ndtr((x - mk) / sk)))
    q025 = optimize.brentq(lambda x: cdf(x) - 0.025, lo_x, hi_x, xtol=1e-14, rtol=1e-13)
    q975 = optimize.brentq(lambda x: cdf(x) - 0.975, lo_x, hi_x, xtol=1e-14, rtol=1e-13)
    het = float((w * np.sqrt(xf + xd)).sum()) if want_het else None
    decl = "above" if p_above >= 0.95 else "below" if p_below >= 0.95 else "equivalent" if p_within >= 0.95 else "inconclusive"
    return dict(p_above=p_above, p_below=p_below, p_within=p_within, mean=float((w * mean).sum()), lo=q025, hi=q975,
                heterogeneity=het, declaration=decl, nodes=len(ll),
                tau_mass_above_5=float(w[(np.sqrt(xd) > 5 * scale) | (np.sqrt(xf) > 5 * scale)].sum()))
