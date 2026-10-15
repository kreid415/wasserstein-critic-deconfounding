# Provenance: written by the independent re-check of 2026-10-14 (evidence archive, artifact version
# 4fc04d2a-1c2e-481c-bb7e-5157820f755f, scripts/ref_mp.py); copied here without change to the code.
"""50-digit versions of the independent reference (see ref_pool.py): direct n x n algebra for validating the block
closed form, and the block closed form with Gauss-Legendre in log(tau) for the headline cases."""
import math
import numpy as np
import mpmath as mp
from scipy import special, optimize
from ref_pool import gl_u

mp.mp.dps = 50


def direct_node(d, s, fam_labels, structure, xf, xd, delta2):
    n = len(d)
    V = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            v = delta2
            if structure == "family" and fam_labels[i] == fam_labels[j]:
                v += xf
            if i == j:
                v += xd + mp.mpf(s[i]) ** 2
            V[i, j] = v
    dv = mp.matrix([mp.mpf(x) for x in d])
    Vi = V ** -1
    vd = Vi * dv
    quad = sum(dv[i] * vd[i] for i in range(n))
    ones = mp.matrix([1] * n)
    one_vi_one = sum(Vi[i, j] for i in range(n) for j in range(n))
    return (-(mp.log(mp.det(V)) + quad) / 2, delta2 * sum(vd), delta2 - delta2 ** 2 * one_vi_one)


def block_node(d, s, fam_labels, structure, xf, xd, delta2):
    labs = sorted(set(fam_labels)) if structure == "family" else [0]
    members = {f: [i for i in range(len(d)) if (fam_labels[i] == f or structure != "family")] for f in labs}
    D = [mp.mpf(si) ** 2 + xd for si in s]
    a = b = q = logdet = mp.mpf(0)
    logdet = sum(mp.log(x) for x in D)
    for f in labs:
        idx = members[f]
        W = sum(1 / D[i] for i in idx)
        Y = sum(mp.mpf(d[i]) / D[i] for i in idx)
        S2 = sum(mp.mpf(d[i]) ** 2 / D[i] for i in idx)
        g = 1 + (xf * W if structure == "family" else 0)
        a += W / g
        b += Y / g
        q += S2 - (xf * Y ** 2 / g if structure == "family" else 0)
        logdet += mp.log(g)
    prec = 1 / delta2 + a
    quad = q - b ** 2 / prec
    return (-(logdet + mp.log(delta2 * prec) + quad) / 2, b / prec, 1 / prec)


def reference_mp(d, s, fam_labels, structure, scale, margin, lo=1e-18, span=12.0, panel_w=1.5, g=8):
    two = structure == "family"
    u, wu = gl_u(lo, span * scale, panel_w, g)
    tau = [mp.exp(mp.mpf(x)) for x in u]
    lwu = [mp.log(mp.mpf(w)) + mp.mpf(x) - t ** 2 / (2 * mp.mpf(scale) ** 2) for w, x, t in zip(wu, u, tau)]
    delta2 = mp.mpf(scale) ** 2
    nodes = []
    rng_f = range(len(u)) if two else [None]
    for i in rng_f:
        for j in range(len(u)):
            xf = tau[i] ** 2 if two else mp.mpf(0)
            lw0 = (lwu[i] if two else 0) + lwu[j]
            ll, mean, var = block_node(d, s, fam_labels, structure, xf, tau[j] ** 2, delta2)
            nodes.append((lw0 + ll, mean, var, xf, tau[j] ** 2))
    L = max(x[0] for x in nodes)
    w = [mp.exp(x[0] - L) for x in nodes]
    Z = sum(w)
    w = [x / Z for x in w]
    r = mp.mpf(margin)
    ncdf = mp.ncdf
    sd = [mp.sqrt(x[2]) for x in nodes]
    mean = [x[1] for x in nodes]
    p_above = sum(wk * ncdf((m - r) / sk) for wk, m, sk in zip(w, mean, sd))
    p_below = sum(wk * ncdf((-r - m) / sk) for wk, m, sk in zip(w, mean, sd))
    p_within = sum(wk * (ncdf((r - m) / sk) - ncdf((-r - m) / sk)) for wk, m, sk in zip(w, mean, sd))
    het = sum(wk * mp.sqrt(x[3] + x[4]) for wk, x in zip(w, nodes))
    mu = sum(wk * m for wk, m in zip(w, mean))
    wf = np.array([float(x) for x in w]); mf = np.array([float(x) for x in mean]); sf = np.array([float(x) for x in sd])
    keep = wf > 1e-18
    wk_, mk_, sk_ = wf[keep] / wf[keep].sum(), mf[keep], sf[keep]
    cdf = lambda x: float(np.sum(wk_ * special.ndtr((x - mk_) / sk_)))
    lo_x, hi_x = float(np.min(mk_ - 12 * sk_)), float(np.max(mk_ + 12 * sk_))
    q025 = optimize.brentq(lambda x: cdf(x) - 0.025, lo_x, hi_x, xtol=1e-14, rtol=1e-13)
    q975 = optimize.brentq(lambda x: cdf(x) - 0.975, lo_x, hi_x, xtol=1e-14, rtol=1e-13)
    return dict(p_above=float(p_above), p_below=float(p_below), p_within=float(p_within), mean=float(mu),
                heterogeneity=float(het), lo=q025, hi=q975, nodes=len(nodes))
