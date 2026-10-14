#!/usr/bin/env python
"""Numerical checks of the pooled model in scripts/analysis_x1.py (rigour review AN-01 / AN-02, 2026-10-14).

1. The review's failing case (synthetic dry-run, local endpoint, decoder 0): the code at 973a97d (evenly spaced
   spread grid from 0), the amended code, and a reference rule (trapezoid in u = log tau, step 0.01, from 1e-18 to
   8 prior scales).
2. 240 random sets of 1-6 datasets (s from 1e-6 to 10^-1.3, three pooling structures, prior scales 0.05-0.25):
   amended code against the reference (step 0.06 for two spreads, 0.01 for one), and against the same reference cut
   at 5 prior scales as the code is (to separate the prior truncation from the integration rule).
3. Per-grid-node precision of the formulas for Delta | d against 40-digit arithmetic (mpmath).
4. The effect of the two planted errors the tests do not catch (M24: 101 grid points, M28: no half end weights).
Inputs are synthetic only (no X1 score is read). Run from the repo root:  python experiments/analysis_numerics_20261014/check_pool_numerics.py
Environment used on 2026-10-14: conda env `python` (Python 3.11.15, numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, mpmath 1.4.1).
"""
import importlib.util, json, math, os, subprocess, sys, tempfile
import numpy as np, pandas as pd
from scipy import special

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analysis_x1 as A  # noqa: E402  (the amended code)


def load_old(sha="973a97d"):
    src = subprocess.run(["git", "-C", ROOT, "show", f"{sha}:scripts/analysis_x1.py"], capture_output=True, text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "analysis_x1_old.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("analysis_x1_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(d, s, levels, scale, margin, step, umin=math.log(1e-18), span=8.0):
    """Trapezoid in u = log(tau) on [umin, ln(span x scale)]; Delta | d, tau by the precision form
    1/v = 1/scale^2 + 1'A^-1 1, m = v 1'A^-1 d (A = covariance of d without Delta's prior variance)."""
    d, s = np.asarray(d, float), np.asarray(s, float)
    n, ds2 = len(d), scale ** 2
    u = np.arange(umin, math.log(span * scale) + 1e-12, step)
    tau = np.exp(u)
    lw1 = np.log(np.r_[step / 2, np.full(len(u) - 2, step), step / 2]) + u - tau ** 2 / (2 * scale ** 2)
    G = [np.asarray(lab)[:, None] == np.asarray(lab)[None, :] for lab in levels]
    S = np.diag(s ** 2)
    parts = []
    for i in ([None] if len(levels) == 1 else range(len(u))):
        if len(levels) == 1:
            Am, lw, het = S[None] + tau[:, None, None] ** 2 * G[0][None], lw1.copy(), tau
        else:
            Am = S[None] + tau[i] ** 2 * G[0][None] + tau[:, None, None] ** 2 * G[1][None]
            lw, het = lw1[i] + lw1, np.hypot(tau[i], tau)
        x = np.linalg.solve(Am, np.broadcast_to(np.stack([np.ones(n), d], axis=1), (len(Am), n, 2)))
        v = ds2 / (1 + ds2 * x[:, :, 0].sum(1))
        m = v * x[:, :, 1].sum(1)
        V = Am + ds2
        lw = lw - 0.5 * (np.linalg.slogdet(V)[1] + np.einsum("j,gjk,k->g", d, np.linalg.inv(V), d))
        sd = np.sqrt(v)
        pa, pb = special.ndtr((m - margin) / sd), special.ndtr((-margin - m) / sd)
        parts.append((lw, np.stack([np.ones_like(m), pa, pb, 1 - pa - pb, m, het])))
    L0 = max(p[0].max() for p in parts)
    tot = sum((np.exp(lw - L0)[None] * q).sum(1) for lw, q in parts)
    tot = tot / tot[0]
    return dict(p_above=tot[1], p_below=tot[2], p_within=tot[3], mean=tot[4], heterogeneity=tot[5])


def declare(r):
    return ("above" if r["p_above"] >= 0.95 else "below" if r["p_below"] >= 0.95
            else "equivalent" if r["p_within"] >= 0.95 else "inconclusive")


KEYS = ("p_above", "p_below", "p_within", "mean", "heterogeneity")
FAM6 = ["pancreas", "lung", "immune", "immune", "atac", "atac"]


def part1(old):
    # rigour review dry run (artifact 3de51e25, rv_out/dry_main/analysis.json): pooled local endpoint, decoder 0
    d = [0.013240398313286439, 0.017744079582730055, 0.016738986549012624, 0.01553635640855433, 0.01880993348817228,
         0.015235264206615806]
    s = [0.00017513232962922487, 0.0014235634824912538, 0.0011280789505213036, 0.000444842801815976,
         0.0006267795699190329, 0.0007926453923105091]
    levels = [FAM6, ["pancreas", "lung", "immune", "immune_hum_mou", "atac_small", "atac_large"]]
    rows = []
    for scale in (0.05, 0.1, 0.2):
        ref = reference(d, s, levels, scale, 0.01, step=0.01)
        for name, mod in (("973a97d", old), ("amended", A)):
            r = mod.pool(d, s, levels, scale, scale, 0.01)
            rows.append(dict(scale=scale, code=name, declaration=r["declaration"], ref_declaration=declare(ref),
                             max_abs_err=max(abs(r[k] - ref[k]) for k in KEYS), **{k: r[k] for k in KEYS},
                             **{f"ref_{k}": float(ref[k]) for k in KEYS}))
    return pd.DataFrame(rows)


def part2():
    rng = np.random.default_rng(20261014)
    rows, worlds = [], []
    for w in range(240):
        n = int(rng.integers(1, 7))
        idx = sorted(rng.choice(6, n, replace=False))
        fam, units = [FAM6[i] for i in idx], [f"t{i}" for i in idx]
        structure = ["family", "independent", "pairs"][w % 3]
        scale = [0.05, 0.1, 0.2, 0.25][(w // 3) % 4]
        margin = 0.05 if scale == 0.25 else 0.01
        s = 10 ** rng.uniform(-6, -1.3, n)
        tau_true = 10 ** rng.uniform(-6, -0.7)
        d = 0.02 * rng.standard_normal() + tau_true * rng.standard_normal(n) + s * rng.standard_normal(n)
        if structure == "family":
            levels = [fam, units]
        elif structure == "independent":
            levels = [units]
        else:
            uu = list(dict.fromkeys(fam))
            g = [[i for i, f in enumerate(fam) if f == x] for x in uu]
            d = np.array([np.mean([d[i] for i in gi]) for gi in g])
            s = np.array([math.sqrt(sum(s[i] ** 2 for i in gi)) / len(gi) for gi in g])
            levels = [uu]
        worlds.append((d, s, levels, scale, margin))
        r = A.pool(d, s, levels, scale, scale, margin)
        step = 0.06 if len(levels) == 2 else 0.01
        ref = reference(d, s, levels, scale, margin, step=step)
        cut = reference(d, s, levels, scale, margin, step=step, span=A.GRID_SPAN)   # cut at 5 scales, as the code is
        rows.append(dict(world=w, n=len(d), structure=structure, scale=scale, s_min=float(np.min(s)),
                         max_abs_err=max(abs(r[k] - ref[k]) for k in KEYS),
                         max_abs_err_cut5=max(abs(r[k] - cut[k]) for k in KEYS), declaration=r["declaration"],
                         ref_declaration=declare(ref)))
    return pd.DataFrame(rows), worlds


def part4(worlds):
    """Effect of two planted errors that the tests do not catch (mutation/run_mut.py M24, M28) on the stress worlds
    0, 4, 8, ...: largest change of any reported probability, mean, interval end or heterogeneity."""
    import inspect
    src = inspect.getsource(A.pool)
    out = {}
    for name, edit in (("M24 GRID_N 101", lambda ns, s_: (ns.update(GRID_N=101), s_)[1]),
                       ("M28 no half end weights", lambda ns, s_: s_.replace(
                           "(indices == 0) | (indices == GRID_N - 1), 0.5, 1.0)", "(indices == 0) | (indices == GRID_N - 1), 1.0, 1.0)"))):
        ns = dict(A.__dict__)
        exec(edit(ns, src), ns)
        worst = 0.0
        for d, s, levels, scale, margin in worlds[::4]:
            a, b = A.pool(d, s, levels, scale, scale, margin), ns["pool"](d, s, levels, scale, scale, margin)
            worst = max(worst, max(abs(a[k] - b[k]) for k in KEYS + ("lo", "hi")))
        out[name] = worst
    return out


def part3():
    try:
        import mpmath as mp
    except ImportError:          # pragma: no cover
        return None
    mp.mp.dps = 40
    fam, units = FAM6, list("abcdef")
    base = np.array([0.013, 0.018, 0.017, 0.0155, 0.019, 0.0152])
    taus = np.geomspace(1e-15, 0.5, 201)[::20]

    def exact(d, s, tf, td, ds):
        n = len(d)
        V = mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                V[i, j] = mp.mpf(ds) ** 2 + (mp.mpf(tf) ** 2 if fam[i] == fam[j] else 0) + (mp.mpf(td) ** 2 if i == j else 0) \
                          + (mp.mpf(s[i]) ** 2 if i == j else 0)
        Vi = V ** -1
        vd = Vi * mp.matrix([mp.mpf(x) for x in d])
        return float(mp.mpf(ds) ** 2 * sum(vd)), float(mp.mpf(ds) ** 2 - mp.mpf(ds) ** 4 * sum(Vi[i, j] for i in range(n) for j in range(n)))

    def floats(d, s, tf, td, ds):
        n = len(d)
        Am = np.diag(np.asarray(s) ** 2) + tf ** 2 * (np.array(fam)[:, None] == np.array(fam)[None, :]) + td ** 2 * np.eye(n)
        V = Am + ds ** 2
        Vi = np.linalg.inv(V)
        old = (ds ** 2 * (Vi @ d).sum(), ds ** 2 - ds ** 4 * Vi.sum())
        x = np.linalg.solve(Am, np.stack([np.ones(n), d], axis=1))
        v = ds ** 2 / (1 + ds ** 2 * x[:, 0].sum())
        return old, (v * x[:, 1].sum(), v)

    rows = []
    cases = [("one s small", lambda q: np.maximum(np.array([q, 1e-3, 1.1e-3, 4e-4, 6e-4, 8e-4]), q), base),
             ("all s equal", lambda q: np.full(6, q), base)]
    for label, smaker, d in cases:
        for q in (1e-3, 1e-4, 1e-5, 1e-6):
            s = smaker(q)
            worst = dict(old_mean=0.0, old_var=0.0, new_mean=0.0, new_var=0.0)
            for tf in taus:
                for td in taus:
                    m_ex, v_ex = exact(d, s, tf, td, 0.1)
                    (mo, vo), (mn, vn) = floats(d, s, tf, td, 0.1)
                    worst["old_mean"] = max(worst["old_mean"], abs(mo - m_ex) / abs(m_ex))
                    worst["old_var"] = max(worst["old_var"], abs(vo - v_ex) / v_ex)
                    worst["new_mean"] = max(worst["new_mean"], abs(mn - m_ex) / abs(m_ex))
                    worst["new_var"] = max(worst["new_var"], abs(vn - v_ex) / v_ex)
            rows.append(dict(case=label, s=q, **worst))
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover  # validation script
    p1 = part1(load_old())
    p1.to_csv(os.path.join(OUT, "review_case.csv"), index=False)
    p2, worlds = part2()
    p4 = part4(worlds)
    p2.to_csv(os.path.join(OUT, "stress_240.csv"), index=False)
    p3 = part3()
    if p3 is not None:
        p3.to_csv(os.path.join(OUT, "precision_mpmath.csv"), index=False)
    summary = dict(review_case=p1[["scale", "code", "p_above", "ref_p_above", "declaration", "ref_declaration",
                                   "max_abs_err"]].to_dict("records"),
                   stress=dict(worlds=len(p2), max_abs_err=float(p2.max_abs_err.max()),
                               max_abs_err_reference_cut_at_5_scales=float(p2.max_abs_err_cut5.max()),
                               declaration_mismatches=int((p2.declaration != p2.ref_declaration).sum()),
                               s_min=float(p2.s_min.min())),
                   precision=None if p3 is None else p3.to_dict("records"), uncaught_planted_errors=p4)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
