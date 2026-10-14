"""Known-answer tests for scripts/analysis_x1.py, the frozen X1 analysis (docs/ANALYSIS_SPEC.md section 14).

Synthetic inputs only: the analysis must not run on any X1 adversarial-arm score before the tag prereg-tier12-v3.
Every expected value is derived by hand (the derivation is stated next to the case) or by an independent method
(scipy, enumeration, brute-force integration), never by the code under test.
"""
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.integrate import cumulative_trapezoid

# ACCEPT_WORKDIR: the tree under test when this file runs from outside the repo (the offload acceptance check)
ROOT = os.environ.get("ACCEPT_WORKDIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tests", "prereg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import analysis_x1 as A  # noqa: E402
import prereg_rules as P  # noqa: E402
import synth  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
X1_MANIFEST = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v3.x1.tsv")
R1_RECORD = os.path.join(ROOT, "docs", "prereg", "r1_record.json")


def pts(spec, fail=()):
    """Points in prereg_rules._cell_points format. spec: {lam: (B, C)}, each a scalar (every seed) or a 5-list;
    fail: (lam, seed) pairs whose run failed."""
    out = []
    for lam in sorted(spec):
        b, c = (v if isinstance(v, list) else [v] * 5 for v in spec[lam])
        ok = [s for s in SEEDS if (lam, s) not in fail]
        out.append(dict(lam=float(lam), n_fail=5 - len(ok), B={s: b[s] for s in ok}, C={s: c[s] for s in ok}))
    return out


def zero(b=0.40, c=0.80):
    b, c = (v if isinstance(v, list) else [v] * 5 for v in (b, c))
    return {s: (b[s], c[s]) for s in SEEDS}


def close(got, want, tol=1e-12):
    if want is None:
        assert got is None
    else:
        assert got == pytest.approx(want, abs=tol)


# ---------------------------------------------------------------------------------------------
# P1 reading (spec section 2)
# ---------------------------------------------------------------------------------------------
C_SPREAD = [0.70, 0.72, 0.74, 0.76, 0.78]          # mean 0.74, SD sqrt(0.001)
REF = {1: (0.45, 0.78), 3: (0.55, 0.70)}           # from the anchor (0.40, 0.80) it crosses b* = 0.50 at C = 0.74
JACK = math.sqrt(0.001 / 5)                         # SD / sqrt(5) = the jackknife SE of a 5-run mean
NOT_SEED_0_1 = [None, None, "x", "x", "x"]          # "x": a reading exists


@pytest.mark.parametrize("ref, dis, mode, want", [
    # interpolated crossing (ref) and an exact hit (dis); without seed i the hit reads (3.70 - C_i) / 4
    (pts(REF), pts({1: (0.50, C_SPREAD)}), "mean",
     dict(d=0.0, s=JACK, m=5, reason=None, bio=(0.74, 0.74), status=("bracketed", "bracketed"), crossings=(1, 1),
          readings=[-0.01, -0.005, 0.0, 0.005, 0.01])),
    # the same runs read one by one (SI-69 sensitivity): d_s = 0.74 - C_s, s = SD / sqrt(5)
    (pts(REF), pts({1: (0.50, C_SPREAD)}), "per_run",
     dict(d=0.0, s=JACK, m=5, reason=None, readings=[0.04, 0.02, 0.0, -0.02, -0.04])),
    # three crossings: the first (lambda 1 to 2) is read, the other two are only counted
    (pts({1: (0.45, 0.78), 2: (0.55, 0.70), 3: (0.45, 0.60), 4: (0.55, 0.50)}), pts({1: (0.50, 0.75)}), "mean",
     dict(d=-0.01, s=0.0, m=5, reason=None, bio=(0.74, 0.75), crossings=(3, 1))),
    # a point whose 5 runs failed is skipped; a point with 1 failed run is the mean of its other 4 (SI-71)
    (pts({1: (0.45, 0.78), 2: (0.9, 0.1), 3: (0.55, 0.70)}, fail=[(2, s) for s in SEEDS]),
     pts({1: (0.50, [0.70, 0.72, 0.74, 0.76, 0.0])}, fail=[(1, 4)]), "mean",
     dict(d=0.01, m=5, reason=None, bio=(0.74, 0.73))),
    # drop_failed (SI-71 sensitivity): the reference's lambda-1 point has a failed run and goes: anchor to lambda 3
    (pts(REF, fail=[(1, 0)]), pts({1: (0.50, 0.75)}), "drop_failed",
     dict(d=0.80 - 0.10 / 1.5 - 0.75, reason=None, bio=(0.80 - 0.10 / 1.5, 0.75))),
    # the reference never reaches b*: not evaluable
    (pts({1: (0.42, 0.79), 3: (0.48, 0.78)}), pts({1: (0.50, 0.75)}), "mean",
     dict(d=None, s=None, reason="unreached", status=("unreached_low", "bracketed"), crossings=(0, 1))),
    (pts({1: (0.42, 0.79), 3: (0.48, 0.78)}), pts({1: (0.50, 0.75)}), "per_run", dict(d=None, m=0, reason="unreached")),
    # mean B 0.506 reaches b*, but without seed 0 or 1 it is 0.4925: 3 of 5 leave-one-seed-out readings (spec 2.3)
    (pts(REF), pts({1: ([0.56, 0.56, 0.47, 0.47, 0.47], 0.75)}), "mean",
     dict(m=3, s=None, reason="readings", readings=NOT_SEED_0_1)),
    # 4 of 5 readings is enough (mean B 0.508; without seed 0 it is 0.48, without any other seed 0.515)
    (pts(REF), pts({1: ([0.62, 0.48, 0.48, 0.48, 0.48], 0.75)}), "mean",
     dict(m=4, s=0.0, reason=None, readings=[None, "x", "x", "x", "x"])),
    # read run by run, only seeds 0 and 1 of the discriminator reach b*
    (pts(REF), pts({1: ([0.56, 0.56, 0.47, 0.47, 0.47], 0.75)}), "per_run",
     dict(m=2, s=None, reason="readings", readings=["x", "x", None, None, None])),
])
def test_p1_reading(ref, dis, mode, want):
    r = A.read_p1(zero(), ref, dis, 0.50, mode=mode)
    assert r["mode"] == mode and r["reason"] == want["reason"]
    for k in ("d", "s", "m"):
        if k in want:
            close(r[k], want[k])
    for k in ("bio", "status", "crossings"):
        if k in want:
            got = (r[k]["ref"], r[k]["dis"])
            assert got == (pytest.approx(want[k], abs=1e-12) if k == "bio" else want[k])
    for g, w in zip(r["readings"], want.get("readings", [])):
        if w == "x":
            assert g is not None
        else:
            close(g, w)
    assert len(r["readings"]) == 5


# ---------------------------------------------------------------------------------------------
# frontier area (spec section 3)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("points, want", [
    # vertical strips from the corner (0.4, 0.6): 0.1 x 0.3 + 0.1 x 0.2 + 0.1 x 0.15
    ([(0.6, 0.8), (0.7, 0.75), (0.5, 0.9)], 0.065),
    # + a dominated point, one below the corner's C, one left of its B and one on its B: no area added
    ([(0.6, 0.8), (0.7, 0.75), (0.5, 0.9), (0.55, 0.7), (0.9, 0.55), (0.35, 0.95), (0.4, 0.99)], 0.065),
    ([], 0.0),
])
def test_area(points, want):  # test-reason: known answer for the union of rectangles
    assert A.area(points, 0.4, 0.6) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("corner, drop_failed, fail, want", [
    # corner (b0, c0 - 0.10) = (0.40, 0.70): ref 0.1 x 0.02 + 0.2 x 0.08, dis 0.25 x 0.05, both / 0.2. Without seed i
    # the corner's C is c0_(i) - 0.10 and d = 0.2025 - 0.25 c_ref, so s = 0.25 x SE(c0)
    ("collapse", False, (), dict(ref=0.09, dis=0.0625, d=0.0275, s=0.25 * JACK,
                                 readings=[0.025, 0.02625, 0.0275, 0.02875, 0.03])),
    ("zero", False, (), dict(ref=1.14, dis=0.9375, d=0.2025, s=0.0)),          # corner (b0, 0), SI-72 sensitivity
    ("collapse", True, [(2, 0)], dict(ref=0.08, dis=0.0625, d=0.0175)),         # drop_failed: lambda 2 goes
    ("collapse", False, [(2, 0)], dict(ref=0.09, dis=0.0625, d=0.0275)),        # kept: the mean of its 4 runs
])
def test_read_area(corner, drop_failed, fail, want):  # test-reason: known answers: moving corner, drop_failed, corner (b0, 0)
    ref, dis = pts({1: (0.6, 0.78), 2: (0.7, 0.72)}, fail=fail), pts({1: (0.65, 0.75)})
    r = A.read_area(zero(c=[0.76, 0.78, 0.80, 0.82, 0.84]), ref, dis, 0.2, corner=corner, drop_failed=drop_failed)
    close(r["area"]["ref"], want["ref"])
    close(r["area"]["dis"], want["dis"])
    close(r["d"], want["d"])
    assert r["m"] == 5
    if "s" in want:
        close(r["s"], want["s"])
    for g, w in zip(r["readings"], want.get("readings", [])):
        close(g, w)


# ---------------------------------------------------------------------------------------------
# combining datasets (spec section 4) and declarations (section 5)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("levels", [[["a", "b", "c"]], [["f", "f", "g"], ["a", "b", "c"]]])
@pytest.mark.parametrize("d, s, declaration", [
    ([0.02, -0.01, 0.03], [0.01, 0.02, 0.015], "inconclusive"),
    ([0.10, 0.10, 0.10], [0.005, 0.005, 0.005], "above"),
    ([-0.10, -0.10, -0.10], [0.005, 0.005, 0.005], "below"),
    ([0.0, 0.001, -0.001], [0.002, 0.002, 0.002], "equivalent"),
])
def test_pool_closed_form(levels, d, s, declaration):
    """Spreads ~0: Delta | d is normal with the inverse-variance weighted mean, shrunk by the N(0, 0.1^2) prior."""
    d, s = np.array(d), np.array(s)
    r = A.pool(d, s, levels, 0.1, 1e-9, 0.01)
    prec = 1 / 0.1 ** 2 + (1 / s ** 2).sum()
    m, sd = (d / s ** 2).sum() / prec, prec ** -0.5
    assert r["n"] == 3 and r["declaration"] == declaration and r["margin"] == 0.01
    close(r["mean"], m, 1e-9)
    assert [r["lo"], r["hi"]] == pytest.approx(stats.norm.ppf([0.025, 0.975], m, sd), abs=1e-8)
    close(r["p_above"], stats.norm.sf(0.01, m, sd), 1e-8)
    close(r["p_below"], stats.norm.cdf(-0.01, m, sd), 1e-8)
    close(r["p_within"], stats.norm.cdf(0.01, m, sd) - stats.norm.cdf(-0.01, m, sd), 1e-8)
    assert [t["mean"] for t in r["theta"]] == pytest.approx([m] * 3, abs=1e-9)
    assert [t["lo"] for t in r["theta"]] == pytest.approx([r["lo"]] * 3, abs=1e-7)


def brute(d, s, fam, scale, r, h=0.004, top=0.8, n_delta=4001):
    """Posterior by brute force: Delta on a fine grid, both spreads by the midpoint rule on [0, top]; the per-dataset
    means by the precision form of the Gaussian model (beta = Delta, family effects, dataset effects)."""
    d, s, n = np.asarray(d, float), np.asarray(s, float), len(d)
    fams = sorted(set(fam))
    Zf = np.array([[float(a == u) for u in fams] for a in fam])
    X = np.hstack([np.ones((n, 1)), Zf, np.eye(n)])
    D = np.linspace(-1, 1, n_delta)
    pD = stats.norm.pdf(D, 0, scale)
    taus = (np.arange(int(round(top / h))) + 0.5) * h
    cum, first, het, theta, mass = np.zeros(n_delta), 0.0, 0.0, np.zeros(n), 0.0
    for tf in taus:
        for td in taus:
            Sig = np.diag(s ** 2 + td ** 2) + tf ** 2 * Zf @ Zf.T
            Si = np.linalg.inv(Sig)
            a, b, c = d @ Si @ d, Si.sum(0) @ d, Si.sum()
            g = pD * np.exp(-0.5 * (a - 2 * D * b + D ** 2 * c)) / math.sqrt(np.linalg.det(Sig))
            g *= math.exp(-(tf ** 2 + td ** 2) / (2 * scale ** 2))
            cg = cumulative_trapezoid(g, D, initial=0)
            cum += cg
            first += np.trapezoid(g * D, D)
            het += cg[-1] * math.hypot(tf, td)
            prec = X.T @ (X / s[:, None] ** 2) + np.diag(np.r_[1 / scale ** 2, [1 / tf ** 2] * len(fams), [1 / td ** 2] * n])
            theta += cg[-1] * (X @ np.linalg.solve(prec, X.T @ (d / s ** 2)))
            mass += cg[-1]
    F = cum / cum[-1]
    cdf = lambda x: float(np.interp(x, D, F))  # noqa: E731
    return dict(mean=first / mass, heterogeneity=het / mass, theta=theta / mass, p_above=1 - cdf(r), p_below=cdf(-r),
                p_within=cdf(r) - cdf(-r), lo=float(np.interp(0.025, F, D)), hi=float(np.interp(0.975, F, D)))


def test_pool_brute_force():  # test-reason: spreads > 0 against brute-force integration (independent algebra and grid)
    d, s, fam = [0.03, 0.05, -0.02, 0.01], [0.02, 0.025, 0.02, 0.03], ["f1", "f1", "f2", "f3"]
    r = A.pool(d, s, [fam, ["a", "b", "c", "d"]], 0.1, 0.1, 0.01)
    b = brute(d, s, fam, 0.1, 0.01)
    close(r["mean"], b["mean"], 1e-8)
    close(r["heterogeneity"], b["heterogeneity"], 1e-5)
    assert [t["mean"] for t in r["theta"]] == pytest.approx(b["theta"], abs=1e-8)
    for k in ("lo", "hi", "p_above", "p_below", "p_within"):
        close(r[k], b[k], 5e-5)
    assert r["declaration"] == "inconclusive" and r["delta_scale"] == 0.1 and r["tau_scale"] == 0.1


def test_combine_structures():
    tasks = list(P.REAL_TASKS)              # pancreas, lung, immune, immune_hum_mou, atac_small, atac_large
    d, s = [0.03, 0.01, 0.05, 0.04, -0.01, 0.02], [0.01, 0.02, 0.015, 0.01, 0.02, 0.025]
    fam = A.combine(tasks, d, s, "family", 0.1, 0.01)
    close(fam["mean"], A.pool(d, s, [[P.TASK_FAMILY[t] for t in tasks], tasks], 0.1, 0.1, 0.01)["mean"])
    assert fam["units"] == tasks and fam["structure"] == "family"
    ind = A.combine(tasks, d, s, "independent", 0.2, 0.01)
    close(ind["mean"], A.pool(d, s, [tasks], 0.2, 0.2, 0.01)["mean"])
    # pairs: each family's datasets averaged first, SE sqrt(sum s^2) / k; one spread over the 4 units
    pairs = A.combine(tasks, d, s, "pairs", 0.1, 0.01)
    units = ["pancreas", "lung", "immune", "atac"]
    du = [0.03, 0.01, 0.045, 0.005]
    su = [0.01, 0.02, math.hypot(0.015, 0.01) / 2, math.hypot(0.02, 0.025) / 2]
    assert pairs["units"] == units and pairs["n"] == 4
    close(pairs["mean"], A.pool(du, su, [units], 0.1, 0.1, 0.01)["mean"])
    assert A.combine([], [], [], "family", 0.1, 0.01) == {"n": 0, "declaration": "not_evaluable"}


# ---------------------------------------------------------------------------------------------
# frequentist companions (sections 4, 5, 7) and P2 (section 6)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d, want", [
    ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 2 / 64),          # T+ = 21, the maximum
    ([-0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 4 / 64),         # T+ = 20: 2 of 64 sign patterns at least as extreme per side
    ([0.0, 0.2, -0.3, 0.4], 6 / 8),                    # the zero is dropped: n = 3, T+ = 4, |T+ - 3| >= 1 in 6 of 8
    ([0.1, 0.1, -0.2, 0.3], 10 / 16),                  # tied |d|: average ranks 1.5, 1.5, 3, 4; T+ = 7
    ([], 1.0),
    ([0.0, 0.0], 1.0),
])
def test_wilcoxon_exact(d, want):
    close(A.wilcoxon_exact(d), want)
    if d and 0.0 not in d and len(set(np.abs(d))) == len(d):
        close(A.wilcoxon_exact(d), stats.wilcoxon(d, method="exact").pvalue)


def test_sign_test_and_holm():  # test-reason: known answers for the sign test and Holm's step-down adjustment
    close(A.sign_test([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 2 / 64)
    close(A.sign_test([0.1, -0.2, 0.3, 0.0, 0.5]), stats.binomtest(3, 4).pvalue)      # the zero is dropped
    assert A.sign_test([]) == 1.0
    assert A.holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2], abs=1e-12)
    assert A.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0], abs=1e-12)                 # capped at 1, monotone


@pytest.mark.parametrize("x, n", [(0, 30), (4, 5), (30, 30), (7, 180)])
def test_clopper_pearson(x, n):  # test-reason: exact interval against scipy, including both edges
    ci = stats.binomtest(x, n).proportion_ci(confidence_level=0.95, method="exact")
    assert A.clopper_pearson(x, n) == pytest.approx((ci.low, ci.high), abs=1e-12)


def test_collapse_counts():  # test-reason: the 0.10 boundary within the tie tolerance and the failure types
    rows = pd.DataFrame(dict(tag=list("abcde"), seed_i=SEEDS, status=["ok", "ok", "failed", "failed", "ok"],
                             C=[0.70, 0.70 + 5e-10, np.nan, np.nan, 0.70 + 2e-9]))
    r = A.collapse_counts(rows, zero(c=0.80), {"c": "diverged", "d": "nonfinite_latent"})
    lo, hi = A.clopper_pearson(4, 5)
    assert r == dict(n=5, collapsed=4, bio=2, failed={"diverged": 1, "nonfinite_latent": 1}, rate=0.8,
                     lo=pytest.approx(lo, abs=1e-12), hi=pytest.approx(hi, abs=1e-12))
    # Agresti-Caffo: one collapse and one non-collapse added per arm: 4/32 - 1/32, SE from the adjusted rates
    assert A.agresti_caffo(3, 30, 0, 30) == pytest.approx((0.09375, math.sqrt(0.1396484375 / 32)), abs=1e-12)


# ---------------------------------------------------------------------------------------------
# single-lambda tables (section 8), best-of-k and frontier positions (9), window width (10)
# ---------------------------------------------------------------------------------------------
def test_transfer_lambda():
    lams = [1.0, 3.0, 10.0]
    S = {t: {1.0: 0.5, 3.0: 0.6, 10.0: 0.4} for t in list(P.REAL_TASKS) + ["sim1"]}
    fails = {t: dict.fromkeys(lams, 0) for t in S}
    S["immune_hum_mou"][10.0] = 5.0      # planted best point; immune (same family) never trains on it
    fails["lung"][3.0] = 1               # a failed lung run: lambda 3 is not eligible wherever lung trains
    r = A.transfer_lambdas(S, fails)
    assert {t: v["lam"] for t, v in r.items()} == {"immune": 1.0, "immune_hum_mou": 1.0, "pancreas": 10.0,
                                                    "lung": 10.0, "atac_small": 10.0, "atac_large": 10.0, "sim1": 10.0}
    assert r["immune"]["train"] == ["atac_large", "atac_small", "lung", "pancreas"]
    assert r["atac_small"]["train"] == ["immune", "immune_hum_mou", "lung", "pancreas"]
    assert r["sim1"]["train"] == sorted(P.REAL_TASKS)
    tie = {"a": {1.0: 0.5, 3.0: 0.5 + 5e-10}}
    assert A.choose_lambda(tie, {"a": {1.0: 0, 3.0: 0}}, ["a"]) == 1.0                # tie -> smaller lambda
    assert A.choose_lambda({"a": {1.0: 0.5, 3.0: 0.5 + 2e-9}}, {"a": {1.0: 0, 3.0: 0}}, ["a"]) == 3.0
    assert A.choose_lambda({"a": {}}, {"a": {1.0: 5}}, ["a"]) is None                # every run failed


@pytest.mark.parametrize("S", [[0.3], [0.5, 0.2, 0.9, 0.4, 0.7, 0.6], [0.4, 0.4, 0.1]])
def test_best_of_k(S):  # test-reason: E_k against enumeration of every k-subset
    want = [np.mean([max(c) for c in itertools.combinations(S, k)]) for k in range(1, len(S) + 1)]
    assert A.best_of_k(S) == pytest.approx(want, abs=1e-12)


def test_dominance():  # test-reason: known answer for the frontier-position counts
    front = [(0.6, 0.8), (0.5, 0.7), (0.4, 0.6), (0.6, 0.6), (0.5, 0.75)]
    assert A.dominance((0.5, 0.7), front) == {"dominated_by": 2, "dominates": 1}


def test_window():
    w = A.window([(1.0, [0.50, 0.52, 0.48, 0.51, 0.49]), (3.0, [0.60, 0.62, 0.58, 0.61, 0.59]),
                  (10.0, [0.59, 0.60, 0.58, 0.595, 0.585]), (30.0, [0.3] * 5), (100.0, [])])
    lower = 0.60 - stats.t.ppf(0.975, 4) * math.sqrt(0.00025 / 5)           # best mean - t x SD / sqrt(n) = 0.5804
    assert w == dict(status="ok", best=3.0, lower=pytest.approx(lower, abs=1e-12), lams=[3.0, 10.0],
                     width=pytest.approx(math.log10(10 / 3), abs=1e-12), n_points=2)
    tie = A.window([(1.0, [0.5, 0.5]), (3.0, [0.5, 0.5])])                   # tie -> smaller lambda; SD 0
    assert tie["best"] == 1.0 and tie["lams"] == [1.0, 3.0] and tie["width"] == pytest.approx(math.log10(3))
    none = dict(lower=None, lams=None, width=None, n_points=None)
    assert A.window([(1.0, [0.5]), (3.0, [0.4, 0.45])]) == dict(none, status="no_interval", best=1.0)
    assert A.window([(1.0, []), (3.0, [])]) == dict(none, status="no_points", best=None)


def test_guards():  # test-reason: refusals at the function boundaries
    with pytest.raises(P.PreregError, match="scale"):
        A.read_area(zero(), pts(REF), pts(REF), 0.0)
    with pytest.raises(P.PreregError, match="positive"):
        A.pool([0.1, 0.2], [0.01, 0.0], [["a", "b"]], 0.1, 0.1, 0.01)
    with pytest.raises(P.PreregError, match="grid"):
        A.choose_lambda({"a": {1.0: 0.5}, "b": {3.0: 0.5}}, {"a": {1.0: 0}, "b": {3.0: 0}}, ["a", "b"])
    row = dict(task="pancreas", status="ok", B=0.5, C=0.7, **dict.fromkeys(P.BATCH_METRICS, 0.5),
               **dict.fromkeys(P.bio_metrics_for("pancreas"), 0.7))
    row["kBET"] = 0.6                     # B is no longer the mean of the 5 batch metrics
    with pytest.raises(P.PreregError, match="local"):
        A.add_scores(pd.DataFrame([row]))


# ---------------------------------------------------------------------------------------------
# end to end on the committed X1 + X13 manifest with synthetic scores
# ---------------------------------------------------------------------------------------------
CENTRE = {("js", 0): 0.0, ("js", 1): 0.5, ("w1", 0): 1.0, ("w1", 1): 1.0, ("mmd", 0): 2.0, ("mmd", 1): 2.0,
          ("sinkhorn", 0): 1.5, ("sinkhorn", 1): 0.5}     # log10 lambda of half batch removal, inside every X1 grid
BONUS = {"reference": 0.20}        # biology the reference critic keeps: P1 d = (2/3)(b* - b0), about +0.097
SPREAD = {"reference": 0.08}       # local-minus-global spread it adds to the batch metrics (B unchanged)
TRAJ_DROP = 0.08                   # trajectory below the other bio metrics: C = c - 0.01 on immune, immune_hum_mou
FAIL = {("lung", "0", "discriminator", "10", "0"): "diverged",
        ("pancreas", "1", "mmd", "3000", "2"): "nonfinite_latent",
        ("lung", "0", "harmony", "8", "0"): "nonfinite_latent",           # an X13 configuration without a run
        **{("sim2", "0", "sinkhorn", "3000", str(s)): "diverged" for s in SEEDS},
        # barycenter, unconditioned: failed runs leave lambda 10, 100 failure-free in pancreas and 0.3, 1, 3, 30 in
        # lung, so no lambda is eligible for a held-out dataset that trains on both (all outside their families);
        # each keeps a high failure-free point, so b_common (R4) barely moves
        **{("pancreas", "0", "barycenter", lam, "3"): "diverged" for lam in ("0.3", "1", "3", "30")},
        **{("lung", "0", "barycenter", lam, "3"): "diverged" for lam in ("10", "100")}}


def _gain(r):
    if (r["task"], r["cond"]) == ("immune", "1"):
        return 0.015                                       # b_common - b0 < 0.02: bstar_low_signal
    lone = {"reference": "1", "discriminator": "0"}.get(r["arm"])
    if (r["task"], r["cond"]) == ("atac_large", "1") and lone:
        return 0.30 if r["seed"] == lone else 0.004        # one responding seed per P1 arm: 3 of 5 readings
    return 0.30


def world(r):
    """(status, B, C, spread) of one manifest row; seed effects are shared by every arm (pairing)."""
    key = f"{r['task']}|{r['cond']}|{r['seed']}"
    B = 0.40 + 0.01 * sorted(synth.BPM.TASKS).index(r["task"]) + synth._noise("b" + key, 0.01)
    C = 0.75 + synth._noise("c" + key, 0.01)
    a = 0.0
    if r["arm"] in P.ADV_ARMS:
        x = math.log10(float(r["lam"])) - CENTRE[(P.FAMILY_OF[r["arm"]], int(r["cond"]))]
        B += _gain(r) * synth.sig(x / 0.3) + synth._noise("B" + r["tag"], 0.003)
        C += -0.25 * synth.sig((x - 1) / 0.3) + BONUS.get(r["arm"], 0.0) * synth.sig(x / 0.3) + synth._noise("C" + r["tag"], 0.003)
        a = SPREAD.get(r["arm"], 0.0) * synth.sig(x / 0.3)
    elif r["experiment"] == "X13":
        B += 0.05 + 0.01 * math.log1p(float(r["lam"]))
        C -= 0.02
    return FAIL.get((r["task"], r["cond"], r["arm"], r["lam"], r["seed"]), "ok"), B, C, a


def score_tables(M):
    S, F = [], []
    for r in M.to_dict("records"):
        status, B, C, a = world(r)
        if status != "ok":
            F.append(dict(tag=r["tag"], status=status, detail="synthetic"))
            continue
        bio = P.bio_metrics_for(r["task"])
        row = {"tag": r["tag"], "kbet_seed": 0, "score_seconds": 1.0, **synth.PROVENANCE}
        for m in synth.ALL_METRICS:
            if m in A.LOCAL:
                row[m] = B + a
            elif m in A.GLOBAL:
                row[m] = B - 1.5 * a
            elif m in bio:
                row[m] = C - TRAJ_DROP if m == "trajectory" else C
            else:
                row[m] = float("nan")
        S.append(row)
    return pd.DataFrame(S), pd.DataFrame(F, columns=["tag", "status", "detail"])


def argv(a):
    return ["--manifest", a["manifest"], "--scores", a["scores"], "--failures", a["failures"],
            "--r1-record", a["r1"], "--r4-record", a["r4"], "--out-json", a["out"]]


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("analysis_x1"))
    M, _ = P.read_manifest(X1_MANIFEST)
    M = M[M.experiment.isin(["X1", "X13"])].reset_index(drop=True)
    man = synth.write_tsv(M, os.path.join(tmp, "manifest.tsv"))
    sp, fp = synth.write_scores(*score_tables(M), os.path.join(tmp, "scores"), "x1")
    S, F = P.read_scores([sp])[0], P.read_failures(fp)
    r1 = P.read_record(R1_RECORD, "R1")
    r4p = os.path.join(tmp, "r4x1.json")          # what freeze_matched_lambda.py --stage x1 records, no extension
    P.write_record(r4p, P.r4_stage(P.outcomes(M, S, F, ["X1"]), "x1", r1, need=set()), dict(scores=[sp]))
    args = dict(manifest=man, scores=sp, failures=fp, r1=R1_RECORD, r4=r4p, out=os.path.join(tmp, "analysis.json"))
    assert A.main(argv(args)) == 0
    return dict(args=args, rec=json.load(open(args["out"])), M=M, out=A.add_scores(P.outcomes(M, S, F, ["X1", "X13"])),
                r1=r1, r4=json.load(open(r4p)))


def test_end_to_end(e2e):  # test-reason: the record's wiring, planted effects and not-evaluable bookkeeping
    rec, out, r1, r4 = e2e["rec"], e2e["out"], e2e["r1"], e2e["r4"]
    assert rec["rule"] == "analysis_x1" and rec["status"] == "done"
    assert set(rec["inputs"]) == {"manifest", "scores", "failures", "r1_record", "r4_record"}
    assert all(len(x["sha256"]) == 64 for v in rec["inputs"].values() for x in v)
    prim = rec["primary"]
    assert prim["settings"] == dict(value="C", mode="mean", corner="collapse", structure="family", scale=0.1)
    assert len(prim["cells"]) == 16 and prim["n_primary"] == 6
    assert set(prim["holm"]) == {f"{e}|{c}" for e in ("p1", "area", "p2") for c in (0, 1)}

    # wiring: the record's cell values are the functions' values on the same runs
    t, c = "pancreas", 1
    grid = {a: r1["families"][f"{P.FAMILY_OF[a]}|{c}"]["x1_grid"] for a in ("reference", "discriminator")}
    tg = r4["targets"][f"{t}|{c}"]

    def read(v):
        z = A.zero_runs(out, t, c, value=v)
        ref, dis = (A.arm_points(out, t, c, a, grid[a], sorted(z), value=v) for a in ("reference", "discriminator"))
        return z, ref, dis, A.read_p1(z, ref, dis, tg["bstar"])
    z, ref, dis, p1 = read("C")
    cell = prim["cells"][f"{t}|{c}"]
    close(cell["p1"]["d"], p1["d"])
    close(cell["p1"]["s"], p1["s"])
    close(cell["area"]["d"], A.read_area(z, ref, dis, tg["b_common"] - tg["b0"])["d"])
    loc, glo = read("L")[3]["d"], read("G")[3]["d"]
    assert abs(3 * loc + 2 * glo) < 1e-12 and loc > 0.03            # 3 local + 2 global = 5 batch, equal at b*
    close(cell["local"]["d"], loc)
    assert prim["cells"]["immune|1"]["low_signal"] and prim["cells"]["immune|1"]["area"] is None

    # planted effects and the not-evaluable bookkeeping
    pooled = prim["pooled"]
    assert pooled["0"]["p1"]["tasks"] == list(P.REAL_TASKS) and pooled["0"]["p1"]["declaration"] == "above"
    assert pooled["1"]["p1"]["not_evaluable"] == [{"task": "immune", "reason": "low_signal"},
                                                 {"task": "atac_large", "reason": "readings"}]
    assert [x["task"] for x in pooled["1"]["p1"]["low_signal"]] == ["immune"]
    assert pooled["1"]["p1"]["declaration"] == "above" and len(pooled["1"]["p1"]["signs"]) == 4
    assert pooled["1"]["area"]["not_evaluable"] == [{"task": "immune", "reason": "low_signal"}]
    assert pooled["0"]["local"]["declaration"] == "above"
    close(pooled["0"]["local"]["sign_test_p"], 2 / 64)
    close(pooled["1"]["local"]["sign_test_p"], 2 / 16)
    close(pooled["0"]["p1"]["wilcoxon_p"], 2 / 64)
    assert pooled["0"]["p2"]["declaration"] == "below" and len(pooled["0"]["p2"]["diffs"]) == 6
    # the pooled results use the frozen prior scales and margins (P1 0.1 and 0.01, SI-73/74; P2 0.25 and 0.05, SI-75)
    p1, p2 = pooled["0"]["p1"], pooled["0"]["p2"]
    for got, d, s, scale, margin in ((p1, [x["d"] for x in p1["signs"]], [x["s"] for x in p1["signs"]], 0.1, 0.01),
                                     (p2, [x["d"] for x in p2["diffs"]], [x["se"] for x in p2["diffs"]], 0.25, 0.05)):
        want = json.loads(json.dumps(A.combine(got["tasks"], d, s, "family", scale, margin)))
        assert {k: got[k] for k in want} == want

    # P2 counts with the planted failures
    p2 = prim["p2"]
    assert p2["0"]["discriminator"]["lung"]["failed"] == {"diverged": 1, "nonfinite_latent": 0}
    assert p2["1"]["mmd"]["pancreas"]["failed"] == {"diverged": 0, "nonfinite_latent": 1}
    assert p2["0"]["discriminator"]["lung"]["n"] == 30 and p2["0"]["discriminator"]["lung"]["collapsed"] == 5
    assert p2["0"]["reference"]["lung"]["collapsed"] == 0 and p2["0"]["discriminator"]["pooled"]["n"] == 180

    # single-lambda tables, best-of-k, anchors, frontier positions, window
    tr = prim["transfer"]["1"]["reference"]
    assert tr["immune"]["train"] == ["atac_large", "atac_small", "lung", "pancreas"]
    assert tr["sim1"]["train"] == sorted(P.REAL_TASKS)
    assert tr["immune"]["lam"] in grid["reference"] and tr["immune"]["at"]["n_ok"] == 5
    bary = prim["transfer"]["0"]["barycenter"]
    assert bary["immune"]["lam"] is None and bary["immune"]["at"] is None and bary["sim1"]["lam"] is None
    assert bary["pancreas"]["lam"] in (0.3, 1.0, 3.0, 30.0) and bary["lung"]["lam"] in (10.0, 100.0)
    assert bary["pancreas"]["best_lam"] in (10.0, 100.0) and bary["pancreas"]["best"]["n_fail"] == 0
    assert prim["cells"]["lung|0"]["best_of_k"]["harmony"]["n"] == 5
    bk = prim["cells"]["sim2|0"]["best_of_k"]
    assert {f: bk[f]["n"] for f in bk} == dict(dict.fromkeys(P.ADV_ARMS, 6), sinkhorn=5, harmony=6, scanorama=6,
                                                 sysvi=6, scanvi=1, none=1, scvi_adv=1)
    assert bk["scanvi"]["stratum"] == "label_aware" and bk["harmony"]["stratum"] == "standard"
    assert np.all(np.diff(bk["reference"]["E"]) >= -1e-12) and len(bk["reference"]["E"]) == 6
    assert set(prim["cells"]["sim2|0"]["anchors"]) == {"pca|0|10", "pca|0|50", "harmony|2|50", "scanorama|20|100"}
    fr = prim["cells"]["pancreas|0"]["frontier"]
    assert len(fr) == 25 and set(fr["harmony|0.5|10"]) == set(P.ADV_ARMS)
    w = prim["cells"]["pancreas|0"]["window"]["reference"]
    assert w["status"] == "ok" and w["n_points"] == len(w["lams"])
    # selection score S = 0.4 B + 0.6 C (SI-77): best-of-k's E_N is the best configuration's S
    g0 = r1["families"]["w1|0"]["x1_grid"]
    pts0 = A.arm_points(out, "pancreas", 0, "reference", g0, sorted(A.zero_runs(out, "pancreas", 0)))
    close(prim["cells"]["pancreas|0"]["best_of_k"]["reference"]["E"][-1],
          max(0.4 * np.mean(list(p["B"].values())) + 0.6 * np.mean(list(p["C"].values())) for p in pts0 if p["B"]))
    pca = out[(out.experiment == "X13") & (out.task == "pancreas") & (out.arm == "pca") & (out.n_latent == "10")]
    close(prim["cells"]["pancreas|0"]["anchors"]["pca|0|10"], float(0.4 * pca.B.mean() + 0.6 * pca.C.mean()))
    at = tr["immune"]["at"]
    close(at["S"][0], 0.4 * at["B"][0] + 0.6 * at["C"][0])

    # sensitivity analyses (spec section 11)
    sens = rec["sensitivity"]
    assert set(sens) == set(A.SENSITIVITY) == {"no_trajectory", "per_run", "drop_failed", "corner_zero", "pairs",
                                                 "independent", "prior_0.05", "prior_0.2"}
    nt, pr = sens["no_trajectory"]["cells"]["immune_hum_mou|0"]["p1"], prim["cells"]["immune_hum_mou|0"]["p1"]
    close(nt["bio"]["ref"] - pr["bio"]["ref"], 0.01, 1e-9)           # same crossing, C 0.01 higher
    close(nt["d"], pr["d"], 1e-9)
    assert sens["per_run"]["cells"]["pancreas|1"]["p1"]["mode"] == "per_run"
    assert sens["prior_0.2"]["pooled"]["0"]["p1"]["tau_scale"] == 0.2
    assert sens["pairs"]["pooled"]["0"]["p1"]["units"] == ["pancreas", "lung", "immune", "atac"]
    assert sens["corner_zero"]["settings"]["corner"] == "zero"


def _rewrite(src, dst, keep):
    df = pd.read_csv(src, sep="\t" if src.endswith(".tsv") else ",", dtype=str, keep_default_na=False, comment="#")
    df[keep(df)].to_csv(dst, sep="\t" if dst.endswith(".tsv") else ",", index=False)
    return dst


@pytest.mark.parametrize("change, match", [
    (lambda e, tmp: dict(scores=_rewrite(e["args"]["scores"], os.path.join(tmp, "s.csv"),
                                         lambda S: S.tag != sorted(e["M"].tag[e["M"].experiment == "X13"])[0])),
     "neither scored nor failed"),                                              # an X13 run is missing
    (lambda e, tmp: dict(manifest=_rewrite(e["args"]["manifest"], os.path.join(tmp, "m.tsv"),
                                           lambda M: ~((M.task == "pancreas") & (M.cond == "0") & (M.arm == "reference")
                                                       & (M.lam == "0.3")))),
     "registered"),                                                             # a registered lambda is missing
    (lambda e, tmp: dict(manifest=_rewrite(e["args"]["manifest"], os.path.join(tmp, "m.tsv"),
                                           lambda M: M.task != "sim2")), "tasks"),
    (lambda e, tmp: dict(manifest=_rewrite(e["args"]["manifest"], os.path.join(tmp, "m.tsv"),
                                           lambda M: ~((M.task == "lung") & (M.arm == "scvi_adv")))), "scvi_adv"),
    (lambda e, tmp: dict(r4=_shift_b0(e["args"]["r4"], os.path.join(tmp, "r4.json"))), "b0"),
    (lambda e, tmp: dict(out=e["args"]["out"]), "exists"),                      # never overwrite a record
])
def test_refusals(e2e, tmp_path, change, match):
    args = dict(e2e["args"], out=str(tmp_path / "a.json"))
    args.update(change(e2e, str(tmp_path)))
    with pytest.raises(P.PreregError, match=match):
        A.main(argv(args))


def _shift_b0(src, dst):
    r4 = json.load(open(src))
    r4["targets"]["lung|0"]["b0"] += 0.01        # an R4 record that was not made from these scores
    json.dump(r4, open(dst, "w"))
    return dst
