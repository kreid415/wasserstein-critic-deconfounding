"""Known-answer tests for scripts/analysis_x1.py, the X1 analysis of the ranking design (docs/ANALYSIS_SPEC.md sections 2-12, 15).

Synthetic inputs only: the analysis must not run on any X1 or X13 score before the ranking version is frozen (SI-78).
Every expected value is derived by hand (the derivation is stated next to the case) or by an independent method
(scipy, enumeration), never by the code under test. The contrast version of this file is at the tag prereg-tier12-v3.
"""
import hashlib
import itertools
import json
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

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
# ranks, standard errors and rank ranges (spec sections 3 and 4)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("readings, want", [
    # the specification's example (section 3.1): 0.70, 0.65, 0.65, 0.60 and two that do not reach b*
    (dict(a=0.70, b=0.65, c=0.65, d=0.60, e=None, f=None), dict(a=1, b=2.5, c=2.5, d=4, e=5.5, f=5.5)),
    (dict.fromkeys("abcdef"), dict.fromkeys("abcdef", 3.5)),                          # all six fail: places 1-6, mean 3.5
    (dict(a=0.5, b=0.4, c=0.3, d=0.2, e=0.1, f=None), dict(a=1, b=2, c=3, d=4, e=5, f=6)),   # one failure takes place 6
    # tolerance 1e-9 (PREREG section 1): c is 1.5e-9 above b (not tied), b is 5e-10 above a (tied); a tie is measured from
    # the first of its group, so the three do not chain into one
    (dict(a=0.5, b=0.5 + 5e-10, c=0.5 + 2e-9), dict(a=2.5, b=2.5, c=1)),
    (dict(a=-0.1, b=0.0, c=None), dict(a=2, b=1, c=3)),                               # negative and zero readings are readings
])
def test_rank_pair(readings, want):  # test-reason: hand-worked ranks, bottom places and ties of section 3.1
    got = A.rank_pair(readings)
    assert got == want and list(got) == list(readings)
    assert sum(got.values()) == len(readings) * (len(readings) + 1) / 2             # the ranks of a pair sum to n(n+1)/2


@pytest.mark.parametrize("xs, s, m", [
    # mean 0.74, squares of deviations 0.0016 + 0.0004 + 0 + 0.0004 + 0.0016 = 0.004; s^2 = (4/5) 0.004 = 0.0032
    ([0.70, 0.72, 0.74, 0.76, 0.78], math.sqrt(0.0032), 5),
    # four readings 0.70 .. 0.76: mean 0.73, squares 0.0009 + 0.0001 + 0.0001 + 0.0009 = 0.002; s^2 = (3/4) 0.002 = 0.0015
    ([None, 0.70, 0.72, 0.74, 0.76], math.sqrt(0.0015), 4),
    ([None, None, 0.1, 0.2, 0.3], None, 3),               # fewer than 4 readings: not available
    ([0.5] * 5, 0.0, 5),
    ([None] * 5, None, 0),
])
def test_loso_se(xs, s, m):  # test-reason: hand-worked jackknife standard error and the m >= 4 rule of section 4 (i)
    got = A.loso_se(xs)
    close(got[0], s)
    assert got[1] == m


def test_rank_one_pair():  # test-reason: hand-worked readings, leave-one-seed-out repeats, bottom ties and rank ranges
    """Four penalties, b* = 0.50, plain scVI's B by seed 0.38 .. 0.42 (mean 0.40) and C 0.80.
    a: one point (0.60, 0.70): reading 0.80 - 0.10 (0.50 - b0) / (0.60 - b0) = 0.75; without seed i b0 is (2.0 - b_i) / 4.
    b: an exact hit (0.50, 0.74): reading 0.74 in every repeat.
    c: never reaches b* (B 0.45, 0.48): no reading.
    d: B [0.56, 0.56, 0.47, 0.47, 0.47] (mean 0.506), C 0.70: reading 0.80 - 0.10 (0.1 / 0.106) = 0.70566. Without seed 0 or 1 its
       mean B is 0.4925, below b*: 3 readings, no standard error. Without seed 2 (b0 0.40) it reads 0.80 - 0.10 (0.1 / 0.115),
       below b."""
    arms = dict(a=pts({1: (0.60, 0.70)}), b=pts({1: (0.50, 0.74)}), c=pts({1: (0.45, 0.79), 3: (0.48, 0.78)}),
                d=pts({1: ([0.56, 0.56, 0.47, 0.47, 0.47], 0.70)}))
    got = A.rank_one_pair(zero(b=[0.38, 0.39, 0.40, 0.41, 0.42]), arms, 0.50)
    b0 = [0.405, 0.4025, 0.40, 0.3975, 0.395]
    a_loso = [0.80 - 0.10 * (0.50 - b) / (0.60 - b) for b in b0]
    assert [(r["status"], r["crossings"], r["m"]) for r in got.values()] == [
        ("bracketed", 1, 5), ("bracketed", 1, 5), ("unreached_low", 0, 0), ("bracketed", 1, 3)]
    close(got["a"]["bio"], 0.75), close(got["b"]["bio"], 0.74), close(got["d"]["bio"], 0.80 - 0.10 * 0.1 / 0.106)
    assert got["c"]["bio"] is None and got["c"]["s"] is None and got["c"]["loso"] == [None] * 5
    assert got["a"]["loso"] == pytest.approx(a_loso, abs=1e-12)
    close(got["a"]["s"], A.loso_se(a_loso)[0])
    assert got["b"]["loso"] == pytest.approx([0.74] * 5) and got["b"]["s"] == pytest.approx(0.0, abs=1e-12)
    assert got["d"]["loso"][:2] == [None, None] and got["d"]["s"] is None
    close(got["d"]["loso"][2], 0.80 - 0.10 * 0.1 / 0.115)
    # ranks: a 0.75, b 0.74, d 0.7057, c bottom. Without seed 0 or 1, d and c share places 3 and 4 (3.5 each)
    assert [got[k]["rank"] for k in "abcd"] == [1, 2, 4, 3]
    assert [(got[k]["rank_lo"], got[k]["rank_hi"]) for k in "abcd"] == [(1, 1), (2, 2), (3.5, 4), (3, 3.5)]


# ---------------------------------------------------------------------------------------------
# frontier area (spec section 5)
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


@pytest.mark.parametrize("points, corner, want", [
    # plain scVI (0.40, mean C 0.80). Corner (b0, c0 - 0.10) = (0.40, 0.70): points (0.6, 0.78), (0.7, 0.72) give
    # 0.3 x 0.02 + 0.2 x 0.06 = 0.018, over the scale 0.2: 0.09; the single point (0.65, 0.75) gives 0.25 x 0.05 / 0.2 = 0.0625
    ({1: (0.6, 0.78), 2: (0.7, 0.72)}, "collapse", 0.09),
    ({1: (0.65, 0.75)}, "collapse", 0.0625),
    # corner (b0, 0), the SI-72 sensitivity: 0.3 x 0.72 + 0.2 x 0.06 = 0.228 / 0.2 = 1.14; 0.25 x 0.75 / 0.2 = 0.9375
    ({1: (0.6, 0.78), 2: (0.7, 0.72)}, "zero", 1.14),
    ({1: (0.65, 0.75)}, "zero", 0.9375),
    # a point whose five runs failed is skipped (SI-71)
    ({1: (0.6, 0.78), 2: (0.9, 0.1)}, "collapse", 0.2 * 0.08 / 0.2),
])
def test_read_area(points, corner, want):  # test-reason: known answers: the moving corner (mean of the plain-scVI C), both corners
    fail = [(2, s) for s in SEEDS] if 2 in points and points[2] == (0.9, 0.1) else []
    close(A.read_area(zero(c=[0.76, 0.78, 0.80, 0.82, 0.84]), pts(points, fail=fail), 0.2, corner=corner), want)


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


# ---------------------------------------------------------------------------------------------
# summaries over tasks, single-strength tables (section 10), best-of-k and frontier positions (11), window width (8)
# ---------------------------------------------------------------------------------------------
def test_rank_means():  # test-reason: hand-worked mean ranks (section 3.2) and per-seed mean ranks (section 12)
    got = A.mean_ranks({"t1": dict(a=1, b=2, c=3), "t2": dict(a=2, b=1, c=3)}, ["a", "b", "c"])
    assert got == dict(a=(1.5, 2), b=(1.5, 2), c=(3.0, 2))
    assert A.mean_ranks({}, ["a", "b"]) == dict(a=(None, 0), b=(None, 0))              # no ranked task: no mean rank
    # seed 1: a 1, b 2, c bottom 3; seed 2: b 1, a 2, c 3; seed 3: nothing read, places 1-3 shared: 2 each
    got = A.per_seed_mean_ranks([dict(a=0.7, b=0.6, c=None), dict(a=0.6, b=0.7, c=0.5), dict(a=None, b=None, c=None)])
    assert got == pytest.approx(dict(a=5 / 3, b=5 / 3, c=8 / 3), abs=1e-12)           # (1+2+2)/3, (2+1+2)/3, (3+3+2)/3


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
def test_best_of_k(S):  # test-reason: E_k against enumeration of every k-subset, and the real-task mean at k above a task's N
    want = [np.mean([max(c) for c in itertools.combinations(S, k)]) for k in range(1, len(S) + 1)]
    assert A.best_of_k(S) == pytest.approx(want, abs=1e-12)
    # task a has N configurations, task b one (E_1 = 0.5): k = 1 averages both tasks, later k only task a (section 11, 16.8)
    got = A.best_of_k_real_mean({"a": want, "b": [0.5]})
    assert got == pytest.approx([((want[0] + 0.5) / 2, 2)] + [(e, 1) for e in want[1:]], abs=1e-12)
    assert A.best_of_k_real_mean({}) == []


def test_dominance():  # test-reason: known answer for the frontier-position counts
    front = [(0.6, 0.8), (0.5, 0.7), (0.4, 0.6), (0.6, 0.6), (0.5, 0.75)]
    assert A.dominance((0.5, 0.7), front) == {"dominated_by": 2, "dominates": 1}


def test_window():
    w = A.window([(1.0, [0.50, 0.52, 0.48, 0.51, 0.49]), (3.0, [0.60, 0.62, 0.58, 0.61, 0.59]),
                  (10.0, [0.59, 0.60, 0.58, 0.595, 0.585]), (30.0, [0.3] * 5), (100.0, [])])
    lower = 0.60 - stats.t.ppf(0.975, 4) * math.sqrt(0.00025 / 5)           # best mean - t x SD / sqrt(n) = 0.5804
    assert w == dict(status="ok", best=3.0, lower=pytest.approx(lower, abs=1e-12), lams=[3.0, 10.0],
                     width=pytest.approx(math.log10(10 / 3), abs=1e-12), n_points=2, gap=False, n_all_failed_between=0)
    tie = A.window([(1.0, [0.5, 0.5]), (3.0, [0.5, 0.5])])                   # tie -> smaller lambda; SD 0
    assert tie["best"] == 1.0 and tie["lams"] == [1.0, 3.0] and tie["width"] == pytest.approx(math.log10(3))
    # lambda 3 lies below the bound between two points above it: the width spans a gap (review AN-14)
    good, other = [0.60, 0.62, 0.58, 0.61, 0.59], [0.59, 0.60, 0.58, 0.595, 0.585]
    gap = A.window([(1.0, good), (3.0, [0.3] * 5), (10.0, other)])
    assert gap["lams"] == [1.0, 10.0] and gap["gap"] is True and gap["n_points"] == 2 and gap["n_all_failed_between"] == 0
    # a configuration with no successful run between the ends is counted and leaves the gap rule alone (AN-R13)
    failed = A.window([(1.0, good), (3.0, []), (10.0, other)])
    assert failed["lams"] == [1.0, 10.0] and failed["gap"] is False and failed["n_all_failed_between"] == 1
    # the low point may lie deep inside the window, not next to its smallest lambda (AN-R09)
    deep = A.window([(1.0, good), (3.0, [0.595] * 5), (10.0, [0.3] * 5), (30.0, []), (100.0, other)])
    assert deep["lams"] == [1.0, 3.0, 100.0] and deep["gap"] is True and deep["n_all_failed_between"] == 1
    none = dict(lower=None, lams=None, width=None, n_points=None, gap=None, n_all_failed_between=None)
    assert A.window([(1.0, [0.5]), (3.0, [0.4, 0.45])]) == dict(none, status="no_interval", best=1.0)
    assert A.window([(1.0, []), (3.0, [])]) == dict(none, status="no_points", best=None)


# ---------------------------------------------------------------------------------------------
# label transfer (spec section 9)
# ---------------------------------------------------------------------------------------------
def test_label_transfer_f1():  # test-reason: hand-worked macro F1 over the types present in the held-out batch, dropped types
    """One-dimensional embedding, k = 1 (nearest neighbour). Cell types A, B in all three batches, D in batches 1 and 2, X only in
    batch 1 (dropped). b1: A 0.0, 0.1, B 10.0, D 20.0, X 30.0; b2: A 0.2, B 10.1, 10.2, D 20.1; b3: A 0.3, B 10.3, B 4.0.
    Holding out b1 or b2 every cell takes its own type (F1 1). Holding out b3, the B cell at 4.0 is nearest to A 0.2: predictions
    A, B, A. A: precision 1/2, recall 1, F1 2/3; B: precision 1, recall 1/2, F1 2/3; D is not in b3 and is not counted
    (macro F1 2/3, not (2/3 + 2/3 + 0) / 3). Mean over the held-out batches (1 + 1 + 2/3) / 3 = 8/9."""
    x = [0.0, 0.1, 10.0, 20.0, 30.0, 0.2, 10.1, 10.2, 20.1, 0.3, 10.3, 4.0]
    types = list("AABDXABBDABB")
    batches = ["b1"] * 5 + ["b2"] * 4 + ["b3"] * 3
    got = A.label_transfer_f1(np.array([[v, 0.0] for v in x]), types, batches, k=1)
    assert got["per_batch"] == pytest.approx(dict(b1=1.0, b2=1.0, b3=2 / 3), abs=1e-12)
    assert got["mean"] == pytest.approx(8 / 9, abs=1e-12) and got["dropped"] == ["X"] and not got["not_evaluable"]
    # every cell type in one batch only: nothing left to evaluate
    alone = A.label_transfer_f1(np.zeros((4, 2)), list("ABCD"), ["b1", "b1", "b2", "b2"])
    assert alone == dict(per_batch={}, mean=None, dropped=list("ABCD"), not_evaluable=True)


def test_guards():  # test-reason: refusals at the function boundaries
    with pytest.raises(P.PreregError, match="scale"):
        A.read_area(zero(), pts({1: (0.6, 0.7)}), 0.0)
    with pytest.raises(P.PreregError, match="corner"):
        A.read_area(zero(), pts({1: (0.6, 0.7)}), 0.2, corner="sideways")
    with pytest.raises(P.PreregError, match="rank_pair"):
        A.rank_pair({})
    with pytest.raises(P.PreregError, match="rank_pair"):
        A.rank_pair(dict(a=0.5, b=float("nan")))
    with pytest.raises(P.PreregError, match="every ranked pair"):
        A.mean_ranks({"t": dict(a=1)}, ["a", "b"])
    with pytest.raises(P.PreregError, match="same penalties"):
        A.per_seed_mean_ranks([dict(a=0.5), dict(b=0.5)])
    with pytest.raises(P.PreregError, match="same cells"):
        A.label_transfer_f1(np.zeros((3, 2)), list("AB"), list("xyz"))
    with pytest.raises(P.PreregError, match="no zero runs"):
        A.read_arms(zero(), {"a": pts({1: (0.6, 0.7)})}, 0.5, [])
    with pytest.raises(P.PreregError, match="grid"):
        A.choose_lambda({"a": {1.0: 0.5}, "b": {3.0: 0.5}}, {"a": {1.0: 0}, "b": {3.0: 0}}, ["a", "b"])
    row = dict(task="pancreas", status="ok", B=0.5, C=0.7, **dict.fromkeys(P.BATCH_METRICS, 0.5),
               **dict.fromkeys(P.bio_metrics_for("pancreas"), 0.7))
    row["kBET"] = 0.6                     # B is no longer the mean of the 5 batch metrics
    with pytest.raises(P.PreregError, match="local"):
        A.add_scores(pd.DataFrame([row]))


# ---------------------------------------------------------------------------------------------
# end to end on the committed X1 + X13 manifest with synthetic scores and latents
# ---------------------------------------------------------------------------------------------
CENTRE = {("js", 0): 0.0, ("js", 1): 0.5, ("w1", 0): 1.0, ("w1", 1): 1.0, ("mmd", 0): 2.0, ("mmd", 1): 2.0,
          ("sinkhorn", 0): 1.5, ("sinkhorn", 1): 0.5}     # log10 lambda of half batch removal, inside every X1 grid
# Biology a penalty keeps (+) or loses (-) at matched batch removal. Every adversarial arm gains 0.30 in B as sig(x / 0.3) and C
# moves by BONUS x sig(x / 0.3), so C is linear in B along a curve and at b* (u* = (b* - b0) / 0.30) it reads 0.75 + BONUS x u*:
# the ranks are reference 1, pooled 2, barycenter 3, mmd and discriminator 4 and 5 (equal bonus), sinkhorn 6.
BONUS = {"reference": 0.20, "pooled": 0.12, "barycenter": 0.06, "sinkhorn": -0.08}
SPREAD = {"reference": 0.08}       # local-minus-global spread it adds to the batch metrics (B unchanged): local - global = 0.2 u*
TRAJ_DROP = 0.08                   # trajectory below the other bio metrics: C = c - 0.01 on immune, immune_hum_mou
FAIL = {("lung", "0", "discriminator", "10", "0"): "diverged",
        ("pancreas", "1", "mmd", "3000", "2"): "nonfinite_latent",
        ("lung", "0", "harmony", "8", "0"): "nonfinite_latent",           # an X13 configuration without a run
        **{("sim2", "0", "sinkhorn", "3000", str(s)): "diverged" for s in SEEDS},
        # barycenter, unconditioned: failed runs leave lambda 10, 100 failure-free in pancreas and 0.3, 1, 3, 30 in
        # lung, so no lambda is eligible for a held-out dataset that trains on both (all outside their families);
        # each keeps a high failure-free point, so b_common (R4) barely moves
        **{("pancreas", "0", "barycenter", lam, "3"): "diverged" for lam in ("0.3", "1", "3", "30")},
        **{("lung", "0", "barycenter", lam, "3"): "diverged" for lam in ("10", "100")},
        ("atac_small", "1", "reference", "10", "2"): "diverged"}    # a failed run at a point that adds area
ZERO_AREA = ("lung", "1")          # every adversarial point 0.35 below in biology: no penalty adds area
# R4-x1 extension points (used for matching only, never as points; SI-55). The first is of the arm that sets b_common in
# atac_small, decoder 0, with a batch score above its six grid points: b_common and so b* rise, so the analysis must see the
# row (AN-R07). The second is of mmd in sim1, decoder 0, whose six grid points gain only 0.05 in B while its extension point is
# far above them: b* is set by the other five penalties, mmd never reaches it and takes the last place.
_GRID = json.load(open(R1_RECORD))["families"]
EXT = ("atac_small", "0", "barycenter", P.fmt_lam(P.lam_step(max(_GRID["w1|0"]["x1_grid"]), +1)))
EXT_MMD = ("sim1", "0", "mmd", P.fmt_lam(P.lam_step(max(_GRID["mmd|0"]["x1_grid"]), +1)))
EXT_BUMP = {EXT: 0.05, EXT_MMD: 0.40}
NOT_EVALUABLE = "atac_small"       # its cell types each occur in one batch only: label transfer has nothing to evaluate
SPECIAL = {("immune", 1), ("lung", 1), ("immune_hum_mou", 1), ("sim1", 0)}     # pairs whose ranks the planted effects change


def _gain(r):
    key = (r["task"], r["cond"])
    if key == ("immune", "1"):
        return 0.015                                       # b_common - b0 < 0.02: bstar_low_signal
    if key == ("sim1", "0") and r["arm"] == "mmd":
        return 0.05
    if key == ("immune_hum_mou", "1") and r["arm"] in ("reference", "discriminator"):
        return 0.30 if r["seed"] == "1" else 0.004         # one responding seed: without it neither reaches b* (4 of 5 readings)
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
        B += EXT_BUMP.get((r["task"], r["cond"], r["arm"], r["lam"]), 0.0)
        C += -0.25 * synth.sig((x - 1) / 0.3) + BONUS.get(r["arm"], 0.0) * synth.sig(x / 0.3) + synth._noise("C" + r["tag"], 0.003)
        C -= 0.35 if (r["task"], r["cond"]) == ZERO_AREA else 0.0
        a = SPREAD.get(r["arm"], 0.0) * synth.sig(x / 0.3)
    elif r["experiment"] == "X13":
        B += 0.05 + 0.01 * math.log1p(float(r["lam"]))      # better with the setting: the best setting is the largest
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


def cells_of(task):
    """(batch, cell type) of each synthetic cell: types A, B, C in three batches (10 cells each) and D in batch 0 only; for the
    NOT_EVALUABLE task every type occurs in one batch only."""
    if task == NOT_EVALUABLE:
        return [(f"b{b}", f"T{b}{t}") for b in range(3) for t in "ABC" for _ in range(10)]
    return [(f"b{b}", t) for b in range(3) for t in "ABC" for _ in range(10)] + [("b0", "D")] * 5


def write_latents(M, r4, d):
    """Latents <tag>.npz in fit_paper_config.py's format (z, obs_names, batch, celltype) of the X1 rows label transfer reads: plain
    scVI, scvi_adv and each penalty at its R4 matched strength. The reference critic's embedding follows the batches, every other
    embedding the cell types (two well-separated clusters per cell type or batch). Returns {tag: sha256}."""
    os.makedirs(d)
    shas, centre = {}, dict(A=(0, 0), B=(10, 0), C=(0, 10), D=(10, 10), b0=(0, 0), b1=(10, 0), b2=(0, 10))
    for r in M[M.experiment == "X1"].to_dict("records"):
        if r["arm"] in P.ADV_ARMS and float(r["lam"]) != r4["cells"][f"{r['task']}|{r['cond']}|{r['arm']}"]["matched"]:
            continue
        if world(r)[0] != "ok":
            continue
        batches, types = zip(*cells_of(r["task"]))
        rng = np.random.default_rng(int(hashlib.sha1(r["tag"].encode()).hexdigest()[:8], 16))
        z = np.array([centre.get(b if r["arm"] == "reference" else t, (20, 20)) for b, t in zip(batches, types)]) \
            + 0.3 * rng.normal(size=(len(types), 2))
        path = os.path.join(d, f"{r['tag']}.npz")
        np.savez_compressed(path, z=z.astype(np.float32), obs_names=np.array([f"c{i}" for i in range(len(types))]),
                            batch=np.array(batches), celltype=np.array(types))
        shas[r["tag"]] = P.sha256(path)
    return shas


def argv(a):
    return ["--manifest", a["manifest"], "--scores", a["scores"], "--failures", a["failures"],
            "--r1-record", a["r1"], "--r4-record", a["r4"], "--latents", a["latents"], "--out-json", a["out"]]


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("analysis_x1"))
    M, _ = P.read_manifest(X1_MANIFEST)
    M = M[M.experiment.isin(["X1", "X13"])].reset_index(drop=True)
    r1 = P.read_record(R1_RECORD, "R1")
    M = pd.concat([M, P.extension_rows(M, [dict(experiment="X1", task=k[0], cond=int(k[1]), arm=k[2], lam=float(k[3]))
                                           for k in EXT_BUMP])], ignore_index=True)
    man = synth.write_tsv(M, os.path.join(tmp, "manifest.tsv"))
    tables = score_tables(M)
    sp, fp = synth.write_scores(*tables, os.path.join(tmp, "scores"), "x1")
    S, F = P.read_scores([sp])[0], P.read_failures(fp)
    r4 = P.r4_stage(P.outcomes(M, S, F, ["X1"]), "x1", r1, need=set())      # what freeze_matched_lambda.py --stage x1 records
    latents = os.path.join(tmp, "latents")
    tables = (tables[0].assign(latent_sha256=tables[0].tag.map(write_latents(M, r4, latents))), tables[1])
    sp, fp = synth.write_scores(*tables, os.path.join(tmp, "scores"), "x1")
    S, F = P.read_scores([sp])[0], P.read_failures(fp)
    r4p = os.path.join(tmp, "r4x1.json")
    P.write_record(r4p, r4, dict(scores=[sp]))
    args = dict(manifest=man, scores=sp, failures=fp, r1=R1_RECORD, r4=r4p, latents=latents, out=os.path.join(tmp, "analysis.json"))
    assert A.main(argv(args)) == 0
    # the same inputs with the rows of every file in another order (review AN-11, AN-R14)
    shuf = dict(args, manifest=synth.write_tsv(M.sample(frac=1, random_state=1), os.path.join(tmp, "manifest_shuffled.tsv")),
                out=os.path.join(tmp, "analysis_shuffled.json"))
    shuf["scores"], shuf["failures"] = synth.write_scores(*(t.sample(frac=1, random_state=1) for t in tables),
                                                          os.path.join(tmp, "scores_shuffled"), "x1")
    assert A.main(argv(shuf)) == 0
    return dict(args=args, rec=json.load(open(args["out"])), M=M, out=A.add_scores(P.outcomes(M, S, F, ["X1", "X13"])),
                r1=r1, r4=json.load(open(r4p)), shuffled=json.load(open(shuf["out"])))


def points_of(out, r1, task, cond, value="C"):
    z = A.zero_runs(out, task, cond, value)
    return z, {a: A.arm_points(out, task, cond, a, r1["families"][f"{P.FAMILY_OF[a]}|{cond}"]["x1_grid"], sorted(z), value)
               for a in P.ADV_ARMS}


def test_end_to_end_ranking(e2e):  # test-reason: the record's wiring of readings, ranks and mean ranks, and the planted ranks
    rec, out, r1, r4 = e2e["rec"], e2e["out"], e2e["r1"], e2e["r4"]
    assert rec["rule"] == "analysis_x1" and rec["status"] == "done"
    assert set(rec["inputs"]) == {"manifest", "scores", "failures", "r1_record", "r4_record", "latents"}
    assert all(len(x["sha256"]) == 64 for v in rec["inputs"].values() for x in v)
    prim = rec["primary"]
    assert prim["settings"] == dict(value="C", mode="mean", corner="collapse")
    assert len(prim["cells"]) == 16 and set(prim) == {"settings", "cells", "ranking", "best_of_k_real_mean", "transfer",
                                                      "label_transfer"}
    # the same inputs with the rows of every file in another order give the same numbers (review AN-11, AN-R14)
    assert e2e["shuffled"]["primary"] == prim and e2e["shuffled"]["sensitivity"] == rec["sensitivity"]
    # the extension rows move b*, and the record is accepted only because the analysis sees them (planted error: a
    # recompute from registered points only): atac_small|0 by about 1.7e-4 against a tolerance of 1e-9; sim1|0 by more than
    # 0.05, because without mmd's extension point mmd's own six points would set b_common
    for key, ext, least in (("atac_small|0", EXT, 1e-5), ("sim1|0", EXT_MMD, 0.05)):
        reg = out[~((out.task == ext[0]) & (out.cond == ext[1]) & (out.arm == ext[2]) & (out.lam == ext[3]))]
        plain = P.r4_stage(reg, "x1", r1, need=set())["targets"][key]
        assert r4["targets"][key]["bstar"] - plain["bstar"] > least

    # wiring: the record's pair values are the functions' values on the same runs
    for t, c in (("pancreas", 1), ("immune_hum_mou", 0)):
        z, points = points_of(out, r1, t, c)
        tg = r4["targets"][f"{t}|{c}"]
        cell = prim["cells"][f"{t}|{c}"]
        assert [cell[k] for k in ("b0", "b_common", "bstar", "low_signal")] == [tg[k] for k in ("b0", "b_common", "bstar", "bstar_low_signal")]
        for arm, want in json.loads(json.dumps(A.rank_one_pair(z, points, tg["bstar"]))).items():
            assert {k: cell["penalties"][arm][k] for k in want} == want
        scale = tg["b_common"] - tg["b0"]
        assert cell["area"] == pytest.approx({a: A.read_area(z, points[a], scale) for a in P.ADV_ARMS}, abs=1e-12)
    # local minus global: 3 local + 2 global = 5 batch, so at the crossing it is 2.5 (local - b*), read from the same curve
    z, points = points_of(out, r1, "pancreas", 1)
    zl, local = points_of(out, r1, "pancreas", 1, "L")
    bstar = r4["targets"]["pancreas|1"]["bstar"]
    for arm, r in A.read_arms(zl, local, bstar, sorted(zl)).items():
        close(prim["cells"]["pancreas|1"]["penalties"][arm]["local_minus_global"], 2.5 * (r["bio"] - bstar), 1e-12)

    # planted ranks. Pairs without a planted twist: reference 1, pooled 2, barycenter 3, sinkhorn 6, mmd and discriminator 4 and 5;
    # the five leave-one-seed-out repeats rank the four penalties of a distinct bonus the same; local minus global is the
    # reference's only (0.2 u*, u* near 0.45: above 0.03) and zero for the others
    for t, c in itertools.product(sorted(synth.BPM.TASKS), (0, 1)):
        if (t, c) in SPECIAL:
            continue
        pen = prim["cells"][f"{t}|{c}"]["penalties"]
        r = {a: pen[a]["rank"] for a in P.ADV_ARMS}
        assert (r["reference"], r["pooled"], r["barycenter"], r["sinkhorn"]) == (1, 2, 3, 6) and sorted((r["mmd"], r["discriminator"])) == [4, 5]
        assert all(pen[a]["rank_lo"] == pen[a]["rank_hi"] == r[a] for a in ("reference", "pooled", "barycenter", "sinkhorn"))
        assert pen["reference"]["local_minus_global"] > 0.03
        assert all(abs(pen[a]["local_minus_global"]) < 1e-9 for a in P.ADV_ARMS if a != "reference")
    # a penalty whose six points stay below b* (mmd in sim1|0; its high point is an extension row) is ranked last and the pair
    # stays ranked; it has no reading and no standard error, and its rank range is 6 to 6
    pen = prim["cells"]["sim1|0"]["penalties"]
    assert {a: pen[a]["rank"] for a in P.ADV_ARMS} == dict(reference=1, pooled=2, barycenter=3, discriminator=4, sinkhorn=5, mmd=6)
    assert (pen["mmd"]["bio"], pen["mmd"]["status"], pen["mmd"]["m"], pen["mmd"]["s"]) == (None, "unreached_low", 0, None)
    assert pen["mmd"]["loso"] == [None] * 5 and (pen["mmd"]["rank_lo"], pen["mmd"]["rank_hi"]) == (6, 6)
    # immune_hum_mou|1: reference and discriminator respond in seed 1 only; without seed 1 neither reaches b*: 4 of 5 readings
    # (a standard error from 4), and in that repeat they share places 5 and 6 (5.5), so the reference ranges from 1 to 5.5
    pen = prim["cells"]["immune_hum_mou|1"]["penalties"]
    for a in ("reference", "discriminator"):
        assert [x is None for x in pen[a]["loso"]] == [False, True, False, False, False] and pen[a]["m"] == 4 and pen[a]["s"] > 0
    assert pen["reference"]["rank"] == 1 and (pen["reference"]["rank_lo"], pen["reference"]["rank_hi"]) == (1, 5.5)
    # lung|1: no point is above the collapse line, so every area is 0 (not none); immune|1 is low-signal: not ranked, no area
    assert prim["cells"]["lung|1"]["area"] == dict.fromkeys(P.ADV_ARMS, 0.0)
    low = prim["cells"]["immune|1"]
    assert low["low_signal"] and low["area"] is None
    assert all(set(r) == {"bio", "status", "crossings", "local_minus_global"} for r in low["penalties"].values())

    # mean ranks: decoder 0 ranks all six real tasks, decoder 1 five (immune is low-signal); the simulations are listed by task
    rk = prim["ranking"]
    assert rk["0"]["low_signal"] == [] and rk["1"]["low_signal"] == ["immune"]
    m0 = {a: (v["mean"], v["n_tasks"]) for a, v in rk["0"]["mean_rank"].items()}
    assert {a: m0[a] for a in ("reference", "pooled", "barycenter", "sinkhorn")} == dict(
        reference=(1.0, 6), pooled=(2.0, 6), barycenter=(3.0, 6), sinkhorn=(6.0, 6))
    assert m0["mmd"][0] + m0["discriminator"][0] == 9 and m0["mmd"][1] == m0["discriminator"][1] == 6
    assert all(v["n_tasks"] == 5 for v in rk["1"]["mean_rank"].values())
    assert sum(v["mean"] for v in rk["1"]["mean_rank"].values()) == pytest.approx(21, abs=1e-9)
    for ck, c in (("0", 0), ("1", 1)):
        ranked = {t: {a: prim["cells"][f"{t}|{c}"]["penalties"][a]["rank"] for a in P.ADV_ARMS} for t in P.REAL_TASKS
                  if not prim["cells"][f"{t}|{c}"]["low_signal"]}
        assert {a: (v["mean"], v["n_tasks"]) for a, v in rk[ck]["mean_rank"].items()} == pytest.approx(
            {a: (m, n) for a, (m, n) in A.mean_ranks(ranked, list(P.ADV_ARMS)).items()}, abs=1e-12)
        assert set(rk[ck]["sim_ranks"]) == {"sim1", "sim2"}
    assert rk["0"]["sim_ranks"]["sim1"] == dict(reference=1, pooled=2, barycenter=3, discriminator=4, sinkhorn=5, mmd=6)


def test_end_to_end_secondary(e2e):  # test-reason: the record's collapse, width, tables, best-of-k and the failure bookkeeping
    rec, out, r1, r4 = e2e["rec"], e2e["out"], e2e["r1"], e2e["r4"]
    prim, cells = rec["primary"], rec["primary"]["cells"]
    # collapse counts with the planted failures, per penalty, task and decoder (no pooled rate)
    c = cells["lung|0"]["collapse"]
    assert c["discriminator"]["failed"] == {"diverged": 1, "nonfinite_latent": 0} and c["discriminator"]["n"] == 30
    assert c["discriminator"]["collapsed"] == 5 and c["reference"]["collapsed"] == 0
    assert cells["pancreas|1"]["collapse"]["mmd"]["failed"] == {"diverged": 0, "nonfinite_latent": 1}

    # single-strength tables: the setting a tool or penalty takes is the best on the other families' tasks, and the best of the
    # tool settings is the largest (B grows with log(1 + setting)); harmony's setting 8 has a failed run in lung, so it is not
    # eligible for any task that trains on lung and lung's own best is 4
    tr = prim["transfer"]["0"]
    assert {t: v["lam"] for t, v in tr["harmony"].items()} == dict.fromkeys(sorted(synth.BPM.TASKS), 4.0) | dict(lung=8.0)
    assert {t: v["best_lam"] for t, v in tr["harmony"].items()} == dict.fromkeys(sorted(synth.BPM.TASKS), 8.0) | dict(lung=4.0)
    assert {v["lam"] for v in tr["scanorama"].values()} == {80.0} and {v["lam"] for v in tr["sysvi"].values()} == {50.0}
    assert tr["scanorama"]["immune"]["train"] == ["atac_large", "atac_small", "lung", "pancreas"]
    tr = prim["transfer"]["1"]["reference"]
    assert tr["immune"]["train"] == ["atac_large", "atac_small", "lung", "pancreas"] and tr["sim1"]["train"] == sorted(P.REAL_TASKS)
    grid = {a: r1["families"][f"{P.FAMILY_OF[a]}|1"]["x1_grid"] for a in ("reference", "discriminator")}
    assert tr["immune"]["lam"] in grid["reference"] and tr["immune"]["at"]["n_ok"] == 5
    bary = prim["transfer"]["0"]["barycenter"]
    assert bary["immune"]["lam"] is None and bary["immune"]["at"] is None and bary["sim1"]["lam"] is None
    assert bary["pancreas"]["lam"] in (0.3, 1.0, 3.0, 30.0) and bary["lung"]["lam"] in (10.0, 100.0)
    assert bary["pancreas"]["best_lam"] in (10.0, 100.0) and bary["pancreas"]["best"]["n_fail"] == 0
    # S = 0.4 B + 0.6 C (SI-77), and the difference from plain scVI's mean S in the same task and decoder
    at = prim["transfer"]["0"]["scanorama"]["pancreas"]["at"]
    close(at["S"][0], 0.4 * at["B"][0] + 0.6 * at["C"][0])
    z = A.zero_runs(out, "pancreas", 0)
    close(at["S_minus_plain_scvi"], at["S"][0] - np.mean([0.4 * b + 0.6 * c for b, c in z.values()]))
    # as they are: plain scVI, scvi_adv, scANVI, PCA and the tools' sensitivity points
    single = cells["sim2|0"]["single"]
    assert {k.split("|")[0] for k in single} == {"none", "scvi_adv", "scanvi", "pca", "harmony", "scanorama"}
    assert {"pca|0|10", "pca|0|50", "harmony|2|50", "scanorama|20|100"} <= set(single)
    pca = out[(out.experiment == "X13") & (out.task == "pancreas") & (out.arm == "pca") & (out.n_latent == "10")]
    close(cells["pancreas|0"]["single"]["pca|0|10"]["S"][0], float(0.4 * pca.B.mean() + 0.6 * pca.C.mean()))

    # best-of-k: failures counted, not penalised (AN-13); the total over configurations, not the largest count (AN-R09)
    lh = cells["lung|0"]["best_of_k"]["harmony"]
    assert (lh["n"], lh["configs"], lh["failed_runs"]) == (5, 6, 1)
    assert [(cells[f"{t}|0"]["best_of_k"]["barycenter"]["configs"],
             cells[f"{t}|0"]["best_of_k"]["barycenter"]["failed_runs"]) for t in ("pancreas", "lung")] == [(6, 4), (6, 2)]
    bk = cells["sim2|0"]["best_of_k"]
    assert {f: bk[f]["n"] for f in bk} == dict(dict.fromkeys(P.ADV_ARMS, 6), sinkhorn=5, harmony=6, scanorama=6,
                                                 sysvi=6, scanvi=1, none=1, scvi_adv=1)
    assert (bk["sinkhorn"]["configs"], bk["sinkhorn"]["failed_runs"], bk["mmd"]["failed_runs"]) == (6, 5, 0)
    assert bk["scanvi"]["stratum"] == "label_aware" and bk["harmony"]["stratum"] == "standard"
    assert np.all(np.diff(bk["reference"]["E"]) >= -1e-12) and len(bk["reference"]["E"]) == 6
    # each E_k minus plain scVI's S (its one configuration, so its E_1), for k = 1 .. N (spec 11)
    assert bk["reference"]["E_minus_plain_scvi"] == pytest.approx([e - bk["none"]["E"][0] for e in bk["reference"]["E"]], abs=1e-12)
    # E_N is the best configuration's S, and the real-task mean of E_k: six tasks, but five at k = 6 for harmony (lung's N is 5)
    pts0 = A.arm_points(out, "pancreas", 0, "reference", r1["families"]["w1|0"]["x1_grid"], sorted(z))
    close(cells["pancreas|0"]["best_of_k"]["reference"]["E"][-1],
          max(0.4 * np.mean(list(p["B"].values())) + 0.6 * np.mean(list(p["C"].values())) for p in pts0 if p["B"]))
    rm = prim["best_of_k_real_mean"]["0"]
    assert [n for _, n in rm["harmony"]] == [6, 6, 6, 6, 6, 5] and [n for _, n in rm["reference"]] == [6] * 6
    assert [n for _, n in rm["none"]] == [6] and [n for _, n in rm["scanvi"]] == [6]
    close(rm["reference"][2][0], np.mean([cells[f"{t}|0"]["best_of_k"]["reference"]["E"][2] for t in P.REAL_TASKS]))
    # frontier positions take (B, C) in that order; each arm's window comes from its own points (review AN-04)
    fr = cells["pancreas|0"]["frontier"]
    assert len(fr) == 25 and set(fr["harmony|0.5|10"]) == set(P.ADV_ARMS)
    h = out[(out.experiment == "X13") & (out.task == "pancreas") & (out.arm == "harmony") & (out.lam == "0.5") & (out.n_latent == "10")]
    bc, front = (float(h.B.mean()), float(h.C.mean())), [(b, c) for _, b, c, _ in P.window_curve(pts0)]
    assert fr["harmony|0.5|10"]["reference"] == A.dominance(bc, front) != A.dominance(bc[::-1], front)
    w = cells["pancreas|0"]["window"]["reference"]
    assert w["status"] == "ok" and w["n_points"] == len(w["lams"])
    dis0 = A.arm_points(out, "pancreas", 0, "discriminator", r1["families"]["js|0"]["x1_grid"], sorted(z))
    wd = A.window([(p["lam"], [0.4 * p["B"][s] + 0.6 * p["C"][s] for s in p["B"]]) for p in dis0])
    got = cells["pancreas|0"]["window"]["discriminator"]
    assert got["lams"] == wd["lams"] != w["lams"] and got["lower"] == pytest.approx(wd["lower"], abs=1e-12)


def test_end_to_end_label_transfer(e2e):  # test-reason: label transfer's matched strength, labels, files, summary and planted effect
    rec, out, r4 = e2e["rec"], e2e["out"], e2e["r4"]
    lt = rec["primary"]["label_transfer"]
    # 8 tasks x 2 decoders x (6 penalties + plain scVI) x 5 seeds, and scvi_adv (conditioned decoder, decoder 1) 8 x 5
    assert lt["k"] == 15 and len(lt["files"]) == 600 and [x["path"] for x in rec["inputs"]["latents"]] == lt["files"]
    assert all(x["sha256"] == P.sha256(x["path"]) for x in rec["inputs"]["latents"])
    for ck, c in (("0", 0), ("1", 1)):
        by_task = lt["decoders"][ck]["tasks"]
        assert all(set(v) == set(P.ADV_ARMS) | {"none"} | ({"scvi_adv"} if c else set()) for v in by_task.values())
        for t, arms in by_task.items():
            for arm, x in arms.items():
                if arm in P.ADV_ARMS:      # the strength R4 matches, and its distance from b* (both from R4's own record)
                    cell = r4["cells"][f"{t}|{c}|{arm}"]
                    assert x["lam"] == cell["matched"] and x["batch_score"] == pytest.approx(cell["matched_B"], abs=1e-12)
                    assert x["distance"] == pytest.approx(cell["distance"], abs=1e-12)
                else:
                    rows = out[(out.experiment == "X1") & (out.task == t) & (out.cond_i == c) & (out.arm == arm)]
                    assert x["lam"] is None and x["batch_score"] == pytest.approx(rows.B.mean(), abs=1e-12) and len(rows) == 5
                assert len(x["per_seed"]) == 5
                if t == NOT_EVALUABLE:
                    assert x["not_evaluable"] and x["mean"] is None and x["sd"] is None and len(x["dropped"]) == 9
                elif arm == "reference":   # the embedding follows the batches: held-out cells are predicted from one other batch
                    assert x["mean"] < 0.7 and x["dropped"] == ["D"]
                else:                      # clusters by cell type: every held-out cell is predicted from its own type
                    assert x["mean"] == 1.0 and x["sd"] == 0.0 and x["dropped"] == ["D"] and not x["not_evaluable"]
        # the summary: real tasks that are not low-signal and evaluable (decoder 0: five; decoder 1: four, immune is low-signal)
        s = lt["decoders"][ck]["summary"]
        n = 5 if c == 0 else 4
        assert s["pooled"] == dict(mean=1.0, n_tasks=n, not_evaluable=[NOT_EVALUABLE]) and s["reference"]["mean"] < 0.7
        assert s["scvi_adv"] == (dict(mean=1.0, n_tasks=n, not_evaluable=[NOT_EVALUABLE]) if c else
                                 dict(mean=None, n_tasks=0, not_evaluable=[]))
    # the sensitivity analyses change nothing label transfer uses, so they do not run it
    assert all(r["label_transfer"] is None for r in rec["sensitivity"].values())


def test_end_to_end_sensitivity(e2e):  # test-reason: each sensitivity analysis changes what section 12 says and nothing else
    rec, out, r1, r4 = e2e["rec"], e2e["out"], e2e["r1"], e2e["r4"]
    prim, sens = rec["primary"], rec["sensitivity"]
    assert set(sens) == set(A.SENSITIVITY) == {"no_trajectory", "per_run", "drop_failed", "corner_zero"}
    # biology without trajectory: C is 0.01 higher on immune_hum_mou at the same crossing, so readings move by 0.01 and the
    # ranks, the local minus global (which does not use C) and the standard errors do not change
    nt, pr = sens["no_trajectory"]["cells"]["immune_hum_mou|0"], prim["cells"]["immune_hum_mou|0"]
    for a in P.ADV_ARMS:
        close(nt["penalties"][a]["bio"] - pr["penalties"][a]["bio"], 0.01, 1e-9)
        assert nt["penalties"][a]["rank"] == pr["penalties"][a]["rank"]
        assert nt["penalties"][a]["local_minus_global"] == pr["penalties"][a]["local_minus_global"]
    assert sens["no_trajectory"]["settings"]["value"] == "C_nt" and sens["corner_zero"]["settings"]["corner"] == "zero"
    # per run: each seed read on its own curve and the penalties ranked within each seed. In immune_hum_mou|1 reference and
    # discriminator respond in seed 1 only (nothing in the other seeds), so in the other four seeds they share places 5 and 6
    # (5.5 each); the rank in seed 1 rests on a single seed's noise and is not asserted. The area has no per-run version
    pen = sens["per_run"]["cells"]["immune_hum_mou|1"]["penalties"]
    assert [x is not None for x in pen["reference"]["readings"]] == [False, True, False, False, False]
    ranks = [A.rank_pair({a: pen[a]["readings"][s] for a in P.ADV_ARMS}) for s in SEEDS]
    assert [r["reference"] for s, r in enumerate(ranks) if s != 1] == [5.5] * 4
    close(pen["reference"]["rank"], np.mean([r["reference"] for r in ranks]), 1e-12)
    z, points = points_of(out, r1, "pancreas", 1)
    tg = r4["targets"]["pancreas|1"]
    runs = [A.read_arms(z, points, tg["bstar"], [s]) for s in sorted(z)]
    want = A.per_seed_mean_ranks([{a: r[a]["bio"] for a in P.ADV_ARMS} for r in runs])
    got = sens["per_run"]["cells"]["pancreas|1"]["penalties"]
    assert {a: got[a]["rank"] for a in P.ADV_ARMS} == pytest.approx(want, abs=1e-12)
    assert got["pooled"]["readings"] == pytest.approx([r["pooled"]["bio"] for r in runs], abs=1e-12)
    assert sens["per_run"]["cells"]["pancreas|1"]["area"] == prim["cells"]["pancreas|1"]["area"]
    # drop_failed reaches the area: the failed reference run in atac_small|1 removes its lambda-10 point (SI-71)
    t, c = "atac_small", 1
    tg, (z, points) = r4["targets"][f"{t}|{c}"], points_of(out, r1, t, c)
    kept = [p for p in points["reference"] if p["n_fail"] == 0]
    assert len(kept) == 5
    dropped = A.read_area(z, kept, tg["b_common"] - tg["b0"])
    close(sens["drop_failed"]["cells"][f"{t}|{c}"]["area"]["reference"], dropped)
    assert abs(dropped - prim["cells"][f"{t}|{c}"]["area"]["reference"]) > 1e-4
    close(sens["corner_zero"]["cells"][f"{t}|{c}"]["area"]["reference"], A.read_area(z, points["reference"], tg["b_common"] - tg["b0"], "zero"))


def _rewrite(src, dst, keep, drop=()):
    df = pd.read_csv(src, sep="\t" if src.endswith(".tsv") else ",", dtype=str, keep_default_na=False, comment="#")
    df[keep(df)].drop(columns=list(drop)).to_csv(dst, sep="\t" if dst.endswith(".tsv") else ",", index=False)
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
    # an R4 record that was not made from these scores: b0, then b* and the low-signal flag (review AN-03)
    (lambda e, tmp: dict(r4=_tamper(e["args"]["r4"], os.path.join(tmp, "r4.json"), "lung|0", "b0", 0.01)), "differ"),
    (lambda e, tmp: dict(r4=_tamper(e["args"]["r4"], os.path.join(tmp, "r4.json"), "atac_large|0", "bstar", 0.03)),
     "differ"),
    (lambda e, tmp: dict(r4=_tamper(e["args"]["r4"], os.path.join(tmp, "r4.json"), "lung|0", "bstar_low_signal",
                                    True)), "differ"),
    (lambda e, tmp: dict(r4=_tamper(e["args"]["r4"], os.path.join(tmp, "r4.json"), "pancreas|1", "b_common", 0.01)),
     "differ"),                                                                 # b_common alone (AN-R07)
    (lambda e, tmp: dict(r4=_tamper(e["args"]["r4"], os.path.join(tmp, "r4.json"), "lung|0", "bstar", 2e-9)),
     "differ"),                                                                 # just outside the 1e-9 tolerance (AN-R07)
    # a record made with an R4 extension row, given without that row: refused, and the message names the cause (AN-R03)
    (lambda e, tmp: dict(manifest=_rewrite(e["args"]["manifest"], os.path.join(tmp, "m.tsv"),
                                           lambda M: ~((M.task == EXT[0]) & (M.cond == EXT[1]) & (M.arm == EXT[2])
                                                       & (M.lam == EXT[3])))),
     r"for atac_small\|0: record \[[-0-9.e]+, [-0-9.e]+, [-0-9.e]+, (True|False)\], recomputed \[[-0-9.e]+, [-0-9.e]+, "
     r"[-0-9.e]+, (True|False)\]; if R4 added extension rows"),
    (lambda e, tmp: _without_seed(e, tmp, "4"), "X1 seeds"),                     # review AN-10
    (lambda e, tmp: dict(out=e["args"]["out"]), "exists"),                      # never overwrite a record
    # label transfer: scores without latent_sha256, a latent that is not the scored one, a missing latent file
    (lambda e, tmp: dict(scores=_rewrite(e["args"]["scores"], os.path.join(tmp, "s.csv"), lambda S: S.tag != "", drop=["latent_sha256"])),
     "lack latent_sha256"),
    (lambda e, tmp: dict(latents=_damaged_latents(e, tmp, remove=False)), r"latent_sha256 differs"),
    (lambda e, tmp: dict(latents=_damaged_latents(e, tmp, remove=True)), r"0 latent files"),
])
def test_refusals(e2e, tmp_path, change, match):
    args = dict(e2e["args"], out=str(tmp_path / "a.json"))
    args.update(change(e2e, str(tmp_path)))
    with pytest.raises(P.PreregError, match=match):
        A.main(argv(args))


def _damaged_latents(e, tmp, remove):
    """A copy of the latents directory without one file, or with one replaced by a different embedding."""
    d = shutil.copytree(e["args"]["latents"], os.path.join(tmp, "latents"))
    f = os.path.join(d, sorted(os.listdir(d))[0])
    if remove:
        os.remove(f)
    else:
        np.savez(f, z=np.zeros((2, 2)))
    return d


def _tamper(src, dst, key, field, change):
    r4 = json.load(open(src))
    t = r4["targets"][key]
    t[field] = change if isinstance(change, bool) else t[field] + change
    json.dump(r4, open(dst, "w"))
    return dst


def _without_seed(e, tmp, seed):
    """X1 without one seed and an R4 record made from the same rows, so only the seed check can refuse."""
    M = e["M"][~((e["M"].experiment == "X1") & (e["M"].seed == seed))]
    S, F = P.read_scores([e["args"]["scores"]])[0], P.read_failures(e["args"]["failures"])
    r4 = os.path.join(tmp, "r4.json")
    P.write_record(r4, P.r4_stage(P.outcomes(M, S, F, ["X1"]), "x1", e["r1"], need=set()),
                   dict(scores=[e["args"]["scores"]]))
    return dict(manifest=synth.write_tsv(M, os.path.join(tmp, "m.tsv")), r4=r4)
