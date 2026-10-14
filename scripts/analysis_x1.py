#!/usr/bin/env python
"""Frozen X1 analysis: matched biology, frontier area, collapse, local mixing and lambda selection,
with deterministic posterior pooling and the registered sensitivities (docs/ANALYSIS_SPEC.md).
This script must not be run on any X1 adversarial-arm score before the tag prereg-tier12-v3.

Usage (repo root):
  python scripts/analysis_x1.py --manifest manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv \
      --scores <dir> --failures <csv> --r1-record docs/prereg/r1_record.json \
      --r4-record docs/prereg/r4_x1_record.json --out-json <analysis.json>
"""
import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_rules as P  # noqa: E402

MARGIN = P.DELTA_MIN
P2_MARGIN = 0.05
PRIOR_SCALE = 0.1
P2_PRIOR_SCALE = 0.25
GRID_N = 201
GRID_SPAN = 5.0
TAU_FLOOR = 1e-15     # each spread is integrated over [TAU_FLOOR, GRID_SPAN x its prior scale], evenly in log(tau)
S_MIN = 1e-6          # smallest standard error pooled (double precision is verified down to it); below: zero_spread
DECLARE = 0.95
MIN_READINGS = 4
X1_SEEDS = [0, 1, 2, 3, 4]
W_B, W_C = 0.4, 0.6
LOCAL = ("iLISI", "kBET", "graph_conn")
GLOBAL = ("PCR_batch", "ASW_label/batch")
REF, DIS = "reference", "discriminator"
SENSITIVITY = {"no_trajectory": {"value": "C_nt"}, "per_run": {"mode": "per_run"},
               "drop_failed": {"mode": "drop_failed"}, "corner_zero": {"corner": "zero"},
               "pairs": {"structure": "pairs"}, "independent": {"structure": "independent"},
               "prior_0.05": {"scale": 0.05}, "prior_0.2": {"scale": 0.2}}


def add_scores(out):
    out = out.copy()
    out["L"] = out[list(LOCAL)].mean(axis=1, skipna=False)
    out["G"] = out[list(GLOBAL)].mean(axis=1, skipna=False)
    for task, rows in out.groupby("task"):
        metrics = [m for m in P.bio_metrics_for(task) if m != "trajectory"]
        out.loc[rows.index, "C_nt"] = rows[metrics].mean(axis=1, skipna=False)
    out.loc[out.status == "failed", ["L", "G", "C_nt"]] = np.nan
    ok = out[out.status == "ok"]
    P._require((abs(ok.B - (3 * ok.L + 2 * ok.G) / 5) <= P.TIE_TOL).all(),
               "local and global scores do not reconstruct B")
    return out


def zero_runs(out, task, cond, value="C"):
    return P._zero_ref(out.assign(C=out[value]), "X1", task, cond)


def arm_points(out, task, cond, arm, grid, seeds, value="C"):
    rows = out[(out.experiment == "X1") & (out.task == task) & (out.cond_i == cond) & (out.arm == arm)]
    rows = rows.loc[[any(math.isclose(lam, g, rel_tol=1e-9) for g in grid) for lam in rows.lamf]]
    P._require(len(grid) == P.N_GRID and sorted(set(rows.lamf)) == list(grid),
               f"{task}|{cond}|{arm}: registered lambda grid is incomplete")
    return P._cell_points(rows.assign(C=rows[value]), "X1", task, cond, arm, seeds)


def read_p1(zero, ref, dis, bstar, mode="mean"):
    P._require(mode in ("mean", "drop_failed", "per_run"), f"unknown reading mode {mode}")
    arms = [ref, dis]
    if mode == "drop_failed":
        arms = [[p for p in points if p["n_fail"] == 0] for points in arms]
    seeds = sorted(zero)

    def read(keep):
        P._require(len(keep), "no zero runs remain for a curve")
        b0, c0 = np.mean([zero[s] for s in keep], axis=0)
        curves = []
        for points in arms:
            selected = [dict(p, B={s: b for s, b in p["B"].items() if s in keep},
                             C={s: c for s, c in p["C"].items() if s in keep}) for p in points]
            curves.append([(0.0, b0, c0, 0)] + P.window_curve(selected))
        readings = [P.bio_at_bstar(curve, bstar) for curve in curves]
        d = readings[0]["bio"] - readings[1]["bio"] if all(r["status"] == "bracketed" for r in readings) else None
        return curves, readings, d

    if mode == "per_run":
        readings = [read([s])[2] for s in seeds]
        existing = [d for d in readings if d is not None]
        d = float(np.mean(existing)) if existing else None
        result = dict(mode=mode, d=d)
    else:
        curves, full, d = read(seeds)
        crossings = []
        for curve in curves:
            dist = np.array([b - bstar for _, b, _, _ in curve])
            hits = abs(dist) <= P.TIE_TOL
            crossings.append(int(hits.sum() + ((~hits[:-1]) & (~hits[1:]) & (dist[:-1] * dist[1:] < 0)).sum()))
        result = dict(mode=mode, bio=dict(zip(("ref", "dis"), [r["bio"] for r in full])),
                      status=dict(zip(("ref", "dis"), [r["status"] for r in full])),
                      crossings=dict(zip(("ref", "dis"), crossings)), d=d)
        readings = [read([s for s in seeds if s != omit])[2] for omit in seeds]
        existing = [x for x in readings if x is not None]
    m = len(existing)
    reason = "unreached" if d is None else ("readings" if m < MIN_READINGS else None)
    s = None
    if reason is None:
        if mode == "per_run":
            s = float(np.std(existing, ddof=1) / math.sqrt(m))
        else:
            s = float(np.sqrt((m - 1) / m * np.sum((existing - np.mean(existing)) ** 2)))
    return dict(result, readings=readings, m=m, s=s, reason=reason)


def area(points, b_ref, c_ref):
    height, total = c_ref, 0.0
    for b, c in sorted(((b, c) for b, c in points if b > b_ref and c > c_ref), reverse=True):
        total += (b - b_ref) * max(0.0, c - height)
        height = max(height, c)
    return float(total)


def read_area(zero, ref, dis, scale, corner="collapse", drop_failed=False):
    P._require(math.isfinite(scale) and scale > 0, "area scale must be positive")
    P._require(corner in ("collapse", "zero"), f"unknown area corner {corner}")
    arms = [[p for p in points if not drop_failed or p["n_fail"] == 0] for points in (ref, dis)]
    seeds = sorted(zero)

    def read(keep):
        P._require(len(keep), "no zero runs remain for an area")
        b0, c0 = np.mean([zero[s] for s in keep], axis=0)
        c_ref = c0 - P.COLLAPSE_BIO if corner == "collapse" else 0.0
        values = []
        for points in arms:
            selected = [dict(p, B={s: b for s, b in p["B"].items() if s in keep},
                             C={s: c for s, c in p["C"].items() if s in keep}) for p in points]
            values.append(area([(b, c) for _, b, c, _ in P.window_curve(selected)], b0, c_ref) / scale)
        return values

    values = read(seeds)
    readings = []
    for omit in seeds:
        r, d = read([s for s in seeds if s != omit])
        readings.append(r - d)
    m = len(readings)
    s = float(np.sqrt((m - 1) / m * np.sum((readings - np.mean(readings)) ** 2)))
    return dict(area=dict(zip(("ref", "dis"), values)), d=values[0] - values[1], readings=readings, m=m, s=s)


def pool(d, s, levels, delta_scale, tau_scale, margin):
    d, s = np.asarray(d, dtype=float), np.asarray(s, dtype=float)
    P._require(d.ndim == s.ndim == 1 and len(d) == len(s) and len(d) >= 1
               and np.isfinite(d).all() and np.isfinite(s).all() and (s >= S_MIN).all(),
               f"pool needs finite equal-length d and positive s (at least {S_MIN:g}), with at least one observation")
    n = len(d)
    P._require(all(len(labels) == n for labels in levels), "pool levels must match the observations")
    P._require(np.isfinite([delta_scale, tau_scale, margin]).all()
               and delta_scale > 0 and tau_scale > 0 and margin >= 0,
               "pool prior scales must be positive and margin nonnegative")
    indices = np.array(list(itertools.product(range(GRID_N), repeat=len(levels))), dtype=int)
    # trapezoid rule in log(tau) (weight x tau): resolves spreads near a standard error however small (review AN-01)
    tau = np.geomspace(TAU_FLOOR, GRID_SPAN * tau_scale, GRID_N)[indices]
    log_weight = np.log(np.where((indices == 0) | (indices == GRID_N - 1), 0.5, 1.0)).sum(axis=1)
    log_weight += np.log(tau).sum(axis=1) - (tau ** 2).sum(axis=1) / (2 * tau_scale ** 2)
    delta2, s2 = delta_scale ** 2, s ** 2
    A = np.zeros((len(indices), n, n)) + np.diag(s2)          # the covariance of d without Delta's prior variance
    for j, labels in enumerate(levels):
        labels = np.asarray(labels)
        A += tau[:, j, None, None] ** 2 * (labels[:, None] == labels[None, :])
    V = A + delta2
    inverse = np.linalg.inv(V)
    vd = inverse @ d
    log_weight -= 0.5 * (np.linalg.slogdet(V)[1] + np.einsum("ni,i->n", vd, d))
    weights = np.exp(log_weight - log_weight.max())
    weights /= weights.sum()
    # Delta | d has precision 1 / delta2 + 1'A^-1 1 and mean 1'A^-1 d / precision: solving with A avoids the
    # cancellation in delta2 - delta2^2 1'V^-1 1 and delta2 1'V^-1 d when the standard errors are small
    x = np.linalg.solve(A, np.broadcast_to(np.stack([np.ones(n), d], axis=1), (len(A), n, 2)))
    delta_var = delta2 / (1 + delta2 * x[:, :, 0].sum(axis=1))
    delta_mean, delta_sd = delta_var * x[:, :, 1].sum(axis=1), np.sqrt(delta_var)
    # Var(theta | d) = S - S V^-1 S with S = diag(s^2) avoids two batched matrix products for the task posteriors.
    theta_mean = d - s2 * vd
    theta_sd = np.sqrt(s2 - s2 ** 2 * np.diagonal(inverse, axis1=1, axis2=2))

    def summary(means, sd):
        def cdf(x):
            return float(np.sum(weights * special.ndtr((x - means) / sd)))

        lower, upper = float(np.min(means - 8 * sd)), float(np.max(means + 8 * sd))
        return dict(mean=float(np.sum(weights * means)),
                    lo=optimize.brentq(lambda x: cdf(x) - 0.025, lower, upper, xtol=1e-12),
                    hi=optimize.brentq(lambda x: cdf(x) - 0.975, lower, upper, xtol=1e-12))

    p_below = float(np.sum(weights * special.ndtr((-margin - delta_mean) / delta_sd)))
    p_above = float(np.sum(weights * special.ndtr((delta_mean - margin) / delta_sd)))
    p_within = float(np.sum(weights * (special.ndtr((margin - delta_mean) / delta_sd)
                                      - special.ndtr((-margin - delta_mean) / delta_sd))))
    if p_above >= DECLARE:
        declaration = "above"
    elif p_below >= DECLARE:
        declaration = "below"
    elif p_within >= DECLARE:
        declaration = "equivalent"
    else:
        declaration = "inconclusive"
    return dict(n=n, **summary(delta_mean, delta_sd), p_below=p_below, p_within=p_within, p_above=p_above,
                declaration=declaration, heterogeneity=float(np.sum(weights * np.sqrt((tau ** 2).sum(axis=1)))),
                theta=[summary(theta_mean[:, i], theta_sd[:, i]) for i in range(n)],
                delta_scale=delta_scale, tau_scale=tau_scale, margin=margin)


def combine(tasks, d, s, structure, scale, margin):
    if not tasks:
        return dict(n=0, declaration="not_evaluable")
    P._require(structure in ("family", "independent", "pairs"), f"unknown pooling structure {structure}")
    P._require(len(tasks) == len(d) == len(s), "pool tasks, d and s must have equal lengths")
    units = list(tasks)
    if structure == "independent":
        levels = [units]
    else:
        P._require(set(tasks) <= set(P.TASK_FAMILY), "pool tasks lack registered families")
        families = [P.TASK_FAMILY[t] for t in tasks]
        if structure == "family":
            levels = [families, units]
        else:
            units = list(dict.fromkeys(families))
            groups = [[i for i, f in enumerate(families) if f == u] for u in units]
            d = [float(np.mean([d[i] for i in g])) for g in groups]
            s = [math.sqrt(sum(s[i] ** 2 for i in g)) / len(g) for g in groups]
            levels = [units]
    return dict(pool(d, s, levels, scale, scale, margin), units=units, structure=structure)


def wilcoxon_exact(d):
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if not len(d):
        return 1.0
    ranks = stats.rankdata(abs(d), method="average")
    center = ranks.sum() / 2
    observed = abs(ranks[d > 0].sum() - center)
    sums = np.array(list(itertools.product((0, 1), repeat=len(d)))) @ ranks
    return float(np.mean(abs(sums - center) >= observed - P.TIE_TOL))


def sign_test(d):
    d = np.asarray(d, dtype=float)
    n = int(np.count_nonzero(d))
    return float(stats.binomtest(int((d > 0).sum()), n, 0.5).pvalue) if n else 1.0


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def clopper_pearson(x, n):
    P._require(n > 0 and 0 <= x <= n, "collapse counts must satisfy 0 <= x <= n with positive n")
    lo = 0.0 if x == 0 else float(stats.beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(0.975, x + 1, n - x))
    return lo, hi


def agresti_caffo(x1, n1, x2, n2):
    P._require(n1 > 0 and n2 > 0 and 0 <= x1 <= n1 and 0 <= x2 <= n2, "invalid collapse counts")
    p1, p2 = (x1 + 1) / (n1 + 2), (x2 + 1) / (n2 + 2)
    return p1 - p2, math.sqrt(p1 * (1 - p1) / (n1 + 2) + p2 * (1 - p2) / (n2 + 2))


def collapse_counts(rows, zero, fail_status):
    P._require(len(rows) > 0 and rows.status.isin(["ok", "failed"]).all(), "collapse rows need ok/failed outcomes")
    P._require(set(rows.seed_i) <= set(zero), "collapse rows lack paired zero seeds")
    failed = rows.loc[rows.status == "failed", "tag"].map(fail_status)
    P._require(failed.isin(P.FAILURE_STATUSES).all(), "collapse rows lack registered failure statuses")
    failures = {status: int((failed == status).sum()) for status in P.FAILURE_STATUSES}
    ok = rows[rows.status == "ok"]
    P._require(np.isfinite(ok.C).all(), "collapse biology scores must be finite")
    bio = int((ok.C <= ok.seed_i.map({s: c for s, (_, c) in zero.items()}) - P.COLLAPSE_BIO + P.TIE_TOL).sum())
    n, collapsed = len(rows), bio + len(failed)
    lo, hi = clopper_pearson(collapsed, n)
    return dict(n=n, collapsed=collapsed, bio=bio, failed=failures, rate=collapsed / n, lo=lo, hi=hi)


def choose_lambda(S, fails, tasks):
    P._require(len(tasks) > 0 and set(tasks) <= set(fails) and set(tasks) <= set(S),
               "lambda grid needs nonempty tasks with scores and failure counts")
    grid = set(fails[tasks[0]])
    P._require(all(set(fails[t]) == grid for t in tasks), "tasks have different lambda grids")
    eligible = [lam for lam in sorted(grid) if all(fails[t][lam] == 0 and lam in S[t] for t in tasks)]
    if not eligible:
        return None
    means = {lam: float(np.mean([S[t][lam] for t in tasks])) for lam in eligible}
    P._require(np.isfinite(list(means.values())).all(), "lambda grid scores must be finite")
    best = max(means.values())
    return min(lam for lam in eligible if means[lam] >= best - P.TIE_TOL)


def transfer_lambdas(S, fails):
    result = {}
    for task in S:
        if task in P.REAL_TASKS:
            train = sorted(u for u in S if u in P.REAL_TASKS and P.TASK_FAMILY[u] != P.TASK_FAMILY[task])
        else:
            train = sorted(u for u in S if u in P.REAL_TASKS)
        result[task] = dict(train=train, lam=choose_lambda(S, fails, train))
    return result


def best_of_k(scores):
    scores = sorted(scores)
    n = len(scores)
    return [sum(scores[i - 1] * math.comb(i - 1, k - 1) for i in range(k, n + 1)) / math.comb(n, k)
            for k in range(1, n + 1)]


def window(points):
    points = [(lam, scores) for lam, scores in points if len(scores)]
    result = dict(status="no_points", best=None, lower=None, lams=None, width=None, n_points=None, gap=None)
    if not points:
        return result
    means = {lam: float(np.mean(scores)) for lam, scores in points}
    top = max(means.values())
    best = min(lam for lam in means if means[lam] >= top - P.TIE_TOL)
    result.update(status="no_interval", best=best)
    scores = dict(points)[best]
    n = len(scores)
    if n < 2:
        return result
    lower = means[best] - stats.t.ppf(0.975, n - 1) * np.std(scores, ddof=1) / math.sqrt(n)
    lams = sorted(lam for lam in means if means[lam] >= lower - P.TIE_TOL)
    P._require(min(lams) > 0, "window lambda must be positive")
    # gap: a point between the window's ends lies below the bound, so the width spans it (review AN-14)
    return dict(status="ok", best=best, lower=float(lower), lams=lams,
                width=math.log10(max(lams)) - math.log10(min(lams)), n_points=len(lams),
                gap=any(min(lams) < lam < max(lams) and lam not in lams for lam in means))


def dominance(config, front):
    def dominates(q, p):
        return (q[0] >= p[0] - P.TIE_TOL and q[1] >= p[1] - P.TIE_TOL
                and (q[0] > p[0] + P.TIE_TOL or q[1] > p[1] + P.TIE_TOL))

    return dict(dominated_by=sum(dominates(q, config) for q in front),
                dominates=sum(dominates(config, q) for q in front))


def analyse(out, r1, r4, fail_status, value="C", mode="mean", corner="collapse", structure="family", scale=PRIOR_SCALE):
    x1, x13 = out[out.experiment == "X1"], out[out.experiment == "X13"]
    tasks = sorted(P.BPM.TASKS)
    P._require(set(zip(x1.task, x1.cond_i)) == set(itertools.product(tasks, (0, 1)))
               and set(x1.task[x1.arm == "scvi_adv"]) == set(x13.task) == set(tasks),
               "X1 and X13 must contain all eight tasks, X1 both decoders and scvi_adv")
    for task in tasks:
        rows = x13[x13.task == task]
        for arm in ("harmony", "scanorama", "sysvi", "scanvi"):
            configs = rows[rows.arm == arm]
            if arm in ("harmony", "scanorama"):
                configs = configs[configs.n_latent == "10"]
            want = 1 if arm == "scanvi" else 6
            P._require(configs.lam.nunique() == want and len(configs[["lam", "n_latent"]].drop_duplicates()) == want,
                       f"X13 {task} {arm}: expected {want} configurations")
    P._require("targets" in r4, "R4 lacks b0 targets")
    targets = P.r4_stage(out, "x1", r1)["targets"]     # the rule that wrote the record, rerun on these scores (AN-03)
    for task, cond in itertools.product(tasks, (0, 1)):
        key = f"{task}|{cond}"
        P._require(key in r4["targets"] and {"b0", "b_common", "bstar", "bstar_low_signal"} <= set(r4["targets"][key]),
                   f"R4 lacks b0 target for {key}")
        P._require(sorted(zero_runs(out, task, cond)) == X1_SEEDS, f"{key}: plain-scVI seeds are not the X1 seeds 0-4")
        got, want = r4["targets"][key], targets[key]
        P._require(got["bstar_low_signal"] == want["bstar_low_signal"]
                   and all(abs(got[k] - want[k]) <= P.TIE_TOL for k in ("b0", "b_common", "bstar")),
                   f"R4 targets (b0, b_common, b* or the low-signal flag) differ from these scores for {key}")
    P._require(value in ("C", "C_nt"), f"unknown biology value {value}")
    out = out.assign(C=out[value])
    x1, x13 = out[out.experiment == "X1"], out[out.experiment == "X13"]
    cells, all_points, all_scores = {}, {}, {}
    for task, cond in itertools.product(tasks, (0, 1)):
        key = f"{task}|{cond}"
        target = r4["targets"][key]
        zero = zero_runs(out, task, cond)
        local_zero = zero_runs(out, task, cond, "L")
        rows = x1[(x1.task == task) & (x1.cond_i == cond)]
        points, local_points = {}, {}
        for arm in P.ADV_ARMS:
            family = f"{P.FAMILY_OF[arm]}|{cond}"
            P._require("families" in r1 and family in r1["families"] and "x1_grid" in r1["families"][family],
                       f"missing registered grid {family}")
            grid = r1["families"][family]["x1_grid"]
            points[arm] = arm_points(rows, task, cond, arm, grid, sorted(zero))
            local_points[arm] = arm_points(rows, task, cond, arm, grid, sorted(zero), "L")
        scores = {arm: [(p["lam"], [W_B * b + W_C * p["C"][s] for s, b in p["B"].items()]) for p in points[arm]]
                  for arm in P.ADV_ARMS}
        all_points[key], all_scores[key] = points, scores
        cell = dict(task=task, cond=cond, b0=target["b0"], b_common=target["b_common"], bstar=target["bstar"],
                    low_signal=target["bstar_low_signal"],
                    p1=read_p1(zero, points[REF], points[DIS], target["bstar"], mode),
                    local=read_p1(local_zero, local_points[REF], local_points[DIS], target["bstar"], mode))
        cell["area"] = None if cell["low_signal"] else read_area(zero, points[REF], points[DIS],
                        cell["b_common"] - cell["b0"], corner, drop_failed=(mode == "drop_failed"))
        cell["collapse"] = {arm: collapse_counts(rows[(rows.arm == arm) & rows.lamf.isin([p["lam"] for p in points[arm]])],
                                                zero, fail_status) for arm in P.ADV_ARMS}
        cell["window"] = {arm: window(scores[arm]) for arm in P.ADV_ARMS}
        family_scores = {arm: [float(np.mean(v)) for _, v in scores[arm] if v] for arm in P.ADV_ARMS}
        family_scores.update({arm: [] for arm in ("harmony", "scanorama", "sysvi", "scanvi", "none", "scvi_adv")})
        baselines = pd.concat([x13[x13.task == task], x1[(x1.task == task) &
                              (((x1.arm == "none") & (x1.cond_i == cond)) | (x1.arm == "scvi_adv"))]])
        anchors, frontier = {}, {}
        fronts = {arm: [(b, c) for _, b, c, _ in P.window_curve(points[arm])] for arm in P.ADV_ARMS}
        failed = {arm: [p["n_fail"] for p in points[arm]] for arm in P.ADV_ARMS}     # failed runs per configuration
        for (arm, lam, n_latent), config in baselines.groupby(["arm", "lam", "n_latent"], sort=True):
            anchor = arm == "pca" or (arm in ("harmony", "scanorama") and n_latent != "10")
            if not anchor:
                failed.setdefault(arm, []).append(int((config.status == "failed").sum()))
            ok = config[config.status == "ok"]
            if ok.empty:
                continue
            b, c = float(ok.B.mean()), float(ok.C.mean())
            score = W_B * b + W_C * c
            name = f"{arm}|{lam}|{n_latent}"
            if anchor:
                anchors[name] = score
            else:
                family_scores[arm].append(score)
            frontier[name] = {a: dominance((b, c), fronts[a]) for a in P.ADV_ARMS}
        # E_k does not penalise failures (a configuration scores the mean of its successful runs): counted (AN-13)
        cell["best_of_k"] = {arm: dict(n=len(v), E=best_of_k(v), configs=len(failed.get(arm, [])),
                                      failed_runs=sum(failed.get(arm, [])),
                                      stratum="label_aware" if arm == "scanvi" else "standard")
                             for arm, v in family_scores.items()}
        cell["anchors"], cell["frontier"] = anchors, frontier
        cells[key] = cell

    pooled, p2, transfer = {}, {}, {}
    for cond in (0, 1):
        ck = str(cond)
        pooled[ck], p2[ck], transfer[ck] = {}, {}, {}
        for endpoint in ("p1", "area", "local"):
            evaluable, not_evaluable, low_signal, signs = [], [], [], []
            d, s = [], []
            for task in P.REAL_TASKS:
                cell = cells[f"{task}|{cond}"]
                reading = cell[endpoint]
                if cell["low_signal"]:
                    not_evaluable.append(dict(task=task, reason="low_signal"))
                    low_signal.append(dict(task=task, d=None if endpoint == "area" else reading["d"]))
                elif reading.get("reason") is not None:
                    not_evaluable.append(dict(task=task, reason=reading["reason"]))
                elif reading["s"] < S_MIN:          # e.g. both arms' areas 0 in every leave-one-seed-out repeat (AN-02)
                    not_evaluable.append(dict(task=task, reason="zero_spread"))
                else:
                    evaluable.append(task)
                    d.append(reading["d"])
                    s.append(reading["s"])
                    q = stats.t.ppf(0.975, reading["m"] - 1)          # s rests on m leave-one-seed-out readings
                    signs.append(dict(task=task, d=d[-1], s=s[-1], lo=d[-1] - q * s[-1], hi=d[-1] + q * s[-1]))
            entry = dict(combine(evaluable, d, s, structure, scale, MARGIN), tasks=evaluable,
                         not_evaluable=not_evaluable, low_signal=low_signal, signs=signs, wilcoxon_p=wilcoxon_exact(d))
            if endpoint == "local":
                entry["sign_test_p"] = sign_test(d)
            pooled[ck][endpoint] = entry
        diffs = []
        for task in P.REAL_TASKS:
            counts = cells[f"{task}|{cond}"]["collapse"]
            r, d = counts[REF], counts[DIS]
            diff, se = agresti_caffo(r["collapsed"], r["n"], d["collapsed"], d["n"])
            diffs.append(dict(task=task, d=diff, se=se))
        d, s = [r["d"] for r in diffs], [r["se"] for r in diffs]
        pooled[ck]["p2"] = dict(combine(P.REAL_TASKS, d, s, structure, P2_PRIOR_SCALE, P2_MARGIN),
                                tasks=list(P.REAL_TASKS), diffs=diffs, wilcoxon_p=wilcoxon_exact(d))
        for arm in P.ADV_ARMS:
            counts = {task: cells[f"{task}|{cond}"]["collapse"][arm] for task in tasks}
            n = sum(counts[t]["n"] for t in P.REAL_TASKS)
            collapsed = sum(counts[t]["collapsed"] for t in P.REAL_TASKS)
            lo, hi = clopper_pearson(collapsed, n)
            p2[ck][arm] = dict(counts, pooled=dict(n=n, collapsed=collapsed, rate=collapsed / n, lo=lo, hi=hi))
            S = {task: {lam: float(np.mean(v)) for lam, v in all_scores[f"{task}|{cond}"][arm] if v} for task in tasks}
            fails = {task: {p["lam"]: p["n_fail"] for p in all_points[f"{task}|{cond}"][arm]} for task in tasks}
            chosen = transfer_lambdas(S, fails)
            transfer[ck][arm] = {}
            for task in tasks:
                def at(lam):
                    if lam is None:
                        return None
                    point = next(p for p in all_points[f"{task}|{cond}"][arm] if p["lam"] == lam)
                    B = pd.Series(point["B"], dtype=float)
                    C = pd.Series(point["C"], dtype=float)
                    return dict(n_ok=len(B), n_fail=point["n_fail"],
                                **{name: [v.mean(), v.std()] for name, v in (("B", B), ("C", C), ("S", W_B * B + W_C * C))})

                best = choose_lambda(S, fails, [task])
                transfer[ck][arm][task] = dict(chosen[task], at=at(chosen[task]["lam"]), best_lam=best, best=at(best))
    primary = [(endpoint, str(cond)) for endpoint in ("p1", "area", "p2") for cond in (0, 1)]
    adjusted = holm([pooled[cond][endpoint]["wilcoxon_p"] for endpoint, cond in primary])
    return dict(settings=dict(value=value, mode=mode, corner=corner, structure=structure, scale=scale),
                cells=cells, pooled=pooled, p2=p2, transfer=transfer,
                holm={f"{endpoint}|{cond}": p for (endpoint, cond), p in zip(primary, adjusted)}, n_primary=6)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--failures", required=True)
    ap.add_argument("--r1-record", required=True)
    ap.add_argument("--r4-record", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    P._require(not os.path.exists(a.out_json), f"analysis record {a.out_json} exists")
    M, _ = P.read_manifest(a.manifest)
    P.check_seed_separation(M)
    r1 = P.read_record(a.r1_record, "R1")
    r4 = P.read_record(a.r4_record, "R4", stage="x1")
    S, score_files = P.read_scores(a.scores)
    F = P.read_failures(a.failures)
    out = add_scores(P.outcomes(M, S, F, ["X1", "X13"]).sort_values("tag", ignore_index=True))  # row order (AN-11)
    fail_status = dict(zip(F.tag, F.status))
    primary = analyse(out, r1, r4, fail_status)
    sensitivity = {name: analyse(out, r1, r4, fail_status, **kw) for name, kw in SENSITIVITY.items()}
    P.write_record(a.out_json, dict(rule="analysis_x1", status="done", spec="docs/ANALYSIS_SPEC.md",
                                  primary=primary, sensitivity=sensitivity),
                   dict(manifest=a.manifest, scores=score_files, failures=[a.failures],
                        r1_record=[a.r1_record], r4_record=[a.r4_record]))
    print(f"X1 analysis: {len(primary['cells'])} cells, {primary['n_primary']} primary results, "
          f"{len(sensitivity)} sensitivities -> {a.out_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
