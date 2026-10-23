#!/usr/bin/env python
"""X1 analysis, ranking design: penalties ranked by biology at matched batch removal, with per-penalty secondary analyses (docs/ANALYSIS_SPEC.md sections 2-12).
Per pair (task, decoder) the six penalties are ranked by their biology at b*, with leave-one-seed-out standard errors and rank ranges;
per penalty: frontier area, collapse rate, good-strength width, local minus global batch removal, label transfer, single-strength
tables and best-of-k; and the sensitivity analyses. The contrast design this script replaced is at the tag prereg-tier12-v3.
This script must not be run on any X1 or X13 score before the ranking version is frozen by its own tag (SI-78: an analysis script
that uses a score examines it).

Usage (repo root):
  python scripts/analysis_x1.py --manifest manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv \
      [--manifest <each R4 extension manifest>] --scores <dir> [<each R4 extension scores dir>] \
      --failures <csv that covers every row> --r1-record docs/prereg/r1_record.json \
      --r4-record docs/prereg/r4_x1_record.json --latents <dir> [<each R4 extension latents dir>] --out-json <analysis.json>
The script recomputes the R4 targets from the rows it is given (prereg_rules.r4_stage) and refuses a record that
differs: when R4 extended the X1 grids, pass the extension manifests and their scores too, or a valid record is refused.
--latents (label transfer, section 9): the directories hold the saved embeddings <tag>.npz of the fits it uses (OUT/latents of
each stage; scripts/fit_paper_config.py writes z, batch and celltype, the posterior mean and the task's labels), each checked
against the latent_sha256 of its score row; the SHA-256 sums of the files read go into the record.
"""
import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_rules as P  # noqa: E402

MIN_READINGS = 4
X1_SEEDS = [0, 1, 2, 3, 4]
W_B, W_C = 0.4, 0.6
LOCAL = ("iLISI", "kBET", "graph_conn")
GLOBAL = ("PCR_batch", "ASW_label/batch")
TOOLS = ("harmony", "scanorama", "sysvi")
LABEL_K = 15
RANK_KEYS = ("rank", "loso", "m", "s", "rank_lo", "rank_hi")      # what a low-signal pair does not get (never ranked)
SENSITIVITY = {"no_trajectory": {"value": "C_nt"}, "per_run": {"mode": "per_run"},
               "drop_failed": {"mode": "drop_failed"}, "corner_zero": {"corner": "zero"}}


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


def config_points(rows):
    """prereg_rules._cell_points format for the X13 rows of one standard tool: one point per setting."""
    points = []
    for lam, g in rows.groupby("lamf"):
        ok = g[g.status == "ok"]
        points.append(dict(lam=float(lam), n_fail=len(g) - len(ok),
                           B={int(s): float(b) for s, b in zip(ok.seed_i, ok.B)},
                           C={int(s): float(c) for s, c in zip(ok.seed_i, ok.C)}))
    return points


def tool_rows(x13, task, arm):
    rows = x13[(x13.task == task) & (x13.arm == arm)]
    return rows[rows.n_latent == "10"] if arm in ("harmony", "scanorama") else rows      # 10 principal components / dimensions


def usable(points, mode):
    """Points the readings and the area use: all of them, or (drop_failed, SI-71 sensitivity) those without a failed run."""
    return {arm: [p for p in pts if mode != "drop_failed" or p["n_fail"] == 0] for arm, pts in points.items()}


def rank_pair(readings):
    """Ranks of one pair's penalties by reading (spec 3.1): 1 = highest. None = did not reach b*: those share the last places.
    Readings within TIE_TOL are tied; tied entries get the mean of the places they occupy."""
    P._require(readings and all(v is None or math.isfinite(v) for v in readings.values()),
               "rank_pair needs at least one penalty, with finite readings or None")
    names = sorted(readings, key=lambda k: (readings[k] is None, -(readings[k] or 0.0)))

    def tied(a, b):          # names are in descending order of reading, None last
        return (a is None and b is None) or (a is not None and b is not None and a - b <= P.TIE_TOL)

    ranks, i = {}, 0
    while i < len(names):
        j = i + 1
        while j < len(names) and tied(readings[names[i]], readings[names[j]]):
            j += 1
        ranks.update(dict.fromkeys(names[i:j], (i + 1 + j) / 2))        # the mean of the places i + 1 .. j
        i = j
    return {k: ranks[k] for k in readings}


def loso_se(xs):
    """Leave-one-seed-out standard error (spec 4 (i)) of the m readings that exist; (None, m) when m < MIN_READINGS."""
    xs = np.array([x for x in xs if x is not None], dtype=float)
    m = len(xs)
    return (float(np.sqrt((m - 1) / m * np.sum((xs - xs.mean()) ** 2))) if m >= MIN_READINGS else None), m


def crossings(curve, bstar):
    dist = np.array([b - bstar for _, b, _, _ in curve])
    hits = abs(dist) <= P.TIE_TOL
    return int(hits.sum() + ((~hits[:-1]) & (~hits[1:]) & (dist[:-1] * dist[1:] < 0)).sum())


def read_arms(zero, arms, bstar, keep):
    """Each penalty's curve and reading (spec 2.1, 2.2) on the seeds in `keep`: the anchor is the mean of the plain-scVI runs of
    those seeds, the points are the means over their runs of those seeds, and a point left without a run is skipped."""
    P._require(len(keep), "no zero runs remain for a curve")
    b0, c0 = np.mean([zero[s] for s in keep], axis=0)
    result = {}
    for arm, points in arms.items():
        selected = [dict(p, B={s: b for s, b in p["B"].items() if s in keep},
                         C={s: c for s, c in p["C"].items() if s in keep}) for p in points]
        curve = [(0.0, b0, c0, 0)] + P.window_curve(selected)
        result[arm] = dict(P.bio_at_bstar(curve, bstar), curve=curve)
    return result


def rank_one_pair(zero, arms, bstar):
    """One task and decoder (spec 3, 4): readings, ranks, and the leave-one-seed-out repeats (each seed left out of the penalty's
    runs and of the plain-scVI runs, b* fixed) with their standard errors and the smallest and largest rank."""
    seeds = sorted(zero)
    full = read_arms(zero, arms, bstar, seeds)
    repeats = [read_arms(zero, arms, bstar, [s for s in seeds if s != omit]) for omit in seeds]
    ranks = rank_pair({arm: r["bio"] for arm, r in full.items()})
    repeat_ranks = [rank_pair({arm: r[arm]["bio"] for arm in arms}) for r in repeats]
    result = {}
    for arm, r in full.items():
        loso = [x[arm]["bio"] for x in repeats]
        s, m = loso_se(loso)
        result[arm] = dict(bio=r["bio"], status=r["status"], crossings=crossings(r["curve"], bstar), rank=ranks[arm],
                           loso=loso, m=m, s=s if r["bio"] is not None else None,       # no reading, no standard error
                           rank_lo=min(x[arm] for x in repeat_ranks), rank_hi=max(x[arm] for x in repeat_ranks))
    return result


def mean_ranks(pair_ranks, penalties):
    P._require(all(set(r) == set(penalties) for r in pair_ranks.values()), "every ranked pair must rank every penalty")
    n = len(pair_ranks)
    return {arm: (float(np.mean([r[arm] for r in pair_ranks.values()])) if n else None, n) for arm in penalties}


def per_seed_mean_ranks(per_seed):
    """Spec 12, per-run reading: the penalties are ranked within each seed's readings and the ranks averaged."""
    P._require(per_seed and all(set(r) == set(per_seed[0]) for r in per_seed), "per-seed readings need the same penalties")
    ranks = [rank_pair(r) for r in per_seed]
    return {arm: float(np.mean([r[arm] for r in ranks])) for arm in per_seed[0]}


def local_minus_global(zero_l, zero_g, arms_l, arms_g, bstar, keep):
    """Local minus global batch removal at each penalty's crossing (spec 7): the B curve read with L and with G in place of C,
    so both are read at the same crossing with the same interpolation weights; None where the penalty does not reach b*."""
    local, glob = read_arms(zero_l, arms_l, bstar, keep), read_arms(zero_g, arms_g, bstar, keep)
    return {arm: None if r["bio"] is None else r["bio"] - glob[arm]["bio"] for arm, r in local.items()}


def read_pair(zero, zero_l, zero_g, arms, arms_l, arms_g, bstar, mode):
    """Readings, ranks and local minus global of the six penalties in one pair; per_run: per seed, ranks averaged over seeds."""
    seeds = sorted(zero)
    if mode == "per_run":
        runs = [read_arms(zero, arms, bstar, [s]) for s in seeds]
        lmg = [local_minus_global(zero_l, zero_g, arms_l, arms_g, bstar, [s]) for s in seeds]
        rank = per_seed_mean_ranks([{arm: r[arm]["bio"] for arm in arms} for r in runs])
        return {arm: dict(readings=[r[arm]["bio"] for r in runs], local_minus_global=[x[arm] for x in lmg], rank=rank[arm])
                for arm in arms}
    lmg = local_minus_global(zero_l, zero_g, arms_l, arms_g, bstar, seeds)
    return {arm: dict(r, local_minus_global=lmg[arm]) for arm, r in rank_one_pair(zero, arms, bstar).items()}


def area(points, b_ref, c_ref):
    height, total = c_ref, 0.0
    for b, c in sorted(((b, c) for b, c in points if b > b_ref and c > c_ref), reverse=True):
        total += (b - b_ref) * max(0.0, c - height)
        height = max(height, c)
    return float(total)


def read_area(zero, points, scale, corner="collapse"):
    """One penalty's frontier area over the common batch range `scale` (spec 5); the corner is (b0, c0 - 0.10) or (b0, 0)."""
    P._require(math.isfinite(scale) and scale > 0, "area scale must be positive")
    P._require(corner in ("collapse", "zero"), f"unknown area corner {corner}")
    b0, c0 = np.mean(list(zero.values()), axis=0)
    c_ref = c0 - P.COLLAPSE_BIO if corner == "collapse" else 0.0
    return area([(b, c) for _, b, c, _ in P.window_curve(points)], b0, c_ref) / scale


def clopper_pearson(x, n):
    P._require(n > 0 and 0 <= x <= n, "collapse counts must satisfy 0 <= x <= n with positive n")
    lo = 0.0 if x == 0 else float(stats.beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(0.975, x + 1, n - x))
    return lo, hi


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


def seed_summary(B, C, n_fail, plain_S):
    """B, C and S = 0.4 B + 0.6 C as [mean, SD over seeds], and the mean S minus plain scVI's mean S in the same task and decoder."""
    B, C = pd.Series(B, dtype=float), pd.Series(C, dtype=float)
    S = W_B * B + W_C * C
    return dict(n_ok=len(B), n_fail=n_fail, S_minus_plain_scvi=float(S.mean() - plain_S),
                **{name: [v.mean(), v.std()] for name, v in (("B", B), ("C", C), ("S", S))})


def best_of_k(scores):
    scores = sorted(scores)
    n = len(scores)
    return [sum(scores[i - 1] * math.comb(i - 1, k - 1) for i in range(k, n + 1)) / math.comb(n, k)
            for k in range(1, n + 1)]


def best_of_k_real_mean(per_task):
    """Spec 11, main figure: per k the mean over the tasks with N >= k of E_k, and the number of those tasks."""
    return [(float(np.mean([e[k - 1] for e in per_task.values() if len(e) >= k])), sum(len(e) >= k for e in per_task.values()))
            for k in range(1, max((len(e) for e in per_task.values()), default=0) + 1)]


def window(points):
    empty = [lam for lam, scores in points if not len(scores)]      # configurations with no successful run (AN-R13)
    points = [(lam, scores) for lam, scores in points if len(scores)]
    result = dict(status="no_points", best=None, lower=None, lams=None, width=None, n_points=None, gap=None,
                  n_all_failed_between=None)
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
                gap=any(min(lams) < lam < max(lams) and lam not in lams for lam in means),
                n_all_failed_between=sum(min(lams) < lam < max(lams) for lam in empty))     # the width spans these too


def dominance(config, front):
    def dominates(q, p):
        return (q[0] >= p[0] - P.TIE_TOL and q[1] >= p[1] - P.TIE_TOL
                and (q[0] > p[0] + P.TIE_TOL or q[1] > p[1] + P.TIE_TOL))

    return dict(dominated_by=sum(dominates(q, config) for q in front),
                dominates=sum(dominates(config, q) for q in front))


def label_transfer_f1(Z, cell_types, batches, k=LABEL_K):
    """Spec 9 for one embedding: cell types present in only one batch are dropped (from the labels alone); each remaining batch is
    held out in turn, its cell types are predicted from the other batches with a k-nearest-neighbour classifier (Euclidean, equal
    weights), and the score is the macro F1 over the cell types present in the held-out batch."""
    Z, cell_types, batches = np.asarray(Z), np.asarray(cell_types), np.asarray(batches)
    P._require(len(Z) == len(cell_types) == len(batches), "embedding and labels must list the same cells")
    dropped = sorted(t for t in set(cell_types) if len(set(batches[cell_types == t])) == 1)
    keep = ~np.isin(cell_types, dropped)
    Z, cell_types, batches = Z[keep], cell_types[keep], batches[keep]
    per_batch = {}
    for batch in sorted(set(batches)):
        test = batches == batch
        predicted = KNeighborsClassifier(n_neighbors=k).fit(Z[~test], cell_types[~test]).predict(Z[test])
        per_batch[batch] = float(f1_score(cell_types[test], predicted, labels=sorted(set(cell_types[test])),
                                          average="macro", zero_division=0))
    return dict(per_batch=per_batch, mean=float(np.mean(list(per_batch.values()))) if per_batch else None,
                dropped=[str(t) for t in dropped], not_evaluable=not per_batch)


def read_latent(tag, dirs, sha):
    """One fit's saved embedding and labels; refused when the file is not the latent its score row was made from."""
    found = [os.path.join(d, f"{tag}.npz") for d in dirs if os.path.isfile(os.path.join(d, f"{tag}.npz"))]
    P._require(len(found) == 1, f"{len(found)} latent files {tag}.npz in {list(dirs)}, expected 1")
    P._require(P.sha256(found[0]) == sha[tag], f"{found[0]} is not the latent its score row was made from (latent_sha256 differs)")
    with np.load(found[0]) as d:
        return found[0], d["z"], d["celltype"], d["batch"]


def label_transfer_point(rows, latents, bstar):
    """Spec 9 for one point or method: label transfer on each seed's embedding, then the mean and SD over seeds, and how far the
    seed-mean batch score is from b*."""
    files, results = [], []
    for tag in rows.sort_values("seed_i").tag:
        path, z, celltype, batch = read_latent(tag, latents["dirs"], latents["sha"])
        files.append(path)
        results.append(label_transfer_f1(z, celltype, batch))
    P._require(len({tuple(r["dropped"]) for r in results}) == 1, "the seeds' embeddings carry different labels")
    f1 = [r["mean"] for r in results]
    ok = not results[0]["not_evaluable"]
    return dict(per_seed=results, mean=float(np.mean(f1)) if ok else None, sd=float(np.std(f1, ddof=1)) if ok else None,
                dropped=results[0]["dropped"], not_evaluable=not ok, batch_score=float(rows.B.mean()),
                distance=abs(float(rows.B.mean()) - bstar)), files


def transfer_labels(out, r4_cells, latents, cells):
    """Spec 9: per penalty, task and decoder at the strength R4 matches (grid-extension points count); plain scVI and scvi_adv as
    they are; and per decoder the mean over the real tasks that are not low-signal."""
    x1 = out[(out.experiment == "X1") & (out.status == "ok")]
    result, files = {}, set()
    for cond in (0, 1):
        by_task = {}
        for task in sorted(P.BPM.TASKS):
            pair = x1[(x1.task == task) & (x1.cond_i == cond)]
            bstar = cells[f"{task}|{cond}"]["bstar"]
            by_task[task] = {}
            for arm in (*P.ADV_ARMS, "none", "scvi_adv"):
                rows, lam = pair[pair.arm == arm], None
                if arm in P.ADV_ARMS:
                    lam = r4_cells[f"{task}|{cond}|{arm}"]["matched"]
                    rows = rows[rows.lamf == lam]
                if len(rows):        # scvi_adv has runs for the conditioned decoder only
                    by_task[task][arm], used = label_transfer_point(rows, latents, bstar)
                    by_task[task][arm]["lam"] = lam
                    files.update(used)
        summary = {}
        for arm in (*P.ADV_ARMS, "none", "scvi_adv"):
            real = [t for t in P.REAL_TASKS if arm in by_task[t] and not cells[f"{t}|{cond}"]["low_signal"]]
            means = [by_task[t][arm]["mean"] for t in real if not by_task[t][arm]["not_evaluable"]]
            summary[arm] = dict(mean=float(np.mean(means)) if means else None, n_tasks=len(means),
                                not_evaluable=[t for t in real if by_task[t][arm]["not_evaluable"]])
        result[str(cond)] = dict(tasks=by_task, summary=summary)
    return dict(k=LABEL_K, decoders=result, files=sorted(files))


def analyse(out, r1, r4, fail_status, latents=None, value="C", mode="mean", corner="collapse"):
    x1, x13 = out[out.experiment == "X1"], out[out.experiment == "X13"]
    tasks = sorted(P.BPM.TASKS)
    P._require(set(zip(x1.task, x1.cond_i)) == set(itertools.product(tasks, (0, 1)))
               and set(x1.task[x1.arm == "scvi_adv"]) == set(x13.task) == set(tasks),
               "X1 and X13 must contain all eight tasks, X1 both decoders and scvi_adv")
    for task, arm in itertools.product(tasks, (*TOOLS, "scanvi")):
        configs = tool_rows(x13, task, arm)
        want = 1 if arm == "scanvi" else 6
        P._require(configs.lam.nunique() == want and len(configs[["lam", "n_latent"]].drop_duplicates()) == want,
                   f"X13 {task} {arm}: expected {want} configurations")
    P._require("targets" in r4, "R4 lacks b0 targets")
    recomputed = P.r4_stage(out, "x1", r1)      # the rule that wrote the record, rerun on these scores (AN-03)
    targets = recomputed["targets"]
    for task, cond in itertools.product(tasks, (0, 1)):
        key = f"{task}|{cond}"
        P._require(key in r4["targets"] and {"b0", "b_common", "bstar", "bstar_low_signal"} <= set(r4["targets"][key]),
                   f"R4 lacks b0 target for {key}")
        P._require(sorted(zero_runs(out, task, cond)) == X1_SEEDS, f"{key}: plain-scVI seeds are not the X1 seeds 0-4")
        got, want = r4["targets"][key], targets[key]
        four = ("b0", "b_common", "bstar", "bstar_low_signal")
        P._require(got["bstar_low_signal"] == want["bstar_low_signal"]
                   and all(abs(got[k] - want[k]) <= P.TIE_TOL for k in ("b0", "b_common", "bstar")),
                   f"R4 targets (b0, b_common, b* or the low-signal flag) differ from these scores for {key}: record "
                   f"{[got[k] for k in four]}, recomputed {[want[k] for k in four]}; if R4 added extension rows, pass "
                   "their manifest as a further --manifest and their scores with --scores")
    P._require(value in ("C", "C_nt"), f"unknown biology value {value}")
    P._require(mode in ("mean", "drop_failed", "per_run"), f"unknown reading mode {mode}")
    out = out.assign(C=out[value])
    x1, x13 = out[out.experiment == "X1"], out[out.experiment == "X13"]
    cells, all_points, all_scores, plain_S = {}, {}, {}, {}
    for task, cond in itertools.product(tasks, (0, 1)):
        key = f"{task}|{cond}"
        target = r4["targets"][key]
        zero, zero_l, zero_g = (zero_runs(out, task, cond, v) for v in ("C", "L", "G"))
        rows = x1[(x1.task == task) & (x1.cond_i == cond)]
        points, local_points, global_points = {}, {}, {}
        for arm in P.ADV_ARMS:
            family = f"{P.FAMILY_OF[arm]}|{cond}"
            P._require("families" in r1 and family in r1["families"] and "x1_grid" in r1["families"][family],
                       f"missing registered grid {family}")
            grid = r1["families"][family]["x1_grid"]
            points[arm] = arm_points(rows, task, cond, arm, grid, sorted(zero))
            local_points[arm] = arm_points(rows, task, cond, arm, grid, sorted(zero), "L")
            global_points[arm] = arm_points(rows, task, cond, arm, grid, sorted(zero), "G")
        used = usable(points, mode)
        penalties = read_pair(zero, zero_l, zero_g, used, usable(local_points, mode), usable(global_points, mode),
                              target["bstar"], mode)
        if target["bstar_low_signal"]:
            penalties = {arm: {k: v for k, v in r.items() if k not in RANK_KEYS} for arm, r in penalties.items()}
        cell = dict(task=task, cond=cond, b0=target["b0"], b_common=target["b_common"], bstar=target["bstar"],
                    low_signal=target["bstar_low_signal"], penalties=penalties,
                    area=None if target["bstar_low_signal"] else
                    {arm: read_area(zero, used[arm], target["b_common"] - target["b0"], corner) for arm in P.ADV_ARMS})
        family_points = dict(points, **{tool: config_points(tool_rows(x13, task, tool)) for tool in TOOLS})
        scores = {arm: [(p["lam"], [W_B * b + W_C * p["C"][s] for s, b in p["B"].items()]) for p in pts]
                  for arm, pts in family_points.items()}
        all_points[key], all_scores[key] = family_points, scores
        plain_S[key] = float(np.mean([W_B * b + W_C * c for b, c in zero.values()]))
        cell["collapse"] = {arm: collapse_counts(rows[(rows.arm == arm) & rows.lamf.isin([p["lam"] for p in points[arm]])],
                                                zero, fail_status) for arm in P.ADV_ARMS}
        cell["window"] = {arm: window(scores[arm]) for arm in P.ADV_ARMS}
        family_scores = {arm: [float(np.mean(v)) for _, v in scores[arm] if v] for arm in P.ADV_ARMS}
        family_scores.update({arm: [] for arm in (*TOOLS, "scanvi", "none", "scvi_adv")})
        baselines = pd.concat([x13[x13.task == task], x1[(x1.task == task) &
                              (((x1.arm == "none") & (x1.cond_i == cond)) | (x1.arm == "scvi_adv"))]])
        single, frontier = {}, {}
        fronts = {arm: [(b, c) for _, b, c, _ in P.window_curve(points[arm])] for arm in P.ADV_ARMS}
        failed = {arm: [p["n_fail"] for p in points[arm]] for arm in P.ADV_ARMS}     # failed runs per configuration
        for (arm, lam, n_latent), config in baselines.groupby(["arm", "lam", "n_latent"], sort=True):
            anchor = arm == "pca" or (arm in ("harmony", "scanorama") and n_latent != "10")
            n_failed = int((config.status == "failed").sum())
            if not anchor:
                failed.setdefault(arm, []).append(n_failed)
            ok = config[config.status == "ok"]
            if ok.empty:
                continue
            b, c = float(ok.B.mean()), float(ok.C.mean())
            name = f"{arm}|{lam}|{n_latent}"
            if not anchor:
                family_scores[arm].append(W_B * b + W_C * c)
            if anchor or arm in ("none", "scvi_adv", "scanvi"):       # single settings, shown as they are (spec 10)
                single[name] = seed_summary(ok.B, ok.C, n_failed, plain_S[key])
            frontier[name] = {a: dominance((b, c), fronts[a]) for a in P.ADV_ARMS}
        # E_k does not penalise failures (a configuration scores the mean of its successful runs): counted (AN-13)
        E = {arm: best_of_k(v) for arm, v in family_scores.items()}
        cell["best_of_k"] = {arm: dict(n=len(v), E=E[arm], E_minus_plain_scvi=[e - plain_S[key] for e in E[arm]],     # spec 11
                                      configs=len(failed.get(arm, [])), failed_runs=sum(failed.get(arm, [])),
                                      stratum="label_aware" if arm == "scanvi" else "standard")
                             for arm, v in family_scores.items()}
        cell["single"], cell["frontier"] = single, frontier
        cells[key] = cell

    ranking, real_mean, transfer = {}, {}, {}
    for cond in (0, 1):
        ck = str(cond)
        pair = {task: cells[f"{task}|{cond}"] for task in tasks}
        ranked = {task: {arm: c["penalties"][arm]["rank"] for arm in P.ADV_ARMS} for task, c in pair.items() if not c["low_signal"]}
        mean = mean_ranks({task: r for task, r in ranked.items() if task in P.REAL_TASKS}, list(P.ADV_ARMS))
        ranking[ck] = dict(mean_rank={arm: dict(mean=m, n_tasks=n) for arm, (m, n) in mean.items()},
                           sim_ranks={task: r for task, r in ranked.items() if task not in P.REAL_TASKS},
                           low_signal=[task for task, c in pair.items() if c["low_signal"]])
        real_mean[ck] = {fam: best_of_k_real_mean({task: pair[task]["best_of_k"][fam]["E"] for task in P.REAL_TASKS})
                         for fam in pair[tasks[0]]["best_of_k"]}
        transfer[ck] = {}
        for fam in (*P.ADV_ARMS, *TOOLS):
            S = {task: {lam: float(np.mean(v)) for lam, v in all_scores[f"{task}|{cond}"][fam] if v} for task in tasks}
            fails = {task: {p["lam"]: p["n_fail"] for p in all_points[f"{task}|{cond}"][fam]} for task in tasks}
            chosen = transfer_lambdas(S, fails)
            transfer[ck][fam] = {}
            for task in tasks:
                def at(lam):
                    if lam is None:
                        return None
                    point = next(p for p in all_points[f"{task}|{cond}"][fam] if p["lam"] == lam)
                    return seed_summary(point["B"], point["C"], point["n_fail"], plain_S[f"{task}|{cond}"])

                best = choose_lambda(S, fails, [task])
                transfer[ck][fam][task] = dict(chosen[task], at=at(chosen[task]["lam"]), best_lam=best, best=at(best))
    return dict(settings=dict(value=value, mode=mode, corner=corner), cells=cells, ranking=ranking,
                best_of_k_real_mean=real_mean, transfer=transfer,
                label_transfer=None if latents is None else transfer_labels(out, recomputed["cells"], latents, cells))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--failures", required=True)
    ap.add_argument("--r1-record", required=True)
    ap.add_argument("--r4-record", required=True)
    ap.add_argument("--latents", nargs="+", required=True, help="directories of the X1 latent files <tag>.npz (label transfer)")
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
    P._require("latent_sha256" in S.columns, "scores lack latent_sha256: label transfer cannot tie the embeddings to the scores")
    primary = analyse(out, r1, r4, fail_status, dict(dirs=a.latents, sha=dict(zip(S.tag, S.latent_sha256))))
    sensitivity = {name: analyse(out, r1, r4, fail_status, **kw) for name, kw in SENSITIVITY.items()}   # label transfer uses none of them
    P.write_record(a.out_json, dict(rule="analysis_x1", status="done", spec="docs/ANALYSIS_SPEC.md",
                                  primary=primary, sensitivity=sensitivity),
                   dict(manifest=a.manifest, scores=score_files, failures=[a.failures], r1_record=[a.r1_record],
                        r4_record=[a.r4_record], latents=primary["label_transfer"]["files"]))
    print(f"X1 analysis: {len(primary['cells'])} pairs, {len(sensitivity)} sensitivities -> {a.out_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
