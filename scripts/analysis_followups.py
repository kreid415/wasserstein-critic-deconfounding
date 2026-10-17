#!/usr/bin/env python
"""Follow-up analyses X3, X6, X7, X8, X12 and X15 (docs/ANALYSIS_SPEC_FOLLOWUPS.md).

Must be committed and tagged before R4-x1 runs and before any X1 result is examined (SI-68, SI-78). It reads only
follow-up rows (seeds 10-12) and the targets (b0, b*, low-signal flag) of the frozen R4 stage-x1 record; it reuses
the frozen helpers of scripts/analysis_x1.py and scripts/prereg_rules.py without changing them.

Usage (repo root):
  python scripts/analysis_followups.py --manifest <resolved follow-up manifest, design v4> --scores <dir> [<dir> ...] \
      --failures <csv that covers every follow-up row> --r4-record docs/prereg/r4_x1_record.json \
      (--prepped-dir <dir of <task>__scib.h5ad> | --celltype-counts <csv: task, draw, batch, cell_type, n>) --out-json <followups.json>
With --prepped-dir the cell-type count table is built from the prepped obs ('batch', 'celltype') with the fitter's own X8 draw
and written next to the record as <out-json>.celltype_counts.csv.
"""
import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_x1 as A  # noqa: E402
import build_paper_manifest as BPM  # noqa: E402
import prereg_rules as P  # noqa: E402

FU_SEEDS = (10, 11, 12)
MIN_READINGS = 3              # all 3 leave-one-seed-out readings must exist (X1 needs 4 of 5, i.e. at most one missing)
MARGIN = P.DELTA_MIN          # r = 0.01 for every follow-up endpoint, in the endpoint's own unit
LEVELS = ("matched_lo", "matched", "matched_hi")
FIXED = ("matched_lo", "matched_hi")
REAL = set(P.REAL_TASKS)
# ---- design, as decided 2026-10-17 (SI-93..SI-98); names the build track must match (spec section 1.4)
EXPECTED = {
    "X3": {(t, c) for t in ("atac_small", "immune_hum_mou", "sim2", "pancreas") for c in (0, 1)},
    "X6": {(t, c) for t in ("atac_small", "immune", "pancreas") for c in (0, 1)},
    "X7": {(t, c) for t in ("pancreas", "sim1", "atac_small") for c in (0, 1)},
    "X8": {(t, c) for t in ("immune_hum_mou", "lung", "sim2") for c in (0, 1)},
    "X12": {(t, c) for t in ("atac_small", "immune", "pancreas") for c in (0, 1)},
    "X15": {(t, 1) for t in ("immune", "atac_large")},
}
EXTRA_KEYS = {"subsample", "sampler", "iw", "factorial_run", "adv_width", "adv_lr", "kl_warmup", "r1_gamma"}
X3_DOSES, X3_IW_DOSES = (0, 50, 80, 95, 100), (50, 80, 95)
X3_ARMS = ("discriminator", "reference", "pooled", "mmd")
X6_COPT = "discriminator_sn_critic_opt"   # SI-98: JS + spectral norm trained with the critic's Adam settings (followup-build)
X6_CONFIGS = {"js": ("discriminator", 1), "js_sn": ("discriminator_sn", 1), "js_r1": ("discriminator_r1", 1),
              "js_sn_copt": (X6_COPT, 1), "w1_gp_1": ("pooled", 1), "w1_gp_5": ("pooled", 5),
              "w1_gp_10": ("pooled", 10), "w1_sn_1": ("pooled_sn", 1), "w1_sn_5": ("pooled_sn", 5)}
X6_CONTRASTS = {   # name: (minuend, subtrahend); "divergence" is the confirmatory one (spec section 4)
    "divergence": ("w1_sn_1", "js_sn_copt"), "divergence_own_optimiser": ("w1_sn_1", "js_sn"),
    "js_optimiser": ("js_sn_copt", "js_sn"), "js_sn": ("js_sn", "js"), "js_r1": ("js_r1", "js"),
    "w1_sn_1": ("w1_sn_1", "w1_gp_1"), "w1_sn_5": ("w1_sn_5", "w1_gp_5"), "w1_steps_5": ("w1_gp_5", "w1_gp_1"),
    "w1_steps_10": ("w1_gp_10", "w1_gp_1"), "w1_sn_steps_5": ("w1_sn_5", "w1_sn_1"), "as_used": ("w1_gp_1", "js")}
X7_FORMS = ("reference", "reference_fixed", "mmd_ref", "discriminator_ref")
X7_COMPARATORS = ("discriminator_ref", "mmd_ref")    # first = primary (same target, JS); second = secondary
X7_CONTROL, X7_DEPLETION = "none", ("two_thirds", "one_third")   # levels of BPM.X7_DEPLETED; 'none' = same-size control
KINDS = {"composition": BPM.X3_DRAW, "batches": BPM.X8_DRAW, "reference_composition": BPM.X7_DRAW}   # subsample kind: draw
X7_MIN_COUNT = 10                                    # a cell type 'covers' when the reference holds >= 10 of its cells
X8_V = (2, 4, 8, 16)
X8_ARMS = ("none", "discriminator", "mmd", "reference", "pooled")
X12_FACTORS = ("A", "B", "C", "D", "E")             # n_latent 10/30, adv width 32/128, adv lr 1e-4/1e-3, batch 128/512, SCVI/LinearSCVI
X15_ARMS = ("discriminator", "pooled", "mmd")
TAG_LEVEL = r"_l(matched_lo|matched_hi|matched|0)_c[01]_s\d+_[0-9a-f]{8}$"


# ---------------------------------------------------------------------------------------------- inputs
def prepare(out, experiment):
    """Follow-up rows of one experiment with the design columns parsed by name from the tag and `extra`."""
    o = out[out.experiment == experiment].copy()
    P._require(len(o), f"{experiment}: no rows")
    got = set(zip(o.task, o.cond_i))
    P._require(got == EXPECTED[experiment], f"{experiment}: task x decoder cells {sorted(got)} != {sorted(EXPECTED[experiment])}")
    P._require(set(o.seed_i) == set(FU_SEEDS), f"{experiment}: seeds {sorted(set(o.seed_i))} != {list(FU_SEEDS)} (SI-26)")
    o["level"] = o.tag.str.extract(TAG_LEVEL)[0]
    P._require(o.level.notna().all(), f"{experiment}: tags without a lambda label, e.g. {o.tag[o.level.isna()].tolist()[:3]}")
    ex = [json.loads(e) if e not in ("", "{}") else {} for e in o.extra]
    bad = sorted({k for e in ex for k in e} - EXTRA_KEYS)
    P._require(not bad, f"{experiment}: unknown extra keys {bad} (spec section 1.4)")
    sub = [e.get("subsample", {}) for e in ex]
    kinds = sorted({(s.get("kind"), s.get("draw")) for s in sub if s} - set(KINDS.items()))
    P._require(not kinds, f"{experiment}: unknown subsample kinds or draws {kinds} (known: {KINDS})")
    o["dose"] = [s["deplete_pct"] if s.get("kind") == "composition" else -1 for s in sub]
    o["iw"] = [e.get("iw") == "depletion_oracle" for e in ex]
    o["V"] = [s["n_batches"] if s.get("kind") == "batches" else 0 for s in sub]
    o["chain"] = [s["subset"] if s.get("kind") == "batches" else -1 for s in sub]
    o["sampler"] = [e.get("sampler", "default") for e in ex]
    o["run"] = [e.get("factorial_run", -1) for e in ex]
    o["adv_width"] = [e.get("adv_width", 0) for e in ex]
    o["adv_lr"] = [e.get("adv_lr", 0.0) for e in ex]
    o["kl"] = [e.get("kl_warmup", "") for e in ex]
    o["ref_level"] = [x7_level(t, ref, s) for t, ref, s in zip(o.task, o.reference, sub)]
    o["k"] = o.n_critic.astype(int)
    return o


def x7_level(task, reference, spec):
    """Depletion level of an X7 row from its removed-type list (BPM.X7_DEPLETED); 'natural' for the full-task rows."""
    if spec.get("kind") != "reference_composition":
        return "natural"
    P._require(task in BPM.X7_DEPLETED, f"X7 {task}: no depleted-reference table entry")
    batch, n_ref, _, levels = BPM.X7_DEPLETED[task]
    P._require(spec["batch"] == batch == reference and int(spec["n_reference"]) == n_ref,
               f"X7 {task}: depleted spec {spec} does not match the table (reference {batch}, n_reference {n_ref})")
    hit = [lv for lv, types in levels.items() if sorted(types) == sorted(spec["types"])]
    P._require(len(hit) == 1, f"X7 {task}: removed types {spec['types']} match no level of the table")
    return hit[0]


def seed_values(rows, value="C"):
    """{seed: value} over the successful runs of one design point; the point must hold exactly seeds 10-12."""
    P._require(sorted(rows.seed_i) == list(FU_SEEDS), f"point {rows.tag.tolist()[:2]}: seeds {sorted(rows.seed_i)} != {list(FU_SEEDS)}")
    ok = rows[rows.status == "ok"]
    return {int(s): float(v) for s, v in zip(ok.seed_i, ok[value])}


def mean(vals, keep):
    xs = [vals[s] for s in keep if s in vals]
    return float(np.mean(xs)) if xs else None


def loso(stat):
    """Average first (SI-69): `stat(keep)` on all seeds, then once without each seed; jackknife standard error
    s^2 = ((m - 1)/m) sum (d_i - mean d)^2 over the m readings that exist (ANALYSIS_SPEC section 2.4)."""
    full = stat(list(FU_SEEDS))
    readings = [stat([s for s in FU_SEEDS if s != o]) for o in FU_SEEDS]
    existing = np.array([x for x in readings if x is not None], dtype=float)
    m = len(existing)
    reason = "not_readable" if full is None else ("readings" if m < MIN_READINGS else None)
    s = None if reason else float(np.sqrt((m - 1) / m * np.sum((existing - existing.mean()) ** 2)))
    return dict(d=full, s=s, m=m, readings=readings, reason=reason)


def slope(x, y):
    """Ordinary least-squares slope of y on x; None when any y is missing."""
    if any(v is None for v in y):
        return None
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))


def pooled(cells, tasks):
    """The ANALYSIS_SPEC section 4 model over the evaluable real datasets; simulations are never pooled.
    `cells`: {task: loso result}. With 1-2 datasets the result is inconclusive by construction (AN-R12)."""
    keep = [t for t in sorted(tasks) if t in REAL and cells[t]["reason"] is None]
    res = A.combine(keep, [cells[t]["d"] for t in keep], [cells[t]["s"] for t in keep], "family", A.PRIOR_SCALE, MARGIN)
    if 0 < len(keep) < 3:     # user answer 2026-10-17: report per dataset, no verdict, with 1-2 real datasets
        res = dict(res, model_declaration=res["declaration"], declaration="not_decidable_1_2_real_datasets")
    p = A.wilcoxon_exact([cells[t]["d"] for t in keep]) if keep else None
    not_eval = {t: cells[t]["reason"] for t in sorted(tasks) if t in REAL and cells[t]["reason"] is not None}
    return dict(res, wilcoxon_p=p, not_evaluable=not_eval,
                sign_table={t: sign_row(cells[t]) for t in sorted(tasks)})


def sign_row(c):
    if c["reason"] is not None:
        return dict(d=c["d"], s=None, reason=c["reason"])
    half = stats.t.ppf(0.975, c["m"] - 1) * c["s"]
    return dict(d=c["d"], s=c["s"], lo=c["d"] - half, hi=c["d"] + half,
                sign=int(np.sign(c["d"])) if abs(c["d"]) > MARGIN else 0)


def collapse_rate(runs):
    """runs: list of (failed, collapsed) booleans; collapsed = failed or C <= same-seed lambda=0 C - 0.10."""
    x, n = sum(1 for f, c in runs if f or c), len(runs)
    lo, hi = A.clopper_pearson(x, n)
    return dict(x=x, n=n, n_failed=sum(1 for f, _ in runs if f), rate=x / n, lo=lo, hi=hi)


def runs_vs_zero(rows, zero):
    out = []
    for r in rows.itertuples():
        failed = r.status == "failed"
        out.append((failed, (not failed) and r.C <= zero[int(r.seed_i)] - P.COLLAPSE_BIO + P.TIE_TOL))
    return out


def counts_by(counts, task, draw):
    c = counts[(counts.task == task) & (counts.draw == draw)]
    P._require(len(c), f"cell-type counts: no rows for {task} draw {draw!r}")
    return c


def coverage(counts, task, draw, batch):
    """P4 coverage (D-09): share of the task's cell types (draw 'full') present in `batch` of `draw` with at least
    X7_MIN_COUNT cells; also the batch's size and cell-type entropy (natural log)."""
    types = set(counts_by(counts, task, "full").cell_type)
    b = counts_by(counts, task, draw)
    b = b[b.batch == batch]
    P._require(len(b), f"cell-type counts: {task} draw {draw!r} has no batch {batch!r}")
    p = b.n.to_numpy(dtype=float) / b.n.sum()
    p = p[p > 0]
    return dict(coverage=len(set(b.cell_type[b.n >= X7_MIN_COUNT])) / len(types), size=int(b.n.sum()),
                entropy=float(-(p * np.log(p)).sum()))


# ---------------------------------------------------------------------------------------------- X3
def x3(out):
    o = prepare(out, "X3")
    cells, contrast, pools = {}, {}, {}
    for t, c in sorted(EXPECTED["X3"]):
        r = o[(o.task == t) & (o.cond_i == c)]
        zero_rows = {d: r[(r.arm == "none") & (r.dose == d)] for d in X3_DOSES}
        for d, z in zero_rows.items():
            P._require((z.status == "ok").all(), f"X3 {t}|{c} dose {d}: lambda=0 fits failed; stock scVI must not fail")
        zero = {v: {d: seed_values(z, v) for d, z in zero_rows.items()} for v in ("B", "C")}
        pt = {(a, lv, d, w): r[(r.arm == a) & (r.level == lv) & (r.dose == d) & (r.iw == w)]
              for a in X3_ARMS for lv in FIXED for d in X3_DOSES for w in (False, True) if not w or d in X3_IW_DOSES}
        vals = {k: {v: seed_values(rows, v) for v in ("B", "C")} for k, rows in pt.items()}
        x = [d / 100 for d in X3_DOSES]

        def arm_slope(a, lv, v):
            def stat(keep):
                y = []
                for d in X3_DOSES:
                    m1, m0 = mean(vals[(a, lv, d, False)][v], keep), mean(zero[v][d], keep)
                    y.append(None if m1 is None else m1 - m0)
                return slope(x, y)
            return stat

        def recovery(a, lv):
            def stat(keep):
                g = [(mean(vals[(a, lv, d, True)]["C"], keep), mean(vals[(a, lv, d, False)]["C"], keep)) for d in X3_IW_DOSES]
                return None if any(w is None or u is None for w, u in g) else float(np.mean([w - u for w, u in g]))
            return stat

        cell = {}
        for a, lv in ((a, lv) for a in X3_ARMS for lv in FIXED):
            cell[f"{a}|{lv}"] = dict(
                slope_C=loso(arm_slope(a, lv, "C")), slope_B=loso(arm_slope(a, lv, "B")), iw_recovery=loso(recovery(a, lv)),
                collapse={d: collapse_rate(runs_vs_zero(pt[(a, lv, d, False)], zero["C"][d])) for d in X3_DOSES})
        for lv in FIXED:
            ref, dis = arm_slope("reference", lv, "C"), arm_slope("discriminator", lv, "C")
            contrast[(t, c, lv)] = loso(lambda keep: None if ref(keep) is None or dis(keep) is None else ref(keep) - dis(keep))
        cells[f"{t}|{c}"] = cell
    for c in (0, 1):
        for lv in FIXED:
            tasks = [t for t, cc in EXPECTED["X3"] if cc == c]
            pools[f"slope_contrast|{c}|{lv}"] = pooled({t: contrast[(t, c, lv)] for t in tasks}, tasks)
            for a in X3_ARMS:
                pools[f"slope_{a}|{c}|{lv}"] = pooled({t: cells[f"{t}|{c}"][f"{a}|{lv}"]["slope_C"] for t in tasks}, tasks)
    return dict(cells=cells, contrast={f"{t}|{c}|{lv}": v for (t, c, lv), v in contrast.items()}, pooled=pools)


# ---------------------------------------------------------------------------------------------- reading at b*
def window_points(rows):
    """The lo / matched / hi points of one arm in one cell, ordered by lambda: [(lam, {seed: B}, {seed: C}, n_fail)]."""
    pts = []
    for lv in LEVELS:
        g = rows[rows.level == lv]
        P._require(len(g) and g.lamf.nunique() == 1, f"{rows.tag.tolist()[:2]}: lambda label {lv} missing or holding several values")
        pts.append((float(g.lamf.iloc[0]), seed_values(g, "B"), seed_values(g, "C"), int((g.status == "failed").sum())))
    return sorted(pts, key=lambda p: p[0])


def read_bstar(pts, bstar, keep):
    """bio@b* on the lambda-ordered window (the R2/R3 reading, prereg_rules.bio_at_bstar); None unless bracketed."""
    curve = [(lam, mean(b, keep), mean(cc, keep), nf) for lam, b, cc, nf in pts if mean(b, keep) is not None]
    r = P.bio_at_bstar(curve, bstar)
    return r["bio"] if r["status"] == "bracketed" else None


def bstar_status(pts, bstar):
    curve = [(lam, mean(b, FU_SEEDS), mean(cc, FU_SEEDS), nf) for lam, b, cc, nf in pts if b]
    return P.bio_at_bstar(curve, bstar)["status"]


def at_level(pts, rows, level, bstar=None):
    """Fixed-lambda summary of one level: mean C, mean B and B - b* (I048: reported for every variant cell)."""
    lam = float(rows[rows.level == level].lamf.iloc[0])
    _, b, cc, nf = next(p for p in pts if p[0] == lam)
    B = mean(b, FU_SEEDS)
    return dict(lam=lam, C=mean(cc, FU_SEEDS), B=B, n_fail=nf, B_minus_bstar=None if bstar is None or B is None else B - bstar)


def bstar_contrast(pa, pb, bstar, low):
    if low:
        return dict(d=None, s=None, m=0, readings=[], reason="low_signal")
    r = loso(lambda keep: None if read_bstar(pa, bstar, keep) is None or read_bstar(pb, bstar, keep) is None
             else read_bstar(pa, bstar, keep) - read_bstar(pb, bstar, keep))
    if r["d"] is None:
        r["reason"] = "unreached"
    return r


def bio_window(pts):
    """R3 window width on the bio score (D-09): points whose mean C is at least the lower 95% t bound of the best point."""
    rows = [(lam, list(cc.values())) for lam, _, cc, _ in pts if len(cc)]
    best = max(rows, key=lambda r: (np.mean(r[1]), -r[0]))
    if len(best[1]) < 2:
        return dict(width=None, n_points=None, reason="best point has fewer than 2 runs")
    bound = np.mean(best[1]) - stats.t.ppf(0.975, len(best[1]) - 1) * np.std(best[1], ddof=1) / math.sqrt(len(best[1]))
    inside = [lam for lam, v in rows if np.mean(v) >= bound - P.TIE_TOL]
    return dict(width=math.log10(max(inside)) - math.log10(min(inside)), n_points=len(inside), bound=float(bound))


# ---------------------------------------------------------------------------------------------- X6
def x6(out, targets):
    o = prepare(out, "X6")
    have = {(a, k) for a, k in zip(o.arm, o.k)}
    P._require(have == set(X6_CONFIGS.values()), f"X6 arms x steps {sorted(have)} != {sorted(X6_CONFIGS.values())}")
    cells, contrast, pools = {}, {}, {}
    for t, c in sorted(EXPECTED["X6"]):
        tg = targets[f"{t}|{c}"]
        r = o[(o.task == t) & (o.cond_i == c)]
        rows = {n: r[(r.arm == a) & (r.k == k)] for n, (a, k) in X6_CONFIGS.items()}
        pts = {n: window_points(g) for n, g in rows.items()}
        cells[f"{t}|{c}"] = {n: dict(status=bstar_status(pts[n], tg["bstar"]), window=bio_window(pts[n]),
                                     **{lv: at_level(pts[n], rows[n], lv, tg["bstar"]) for lv in LEVELS})
                             for n in X6_CONFIGS}
        for name, (a, b) in X6_CONTRASTS.items():
            res = bstar_contrast(pts[a], pts[b], tg["bstar"], tg["bstar_low_signal"])
            fa, fb = cells[f"{t}|{c}"][a]["matched"], cells[f"{t}|{c}"][b]["matched"]
            res["fixed_matched"] = dict(dC=None if None in (fa["C"], fb["C"]) else fa["C"] - fb["C"],
                                        dB=None if None in (fa["B"], fb["B"]) else fa["B"] - fb["B"])
            contrast[(t, c, name)] = res
    for c in (0, 1):
        tasks = [t for t, cc in EXPECTED["X6"] if cc == c]
        for name in X6_CONTRASTS:
            pools[f"{name}|{c}"] = pooled({t: contrast[(t, c, name)] for t in tasks}, tasks)
    return dict(cells=cells, contrast={f"{t}|{c}|{n}": v for (t, c, n), v in contrast.items()}, pooled=pools)


# ---------------------------------------------------------------------------------------------- X7
def ols(X, y):
    """Coefficients and classical standard errors of y on X (X has an intercept column); df = n - p."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    if n <= p or np.linalg.matrix_rank(X) < p:     # no residual degrees of freedom, or a predictor that does not vary
        return beta, None
    sigma2 = float(np.sum((y - X @ beta) ** 2) / (n - p))
    return beta, np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))


def x7(out, targets, counts):
    o = prepare(out, "X7")
    P._require(set(o.arm) == set(X7_FORMS), f"X7 arms {sorted(set(o.arm))} != {sorted(X7_FORMS)}")
    cells, assoc, causal, pools = {}, {}, {}, {}
    for t, c in sorted(EXPECTED["X7"]):
        tg = targets[f"{t}|{c}"]
        names = sorted(set(counts_by(counts, t, "full").batch))   # X7 references are indices into the sorted names
        r = o[(o.task == t) & (o.cond_i == c)]
        nat = r[r.ref_level == "natural"]
        refs = sorted(set(nat.reference), key=int)
        P._require([int(x) for x in refs] == list(range(len(names))), f"X7 {t}: references {refs} != every batch")
        pts = {(f, ref): window_points(nat[(nat.arm == f) & (nat.reference == ref)]) for f in X7_FORMS for ref in refs}
        cov = {ref: coverage(counts, t, "full", names[int(ref)]) for ref in refs}
        bio = {k: (None if tg["bstar_low_signal"] else read_bstar(p, tg["bstar"], FU_SEEDS)) for k, p in pts.items()}
        cell = {f"{f}|{ref}": dict(bio_at_bstar=bio[(f, ref)], status=bstar_status(pts[(f, ref)], tg["bstar"]),
                                   n_failed=sum(p[3] for p in pts[(f, ref)]),
                                   fixed_matched=at_level(pts[(f, ref)], nat[(nat.arm == f) & (nat.reference == ref)],
                                                          "matched", tg["bstar"]), **cov[ref])
                for f, ref in pts}
        spread = {}
        for f in X7_FORMS:
            ys = [bio[(f, ref)] for ref in refs if bio[(f, ref)] is not None]
            # seed-noise floor: variance of a 3-run mean of C, averaged over the window points and references
            seed_var = [np.var(list(p[2].values()), ddof=1) / len(p[2]) for ref in refs for p in pts[(f, ref)] if len(p[2]) > 1]
            spread[f] = dict(n=len(ys), between_sd=float(np.std(ys, ddof=1)) if len(ys) > 1 else None,
                             seed_floor_sd=float(math.sqrt(np.mean(seed_var))))
        for comp in X7_COMPARATORS:
            use = [ref for ref in refs if bio[("reference", ref)] is not None and bio[(comp, ref)] is not None]
            res = dict(n_refs=len(use), excluded=[ref for ref in refs if ref not in use], d=None, s=None, m=len(use))
            if tg["bstar_low_signal"]:
                res["reason"] = "low_signal"
            elif len(use) < 3:
                res["reason"] = "fewer_than_3_references"
            else:
                y = [bio[("reference", ref)] - bio[(comp, ref)] for ref in use]
                X = [[1.0, cov[ref]["coverage"]] for ref in use]
                beta, se = ols(X, y)
                res.update(d=float(beta[1]) if se is not None else None, s=None if se is None else float(se[1]),
                           m=len(use) - 1, reason=None if se is not None else "coverage_does_not_vary")   # m - 1 = OLS df n - 2
                if len(use) >= 6:      # covariate-adjusted (size, entropy): descriptive, only with >= 6 references
                    Xa = [[1.0, cov[ref]["coverage"], math.log(cov[ref]["size"]), cov[ref]["entropy"]] for ref in use]
                    beta_a, se_a = ols(Xa, y)
                    res["adjusted_coverage_coef"] = None if se_a is None else float(beta_a[1])   # None: collinear covariates
            assoc[(t, c, comp)] = res
        dep, dcell = r[r.ref_level != "natural"], {}
        P._require(len(dep) == 0 or (set(dep.ref_level) == {X7_CONTROL, *X7_DEPLETION} and set(dep.arm) == set(X7_FORMS)),
                   f"X7 {t}|{c}: depleted rows need levels {X7_CONTROL}, {X7_DEPLETION} and every formulation")
        if len(dep):
            # SI-96: each depleted level against the same-size control (level 'none'), so only coverage differs
            # read at the full task's b* (SI-95; spec 1.4, review F10), with a fixed-'matched' fallback for every contrast
            drows = {(f, lv): dep[(dep.arm == f) & (dep.ref_level == lv)] for f in X7_FORMS for lv in (X7_CONTROL,) + X7_DEPLETION}
            dpts = {k: window_points(g) for k, g in drows.items()}
            fixed = {k: drows[k][drows[k].level == "matched"] for k in drows}
            fixed = {k: {v: seed_values(g, v) for v in ("B", "C")} for k, g in fixed.items()}
            dcell = {f"{f}|{lv}": dict(status=bstar_status(dpts[(f, lv)], tg["bstar"]),
                                       fixed_matched=at_level(dpts[(f, lv)], drows[(f, lv)], "matched", tg["bstar"]))
                     for f, lv in drows}
            removed = {lv: len(types) for lv, types in BPM.X7_DEPLETED[t][3].items()}
            for comp, lv in itertools.product(X7_COMPARATORS, X7_DEPLETION):
                def gap(keep, level, comp=comp):
                    a, b = read_bstar(dpts[("reference", level)], tg["bstar"], keep), read_bstar(dpts[(comp, level)], tg["bstar"], keep)
                    return None if a is None or b is None else a - b

                def stat(keep, comp=comp, lv=lv):
                    g1, g0 = gap(keep, lv, comp), gap(keep, X7_CONTROL, comp)
                    return None if g1 is None or g0 is None else g1 - g0

                def fixed_stat(v, comp=comp, lv=lv):
                    def s(keep):
                        m = [mean(fixed[(f, level)][v], keep) for level in (lv, X7_CONTROL) for f in ("reference", comp)]
                        return None if None in m else (m[0] - m[1]) - (m[2] - m[3])
                    return s
                res = dict(reason="low_signal", d=None, s=None, m=0) if tg["bstar_low_signal"] else loso(stat)
                if res["d"] is None and res["reason"] != "low_signal":
                    res["reason"] = "unreached"
                causal[(t, c, f"{lv}_{comp}")] = dict(res, reference=BPM.X7_DEPLETED[t][0], n_reference=BPM.X7_DEPLETED[t][1],
                                                      types_removed=removed[lv],
                                                      fixed_matched=dict(C=loso(fixed_stat("C")), dB=fixed_stat("B")(list(FU_SEEDS))))
        cells[f"{t}|{c}"] = dict(points=cell, spread=spread, depleted_points=dcell)
    for c in (0, 1):
        tasks = [t for t, cc in EXPECTED["X7"] if cc == c]
        for comp in X7_COMPARATORS:
            pools[f"association_{comp}|{c}"] = pooled({t: assoc[(t, c, comp)] for t in tasks}, tasks)
            for lv in X7_DEPLETION:
                dtasks = [t for t in tasks if (t, c, f"{lv}_{comp}") in causal]
                pools[f"depleted_{lv}_{comp}|{c}"] = pooled({t: causal[(t, c, f"{lv}_{comp}")] for t in dtasks}, dtasks)
    key = lambda d: {f"{t}|{c}|{k}": v for (t, c, k), v in d.items()}  # noqa: E731
    return dict(cells=cells, association=key(assoc), depleted=key(causal), pooled=pools)


# ---------------------------------------------------------------------------------------------- X8
def ilisi_ceiling(n_per_batch):
    """Largest scIB-scaled iLISI, (1/sum p_b^2 - 1)/(V - 1), for batch proportions p_b (1 when batches are equal)."""
    p = np.asarray(n_per_batch, dtype=float) / np.sum(n_per_batch)
    return float((1 / np.sum(p ** 2) - 1) / (len(p) - 1))


def x8(out, counts):
    o = prepare(out, "X8")
    P._require(set(o.V) == set(X8_V), f"X8 batch counts {sorted(set(o.V))} != {list(X8_V)}")
    cells, slopes, pools = {}, {}, {}
    for t, c in sorted(EXPECTED["X8"]):
        r = o[(o.task == t) & (o.cond_i == c)]
        chains = sorted(set(r.chain))
        P._require(len(chains) >= 1, f"X8 {t}: no draw chains")
        draws, cov = {}, {}
        for ch in chains:
            base = counts_by(counts, t, f"x8_V2_s{ch}")
            p2 = base.groupby("cell_type").n.sum() / base.n.sum()
            for V in X8_V:
                d = counts_by(counts, t, f"x8_V{V}_s{ch}")
                per = d.groupby("batch").n.sum()
                P._require(len(per) == V, f"X8 {t} V={V} chain {ch}: {len(per)} batches in the counts table")
                p = d.groupby("cell_type").n.sum() / d.n.sum()
                draws[(ch, V)] = set(per.index)
                cov[(ch, V)] = dict(total_cells=int(per.sum()), ceiling=ilisi_ceiling(per.to_numpy()),
                                    cells_per_batch_per_minibatch=float(int(r.batch_size.iloc[0]) / V),
                                    tv_from_V2=float(0.5 * p.sub(p2, fill_value=0).abs().sum()))
        identical = [V for V in X8_V if len(chains) > 1 and len({frozenset(draws[(ch, V)]) for ch in chains}) == 1]
        vals = {}
        for (ch, V, a, smp), g in r.groupby(["chain", "V", "arm", "sampler"]):
            vals[(ch, V, a, smp)] = {v: seed_values(g, v) for v in ("B", "C", "iLISI")}
            vals[(ch, V, a, smp)]["iLISI_cn"] = {s: x / cov[(ch, V)]["ceiling"] for s, x in vals[(ch, V, a, smp)]["iLISI"].items()}
        P._require({(a, s) for _, _, a, s in vals} == {(a, "default") for a in X8_ARMS} | {("reference", "stratified"), ("pooled", "stratified")},
                   f"X8 {t}|{c}: arms x samplers {sorted({(a, s) for _, _, a, s in vals})}")

        def gap(a, b, v):
            def stat(keep):
                per_chain = []
                for ch in chains:
                    g = []
                    for V in X8_V:
                        ma, mb = mean(vals[(ch, V) + a][v], keep), mean(vals[(ch, V) + b][v], keep)
                        g.append(None if ma is None or mb is None else ma - mb)
                    per_chain.append(slope(np.log2(X8_V), g))
                return None if any(x is None for x in per_chain) else float(np.mean(per_chain))
            return stat

        D, R, Pl = ("discriminator", "default"), ("reference", "default"), ("pooled", "default")
        named = {"ref_dis": (R, D), "pooled_dis": (Pl, D), "ref_strat_default": (("reference", "stratified"), R),
                 "pooled_strat_default": (("pooled", "stratified"), Pl)}
        for name, (a, b) in named.items():
            for v in ("iLISI_cn", "B", "C"):
                slopes[(t, c, f"{name}|{v}")] = loso(gap(a, b, v))
        # exploratory (SI-95, I078): bio gap on log2 V with the achieved batch-score gap and the composition distance from
        # the V = 2 draw as covariates; point means, no interval; None when the covariates are collinear
        pts = [(math.log2(V), mean(vals[(ch, V) + R]["B"], FU_SEEDS) - mean(vals[(ch, V) + D]["B"], FU_SEEDS),
                cov[(ch, V)]["tv_from_V2"], mean(vals[(ch, V) + R]["C"], FU_SEEDS) - mean(vals[(ch, V) + D]["C"], FU_SEEDS))
               for ch in chains for V in X8_V]
        adj, adj_se = ols([[1.0, x, gb, tv] for x, gb, tv, _ in pts], [gc for *_, gc in pts])
        zero = {(ch, V): vals[(ch, V, "none", "default")]["C"] for ch in chains for V in X8_V}
        coll = {f"{a}|{smp}": collapse_rate([x for ch in chains for V in X8_V for x in runs_vs_zero(
                    r[(r.chain == ch) & (r.V == V) & (r.arm == a) & (r.sampler == smp)], zero[(ch, V)])])
                for (a, smp) in {(a, s) for _, _, a, s in vals} if a != "none"}
        cells[f"{t}|{c}"] = dict(covariates={f"{ch}|{V}": v for (ch, V), v in cov.items()}, identical_chains_at_V=identical,
                                 exploratory_adjusted=None if adj_se is None else dict(log2V=float(adj[1]), batch_gap=float(adj[2]), tv=float(adj[3])), collapse=coll,
                                 B_minus_none={f"{ch}|{V}|{a}|{s}": mean(vals[(ch, V, a, s)]["B"], FU_SEEDS) - mean(vals[(ch, V, "none", "default")]["B"], FU_SEEDS)
                                               for (ch, V, a, s) in vals if a != "none"})
    for c in (0, 1):
        tasks = [t for t, cc in EXPECTED["X8"] if cc == c]
        for key in {k for _, _, k in slopes}:
            pools[f"{key}|{c}"] = pooled({t: slopes[(t, c, key)] for t in tasks}, tasks)
    return dict(cells=cells, slopes={f"{t}|{c}|{k}": v for (t, c, k), v in slopes.items()}, pooled=pools)


# ---------------------------------------------------------------------------------------------- X12
def x12_codes(r):
    """+1/-1 codes of A..E from a row's settings; the run index and E = ABCD (resolution V, SI-97) are checked."""
    a = dict(A=int(r.n_latent) == 30, B=int(r.adv_width) == 128, C=math.isclose(float(r.adv_lr), 1e-3),
             D=int(r.batch_size) == 512, E=r.decoder == "LinearSCVI")
    run = int(r.run)
    bits = dict(A=run & 1, B=(run >> 1) & 1, C=(run >> 2) & 1, D=(run >> 3) & 1)
    P._require(all(int(a[k]) == bits[k] for k in bits) and int(a["E"]) == bits["A"] ^ bits["B"] ^ bits["C"] ^ bits["D"],
               f"{r.tag}: settings do not match run {run} of the 2^(5-1) design with E = ABCD")
    return {k: 1 if v else -1 for k, v in a.items()}


def x12(out, targets):
    o = prepare(out, "X12")
    P._require(set(o.arm) == {"discriminator", "reference"} and set(o.run) == set(range(16)),
               f"X12: arms {sorted(set(o.arm))}, runs {sorted(set(o.run))} (16 runs, SI-97)")
    codes = {}
    for r in o.itertuples():
        codes.setdefault(int(r.run), x12_codes(r))
        P._require(codes[int(r.run)] == x12_codes(r), f"{r.tag}: run {r.run} has other settings than its first row")
    terms = list(X12_FACTORS) + [a + b for i, a in enumerate(X12_FACTORS) for b in X12_FACTORS[i + 1:]]
    xcode = {run: {k: int(np.prod([cd[f] for f in k])) for k in terms} for run, cd in codes.items()}
    cells, effects, pools = {}, {}, {}
    for t, c in sorted(EXPECTED["X12"]):
        tg = targets[f"{t}|{c}"]
        r = o[(o.task == t) & (o.cond_i == c)]
        for lv in FIXED:
            per_run, resid, df = {}, 0.0, 0
            for run in range(16):
                g = {a: r[(r.run == run) & (r.arm == a) & (r.level == lv)] for a in ("reference", "discriminator")}
                cv = {a: seed_values(x, "C") for a, x in g.items()}
                bv = {a: seed_values(x, "B") for a, x in g.items()}
                paired = [cv["reference"][s] - cv["discriminator"][s] for s in FU_SEEDS if s in cv["reference"] and s in cv["discriminator"]]
                per_run[run] = dict(d=float(np.mean(paired)) if paired else None, n=len(paired),
                                    n_failed={a: int((x.status == "failed").sum()) for a, x in g.items()},
                                    dB=None if not bv["reference"] or not bv["discriminator"] else
                                    mean(bv["reference"], FU_SEEDS) - mean(bv["discriminator"], FU_SEEDS),
                                    B_minus_bstar={a: None if not bv[a] else mean(bv[a], FU_SEEDS) - tg["bstar"] for a in bv})
                resid += float(np.sum((np.array(paired) - np.mean(paired)) ** 2)) if paired else 0.0
                df += max(len(paired) - 1, 0)
            ds = [v["d"] for v in per_run.values()]
            stable = dict(n_above=sum(d is not None and d > MARGIN for d in ds), n_below=sum(d is not None and d < -MARGIN for d in ds),
                          n_within=sum(d is not None and abs(d) <= MARGIN for d in ds), n_missing=sum(d is None for d in ds))
            stable["sign_stable"] = stable["n_missing"] == 0 and (stable["n_above"] == 0 or stable["n_below"] == 0)
            for k in terms:
                key = (t, c, lv, k)
                if stable["n_missing"]:
                    effects[key] = dict(d=None, s=None, m=0, reason="run_without_paired_seeds")
                    continue
                P._require(df > 0, f"X12 {t}|{c}|{lv}: no run has two paired seeds (no pure error)")
                plus = [run for run in range(16) if xcode[run][k] == 1]
                minus = [run for run in range(16) if xcode[run][k] == -1]
                e = np.mean([per_run[x]["d"] for x in plus]) - np.mean([per_run[x]["d"] for x in minus])
                var = resid / df * (sum(1 / per_run[x]["n"] for x in plus) / 64 + sum(1 / per_run[x]["n"] for x in minus) / 64)
                effects[key] = dict(d=float(e), s=float(math.sqrt(var)), m=df + 1, reason=None)
            cells[f"{t}|{c}|{lv}"] = dict(runs=per_run, sign=stable, pure_error_df=df)
    for c in (0, 1):
        tasks = [t for t, cc in EXPECTED["X12"] if cc == c]
        for lv in FIXED:
            for k in X12_FACTORS:     # main effects pooled (secondary); two-factor interactions per dataset only
                pools[f"{k}|{c}|{lv}"] = pooled({t: effects[(t, c, lv, k)] for t in tasks}, tasks)
    return dict(cells=cells, codes={str(k): v for k, v in codes.items()},
                effects={f"{t}|{c}|{lv}|{k}": v for (t, c, lv, k), v in effects.items()}, pooled=pools)


# ---------------------------------------------------------------------------------------------- X15
def x15(out, targets):
    o = prepare(out, "X15")
    P._require(set(o.kl) == {"stock", "complete"}, f"X15 warm-up values {sorted(set(o.kl))}")
    res, pools, batch, coll = {}, {}, {}, {}
    for t, c in sorted(EXPECTED["X15"]):
        r = o[(o.task == t) & (o.cond_i == c)]
        v = {}
        for (kl, a, lv), g in r[r.arm != "none"].groupby(["kl", "arm", "level"]):   # I048: achieved B - b* of every point
            batch[f"{t}|{c}|{kl}|{a}|{lv}"] = mean(seed_values(g, "B"), FU_SEEDS) - targets[f"{t}|{c}"]["bstar"]
        for kl in ("stock", "complete"):
            z = r[(r.arm == "none") & (r.kl == kl)]
            P._require((z.status == "ok").all(), f"X15 {t}|{kl}: lambda=0 fits failed")
            v[(kl, "none", "0")] = seed_values(z)
            for a in X15_ARMS:
                for lv in FIXED:
                    g = r[(r.arm == a) & (r.kl == kl) & (r.level == lv)]
                    v[(kl, a, lv)] = seed_values(g)
                    coll[f"{t}|{c}|{kl}|{a}|{lv}"] = collapse_rate(runs_vs_zero(g, v[(kl, "none", "0")]))

        def interaction(a, b, lv):
            def stat(keep):
                g = []
                for kl in ("complete", "stock"):
                    ma, mb = mean(v[(kl, a, lv)], keep), mean(v[(kl, b, "0" if b == "none" else lv)], keep)
                    g.append(None if ma is None or mb is None else ma - mb)
                return None if None in g else g[0] - g[1]
            return stat

        for lv in FIXED:
            res[(t, c, f"pooled_minus_discriminator|{lv}")] = loso(interaction("pooled", "discriminator", lv))
            for a in X15_ARMS:
                res[(t, c, f"{a}_cost|{lv}")] = loso(interaction(a, "none", lv))
    for key in {k for _, _, k in res}:
        tasks = [t for t, _ in EXPECTED["X15"]]
        pools[f"{key}|1"] = pooled({t: res[(t, 1, key)] for t in tasks}, tasks)
    return dict(cells={f"{t}|{c}|{k}": v for (t, c, k), v in res.items()}, B_minus_bstar=batch, collapse=coll, pooled=pools)


# ---------------------------------------------------------------------------------------------- inputs: cell-type counts
def celltype_counts(obs_by_task, M):
    """The cell-type count table from prepped obs ('batch', 'celltype'): draw 'full' of every task and, for every X8
    subsample spec of manifest M, the cells of the fitter's own draw (fit_paper_config.x8_draw), draw 'x8_V<V>_s<subset>'."""
    import fit_paper_config as FPC

    def table(obs, task, draw):
        g = obs.groupby(["batch", "cell_type"], observed=True).size()
        return g[g > 0].rename("n").reset_index().assign(task=task, draw=draw)
    frames = []
    for t, obs in sorted(obs_by_task.items()):
        obs = pd.DataFrame({"batch": obs["batch"].astype(str).to_numpy(), "cell_type": obs["celltype"].astype(str).to_numpy()})
        frames.append(table(obs, t, "full"))
        specs = {json.dumps(json.loads(e)["subsample"], sort_keys=True) for e in M.extra[(M.experiment == "X8") & (M.task == t)]}
        for spec in map(json.loads, sorted(specs)):
            frames.append(table(obs.iloc[FPC.x8_draw(obs, spec)], t, f"x8_V{spec['n_batches']}_s{spec['subset']}"))
    out = pd.concat(frames, ignore_index=True)[["task", "draw", "batch", "cell_type", "n"]]
    P._require(not out.duplicated(["task", "draw", "batch", "cell_type"]).any(), "cell-type counts: duplicate rows")
    return out.sort_values(["task", "draw", "batch", "cell_type"], ignore_index=True)


# ---------------------------------------------------------------------------------------------- record
CONFIRMATORY = [f"X3|slope_contrast|{c}|{lv}" for c in (0, 1) for lv in FIXED] + [f"X6|divergence|{c}" for c in (0, 1)] + \
               [f"X7|depleted_{lv}_discriminator_ref|{c}" for c in (0, 1) for lv in X7_DEPLETION] + [f"X8|ref_dis|iLISI_cn|{c}" for c in (0, 1)]


def holm_all(p):
    """Holm over the whole family (spec 1.7): a missing p (no evaluable dataset) enters as 1 so m never shrinks; its
    adjusted value is reported as None."""
    adj = A.holm([1.0 if x is None else x for x in p])
    return [None if x is None else a for x, a in zip(p, adj)]


def analyse(out, targets, counts):
    res = dict(X3=x3(out), X6=x6(out, targets), X7=x7(out, targets, counts), X8=x8(out, counts),
               X12=x12(out, targets), X15=x15(out, targets))
    p = [res[k.split("|")[0]]["pooled"][k.split("|", 1)[1]]["wilcoxon_p"] for k in CONFIRMATORY]
    adj = holm_all(p)
    res["confirmatory"] = {k: dict(declaration=res[k.split("|")[0]]["pooled"][k.split("|", 1)[1]]["declaration"],
                                   wilcoxon_p=p[i], holm_p=adj[i]) for i, k in enumerate(CONFIRMATORY)}
    res["n_confirmatory"] = res["holm_m"] = len(CONFIRMATORY)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--failures", required=True)
    ap.add_argument("--r4-record", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--celltype-counts")
    src.add_argument("--prepped-dir")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    P._require(not os.path.exists(a.out_json), f"analysis record {a.out_json} exists")
    M, _ = P.read_manifest(a.manifest)
    if a.prepped_dir:
        import anndata as ad
        tasks = sorted({t for e in ("X7", "X8") for t, _ in EXPECTED[e]})
        obs = {t: ad.read_h5ad(os.path.join(a.prepped_dir, f"{t}__scib.h5ad"), backed="r").obs for t in tasks}
        a.celltype_counts = a.out_json + ".celltype_counts.csv"
        P._require(not os.path.exists(a.celltype_counts), f"{a.celltype_counts} exists")
        celltype_counts(obs, M).to_csv(a.celltype_counts, index=False)
    P.check_seed_separation(M)
    r4 = P.read_record(a.r4_record, "R4", stage="x1")
    S, score_files = P.read_scores(a.scores)
    F = P.read_failures(a.failures)
    counts = pd.read_csv(a.celltype_counts, dtype={"task": str, "draw": str, "batch": str, "cell_type": str})
    P._require(list(counts.columns) == ["task", "draw", "batch", "cell_type", "n"], f"cell-type counts columns {list(counts.columns)}")
    counts["n"] = pd.to_numeric(counts.n, errors="raise").astype(int)
    out = P.outcomes(M, S, F, list(P.FOLLOWUPS)).sort_values("tag", ignore_index=True)    # row order (AN-11)
    rec = analyse(out, r4["targets"], counts)
    P.write_record(a.out_json, dict(rule="analysis_followups", status="done", spec="docs/ANALYSIS_SPEC_FOLLOWUPS.md", **rec),
                   dict(manifest=a.manifest, scores=score_files, failures=[a.failures], r4_record=[a.r4_record],
                        celltype_counts=[a.celltype_counts]))
    print(f"follow-up analysis: {rec['n_confirmatory']} confirmatory results -> {a.out_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
