#!/usr/bin/env python
"""Follow-up analysis X6 (docs/ANALYSIS_SPEC_FOLLOWUPS.md; X3, X7, X8, X12 and X15 were dropped, SI-103).

Must be committed and tagged (tag prereg-x6-v1) before R4-x1 runs and before any X1 result is examined (SI-68, SI-78).
It reads only follow-up rows (seeds 10-12) and the targets (b0, b*, low-signal flag) of the frozen R4 stage-x1 record; it
reuses the frozen helpers of scripts/analysis_x1.py (combine, window, wilcoxon_exact, holm) and scripts/prereg_rules.py
without changing them. The design names it checks are hard-coded below (spec section 1.9).

Usage (repo root):
  python scripts/analysis_followups.py --manifest <resolved follow-up manifest, design v5> --scores <dir> [<dir> ...] \
      --failures <csv that covers every follow-up row> --r4-record docs/prereg/r4_x1_record.json --out-json <followups.json>
The score directories and the failures table must hold X6 rows only; any other row stops the script.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_x1 as A  # noqa: E402
import prereg_rules as P  # noqa: E402

FU_SEEDS = (10, 11, 12)
MIN_READINGS = 3              # all 3 leave-one-seed-out readings must exist (X1 needs 4 of 5, i.e. at most one missing)
MARGIN = P.DELTA_MIN          # r = 0.01 for every follow-up endpoint, in the endpoint's own unit
LEVELS = ("matched_lo", "matched", "matched_hi")
REAL = set(P.REAL_TASKS)
# ---- design, as decided 2026-10-17 (SI-93..SI-98); names the build track must match (spec section 1.9)
EXPECTED = {
    "X6": {(t, c) for t in ("atac_small", "immune", "pancreas") for c in (0, 1)},
}
R1_EXTRA = {"r1_gamma": 10}               # `extra` of the discriminator_r1 rows; every other row has none
X6_COPT = "discriminator_sn_critic_opt"   # SI-98: JS + spectral norm trained with the critic's Adam settings (followup-build)
X6_CONFIGS = {"js": ("discriminator", 1), "js_sn": ("discriminator_sn", 1), "js_r1": ("discriminator_r1", 1),
              "js_sn_copt": (X6_COPT, 1), "w1_gp_1": ("pooled", 1), "w1_gp_5": ("pooled", 5),
              "w1_gp_10": ("pooled", 10), "w1_sn_1": ("pooled_sn", 1), "w1_sn_5": ("pooled_sn", 5)}
X6_CONTRASTS = {   # name: (minuend, subtrahend); "divergence" is the confirmatory one (spec section 3.4)
    "divergence": ("w1_sn_1", "js_sn_copt"), "divergence_own_optimiser": ("w1_sn_1", "js_sn"),
    "js_optimiser": ("js_sn_copt", "js_sn"), "js_sn": ("js_sn", "js"), "js_r1": ("js_r1", "js"),
    "w1_sn_1": ("w1_sn_1", "w1_gp_1"), "w1_sn_5": ("w1_sn_5", "w1_gp_5"), "w1_steps_5": ("w1_gp_5", "w1_gp_1"),
    "w1_steps_10": ("w1_gp_10", "w1_gp_1"), "w1_sn_steps_5": ("w1_sn_5", "w1_sn_1"), "as_used": ("w1_gp_5", "js")}
FIXED_LABEL = "fixed-strength, not matched batch removal"     # rule I048: a single-strength contrast is never "matched"
TAG_LEVEL = r"_l(matched_lo|matched_hi|matched)_c[01]_s\d+_[0-9a-f]{8}$"


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
    bad = [t for t, e, a in zip(o.tag, ex, o.arm) if e != (R1_EXTRA if a == "discriminator_r1" else {})]
    P._require(not bad, f"{experiment}: `extra` must be {R1_EXTRA} on discriminator_r1 rows and empty on every other row "
                        f"(spec section 1.9); {len(bad)} rows differ, e.g. {bad[:3]}")
    o["k"] = o.n_critic.astype(int)
    return o


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
    if r["d"] is None:     # the reason is the status of the first arm whose curve does not bracket b*
        r["reason"] = next(s for s in (bstar_status(pa, bstar), bstar_status(pb, bstar)) if s != "bracketed")
    return r


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
        cells[f"{t}|{c}"] = {n: dict(status=bstar_status(pts[n], tg["bstar"]),
                                     window=A.window([(lam, list(cc.values())) for lam, _, cc, _ in pts[n]]),   # R3 window, frozen rule
                                     **{lv: at_level(pts[n], rows[n], lv, tg["bstar"]) for lv in LEVELS})
                             for n in X6_CONFIGS}
        for name, (a, b) in X6_CONTRASTS.items():
            res = bstar_contrast(pts[a], pts[b], tg["bstar"], tg["bstar_low_signal"])
            fa, fb = cells[f"{t}|{c}"][a]["matched"], cells[f"{t}|{c}"][b]["matched"]
            res["fixed_strength"] = dict(label=FIXED_LABEL,    # each arm at its own inherited `matched` strength
                                         dC=None if None in (fa["C"], fb["C"]) else fa["C"] - fb["C"],
                                         dB=None if None in (fa["B"], fb["B"]) else fa["B"] - fb["B"])
            contrast[(t, c, name)] = res
    for c in (0, 1):
        tasks = [t for t, cc in EXPECTED["X6"] if cc == c]
        for name in X6_CONTRASTS:
            pools[f"{name}|{c}"] = pooled({t: contrast[(t, c, name)] for t in tasks}, tasks)
    return dict(cells=cells, contrast={f"{t}|{c}|{n}": v for (t, c, n), v in contrast.items()}, pooled=pools)


# ---------------------------------------------------------------------------------------------- record
CONFIRMATORY = [f"X6|divergence|{c}" for c in (0, 1)]


def holm_all(p):
    """Holm over the whole family (spec 1.7): a missing p (no evaluable dataset) enters as 1 so m never shrinks; its
    adjusted value is reported as None."""
    adj = A.holm([1.0 if x is None else x for x in p])
    return [None if x is None else a for x, a in zip(p, adj)]


def analyse(out, targets):
    res = dict(X6=x6(out, targets))
    p = [res[k.split("|")[0]]["pooled"][k.split("|", 1)[1]]["wilcoxon_p"] for k in CONFIRMATORY]
    adj = holm_all(p)
    res["confirmatory"] = {k: dict(declaration=res[k.split("|")[0]]["pooled"][k.split("|", 1)[1]]["declaration"],
                                   wilcoxon_p=p[i], holm_p=adj[i]) for i, k in enumerate(CONFIRMATORY)}
    res["n_confirmatory"] = res["holm_m"] = len(CONFIRMATORY)
    return res


def only_x6(M, S, F):
    """Refuse score or failure rows that are not X6 manifest rows (blinding: no X1 or X13 row is to be read, SI-78)."""
    x6_tags = set(M.tag[M.experiment == "X6"])
    other = len(set(S.tag) - x6_tags) + len(set(F.tag) - x6_tags)
    P._require(not other, f"{other} score or failure rows are not X6 manifest rows; pass X6 score and failure files only")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--failures", required=True)
    ap.add_argument("--r4-record", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    P._require(not os.path.exists(a.out_json), f"analysis record {a.out_json} exists")
    M, _ = P.read_manifest(a.manifest)
    P.check_seed_separation(M)
    r4 = P.read_record(a.r4_record, "R4", stage="x1")
    S, score_files = P.read_scores(a.scores)
    F = P.read_failures(a.failures)
    only_x6(M, S, F)
    out = P.outcomes(M, S, F, ["X6"]).sort_values("tag", ignore_index=True)    # row order (AN-11)
    rec = analyse(out, r4["targets"])
    P.write_record(a.out_json, dict(rule="analysis_followups", status="done", spec="docs/ANALYSIS_SPEC_FOLLOWUPS.md", **rec),
                   dict(manifest=a.manifest, scores=score_files, failures=[a.failures], r4_record=[a.r4_record]))
    print(f"follow-up analysis: {rec['n_confirmatory']} confirmatory results -> {a.out_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
