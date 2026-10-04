#!/usr/bin/env python
"""Pre-registered decision rules for the Tier 1+2 rerun (docs/PREREG.md, user sign-off 2026-10-02).

    R1  A1 -> 6-point X1 lambda grid per arm family x decoder        scripts/freeze_a1_grid.py
    R2  A2 -> X1 adversary input (posterior mean vs sample)           scripts/decide_a2_a3.py
    R3  A3 -> X1 per-dimension standardisation of the adversary input scripts/decide_a2_a3.py
    R4  matched lambda (lo / matched / hi); stage a1 -> A2/A3 windows, stage x1 -> follow-ups
                                                                      scripts/freeze_matched_lambda.py
    power report of the fixed design (5 seeds, SI-16)                 scripts/freeze_a1_grid.py

Every rule is a deterministic function of the manifest (scripts/build_paper_manifest.py), the raw
scIB metrics written by scripts/score_scib_native.py (one CSV row per fit, joined by tag) and a
failures table (tag, status in {diverged, nonfinite_latent}, detail). A row that is neither scored
nor failed, a tag scored twice, a NaN in a required metric or an incomplete design raises
PreregError: nothing is imputed, dropped or defaulted.
"""
import ast
import glob
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import build_paper_manifest as BPM  # noqa: E402  (COLS, LAMBDA_GRID, placeholders, finalize)

EXIT_FROZEN, EXIT_EXTEND = 0, 3

# ---- constants of the pre-registration (docs/PREREG.md section 1); changing one changes the rules ----
DELTA_MIN = 0.01          # smallest effect on the unscaled batch / bio score (PAPER_PLAN section 5 ROPE floor)
K_NOISE = 3.0             # noise multiple for 'no effect'
COLLAPSE_BIO = 0.10       # collapse: bio loss vs lambda=0 (unscaled)
Q_BSTAR = 0.5             # b* = b0 + Q (b_common - b0)
ROPE = 0.01               # R2 / R3 switching margin
N_GRID = 6                # X1 lambda points per family x decoder (SI-16)
TIE_TOL = 1e-9
R1_EXT_POINTS, R1_MAX_ROUNDS = 2, 2
R4_EXT_POINTS, R4_MAX_ROUNDS = 1, 2
POWER_DELTA, POWER_SEEDS, EQUIV_BOUNDS, N_ENDPOINTS = 0.02, 5, (0.01, 0.02), 4

FAMILIES = {"js": ("discriminator",), "w1": ("reference", "pooled", "barycenter"),
            "mmd": ("mmd",), "sinkhorn": ("sinkhorn",)}          # SI-18: divergence x decoder
ADV_ARMS = tuple(a for arms in FAMILIES.values() for a in arms)
FAMILY_OF = {a: f for f, arms in FAMILIES.items() for a in arms}
# R4: variant arms inherit the matched lambda of their base arm (same task and decoder).
BASE_ARM = dict({a: a for a in ADV_ARMS}, discriminator_sn="discriminator", reference_sn="reference",
                reference_fixed="reference", pooled_sn="pooled", barycenter_sn="barycenter", mmd_ref="mmd",
                discriminator_r1="discriminator", discriminator_ref="discriminator")   # SPECS_missing_arms.md (SI-22, SI-23)
NO_ADVERSARY = ("none", "scvi_adv", "scanvi", "sysvi")
FAILURE_STATUSES = ("diverged", "nonfinite_latent")
CC_TASKS = ("pancreas", "lung", "immune", "immune_hum_mou")   # uns modality rna, organism human (prepped, 2026-10-02)
TRAJ_TASKS = ("immune", "immune_hum_mou")                     # obs dpt_pseudotime present (prepped, 2026-10-02)
REAL_TASKS = ("pancreas", "lung", "immune", "immune_hum_mou", "atac_small", "atac_large")
TASK_FAMILY = {"pancreas": "pancreas", "lung": "lung", "immune": "immune", "immune_hum_mou": "immune",
               "atac_small": "atac", "atac_large": "atac"}
FOLLOWUPS = ("X3", "X6", "X7", "X8", "X12", "X15")   # X15: KL warm-up sensitivity (SI-44)


class PreregError(RuntimeError):
    """An input violates the pre-registered contract; the rule cannot be applied."""


def _require(cond, msg):
    if not cond:
        raise PreregError(msg)


# ---------------------------------------------------------------------------------------------
# metric definitions: read from score_scib_native.py (one definition, no import of scanpy/scib)
# ---------------------------------------------------------------------------------------------
def metric_lists(path=os.path.join(HERE, "score_scib_native.py")):
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("BATCH_METRICS", "BIO_METRICS"):
            out[node.targets[0].id] = list(ast.literal_eval(node.value))
    _require(set(out) == {"BATCH_METRICS", "BIO_METRICS"}, f"{path}: BATCH_METRICS / BIO_METRICS not found")
    return out["BATCH_METRICS"], out["BIO_METRICS"]


BATCH_METRICS, BIO_METRICS = metric_lists()


def bio_metrics_for(task):
    _require(task in BPM.TASKS, f"unknown task {task!r}")
    m = [x for x in BIO_METRICS if x not in ("cell_cycle_conservation", "trajectory")]
    if "cell_cycle_conservation" in BIO_METRICS and task in CC_TASKS:
        m.append("cell_cycle_conservation")
    if "trajectory" in BIO_METRICS and task in TRAJ_TASKS:
        m.append("trajectory")
    return m


# ---------------------------------------------------------------------------------------------
# lambda sequence 0.1, 0.3, 1, 3, ... and its canonical string form (as the builder writes it)
# ---------------------------------------------------------------------------------------------
def _decompose(v):
    v = float(v)
    _require(v > 0 and math.isfinite(v), f"lambda {v} is not positive")
    e = math.floor(math.log10(v) + 1e-12)
    m = v / 10.0 ** e
    for mm in (1, 3):
        if math.isclose(m, mm, rel_tol=1e-9):
            return mm, e
    if math.isclose(m, 10, rel_tol=1e-9):
        return 1, e + 1
    raise PreregError(f"lambda {v} is not on the 1-3 sequence")


def lam_step(v, direction):
    """One step along ..., 0.1, 0.3, 1, 3, 10, ... (direction +1 up, -1 down)."""
    m, e = _decompose(v)
    if direction > 0:
        return float(f"3e{e}") if m == 1 else float(f"1e{e + 1}")
    return float(f"1e{e}") if m == 3 else float(f"3e{e - 1}")


def lam_beyond(v, direction, n):
    out = []
    for _ in range(n):
        v = lam_step(v, direction)
        out.append(v)
    return out


def fmt_lam(v):
    v = float(v)
    return str(int(v)) if v.is_integer() else repr(v)


def _lamf(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(paths):
    """Concatenate manifests (the builder's TSV, extension manifests). Returns (DataFrame of str, header lines)."""
    paths = [paths] if isinstance(paths, str) else list(paths)
    frames, header = [], []
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                header.append(line.rstrip("\n"))
        m = pd.read_csv(p, sep="\t", comment="#", dtype=str, keep_default_na=False)
        missing = [c for c in BPM.COLS if c not in m.columns]
        _require(not missing, f"{p}: manifest lacks columns {missing}")
        frames.append(m)
    M = pd.concat(frames, ignore_index=True)
    dup = M.tag[M.tag.duplicated()].tolist()
    _require(not dup, f"duplicate manifest tags: {dup[:5]}")
    return M, header


def write_manifest(M, header, path, note):
    """Write in the builder's format (tab-joined, unquoted); tags are never changed."""
    cols = list(M.columns)
    for c in cols:
        bad = M[c].astype(str).str.contains("\t|\n", regex=True)
        _require(not bad.any(), f"column {c} holds a tab or newline")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for h in header:
            f.write(h + "\n")
        f.write(f"# {note}\n")
        f.write("\t".join(cols) + "\n")
        for r in M.itertuples(index=False):
            f.write("\t".join(str(x) for x in r) + "\n")
    os.replace(tmp, path)


def read_scores(paths):
    files = []
    for p in ([paths] if isinstance(paths, str) else paths):
        if os.path.isdir(p):
            fs = sorted(glob.glob(os.path.join(p, "*.csv")))
            _require(fs, f"no score CSVs in {p}")
            files += fs
        else:
            _require(os.path.isfile(p), f"score path {p} does not exist")
            files.append(p)
    S = pd.concat([pd.read_csv(f, dtype={"tag": str}) for f in files], ignore_index=True)
    need = ["tag", "kbet_seed"] + BATCH_METRICS + BIO_METRICS
    missing = [c for c in need if c not in S.columns]
    _require(not missing, f"score files lack columns {missing} (kbet_seed: written by score_scib_native.py since "
                          f"2026-10-03; earlier, unseeded score files are not valid inputs)")
    for c in BATCH_METRICS + BIO_METRICS:
        S[c] = pd.to_numeric(S[c], errors="raise")
    return S, files


def read_failures(path):
    F = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ("tag", "status", "detail") if c not in F.columns]
    _require(not missing, f"{path}: failures table lacks columns {missing}")
    bad = sorted(set(F.status) - set(FAILURE_STATUSES))
    _require(not bad, f"{path}: failure status {bad} not in {FAILURE_STATUSES} (infrastructure errors are rerun, not recorded)")
    _require(not F.tag.duplicated().any(), f"{path}: a tag is listed twice")
    _require((F.detail.str.len() > 0).all(), f"{path}: every failure needs a detail")
    return F


def outcomes(M, S, F, experiments):
    """One row per manifest row of `experiments`: status ok|failed, batch score B, bio score C."""
    rows = M[M.experiment.isin(experiments)].copy()
    _require(len(rows), f"no manifest rows for {experiments}")
    tags = set(rows.tag)
    s = S[S.tag.isin(tags)]
    dup = sorted(set(s.tag[s.tag.duplicated()]))
    _require(not dup, f"tags scored more than once: {dup[:5]}")
    f = F[F.tag.isin(tags)]
    both = sorted(set(s.tag) & set(f.tag))
    _require(not both, f"tags both scored and failed: {both[:5]}")
    missing = sorted(tags - set(s.tag) - set(f.tag))
    _require(not missing, f"{len(missing)} of {len(tags)} rows of {experiments} are neither scored nor failed, "
                          f"e.g. {missing[:5]}")
    _require("kbet_seed" in s.columns, "scores lack kbet_seed (unseeded kBET, pre-2026-10-03 scorer output)")
    ks = s.kbet_seed                     # score_scib_native.py seeds R's kBET; one rule never mixes kBET seeds
    if len(ks):
        _require(ks.notna().all() and ks.nunique() == 1,
                 f"scores of {experiments} mix kBET seeds {sorted(ks.dropna().unique().tolist())} "
                 f"({int(ks.isna().sum())} rows without one); measurement-SD re-scores belong in their own directory")
    rows = rows.merge(s[["tag"] + BATCH_METRICS + BIO_METRICS], on="tag", how="left", validate="one_to_one")
    rows["status"] = np.where(rows.tag.isin(set(f.tag)), "failed", "ok")
    B, C = [], []
    for d in rows.to_dict("records"):     # by column name (metric names are not identifiers)
        if d["status"] == "failed":
            B.append(np.nan); C.append(np.nan)
            continue
        need = BATCH_METRICS + bio_metrics_for(d["task"])
        vals = {m: float(d[m]) for m in need}
        nonfinite = [m for m, v in vals.items() if not np.isfinite(v)]
        _require(not nonfinite, f"{d['tag']}: non-finite required metrics {nonfinite} (pipeline fault, not an outcome)")
        out_b = [m for m in BATCH_METRICS if not (-1e-6 <= vals[m] <= 1 + 1e-6)]
        out_c = [m for m in need if not (-1 - 1e-6 <= vals[m] <= 1 + 1e-6)]
        _require(not out_b and not out_c, f"{d['tag']}: metrics outside the scIB scaled range: {out_b + out_c}")
        B.append(float(np.mean([vals[m] for m in BATCH_METRICS])))
        C.append(float(np.mean([vals[m] for m in need if m not in BATCH_METRICS])))
    rows["B"], rows["C"] = B, C
    rows["cond_i"] = rows.cond.astype(int)
    rows["seed_i"] = rows.seed.astype(int)
    rows["lamf"] = [_lamf(x) for x in rows.lam]
    return rows


def check_seed_separation(M):
    """Hard requirements: A1/A2/A3 seeds, X1 seeds and the lambda-matched follow-ups' seeds are pairwise
    disjoint, so no lambda chosen on one stage is evaluated on the fits it was chosen from (SI-13, SI-26)."""
    pilot = set(M.seed[M.experiment.isin(["A1", "A2", "A3"])].astype(int))
    x1 = set(M.seed[M.experiment == "X1"].astype(int))
    fu = set(M.seed[M.experiment.isin(FOLLOWUPS)].astype(int))
    _require(not (pilot & x1), f"A1/A2/A3 seeds {sorted(pilot & x1)} are also X1 seeds")
    _require(not (fu & x1), f"follow-up seeds {sorted(fu & x1)} are also X1 seeds")
    _require(not (fu & pilot), f"follow-up seeds {sorted(fu & pilot)} are also A1/A2/A3 seeds")


# ---------------------------------------------------------------------------------------------
# per-point summaries
# ---------------------------------------------------------------------------------------------
def _zero_ref(out, experiment, task, cond):
    z = out[(out.experiment == experiment) & (out.task == task) & (out.cond_i == cond) & (out.arm == "none")]
    _require(len(z), f"{experiment} {task} cond={cond}: no lambda=0 ('none') rows")
    bad = z.tag[z.status != "ok"].tolist()
    _require(not bad, f"lambda=0 fits failed {bad}: stock scVI must not fail; fix the pipeline")
    _require(not z.seed_i.duplicated().any(), f"{experiment} {task} cond={cond}: lambda=0 seed listed twice")
    return {int(s): (float(b), float(c)) for s, b, c in zip(z.seed_i, z.B, z.C)}


def _cell_points(out, experiment, task, cond, arm, seeds, **match):
    """Points of one (task, cond, arm) cell, sorted by lambda; checks the lambda x seed design is complete."""
    sel = (out.experiment == experiment) & (out.task == task) & (out.cond_i == cond) & (out.arm == arm)
    for k, v in match.items():
        sel &= out[k].astype(str) == str(v)
    c = out[sel]
    _require(len(c), f"{experiment} {task} cond={cond} {arm} {match}: no rows")
    _require(c.lamf.notna().all(), f"{experiment} {task} cond={cond} {arm}: unresolved lambda {sorted(set(c.lam[c.lamf.isna()]))}")
    pts = []
    for lam, g in sorted(c.groupby("lamf"), key=lambda kv: kv[0]):
        got = sorted(g.seed_i.tolist())
        _require(got == sorted(seeds), f"{experiment} {task} cond={cond} {arm} lam={fmt_lam(lam)}: seeds {got} != {sorted(seeds)}")
        ok = g[g.status == "ok"]
        pts.append(dict(lam=float(lam), n_fail=int((g.status == "failed").sum()),
                        B={int(s): float(b) for s, b in zip(ok.seed_i, ok.B)},
                        C={int(s): float(x) for s, x in zip(ok.seed_i, ok.C)}))
    return pts


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


# ---------------------------------------------------------------------------------------------
# R1
# ---------------------------------------------------------------------------------------------
def robust_noise_sd(diff_lists):
    """Median over grid points of the seed SD of the paired difference, each divided by its
    median-unbiasing constant sqrt(chi2_{n-1, 0.5} / (n-1))."""
    vals = []
    for d in diff_lists:
        n = len(d)
        if n >= 2:
            vals.append(float(np.std(d, ddof=1)) / math.sqrt(stats.chi2.ppf(0.5, n - 1) / (n - 1)))
    _require(len(vals) >= 3, "fewer than 3 grid points with >= 2 paired seeds: no noise estimate")
    return float(np.median(vals))


def r1_block(points, zero):
    """Floor, batch peak and upper end of one block (docs/PREREG.md section 2). points sorted by lambda."""
    P = []
    for p in points:
        seeds = sorted(set(p["B"]) & set(zero))
        P.append(dict(lam=p["lam"], n_fail=p["n_fail"], n=len(seeds),
                      d_b=[p["B"][s] - zero[s][0] for s in seeds], d_c=[p["C"][s] - zero[s][1] for s in seeds]))
    sig_b = robust_noise_sd([q["d_b"] for q in P])
    sig_c = robust_noise_sd([q["d_c"] for q in P])
    for q in P:
        q["db"], q["dc"] = _mean(q["d_b"]), _mean(q["d_c"])
        q["del_b"] = max(DELTA_MIN, K_NOISE * sig_b / math.sqrt(q["n"])) if q["n"] else float("nan")
        q["del_c"] = max(DELTA_MIN, K_NOISE * sig_c / math.sqrt(q["n"])) if q["n"] else float("nan")
        q["no_effect"] = q["n_fail"] == 0 and q["n"] >= 1 and abs(q["db"]) < q["del_b"] and abs(q["dc"]) < q["del_c"]
        q["collapsed"] = q["n_fail"] > 0 or (q["n"] >= 1 and q["dc"] <= -COLLAPSE_BIO + TIE_TOL)
    top = len(P) - 1
    f = -1
    while f + 1 <= top and P[f + 1]["no_effect"]:
        f += 1
    with_mean = [i for i in range(len(P)) if P[i]["n"] >= 1]
    _require(with_mean, "every grid point failed in every seed")
    imax = max(with_mean, key=lambda i: (P[i]["db"], -i))
    peak = None
    if P[imax]["db"] >= P[imax]["del_b"] - TIE_TOL:
        peak = next((j for j in range(f + 1, len(P))
                     if P[j]["n"] >= 1 and P[j]["db"] >= P[imax]["db"] - P[j]["del_b"] - TIE_TOL), None)

    def upper(j):
        return P[j]["collapsed"] or (peak is not None and j >= peak)

    i = len(P)
    while i - 1 > f and upper(i - 1):
        i -= 1
    h = i if i <= top else None
    nonresponsive = f == top
    low_edge = f < 0
    high_edge = nonresponsive or h is None or (h == top and not P[top]["collapsed"])
    return dict(floor=(None if low_edge else f), peak=peak, upper=h, low_edge=low_edge, high_edge=high_edge,
                nonresponsive=nonresponsive, sigma_b=sig_b, sigma_c=sig_c,
                points=[{k: q[k] for k in ("lam", "n", "n_fail", "db", "dc", "del_b", "del_c", "no_effect", "collapsed")}
                        for q in P])


def grid_indices(i_lo, i_hi, n_total, n=N_GRID):
    """6 indices spanning [i_lo, i_hi] evenly in log lambda (exact integer rounding, half up); if the
    interval has fewer than 6 points, pad alternately above and below, starting above."""
    _require(0 <= i_lo < i_hi < n_total, f"bad interval [{i_lo}, {i_hi}] on {n_total} points")
    _require(n_total >= n, f"grid has {n_total} < {n} points")
    span = i_hi - i_lo
    if span + 1 >= n:
        return [i_lo + (2 * j * span + (n - 1)) // (2 * (n - 1)) for j in range(n)]
    idx, up, down, side = list(range(i_lo, i_hi + 1)), i_hi + 1, i_lo - 1, "up"
    while len(idx) < n:
        if side == "up" and up >= n_total:
            side = "down"
        if side == "down" and down < 0:
            side = "up"
        if side == "up":
            idx.append(up); up += 1; side = "down"
        else:
            idx.append(down); down -= 1; side = "up"
    return sorted(idx)


def _rounds(grid, side):
    base = [float(x) for x in BPM.LAMBDA_GRID]
    beyond = [v for v in grid if (v < min(base) * (1 - 1e-9) if side == "low" else v > max(base) * (1 + 1e-9))]
    _require(len(beyond) % R1_EXT_POINTS == 0, f"{len(beyond)} A1 extension points on the {side} side: not whole rounds")
    return len(beyond) // R1_EXT_POINTS


def r1_family(out, fam, cond):
    """R1 for one arm family x decoder on the A1 outcomes."""
    arms = FAMILIES[fam]
    a1 = out[(out.experiment == "A1") & (out.cond_i == cond)]
    tasks = sorted(set(a1.task))
    _require(tasks, f"A1 has no rows for cond={cond}")
    grids = {}
    for t in tasks:
        for a in arms:
            grids[(t, a)] = tuple(sorted(set(a1.lamf[(a1.task == t) & (a1.arm == a)])))
    grid = sorted(set(next(iter(grids.values()))))
    bad = [k for k, g in grids.items() if list(g) != grid]
    _require(not bad, f"A1 family {fam} cond={cond}: blocks {bad} have a different lambda grid than {grid}")
    base = [float(x) for x in BPM.LAMBDA_GRID]
    _require(all(any(math.isclose(b, g, rel_tol=1e-9) for g in grid) for b in base),
             f"A1 family {fam} cond={cond}: grid {grid} lacks part of the A1 base grid {base}")
    rounds = {"low": _rounds(grid, "low"), "high": _rounds(grid, "high")}
    blocks = {}
    for t in tasks:
        zero = _zero_ref(out, "A1", t, cond)
        for a in arms:
            blocks[f"{t}|{a}"] = r1_block(_cell_points(out, "A1", t, cond, a, sorted(zero)), zero)
    extend = {}
    if any(b["low_edge"] for b in blocks.values()) and rounds["low"] < R1_MAX_ROUNDS:
        extend["low"] = lam_beyond(grid[0], -1, R1_EXT_POINTS)
    if any(b["high_edge"] for b in blocks.values()) and rounds["high"] < R1_MAX_ROUNDS:
        extend["high"] = lam_beyond(grid[-1], +1, R1_EXT_POINTS)
    res = dict(family=fam, arms=list(arms), cond=cond, a1_grid=grid, rounds=rounds, blocks=blocks, extend=extend)
    if extend:
        res["status"] = "extend"
        return res
    flags, lo, hi = [], [], []
    top = len(grid) - 1
    for k, b in blocks.items():
        if b["nonresponsive"]:
            flags.append(f"{k}: non_responsive (no effect at any lambda up to {fmt_lam(grid[-1])}); left out")
            continue
        f, h = b["floor"], b["upper"]
        if b["low_edge"]:
            f = 0
            flags.append(f"{k}: low_edge_unresolved after {rounds['low']} rounds")
        if b["high_edge"]:
            h = top
            flags.append(f"{k}: high_edge_unresolved after {rounds['high']} rounds")
        lo.append(f); hi.append(h)
    _require(lo, f"A1 family {fam} cond={cond}: every block is non-responsive; a human decision is needed")
    i_lo, i_hi = min(lo), max(hi)
    _require(i_hi > i_lo, f"A1 family {fam} cond={cond}: degenerate interval [{i_lo}, {i_hi}]")
    idx = grid_indices(i_lo, i_hi, len(grid))
    res.update(status="frozen", interval=[grid[i_lo], grid[i_hi]], indices=idx, x1_grid=[grid[i] for i in idx], flags=flags)
    return res


def freeze_a1_grid(out):
    """R1 over every family x decoder present in A1. status 'frozen' only if no family needs an extension."""
    conds = sorted(set(out.cond_i[out.experiment == "A1"]))
    fams = {f"{fam}|{c}": r1_family(out, fam, c) for fam in FAMILIES for c in conds}
    status = "extend" if any(r["status"] == "extend" for r in fams.values()) else "frozen"
    return dict(rule="R1", status=status, families=fams)


# ---------------------------------------------------------------------------------------------
# R4
# ---------------------------------------------------------------------------------------------
def _closest(points, target):
    """Index of the failure-free point whose seed-mean B is closest to target; ties -> smaller lambda."""
    best, best_d = None, None
    for i, p in enumerate(points):
        if p["n_fail"] or not p["B"]:
            continue
        d = abs(_mean(list(p["B"].values())) - target)
        if best is None or d < best_d - TIE_TOL:
            best, best_d = i, d
    return best, best_d


def r4_stage(out, stage, r1_record, need=None):
    """Matched lambda for every (task, cond, adversarial arm) of the stage (a1: A1 fits; x1: X1 fits).
    need: the 'task|cond|arm' cells that rows of the stage will take a value from; only those can ask for
    an edge extension (None = all cells). Other edge cells get the next lambda value, flagged."""
    _require(stage in ("a1", "x1"), stage)
    _require(r1_record.get("rule") == "R1" and r1_record.get("status") == "frozen", "R1 record is not frozen")
    exp = "A1" if stage == "a1" else "X1"
    rows = out[out.experiment == exp]
    cells, targets, extensions = {}, {}, []
    for (t, c) in sorted(set(zip(rows.task, rows.cond_i))):
        zero = _zero_ref(out, exp, t, c)
        b0 = _mean([v[0] for v in zero.values()])
        pts, M = {}, {}
        for a in ADV_ARMS:
            pts[a] = _cell_points(out, exp, t, c, a, sorted(zero))
            ok = [_mean(list(p["B"].values())) for p in pts[a] if not p["n_fail"] and p["B"]]
            _require(ok, f"{exp} {t} cond={c} {a}: no failure-free lambda")
            M[a] = max(ok)
        b_common = min(M.values())
        bstar = b0 + Q_BSTAR * (b_common - b0)
        targets[f"{t}|{c}"] = dict(b0=b0, M=M, b_common=b_common, bstar=bstar,
                                   bstar_low_signal=bool(b_common - b0 < 2 * DELTA_MIN))
        for a in ADV_ARMS:
            P = pts[a]
            grid = [p["lam"] for p in P]
            base = r1_record["families"][f"{FAMILY_OF[a]}|{c}"]["a1_grid" if stage == "a1" else "x1_grid"]
            beyond_lo = [v for v in grid if v < min(base) * (1 - 1e-9)]
            beyond_hi = [v for v in grid if v > max(base) * (1 + 1e-9)]
            i, dist = _closest(P, bstar)
            cell = dict(task=t, cond=c, arm=a, bstar=bstar, matched=grid[i], distance=dist,
                        matched_B=_mean(list(P[i]["B"].values())), flags=[],
                        lo=grid[i - 1] if i > 0 else None, hi=grid[i + 1] if i < len(grid) - 1 else None)
            for side, nbr, beyond, d in (("low", "lo", beyond_lo, -1), ("high", "hi", beyond_hi, +1)):
                if cell[nbr] is not None:
                    continue
                needed = need is None or f"{t}|{c}|{a}" in need
                if needed and len(beyond) < R4_MAX_ROUNDS * R4_EXT_POINTS:
                    extensions.append(dict(experiment=exp, task=t, cond=c, arm=a, lam=lam_step(grid[i], d)))
                else:
                    cell[nbr] = lam_step(grid[i], d)
                    why = "edge_unresolved" if needed else "edge_not_extended (no row uses this cell)"
                    cell["flags"].append(f"{why}_{side}: {nbr} = {fmt_lam(cell[nbr])} has no {exp} fit")
            cells[f"{t}|{c}|{a}"] = cell
    return dict(rule="R4", stage=stage, status=("extend" if extensions else "frozen"),
                targets=targets, cells=cells, extensions=extensions)


# ---------------------------------------------------------------------------------------------
# R2 / R3
# ---------------------------------------------------------------------------------------------
def window_curve(points):
    """[(lam, seed-mean B, seed-mean C, n_fail)] over points with at least one successful seed."""
    return [(p["lam"], _mean(list(p["B"].values())), _mean(list(p["C"].values())), p["n_fail"])
            for p in sorted(points, key=lambda p: p["lam"]) if p["B"]]


def bio_at_bstar(curve, bstar):
    """Seed-mean bio at B = b* along the lambda-ordered curve: walking up in lambda, the first point equal
    to b* (within TIE_TOL) or the first adjacent pair that strictly brackets b* (linear interpolation)."""
    if not curve:
        return dict(status="no_points", bio=None)
    for i, (_, b1, c1, _) in enumerate(curve):
        if abs(b1 - bstar) <= TIE_TOL:
            return dict(status="bracketed", bio=c1)
        if i + 1 < len(curve):
            _, b2, c2, _ = curve[i + 1]
            if abs(b2 - bstar) > TIE_TOL and (b1 - bstar) * (b2 - bstar) < 0:
                return dict(status="bracketed", bio=c1 + (bstar - b1) * (c2 - c1) / (b2 - b1))
    if all(b < bstar for _, b, _, _ in curve):
        return dict(status="unreached_low", bio=None)
    return dict(status="unreached_high", bio=None)


def _group(c):
    """Unit of the coverage and sign criteria (docs/PREREG.md section 3): task x decoder."""
    return c["group"]


def decide_binary(cells, default, alt, masking_setting=None, safe_setting=None):
    """R2 / R3 decision (docs/PREREG.md section 3), ONE pooled decision over all cells.
    cells: [{task, cond, arm, group, <setting>: {status, bio, n_fail}}], group = 'task|decoder'."""
    n = len(cells)
    _require(n > 0, "no cells")
    missing = [c.get("task") for c in cells if "group" not in c]
    _require(not missing, f"cells without a task x decoder group: {missing[:3]}")
    need = math.ceil(2 * n / 3)
    groups = sorted({_group(c) for c in cells})
    ev = [c for c in cells if c[default]["status"] == "bracketed" and c[alt]["status"] == "bracketed"]
    pairs = [(c, c[alt]["bio"] - c[default]["bio"]) for c in ev]
    D = {f"{c['group']}|{c['arm']}": d for c, d in pairs}
    per_group = {g: [d for c, d in pairs if _group(c) == g] for g in groups}
    mean_D = _mean([d for _, d in pairs])
    coverage = len(ev) >= need and all(per_group[g] for g in groups)
    beats_rope = bool(ev) and mean_D - ROPE > TIE_TOL
    per_group_pos = coverage and all(_mean(per_group[g]) > TIE_TOL for g in groups)
    f_def, f_alt = sum(c[default]["n_fail"] for c in cells), sum(c[alt]["n_fail"] for c in cells)
    switch = bool(coverage and beats_rope and per_group_pos and f_alt <= f_def)
    choice, reason = (alt, "all four criteria hold") if switch else (default, "criteria not met")
    masked = []
    if masking_setting is not None:
        other = alt if masking_setting == default else default
        masked = [f"{c['group']}|{c['arm']}" for c in cells
                  if c[masking_setting]["status"] == "unreached_low" and c[other]["status"] == "bracketed"]
        if masked:
            choice, reason = safe_setting, f"masking: {masking_setting} misses b* in {masked}"
    return dict(choice=choice, switch=(choice == alt), reason=reason, n_cells=n, n_evaluable=len(ev),
                need_evaluable=need, groups=groups, D=D, mean_D=mean_D,
                per_group_mean_D={g: _mean(v) for g, v in per_group.items()},
                coverage=coverage, beats_rope=beats_rope, per_group_positive=per_group_pos,
                failures={default: f_def, alt: f_alt}, masked=masked)


# ---------------------------------------------------------------------------------------------
# power of the fixed design
# ---------------------------------------------------------------------------------------------
def _power(delta, se, df):
    tcrit = stats.t.ppf(0.95, df)
    return float(1 - stats.nct.cdf(tcrit, df, delta / se))


def power_report(out, r4a1):
    """P1 contrast (reference - discriminator) at the R4-a1 matched lambda, per A1 task x decoder."""
    blocks, var_num, var_den, by_cond = {}, 0.0, 0, {}
    a1 = out[out.experiment == "A1"]
    for (t, c) in sorted(set(zip(a1.task, a1.cond_i))):
        zero = _zero_ref(out, "A1", t, c)
        seeds = sorted(zero)
        ref = {p["lam"]: p for p in _cell_points(out, "A1", t, c, "reference", seeds)}
        dis = {p["lam"]: p for p in _cell_points(out, "A1", t, c, "discriminator", seeds)}
        pr, pd_ = ref[r4a1["cells"][f"{t}|{c}|reference"]["matched"]], dis[r4a1["cells"][f"{t}|{c}|discriminator"]["matched"]]
        common = sorted(set(pr["C"]) & set(pd_["C"]))
        d = [pr["C"][s] - pd_["C"][s] for s in common]
        _require(len(d) >= 2, f"A1 {t} cond={c}: fewer than 2 paired seeds for the P1 contrast")
        blocks[f"{t}|{c}"] = dict(n=len(d), mean=_mean(d), sd=float(np.std(d, ddof=1)))
        var_num += (len(d) - 1) * float(np.var(d, ddof=1))
        var_den += len(d) - 1
        by_cond.setdefault(c, []).append((_mean(d), len(d)))
    sigma = math.sqrt(var_num / var_den)
    taus = []
    for c, v in by_cond.items():
        if len(v) >= 2:
            means = [m for m, _ in v]
            nbar = float(np.mean([k for _, k in v]))
            taus.append(max(0.0, float(np.var(means, ddof=1)) - sigma ** 2 / nbar))
    _require(taus, "need >= 2 A1 tasks per decoder for tau")
    tau = math.sqrt(float(np.mean(taus)))
    designs = {}
    for label, D in (("real_tasks", len(REAL_TASKS)), ("families", len(set(TASK_FAMILY.values())))):
        se = math.sqrt((tau ** 2 + sigma ** 2 / POWER_SEEDS) / D)
        df = D - 1
        hi = 1.0
        while _power(hi, se, df) < 0.8:
            hi *= 2
            _require(hi < 1e6, f"power never reaches 0.8 (se={se})")
        mde = optimize.brentq(lambda x: _power(x, se, df) - 0.8, 1e-9, hi)
        designs[label] = dict(D=D, df=df, se=se, power_at_delta=_power(POWER_DELTA, se, df), mde80=mde,
                              equivalence_feasible={str(h): bool(stats.t.ppf(0.975, df) * se < h) for h in EQUIV_BOUNDS})
    p_min = 2.0 / 2 ** len(REAL_TASKS)
    holm1 = 0.05 / N_ENDPOINTS
    main = designs["real_tasks"]
    underpowered = bool(main["mde80"] > POWER_DELTA or not main["equivalence_feasible"][str(POWER_DELTA)])
    return dict(contrast="P1: bio(reference) - bio(discriminator) at the R4-a1 matched lambda, unscaled",
                blocks=blocks, sigma=sigma, sigma_df=var_den, tau=tau, tau_df=sum(len(v) - 1 for v in by_cond.values()),
                seeds=POWER_SEEDS, delta=POWER_DELTA, designs=designs,
                wilcoxon_min_p_two_sided=p_min, holm_first_threshold=holm1,
                frequentist_can_reject=bool(p_min <= holm1), underpowered=underpowered)


# ---------------------------------------------------------------------------------------------
# manifest resolution and extension rows
# ---------------------------------------------------------------------------------------------
def extension_rows(M, specs):
    """Manifest rows for extension points: copy every seed's row of the same (experiment, task, cond, arm)
    at its lowest lambda, set the new lambda, and tag with the builder's finalize()."""
    new = []
    for sp in specs:
        sel = (M.experiment == sp["experiment"]) & (M.task == sp["task"]) & (M.cond.astype(int) == int(sp["cond"])) \
              & (M.arm == sp["arm"])
        src = M[sel].copy()
        src = src[[not math.isnan(_lamf(x)) for x in src.lam]]
        _require(len(src), f"no template rows for {sp}")
        lam0 = min(_lamf(x) for x in src.lam)
        src = src[[math.isclose(_lamf(x), lam0) for x in src.lam]]
        for r in src.to_dict("records"):
            r = {c: r[c] for c in BPM.COLS}
            r["lam"] = fmt_lam(sp["lam"])
            new.append(r)
    out = BPM.finalize(new)
    clash = sorted(set(r["tag"] for r in out) & set(M.tag))
    _require(not clash, f"extension tags already in the manifest: {clash[:3]}")
    return pd.DataFrame([{c: str(r[c]) for c in BPM.COLS} for r in out], columns=BPM.COLS)


def settings_key(r):
    return "|".join(f"{c}={r[c]}" for c in BPM.COLS if c not in ("tag", "experiment"))


# ---------------------------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------------------------
def git_state():
    sha = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip() != ""
    return dict(git_sha=sha, git_dirty=dirty)


def constants():
    return dict(DELTA_MIN=DELTA_MIN, K_NOISE=K_NOISE, COLLAPSE_BIO=COLLAPSE_BIO, Q_BSTAR=Q_BSTAR, ROPE=ROPE,
                N_GRID=N_GRID, TIE_TOL=TIE_TOL, R1_EXT_POINTS=R1_EXT_POINTS, R1_MAX_ROUNDS=R1_MAX_ROUNDS,
                R4_EXT_POINTS=R4_EXT_POINTS, R4_MAX_ROUNDS=R4_MAX_ROUNDS, FAMILIES=FAMILIES, BASE_ARM=BASE_ARM,
                BATCH_METRICS=BATCH_METRICS, BIO_METRICS=BIO_METRICS, A1_BASE_GRID=list(BPM.LAMBDA_GRID))


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_record(path, payload, inputs):
    rec = dict(payload, inputs={k: [dict(path=p, sha256=sha256(p)) for p in v] for k, v in inputs.items()},
               constants=constants(), **git_state())
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(rec), f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)
    return rec


def read_record(path, rule, stage=None):
    rec = json.load(open(path))
    _require(rec.get("rule") == rule, f"{path} is not an {rule} record")
    _require(rec.get("status") == "frozen", f"{path}: {rule} is not frozen (status {rec.get('status')})")
    if stage is not None:
        _require(rec.get("stage") == stage, f"{path}: stage {rec.get('stage')} != {stage}")
    return rec
