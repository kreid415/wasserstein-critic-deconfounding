#!/usr/bin/env python
"""Agreement test of the fresh-seed replication against the sweep prediction.

Rule fixed before any replication score: experiments/sweep_replication/AGREEMENT.md, prediction PREDICTION.json (tag
replication-prediction-v1). The replication AGREES with the sweep if
  (1) each large sweep gap (|within-pair penalty difference| >= 0.04, PREDICTION.json large_gaps) keeps its sign, and
  (2) the mean over ranked task x decoder pairs of Kendall tau-b between the sweep and replication ranks of the six penalties is > 0
      by a one-sided sign-flip permutation test over pairs (all 2^n sign patterns), p < 0.05.
Reading (primary, raw new-seed): C (biology) at the sweep's b* on the replication's own curve, walking up from plain scVI
(lambda = 0, seeds 0-4) to the first crossing, linear interpolation in B (prereg_rules.bio_at_bstar, ANALYSIS_SPEC 2.2). Penalties that
do not reach b* take the bottom places (ANALYSIS_SPEC 3.1). Slope-adjusted reading (SENSITIVITY ONLY, defined here because the
rule text names it without a formula): each penalty's replication point nearest b* in B, moved to b* with one common within-penalty
slope dC/dB per pair (pooled over the pair's penalties); it also gives a reading when the raw curve does not bracket b*.
Scorer provenance is not required to match across sites (user decision 2026-10-28, SI-115): the check is replaced by a no-op, as in
replication_design.py; the site mix is reported instead.
Usage: python agreement_test.py --rules-scripts DIR --manifest TSV --scores DIR [DIR ...] [--failures CSV ...] --out DIR
"""
import argparse, collections, itertools, json, os, sys
import numpy as np, pandas as pd
from scipy import stats

SEEDS = (0, 1, 2, 3, 4)
LARGE_GAP, NEAR_TIE, NEAR_TIE_REP_TOL, ALPHA = 0.04, 0.02, 0.04, 0.05


def rank_readings(bios, tol=1e-9):
    """Rank 1 = highest biology; readings within tol tie; None (not reached b*) shares the last places (ANALYSIS_SPEC 3.1)."""
    arms = list(bios)
    reached = [a for a in arms if bios[a] is not None]
    order = sorted(reached, key=lambda a: -bios[a])
    ranks, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(bios[order[j + 1]] - bios[order[i]]) <= tol:
            j += 1
        for a in order[i:j + 1]:
            ranks[a] = (i + j) / 2 + 1
        i = j + 1
    n_lost = len(arms) - len(reached)
    for a in arms:
        if bios[a] is None:
            ranks[a] = len(reached) + (n_lost + 1) / 2
    return ranks


def sign_flip_p(taus):
    """One-sided exact sign-flip permutation p of mean(taus) > 0: share of all 2^n sign patterns with mean >= observed."""
    t = np.asarray(taus, float)
    if not len(t) or not np.isfinite(t).all():
        raise ValueError(f"taus must be a non-empty finite list, got {taus}")
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(t))))
    return float(np.mean(signs @ t / len(t) >= t.mean() - 1e-12))


def slope_adjusted(points, bstar):
    """points: {arm: [(B, C), ...]} replication seed means. Reading at b* from the point nearest b* with one pooled within-arm slope."""
    dB, dC = [], []
    for pts in points.values():
        dB += [b - np.mean([p[0] for p in pts]) for b, _ in pts]
        dC += [c - np.mean([p[1] for p in pts]) for _, c in pts]
    den = float(np.dot(dB, dB))
    if den <= 0:
        raise ValueError("no within-arm batch-score variation: slope undefined")
    s = float(np.dot(dB, dC)) / den
    out = {}
    for a, pts in points.items():
        b, c = min(pts, key=lambda p: abs(p[0] - bstar))
        out[a] = c + s * (bstar - b)
    return out, s


def read_pairs(P, out, bstar, arms):
    """Replication reading of every arm in every ranked pair. out = prereg_rules.outcomes() frame of the replication."""
    rows = []
    for key, bs in bstar.items():
        t, c = key.split("|")
        c = int(c)
        zero = P._zero_ref(out, "X1", t, c)
        assert sorted(zero) == list(SEEDS), (key, sorted(zero))
        b0 = float(np.mean([v[0] for v in zero.values()])); c0 = float(np.mean([v[1] for v in zero.values()]))
        cell, pts_all = [], {}
        for a in arms:
            pts = P._cell_points(out, "X1", t, c, a, list(SEEDS))
            curve = [(0.0, b0, c0, 0)] + P.window_curve(pts)
            rd = P.bio_at_bstar(curve, bs)
            lo = hi = None
            for i, (l1, b1, _, _) in enumerate(curve):
                if rd["status"] == "bracketed" and i + 1 < len(curve) and (b1 - bs) * (curve[i + 1][1] - bs) < 0:
                    lo, hi = curve[i], curve[i + 1]; break
            pts_all[a] = [(b, cc) for _, b, cc, _ in curve[1:]]
            cell.append(dict(pair=key, task=t, cond=c, arm=a, bstar=bs, b0_rep=b0, status=rd["status"], rep_bio=rd["bio"],
                             lam_lo=None if lo is None else lo[0], lam_hi=None if hi is None else hi[0],
                             B_lo=None if lo is None else lo[1], B_hi=None if hi is None else hi[1],
                             C_lo=None if lo is None else lo[2], C_hi=None if hi is None else hi[2],
                             n_fail=int(sum(p["n_fail"] for p in pts)), B_max=max(p[1] for p in curve), n_points=len(curve) - 1, lam_max=max(p["lam"] for p in pts)))
        adj, slope = slope_adjusted(pts_all, bs)
        for d in cell:
            d["rep_bio_slope_adj"] = adj[d["arm"]]; d["slope_common"] = slope
        rows += cell
    return pd.DataFrame(rows)


def evaluate(pred, R):
    """Agreement verdict from the prediction dict and the replication readings frame R (read_pairs output)."""
    arms = list(next(iter(pred["readings"].values())))
    R = R.copy(); R["sweep_bio"] = np.nan; R["sweep_rank"] = np.nan
    for key, rd in pred["readings"].items():
        sw = {a: rd[a]["bio_at_bstar"] for a in arms}
        assert rank_readings(sw) == {a: rd[a]["rank"] for a in arms}, f"sweep ranks of {key} do not follow from its readings"
        for a in arms:
            m = (R.pair == key) & (R.arm == a)
            R.loc[m, "sweep_bio"] = sw[a]; R.loc[m, "sweep_rank"] = rd[a]["rank"]
    assert len(R) == len(pred["readings"]) * len(arms) and R.sweep_bio.notna().all(), "replication readings do not cover the prediction"
    R["rep_rank"] = np.nan; R["rep_rank_slope_adj"] = np.nan
    pair_rows = []
    for key, g in R.groupby("pair", sort=False):
        raw = rank_readings({r.arm: (r.rep_bio if r.status == "bracketed" else None) for r in g.itertuples()})
        adj = rank_readings({r.arm: r.rep_bio_slope_adj for r in g.itertuples()})
        R.loc[g.index, "rep_rank"] = g.arm.map(raw); R.loc[g.index, "rep_rank_slope_adj"] = g.arm.map(adj)
        tau = stats.kendalltau(g.sweep_rank, g.arm.map(raw), variant="b")[0]
        tau_adj = stats.kendalltau(g.sweep_rank, g.arm.map(adj), variant="b")[0]
        if not (np.isfinite(tau) and np.isfinite(tau_adj)):
            raise ValueError(f"{key}: Kendall tau-b undefined (constant ranks)")
        g = R.loc[g.index]  # with the ranks just assigned
        best = lambda col: ",".join(g.arm[g[col] == g[col].min()])
        pair_rows.append(dict(pair=key, task=g.task.iloc[0], cond=int(g.cond.iloc[0]), bstar=g.bstar.iloc[0], b0_rep=g.b0_rep.iloc[0],
                              n_bracketed=int((g.status == "bracketed").sum()), tau=tau, tau_slope_adj=tau_adj,
                              top_sweep=best("sweep_rank"), top_rep=best("rep_rank"),
                              spread_sweep=g.sweep_bio.max() - g.sweep_bio.min(),
                              spread_rep=g.rep_bio.max() - g.rep_bio.min() if (g.status == "bracketed").all() else np.nan,
                              mean_abs_reading_diff=(g.rep_bio - g.sweep_bio).abs().mean() if (g.status == "bracketed").all() else np.nan,
                              slope_common=g.slope_common.iloc[0]))
    P_ = pd.DataFrame(pair_rows)
    # criterion 1: large gaps keep their sign (a gap with an unreached side has no replication sign: counted as not kept)
    gaps, rb = [], R.set_index(["pair", "arm"])
    for g in pred["large_gaps"]:
        a, b = rb.loc[(g["pair"], g["a"])], rb.loc[(g["pair"], g["b"])]
        ok = a.status == b.status == "bracketed"
        d = a.rep_bio - b.rep_bio if ok else np.nan
        gaps.append(dict(pair=g["pair"], a=g["a"], b=g["b"], sweep_diff=g["sweep_diff"], rep_diff=d, sign_kept=bool(ok and np.sign(d) == g["sign"]),
                         sign_kept_slope_adj=bool(np.sign(a.rep_bio_slope_adj - b.rep_bio_slope_adj) == g["sign"]), defined=bool(ok)))
    G = pd.DataFrame(gaps)
    # reported alongside: sweep near-ties whose replication difference stays within +/-0.04
    nt = []
    for key, rd in pred["readings"].items():
        for a, b in itertools.combinations(arms, 2):
            if abs(rd[a]["bio_at_bstar"] - rd[b]["bio_at_bstar"]) < NEAR_TIE:
                x, y = rb.loc[(key, a)], rb.loc[(key, b)]
                nt.append(abs(x.rep_bio - y.rep_bio) <= NEAR_TIE_REP_TOL if x.status == y.status == "bracketed" else False)
    p = sign_flip_p(P_.tau); p_adj = sign_flip_p(P_.tau_slope_adj)
    kept = int(G.sign_kept.sum())
    S = dict(n_pairs=len(P_), n_large_gaps=len(G), sign_kept=kept, sign_kept_slope_adj=int(G.sign_kept_slope_adj.sum()),
             mean_tau=float(P_.tau.mean()), p_value=p, mean_tau_slope_adj=float(P_.tau_slope_adj.mean()), p_value_slope_adj=p_adj,
             n_unreached=int((R.status != "bracketed").sum()), n_readings=len(R), n_near_ties=len(nt), n_near_ties_prediction=pred["n_near_ties"],
             near_ties_within_0p04=int(sum(nt)), n_pairs_tau_positive=int((P_.tau > 0).sum()),
             agree=bool(kept == len(G) and P_.tau.mean() > 0 and p < ALPHA),
             agree_slope_adj=bool(int(G.sign_kept_slope_adj.sum()) == len(G) and P_.tau_slope_adj.mean() > 0 and p_adj < ALPHA))
    assert S["n_near_ties"] == S["n_near_ties_prediction"], "sweep near-tie count differs from PREDICTION.json"
    assert S["n_large_gaps"] == len(pred["large_gaps"])
    return R, P_, G, S


def load_outcomes(P, manifest, score_dirs, failure_files):
    """Replication outcomes; fails loudly (prereg_rules) on any manifest row that is neither scored nor failed."""
    M, _ = P.read_manifest(manifest)
    S, files = P.read_scores(score_dirs)
    extra = sorted(set(S.tag) - set(M.tag))
    if extra:
        raise ValueError(f"{len(extra)} scored tags are not in the manifest, e.g. {extra[:3]}")
    F = pd.concat([P.read_failures(f) for f in failure_files], ignore_index=True) if failure_files else pd.DataFrame(columns=["tag", "status", "detail"])
    out = P.outcomes(M, S[S.tag.isin(M.tag)], F, ["X1"])
    site = collections.Counter(zip(S.get("score_site", pd.Series("workstation", index=S.index)).fillna("workstation"), S.scorer_git_sha))
    return out, dict(n_manifest=len(M), n_scored=len(S), n_failed_rows=len(F), score_sites={f"{k[0]}|{k[1][:7]}": v for k, v in site.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-scripts", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--scores", nargs="+", required=True); ap.add_argument("--failures", nargs="*", default=[])
    ap.add_argument("--prediction", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREDICTION.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.rules_scripts)
    import prereg_rules as P
    P.check_provenance = lambda *x, **k: None  # site mixing allowed (SI-115, user 2026-10-28); site mix is reported
    pred = json.load(open(a.prediction))
    out, info = load_outcomes(P, a.manifest, a.scores, a.failures)
    R = read_pairs(P, out, pred["bstar"], list(next(iter(pred["readings"].values()))))
    R, PR, G, S = evaluate(pred, R)
    S.update(info, prediction_sha_note="PREDICTION.json at tag replication-prediction-v1")
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(f"{a.out}/replication_readings.csv", index=False); PR.to_csv(f"{a.out}/replication_task_readings.csv", index=False)
    G.to_csv(f"{a.out}/replication_large_gaps.csv", index=False)
    json.dump(S, open(f"{a.out}/replication_agreement_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    main()
