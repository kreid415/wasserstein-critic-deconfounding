#!/usr/bin/env python
"""Replication design from the eight-task penalty-strength sweep (rescope 2026-11-02, user choice 'Two bracketing strengths').

For every task x decoder pair: b0 (plain scVI seed mean), M (best failure-free seed-mean batch score per penalty, all sweep strengths),
b_common = min M, b* = b0 + 0.5 (b_common - b0), low-signal if b_common - b0 < 0.02 (prereg_rules constants). For every ranked pair x
penalty: walk up the curve (lambda = 0 anchor, then every sweep strength, seed means over successful runs) to the FIRST crossing of b*
(prereg_rules.bio_at_bstar logic) and take the two strengths that bracket it. Replication rows = X1 rows (seeds 0-4) at those strengths
(existing committed X1 rows where the strength is on the X1 grid, else built with prereg_rules.extension_rows), plus the 80 X1 plain-scVI
rows. Also writes the sweep reading C at b* per penalty and the per-pair ranks (input for the prediction file) and a cost estimate from
the sweep's measured fit seconds.

Scorer-provenance checks are skipped here (prereg_rules.check_provenance replaced by a no-op) because this script uses only seed-mean
batch and biology scores to place strengths; site mixing is governed by SI-115 and checked in the analysis pipeline.
Usage: <scvi-api python> replication_design.py --out DIR   (run from the prereg-tier12-v3 checkout's scripts/ directory)
"""
import argparse, glob, json, math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import prereg_rules as P

D = "/home/kendall/experiment_data/wasserstein-critic-deconfounding"
P.check_provenance = lambda *a, **k: None

def load():
    r4 = f"{D}/tier12_v3/rules/r4_pass2"
    M1, _ = P.read_manifest(f"{r4}/m.r4a1.tsv")
    M1 = M1[(M1.experiment == "A1") & M1.task.isin(["atac_small", "immune", "sim1"])]
    files1 = [l.strip() for l in open(f"{r4}/scores_list.txt") if l.strip()]
    S1, _ = P.read_scores(files1)
    F1 = P.read_failures(f"{r4}/failures_merged.csv")
    M2, _ = P.read_manifest(f"{D}/code/wt_sweep_paper/manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv")
    M2 = M2[M2.experiment == "A1"]
    S2, _ = P.read_scores(f"{D}/jhpce_tier12/pilot5_jhpce/scores")
    F2 = P.read_failures(f"{D}/jhpce_tier12/pilot5_jhpce/failures.csv")
    o1 = P.outcomes(M1, S1[S1.tag.isin(M1.tag)], F1, ["A1"])
    o2 = P.outcomes(M2, S2[S2.tag.isin(M2.tag)], F2, ["A1"])
    return pd.concat([o1, o2], ignore_index=True)

def first_crossing(curve, bstar):
    """curve: [(lam, B, C, n_fail)] starting with the lambda=0 anchor. Returns (lo, hi, i) of the first bracket, or None."""
    for i, (l1, b1, c1, _) in enumerate(curve):
        if abs(b1 - bstar) <= P.TIE_TOL:
            return (l1, l1, i)
        if i + 1 < len(curve):
            l2, b2, c2, _ = curve[i + 1]
            if abs(b2 - bstar) > P.TIE_TOL and (b1 - bstar) * (b2 - bstar) < 0:
                return (l1, l2, i)
    return None

def fit_seconds():
    secs = {}
    for f in glob.glob(f"{D}/tier12_v*/**/ledger/*.csv", recursive=True) + glob.glob(f"{D}/jhpce_tier12/pilot5_jhpce/ledger/*.csv"):
        try:
            L = pd.read_csv(f, dtype=str, keep_default_na=False)
        except Exception:
            continue
        if "tag" in L and "fit_seconds" in L:
            for t, s in zip(L.tag, L.fit_seconds):
                try:
                    v = float(s)
                except ValueError:
                    continue
                if t.startswith("A1_") and v > 0:
                    secs[t] = v
    return secs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = load()
    seeds = (100, 101, 102)
    pairs, cells, specs = {}, [], []
    for (t, c) in sorted(set(zip(out.task, out.cond_i))):
        zero = P._zero_ref(out, "A1", t, c)
        assert sorted(zero) == list(seeds), (t, c, sorted(zero))
        b0 = float(np.mean([v[0] for v in zero.values()])); c0 = float(np.mean([v[1] for v in zero.values()]))
        pts, Mx = {}, {}
        for arm in P.ADV_ARMS:
            pts[arm] = P._cell_points(out, "A1", t, c, arm, list(seeds))
            ok = [P._mean(list(p["B"].values())) for p in pts[arm] if not p["n_fail"] and p["B"]]
            Mx[arm] = max(ok)
        b_common = min(Mx.values()); bstar = b0 + P.Q_BSTAR * (b_common - b0)
        low = bool(b_common - b0 < 2 * P.DELTA_MIN)
        pairs[f"{t}|{c}"] = dict(b0=b0, c0=c0, M=Mx, b_common=b_common, bstar=bstar, low_signal=low,
                                 set_by=min(Mx, key=Mx.get))
        for arm in P.ADV_ARMS:
            curve = [(0.0, b0, c0, 0)] + P.window_curve(pts[arm])
            fc = first_crossing(curve, bstar)
            n_cross = sum(1 for i in range(len(curve) - 1) if (curve[i][1] - bstar) * (curve[i + 1][1] - bstar) < 0)
            rd = P.bio_at_bstar(curve, bstar)
            cell = dict(task=t, cond=c, arm=arm, low_signal=low, n_crossings=n_cross, reading_status=rd["status"],
                        sweep_bio_at_bstar=rd["bio"], lo=None, hi=None)
            if fc:
                lo, hi, i = fc
                cell.update(lo=lo, hi=hi, lo_B=curve[i][1], hi_B=curve[i + 1][1] if lo != hi else curve[i][1],
                            fails_in_bracket=int(curve[i][3] + (curve[i + 1][3] if lo != hi else 0)))
                if not low:
                    for lam in sorted({lo, hi}):
                        if lam > 0:
                            specs.append(dict(experiment="X1", task=t, cond=int(c), arm=arm, lam=float(lam)))
            cells.append(cell)
    # ranks of the sweep readings per ranked pair (higher bio = better; unreached at the bottom, as the spec)
    C = pd.DataFrame(cells)
    C["sweep_rank"] = np.nan
    for (t, c), g in C[~C.low_signal].groupby(["task", "cond"]):
        r = g.sweep_bio_at_bstar.astype(float).rank(ascending=False, method="average")
        C.loc[g.index, "sweep_rank"] = r
    C.to_csv(f"{a.out}/cells.csv", index=False)
    json.dump(pairs, open(f"{a.out}/pairs.json", "w"), indent=1)
    # replication rows
    X, _ = P.read_manifest(f"{D}/code/wcd_final-review-fixes/manifests/paper_manifest_stock_pilot_u5_b10_v3.x1_sinkhorn_redo.tsv")
    X1 = X[X.experiment == "X1"]
    have, need = [], []
    for sp in specs:
        sel = (X1.task == sp["task"]) & (X1.cond.astype(int) == sp["cond"]) & (X1.arm == sp["arm"]) & \
              [math.isclose(P._lamf(x), sp["lam"], rel_tol=1e-9) if not math.isnan(P._lamf(x)) else False for x in X1.lam]
        (have.append(X1[sel]) if sel.any() else need.append(sp))
    parts = have + ([P.extension_rows(X1, need)] if need else [])
    none = X1[X1.arm == "none"]
    R = pd.concat(parts + [none], ignore_index=True)[P.BPM.COLS]
    assert not R.tag.duplicated().any()
    R.to_csv(f"{a.out}/replication_rows.tsv", sep="\t", index=False)
    # cost from sweep fit seconds (seed mean at the same task|cond|arm|lam) x 5 seeds
    secs = fit_seconds(); A = out[["tag", "task", "cond_i", "arm", "lamf"]]
    A = A.assign(s=A.tag.map(secs))
    per = A.groupby(["task", "cond_i", "arm", "lamf"]).s.mean()
    est = []
    for r in R.itertuples():
        k = (r.task, int(r.cond), r.arm, P._lamf(r.lam) if r.arm != "none" else 0.0)
        v = per.get(k, np.nan)
        est.append(v)
    R["est_fit_s"] = est
    summ = dict(n_rows=len(R), n_adversarial=int((R.arm != "none").sum()), n_plain=int((R.arm == "none").sum()),
                n_from_committed_x1=int(sum(len(h) for h in have)), n_new_strength_rows=int(len(R) - len(none) - sum(len(h) for h in have)),
                ranked_pairs=int(sum(not p["low_signal"] for p in pairs.values())), low_signal_pairs=[k for k, p in pairs.items() if p["low_signal"]],
                cells_without_crossing=C[(~C.low_signal) & C.lo.isna()][["task", "cond", "arm"]].to_dict("records"),
                multi_crossing_cells=int(((~C.low_signal) & (C.n_crossings > 1)).sum()),
                est_fit_seconds_total=float(np.nansum(est)), est_missing=int(np.isnan(est).sum()),
                est_by_task_h=R.groupby("task").est_fit_s.sum().div(3600).round(1).to_dict())
    R.to_csv(f"{a.out}/replication_rows_with_cost.tsv", sep="\t", index=False)
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))

if __name__ == "__main__":
    main()
