#!/usr/bin/env python
"""Cross-host gate, score parts.

g4  Scoring equivalence: the same latent files scored on two hosts. Every metric present in either file must agree
    within --tol (default 1e-6, absolute); NaN must be NaN on both. Prints per-metric max |diff|; exit 1 on failure.
g3  Hardware effect on scores: ONE scoring run over the latents of both hosts (same scorer, same env, same prepped
    file). Per latent: raw batch mean and raw bio mean = means of the raw scIB metrics in BATCH_METRICS / BIO_METRICS
    of scripts/score_scib_native.py (read from its source, the analysis definition of record). Per config: paired
    difference remote - local. Per (arm, cond): mean paired difference over seeds, its SE (sd/sqrt(n)), the local seed
    SD, and flag = |mean diff| > 2 SE AND |mean diff| > 0.5 x local seed SD.

Usage:
  python scripts/gate_compare_scores.py g4 --a local.csv --b remote.csv [--tol 1e-6] --out g4.csv
  python scripts/gate_compare_scores.py g3 --scores scores.csv --local-prefix L_ --remote-prefix J_ --manifest M.tsv --out-dir DIR
"""
import argparse
import ast
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
META_COLS = {"tag", "host", "prepped", "score_seconds", "pins", "host_fit", "host_score"}


def metric_lists():
    src = open(os.path.join(HERE, "score_scib_native.py")).read()
    vals = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in ("BATCH_METRICS", "BIO_METRICS"):
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    if set(vals) != {"BATCH_METRICS", "BIO_METRICS"}:
        raise ValueError("BATCH_METRICS / BIO_METRICS not found in score_scib_native.py")
    return vals["BATCH_METRICS"], vals["BIO_METRICS"]


def g4(a):
    A, B = pd.read_csv(a.a), pd.read_csv(a.b)
    A, B = A[A.tag.str.startswith(a.a_prefix)].copy(), B[B.tag.str.startswith(a.b_prefix)].copy()
    if len(A) == 0 or len(B) == 0:
        raise ValueError("no rows after prefix filtering")
    key = lambda d: d.tag.str.replace(r"^[A-Za-z0-9]+_(?=Q_|S_)", "", regex=True)     # G4_/G4b_/L_/J_ prefix -> config tag
    A.index, B.index = key(A), key(B)
    if sorted(A.index) != sorted(B.index):
        raise ValueError(f"latent sets differ: {sorted(A.index)} vs {sorted(B.index)}")
    metrics = sorted((set(A.columns) | set(B.columns)) - META_COLS)
    rows, bad = [], 0
    for m in metrics:
        if m not in A or m not in B:
            rows.append(dict(metric=m, max_abs_diff=np.inf, status="missing on one host")); bad += 1; continue
        x, y = pd.to_numeric(A.loc[B.index, m], errors="raise").to_numpy(float), pd.to_numeric(B[m], errors="raise").to_numpy(float)
        nan_mismatch = bool((np.isnan(x) != np.isnan(y)).any())
        both = ~np.isnan(x) & ~np.isnan(y)
        if not both.any() and not nan_mismatch:          # not computed on either host (e.g. hvg_overlap for embeddings)
            rows.append(dict(metric=m, n=0, all_nan=True, max_abs_diff=np.nan, status="NaN on both (not computed)"))
            continue
        d = float(np.abs(x[both] - y[both]).max()) if both.any() else np.nan
        ok = (not nan_mismatch) and d <= a.tol
        bad += not ok
        rows.append(dict(metric=m, n=int(both.sum()), all_nan=False, max_abs_diff=d,
                         status="ok" if ok else ("NaN mismatch" if nan_mismatch else "exceeds tol")))
    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)
    print(R.to_string(index=False))
    n_comp = int((~R.all_nan).sum())
    print(f"G4 {'PASS' if bad == 0 else 'FAIL'}: {n_comp - bad}/{n_comp} computed metrics within {a.tol} "
          f"({int(R.all_nan.sum())} not computed on either host)")
    sys.exit(1 if bad else 0)


def sign_flip_null(P):
    """Calibration of the flag rule under 'no systematic host effect': paired differences symmetric about 0. For every
    (arm, cond) x {batch_mean, bio_mean} test, all 2^n sign assignments of the n paired differences are enumerated
    (local seed SD fixed) and the fraction flagged is that test's null flag probability; the number of flags over all
    tests is their convolution (tests treated as independent)."""
    probs, obs = [], 0
    for _, g in P.groupby(["arm", "cond"]):
        for m in ("batch_mean", "bio_mean"):
            d = g[f"d_{m}"].to_numpy(float)
            n = len(d)
            if n < 2:
                continue
            sd = float(g[f"{m}_local"].std(ddof=1))
            def fl(x):
                md, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
                return bool(abs(md) > 2 * se and abs(md) > 0.5 * sd)
            signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
            probs.append(float(np.mean([fl(d * sg) for sg in signs])))
            obs += fl(d)
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, [1 - p, p])
    return dict(n_tests=len(probs), observed_flags=int(obs), expected_flags=round(float(sum(probs)), 3),
                p_at_least_observed=round(float(dist[obs:].sum()), 4))


def g3(a):
    BATCH, BIO = metric_lists()
    S = pd.read_csv(a.scores)
    M = pd.read_csv(a.manifest, sep="\t", dtype=str, keep_default_na=False).set_index("tag")
    for m in BATCH + BIO:
        if m not in S:
            raise ValueError(f"metric {m} missing from {a.scores}")
    used_bio = [m for m in BIO if S[m].notna().any()]
    if S[BATCH].isna().any().any():
        raise ValueError("NaN in a batch metric")
    if S[used_bio].isna().any().any():
        raise ValueError("bio metric NaN for some latents but not others")
    S["batch_mean"] = S[BATCH].mean(axis=1)
    S["bio_mean"] = S[used_bio].mean(axis=1)
    loc = S[S.tag.str.startswith(a.local_prefix)].copy(); rem = S[S.tag.str.startswith(a.remote_prefix)].copy()
    loc["cfg"] = loc.tag.str[len(a.local_prefix):]; rem["cfg"] = rem.tag.str[len(a.remote_prefix):]
    if sorted(loc.cfg) != sorted(rem.cfg) or len(loc) != len(set(loc.cfg)):
        raise ValueError("local / remote config sets differ or repeat")
    P = loc.set_index("cfg").join(rem.set_index("cfg"), lsuffix="_local", rsuffix="_remote")
    P["arm"] = M.loc[P.index, "arm"].values
    P["cond"] = M.loc[P.index, "cond"].astype(int).values
    P["seed"] = M.loc[P.index, "seed"].astype(int).values
    extra = [m for m in ("trajectory",) if m not in used_bio and m in S and S[m].notna().any()]   # scorer before 2026-10-03 kept it out
    per_metric = BATCH + used_bio + extra + ["batch_mean", "bio_mean"]
    for m in per_metric:
        P[f"d_{m}"] = P[f"{m}_remote"] - P[f"{m}_local"]
    P.reset_index().to_csv(os.path.join(a.out_dir, "g3_paired_scores.csv"), index=False)
    out = []
    for (arm, cond), g in P.groupby(["arm", "cond"]):
        n = len(g)
        rec = dict(arm=arm, cond=cond, n_seeds=n)
        for m in ("batch_mean", "bio_mean"):
            d = g[f"d_{m}"].to_numpy(float)
            md = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            sd_loc = float(g[f"{m}_local"].std(ddof=1)) if n > 1 else float("nan")
            flag = bool(abs(md) > 2 * se and abs(md) > 0.5 * sd_loc) if n > 1 else False
            rec.update({f"{m}_mean_diff": md, f"{m}_se": se, f"{m}_local_seed_sd": sd_loc,
                        f"{m}_abs_diff_over_se": abs(md) / se if se > 0 else (float("inf") if md != 0 else 0.0),
                        f"{m}_flag": flag})
        out.append(rec)
    G = pd.DataFrame(out)
    G.to_csv(os.path.join(a.out_dir, "g3_group_differences.csv"), index=False)
    pm = pd.DataFrame([dict(metric=m, mean_diff=float(P[f"d_{m}"].mean()), mean_abs_diff=float(P[f"d_{m}"].abs().mean()),
                            max_abs_diff=float(P[f"d_{m}"].abs().max()),
                            local_sd_over_all_latents=float(P[f"{m}_local"].std(ddof=1))) for m in per_metric])
    pm.to_csv(os.path.join(a.out_dir, "g3_per_metric_differences.csv"), index=False)
    summ = dict(n_configs=len(P), batch_metrics=BATCH, bio_metrics_used=used_bio, reported_not_in_means=extra,
                groups=len(G), flagged=G[(G.batch_mean_flag) | (G.bio_mean_flag)][["arm", "cond"]].to_dict("records"),
                null_sign_flip=sign_flip_null(P))
    json.dump(summ, open(os.path.join(a.out_dir, "g3_summary.json"), "w"), indent=1)
    print(G.round(5).to_string(index=False))
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p4 = sub.add_parser("g4"); p4.add_argument("--a", required=True); p4.add_argument("--b", required=True)
    p4.add_argument("--tol", type=float, default=1e-6); p4.add_argument("--out", required=True)
    p4.add_argument("--a-prefix", default="G4_"); p4.add_argument("--b-prefix", default="G4_")
    p3 = sub.add_parser("g3"); p3.add_argument("--scores", required=True); p3.add_argument("--manifest", required=True)
    p3.add_argument("--local-prefix", default="L_"); p3.add_argument("--remote-prefix", default="J_")
    p3.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "g4":
        g4(a)
    else:
        os.makedirs(a.out_dir, exist_ok=True)
        g3(a)


if __name__ == "__main__":
    main()
