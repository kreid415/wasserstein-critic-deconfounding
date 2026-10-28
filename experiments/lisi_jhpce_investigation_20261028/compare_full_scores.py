"""Exact comparison of the one-row score files of scripts/score_scib_native.py (all metric columns, float64, NaN equals NaN
only against NaN), same rule as cluster/jhpce/compare_scores.py. Usage: compare_full_scores.py RESULTS/full_scorer LATENTS.sha256
Rows of ws_stored_production are used only for latents whose stored latent_sha256 equals the one in LATENTS.sha256.
Exit 1 if the two shim runs differ, if a shim run differs from a usable stored row, or if the no-shim run is NOT flagged."""
import csv, math, os, sys
IDENT = {"tag", "latent_sha256", "experiment", "task", "arm", "lam", "cond", "seed"}
PROV = {"scorer_git_sha", "scorer_dirty", "score_host", "cpu_model", "cpu_simd", "numba_cpu_name", "scorer_versions", "score_seconds"}
def rd(d):
    out = {}
    for f in sorted(os.listdir(d)):
        r = list(csv.DictReader(open(os.path.join(d, f)))); assert len(r) == 1, f; out[f[:-4]] = r[0]
    return out
def diff(X, Y, tags):
    bad = {}
    for t in tags:
        for c in X[t]:
            if c in IDENT | PROV: continue
            a, b = (float(X[t][c]) if X[t][c] != "" else math.nan), (float(Y[t][c]) if Y[t][c] != "" else math.nan)
            if not ((math.isnan(a) and math.isnan(b)) or a == b): bad.setdefault(c, []).append(t)
    return {c: len(v) for c, v in bad.items()}
if __name__ == "__main__":
    base, shaf = sys.argv[1:3]
    A, B, N, W = (rd(f"{base}/{d}") for d in ("jh_shim_runA", "jh_shim_runB", "jh_noshim_run1_20261023", "ws_stored_production"))
    sha = {l.split()[1][:-4]: l.split()[0] for l in open(shaf) if l.strip()}
    tags = sorted(A); assert set(tags) == set(B) == set(N) == set(W) and len(tags) == 6, tags
    use = [t for t in tags if W[t]["latent_sha256"] == sha[t] == A[t]["latent_sha256"]]
    ncols = sum(c not in IDENT | PROV for c in A[tags[0]])
    res = {"n_latents": len(tags), "n_metric_columns": ncols, "n_stored_rows_usable": len(use),
           "shim_runA_vs_runB": diff(A, B, tags), "shim_runA_vs_stored": diff(A, W, use), "shim_runB_vs_stored": diff(B, W, use),
           "NOSHIM_run1_vs_stored (negative control)": diff(N, W, use)}
    for k, v in res.items(): print(k, v)
    ok = not res["shim_runA_vs_runB"] and not res["shim_runA_vs_stored"] and not res["shim_runB_vs_stored"] and bool(res["NOSHIM_run1_vs_stored (negative control)"]) and len(use) >= 5
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
