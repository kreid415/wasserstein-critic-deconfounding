#!/usr/bin/env python
"""Compare two runs of check_cslocal_identity.sh (MODE inchat vs MODE cslocal). Criteria, fixed before the cs-local run
(2026-10-12):
  C1 env      fit and score interpreters: executable, Python version, every probed package version, torch CUDA/cuDNN,
              GPU name, threadpools, numba CPU name/features equal; thread/BLAS/numba/CUDA/R env vars that the runner
              does not set itself are reported (the runner overrides OMP/MKL/OPENBLAS/NUMBA/KMP for its children).
              CPU model and flag-set hash equal; GPU driver equal. Process placement (namespaces, cgroup, session) is
              reported only.
  C2 cpu      both runs' switch checks PASS, and every one-epoch CPU latent is bit-identical between the runs.
  C3 gpu      within each run, rep1 == rep2 bit for bit (GPU fits deterministic); between runs: bit-identical, else
              reported with max |dz| (GPU kernels may not be bit-reproducible across processes; the bar is then that
              between-run differences are no larger than the within-run ones, which are zero).
  C4 score    both runners' gates PASS 2/2; every metric column of the re-scores equal between the runs and equal to the
              original A1 score files (same latent, same tagged scorer); provenance columns equal.
PASS = C1-C4 all hold. Writes identity_compare.json and prints one line per check.
Usage: compare_identity.py INCHAT_DIR CSLOCAL_DIR [--a1-scores DIR] [--out PATH]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

RUNNER_SET = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "KMP_AFFINITY",
              "PYTHONWARNINGS", "PYTHONUNBUFFERED", "MKL_THREADING_LAYER", "R_HOME", "R_LIBS", "CUDA_VISIBLE_DEVICES")
STEERING = ("OMP_", "MKL_", "OPENBLAS_", "KMP_", "NUMBA_", "CUDA_", "CUBLAS_", "PYTHONHASHSEED", "R_", "LD_LIBRARY_PATH",
            "LD_PRELOAD")
PROV = ("scorer_git_sha", "scorer_dirty", "cpu_simd", "numba_cpu_name", "scorer_versions", "cpu_model", "kbet_seed")
NOT_METRIC = ("tag", "latent_sha256", "experiment", "task", "arm", "lam", "cond", "seed", "score_host", "score_seconds") + PROV


def load_z(p):
    return np.load(p, allow_pickle=False)["z"] if os.path.exists(p) else None


def zcmp(a, b):
    if a is None or b is None:
        return dict(equal=False, max_abs_diff=None, why="missing")
    if a.shape != b.shape:
        return dict(equal=False, max_abs_diff=None, why=f"shape {a.shape} vs {b.shape}")
    return dict(equal=bool(np.array_equal(a, b)), max_abs_diff=float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inchat"); ap.add_argument("cslocal")
    ap.add_argument("--a1-scores", default="/home/kendall/experiment_data/wasserstein-critic-deconfounding/tier12_v2/A1/scores")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    X, Y = a.inchat, a.cslocal
    for d, want in ((X, "inchat"), (Y, "cslocal")):
        got = open(os.path.join(d, "mode")).read().strip()
        if got != want:
            raise SystemExit(f"{d} is mode {got}, expected {want}")
    res = {}
    # ---- C1 env -------------------------------------------------------------------------------------------------
    ex, ey = (json.load(open(os.path.join(d, "env", "env.json"))) for d in (X, Y))
    diffs = []
    for k in ("fit", "score"):
        ix, iy = ex["interpreters"][k], ey["interpreters"][k]
        for f in sorted(set(ix) | set(iy)):
            if f in ("modules",):
                for m in sorted(set(ix[f]) | set(iy[f])):
                    if ix[f].get(m) != iy[f].get(m):
                        diffs.append(f"{k}.{m}: {ix[f].get(m)} vs {iy[f].get(m)}")
            elif ix.get(f) != iy.get(f):
                diffs.append(f"{k}.{f}: {str(ix.get(f))[:80]} vs {str(iy.get(f))[:80]}")
    for f in ("model", "flags_sha256", "n_flags"):
        if ex["cpu"].get(f) != ey["cpu"].get(f):
            diffs.append(f"cpu.{f}: {ex['cpu'].get(f)} vs {ey['cpu'].get(f)}")
    if ex.get("gpu_driver") != ey.get("gpu_driver"):
        diffs.append(f"gpu_driver: {ex.get('gpu_driver')} vs {ey.get('gpu_driver')}")
    envd = {k: (ex["env"].get(k), ey["env"].get(k)) for k in sorted(set(ex["env"]) | set(ey["env"]))
            if k.startswith(STEERING) and ex["env"].get(k) != ey["env"].get(k)}
    unmanaged = {k: v for k, v in envd.items() if k not in RUNNER_SET}
    res["C1_env"] = dict(ok=not diffs and not unmanaged, differences=diffs, steering_env_differences=envd,
                         steering_env_not_overridden_by_runner=unmanaged,
                         placement=dict(inchat=ex.get("where"), cslocal=ey.get("where")),
                         cpu_affinity=(ex["cpu"].get("affinity"), ey["cpu"].get("affinity")))
    # ---- C2 cpu -------------------------------------------------------------------------------------------------
    sx, sy = (pd.read_csv(os.path.join(d, "cpu", "switch_check.csv")) for d in (X, Y))
    lat = sorted({os.path.relpath(p, os.path.join(X, "cpu")) for p in glob.glob(os.path.join(X, "cpu", "run*", "latents", "*.npz"))})
    c2 = {p: zcmp(load_z(os.path.join(X, "cpu", p)), load_z(os.path.join(Y, "cpu", p))) for p in lat}
    res["C2_cpu"] = dict(ok=bool(sx.ok.all()) and bool(sy.ok.all()) and len(lat) == 7 and all(v["equal"] for v in c2.values()),
                         switch_check_pass=(bool(sx.ok.all()), bool(sy.ok.all())), n_latents=len(lat), latents=c2)
    # ---- C3 gpu -------------------------------------------------------------------------------------------------
    tags = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(X, "gpu", "rep1", "latents", "*.npz")))
    c3 = {}
    for t in tags:
        z = {(r, d): load_z(os.path.join(d, "gpu", r, "latents", f"{t}.npz")) for r in ("rep1", "rep2") for d in (X, Y)}
        c3[t] = dict(within_inchat=zcmp(z[("rep1", X)], z[("rep2", X)]), within_cslocal=zcmp(z[("rep1", Y)], z[("rep2", Y)]),
                     between_rep1=zcmp(z[("rep1", X)], z[("rep1", Y)]), between_rep2=zcmp(z[("rep2", X)], z[("rep2", Y)]))
    ok3 = len(tags) == 2 and all(v["within_inchat"]["equal"] and v["within_cslocal"]["equal"] and v["between_rep1"]["equal"]
                                 and v["between_rep2"]["equal"] for v in c3.values())
    res["C3_gpu"] = dict(ok=ok3, n_tags=len(tags), tags=c3)
    # ---- C4 score -----------------------------------------------------------------------------------------------
    gates = {}
    for d in (X, Y):
        lg = open(os.path.join(d, "score", "runner.log")).read()
        gates[d] = "[gate] PASS: 2/2 final" in lg
    c4, ok4 = {}, all(gates.values())
    for p in sorted(glob.glob(os.path.join(X, "score", "out", "scores", "*.csv"))):
        t = os.path.basename(p)[:-4]
        sx_, sy_ = pd.read_csv(p), pd.read_csv(os.path.join(Y, "score", "out", "scores", f"{t}.csv"))
        s0 = pd.read_csv(os.path.join(a.a1_scores, f"{t}.csv"))
        mets = [c for c in s0.columns if c not in NOT_METRIC]
        def same(u, v, cols):
            return [c for c in cols if not (str(u[c].iloc[0]) == str(v[c].iloc[0]) or
                                            (pd.isna(u[c].iloc[0]) and pd.isna(v[c].iloc[0])))]
        c4[t] = dict(n_metrics=len(mets), inchat_vs_cslocal=same(sx_, sy_, mets), inchat_vs_A1=same(sx_, s0, mets),
                     cslocal_vs_A1=same(sy_, s0, mets), provenance_differs=same(sx_, sy_, [c for c in PROV if c in s0.columns]),
                     latent_sha_equal=bool(sx_.latent_sha256.iloc[0] == sy_.latent_sha256.iloc[0] == s0.latent_sha256.iloc[0]))
        ok4 &= not (c4[t]["inchat_vs_cslocal"] or c4[t]["inchat_vs_A1"] or c4[t]["cslocal_vs_A1"] or c4[t]["provenance_differs"]) \
            and c4[t]["latent_sha_equal"]
    res["C4_score"] = dict(ok=ok4 and len(c4) == 2, gates=dict(inchat=gates[X], cslocal=gates[Y]), tags=c4)
    res["PASS"] = all(res[k]["ok"] for k in ("C1_env", "C2_cpu", "C3_gpu", "C4_score"))
    out = a.out or os.path.join(Y, "identity_compare.json")
    json.dump(res, open(out, "w"), indent=1, default=str)
    for k in ("C1_env", "C2_cpu", "C3_gpu", "C4_score"):
        print(("PASS" if res[k]["ok"] else "FAIL"), k)
    print("IDENTITY", "PASS" if res["PASS"] else "FAIL", "|", out)
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
