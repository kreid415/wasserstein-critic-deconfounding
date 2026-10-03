#!/usr/bin/env python
"""Cost of the non-finite-loss guard on the GPU, and GPU bit identity of the guarded fitter (engineering check).

The guard (scripts/scvi_adversarial_plan.py, commit b0564fb) adds one device-to-host sync per training step. This
script fits the same manifest row with the fitter of a base commit (no guard; a `git worktree` of it) and with the
fitter of this checkout, alternating the order in every repeat (repeat 1 base first, repeat 2 guard first, ...), one
fit at a time, and compares fit_seconds (the fitter's own timer: training + posterior mean, no process start) and the
latents (same seed, same GPU: expected max|dz| = 0).

Rows: atac_small, seed 100, conditioned, design backbone (rows of experiments/stage_runner_e2e/manifest.tsv), arms
`none` (scvi-tools' own training step) and `discriminator` lambda 1 (fewest GPU operations per step, so the largest
relative cost of a sync), --epochs epochs.

Usage: python experiments/stage_runner_e2e/time_guard.py --base-root WORKTREE --fit-python PY --prepped-dir DIR \
           --out OUT [--epochs 20] [--repeats 3]
Out: OUT/fits.csv (one row per fit), OUT/summary.json (median fit_seconds per arm and version, ratio guard/base,
     max|dz| between all fits of an arm). Exit 1 if a fit fails or any latent differs.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ARMS = {"none": "E2E_none_l0_c1_s100", "discriminator": "E2E_discriminator_l1_c1_s100"}


def git_sha(root):
    return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--base-root", required=True, help="checkout of the commit without the guard (e.g. 878d8ea)")
    ap.add_argument("--fit-python", required=True)
    ap.add_argument("--prepped-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise FileExistsError(f"{a.out} exists: use a fresh directory")
    roots = {"base": os.path.abspath(a.base_root), "guard": REPO}
    shas = {k: git_sha(v) for k, v in roots.items()}
    m = pd.read_csv(os.path.join(HERE, "manifest.tsv"), sep="\t", comment="#", dtype=str, keep_default_na=False)
    rows = m[m.tag.isin(ARMS.values())].copy()
    if len(rows) != len(ARMS):
        raise ValueError(f"expected rows {sorted(ARMS.values())}, found {rows.tag.tolist()}")
    rows["max_epochs"] = str(a.epochs)
    os.makedirs(a.out)
    man = os.path.join(a.out, "manifest.tsv")
    rows.to_csv(man, sep="\t", index=False)
    env0 = dict(os.environ, KMP_AFFINITY="disabled", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                MKL_THREADING_LAYER="SEQUENTIAL", PYTHONWARNINGS="ignore", MANIFEST=man, PREPPED_DIR=a.prepped_dir)
    recs, z = [], {}
    for r, arm in itertools.product(range(1, a.repeats + 1), ARMS):
        order = ["base", "guard"] if r % 2 else ["guard", "base"]
        for pos, ver in enumerate(order):
            out = os.path.join(a.out, f"{arm}_{ver}_r{r}")
            env = dict(env0, TAG=ARMS[arm], OUT_DIR=out, WCD_SRC=os.path.join(roots[ver], "src"))
            t0 = time.time()
            with open(out + ".log", "w") as log:
                p = subprocess.run([a.fit_python, os.path.join(roots[ver], "scripts", "fit_paper_config.py")], env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
            wall = time.time() - t0
            if p.returncode != 0:
                raise RuntimeError(f"{arm} {ver} repeat {r}: fitter exit {p.returncode}; log {out}.log")
            with np.load(os.path.join(out, "latents", f"{ARMS[arm]}.npz"), allow_pickle=False) as d:
                cfg = json.loads(str(d["config"]))
                z[(arm, ver, r)] = d["z"].astype(np.float64)
            if cfg["git_sha"] != shas[ver] or cfg["git_dirty"]:
                raise RuntimeError(f"{arm} {ver}: fitted at {cfg['git_sha']} dirty={cfg['git_dirty']}, expected {shas[ver]}")
            recs.append(dict(arm=arm, version=ver, repeat=r, position=pos, fit_seconds=cfg["fit_seconds"],
                             wall_s=round(wall, 1), gpu=cfg["gpu"], git_sha=cfg["git_sha"],
                             finite=bool(np.isfinite(z[(arm, ver, r)]).all())))
            print(f"{arm} {ver} r{r}: fit {cfg['fit_seconds']} s, wall {wall:.1f} s", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(a.out, "fits.csv"), index=False)
    summary = dict(epochs=a.epochs, repeats=a.repeats, git=shas, gpu=sorted(set(df.gpu)), arms={})
    bad = []
    for arm in ARMS:
        g = df[df.arm == arm]
        med = g.groupby("version").fit_seconds.median().to_dict()
        keys = [k for k in z if k[0] == arm]
        dz = max(float(np.abs(z[k1] - z[k2]).max()) for k1, k2 in itertools.combinations(keys, 2))
        summary["arms"][arm] = dict(median_fit_seconds=med, ratio_guard_over_base=med["guard"] / med["base"],
                                    fit_seconds={v: g[g.version == v].fit_seconds.tolist() for v in ("base", "guard")},
                                    max_abs_dz_all_pairs=dz, all_finite=bool(g.finite.all()))
        if dz != 0.0 or not g.finite.all():
            bad.append(arm)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["arms"], indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
