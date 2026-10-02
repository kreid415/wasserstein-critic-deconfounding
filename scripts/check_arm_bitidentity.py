#!/usr/bin/env python
"""Before/after bit-identity gate for the adversary plan (docs/SPECS_missing_arms.md, 2026-10-02).

CLAIM UNDER TEST: every arm that existed at the base commit gives the SAME latent, bit for bit, with the base
code and with the working-tree code, at a fixed seed (max|dz| = 0). The new arms and options (discriminator_r1,
discriminator_ref, sampler, importance weights) must not touch the existing code paths.

How: the base commit's scripts/scvi_adversarial_plan.py and src/wcd_vae/wcd/ are exported with `git show`
into a temporary tree; one subprocess per code version fits every configuration on the same synthetic data
(CPU only, CUDA hidden, 1 thread, so floating-point reductions are deterministic) and writes the latents; the
latents are compared exactly. Exit 1 on any difference or on a configuration that failed in either version.

Usage: python scripts/check_arm_bitidentity.py --base 8cda6de --out docs/bitidentity_existing_arms.csv
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# every arm of the base commit (_ARMS + valid _sn variants), conditioned and unconditioned, + option variants
BASE_ARMS = ["none", "scvi_adv", "discriminator", "discriminator_sn", "reference", "reference_fixed", "pooled",
             "barycenter", "mmd", "sinkhorn", "mmd_ref", "reference_sn", "reference_fixed_sn", "pooled_sn",
             "barycenter_sn"]
CRITICS = ("reference", "reference_fixed", "pooled", "barycenter")


def configs():
    out = []
    for arm in BASE_ARMS:
        for cond in (True, False):
            if arm == "scvi_adv" and not cond:
                continue      # scvi-tools disables its classifier without batch conditioning
            out.append(dict(arm=arm, cond=cond, adv_input="mean", zstd=False, model="SCVI"))
    out += [dict(arm="pooled", cond=True, adv_input="sample", zstd=True, model="SCVI"),
            dict(arm="discriminator", cond=True, adv_input="sample", zstd=True, model="SCVI"),
            dict(arm="mmd", cond=False, adv_input="mean", zstd=True, model="SCVI"),
            dict(arm="pooled", cond=True, adv_input="mean", zstd=False, model="LinearSCVI"),
            dict(arm="discriminator", cond=False, adv_input="mean", zstd=False, model="LinearSCVI")]
    for c in out:
        c["key"] = f"{c['arm']}|c{int(c['cond'])}|{c['adv_input']}|z{int(c['zstd'])}|{c['model']}"
    return out


CHILD = r'''
import json, os, sys, numpy as np
cfgs = json.loads(os.environ["BI_CONFIGS"])
sys.path.insert(0, os.environ["BI_SCRIPTS"])
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_available(), "CUDA must be hidden for a deterministic comparison"
import anndata as ad
import scvi_adversarial_plan as plan

def toy(n=600, g=60, k=3, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.integers(0, k, n); ct = rng.integers(0, 3, n)
    mu = np.exp(rng.normal(0, 1, (3, g)))[ct] * np.exp(rng.normal(0, 0.5, (k, g)))[b]
    X = rng.poisson(mu).astype(np.float32)
    a = ad.AnnData(X); a.layers["counts"] = X.copy()
    a.obs["batch"] = [f"b{i}" for i in b]; a.obs["batch"] = a.obs["batch"].astype("category")
    a.obs["celltype"] = [f"t{i}" for i in ct]
    return a

a = toy()
res = {}
for c in cfgs:
    base = c["arm"][:-3] if c["arm"].endswith("_sn") else c["arm"]
    n_critic = 5 if base in ("reference", "reference_fixed", "pooled", "barycenter") else None
    # options of the new code (absent from the base configurations, so the base code never sees them)
    opt = {}
    if "iw" in c:   # 'ones' or a constant weight for (b1, t0); 1 for every other (batch, cell type) pair
        val = 1.0 if c["iw"] == "ones" else float(c["iw"])
        opt["iw_weights"] = {f"b{i}": {f"t{j}": (val if (i, j) == (1, 0) else 1.0) for j in range(3)} for i in range(3)}
    for k in ("r1_gamma", "sampler"):
        if k in c:
            opt[k] = c[k]
    try:
        z, _ = plan.fit_adversarial_scvi(a, "batch", adversary=c["arm"], d_coef=1.0, n_critic=n_critic,
                                         reference_batch="b0", adv_input=c["adv_input"], zstd=c["zstd"],
                                         n_latent=4, max_epochs=2, batch_size=128, seed=0,
                                         conditioned=c["cond"], model_name=c["model"], **opt)
        res[c["key"]] = np.asarray(z, dtype=np.float32)
    except Exception as e:  # fl: allow FL101 recorded per config and turned into a failed gate row by the parent
        res[c["key"] + "::error"] = np.array(repr(e))
np.savez(os.environ["BI_OUT"], **res)
'''


def export_base(base, dest):
    """Write the base commit's plan + wcd modules into dest/{scripts,src/wcd_vae/wcd}."""
    files = subprocess.run(["git", "-C", ROOT, "ls-tree", "-r", "--name-only", base, "src/wcd_vae/wcd"],
                           check=True, capture_output=True, text=True).stdout.split()
    files.append("scripts/scvi_adversarial_plan.py")
    for f in files:
        blob = subprocess.run(["git", "-C", ROOT, "show", f"{base}:{f}"], check=True, capture_output=True).stdout
        p = os.path.join(dest, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(blob)
    return dest


def run_version(code_root, cfgs, out_npz):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               KMP_AFFINITY="disabled", WCD_SRC=os.path.join(code_root, "src"),
               BI_SCRIPTS=os.path.join(code_root, "scripts"), BI_CONFIGS=json.dumps(cfgs), BI_OUT=out_npz)
    subprocess.run([sys.executable, "-c", CHILD], check=True, env=env)
    return dict(np.load(out_npz, allow_pickle=False))


def compare(z_old, z_new):
    """Exact comparison used by the gate: same shape, finite, max|dz| == 0."""
    if z_old.shape != z_new.shape:
        raise AssertionError(f"shape {z_old.shape} vs {z_new.shape}")
    if not (np.isfinite(z_old).all() and np.isfinite(z_new).all()):
        raise AssertionError("non-finite latent")
    d = float(np.abs(z_old.astype(np.float64) - z_new.astype(np.float64)).max())
    if d != 0.0:
        raise AssertionError(f"max|dz| = {d:.3e}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="commit whose arms must be reproduced")
    ap.add_argument("--out", required=True, help="CSV of max|dz| per configuration")
    a = ap.parse_args()
    cfgs = configs()
    with tempfile.TemporaryDirectory() as tmp:
        old = run_version(export_base(a.base, os.path.join(tmp, "base")), cfgs, os.path.join(tmp, "old.npz"))
        new = run_version(ROOT, cfgs, os.path.join(tmp, "new.npz"))
    rows, bad = [], 0
    for c in cfgs:
        k = c["key"]
        if k not in old or k not in new:
            err = str(old.get(k + "::error", "")) + " | " + str(new.get(k + "::error", ""))
            rows.append(dict(config=k, status="error", max_abs_dz="", detail=err))
            bad += 1
            continue
        try:
            d = compare(old[k], new[k])
            rows.append(dict(config=k, status="identical", max_abs_dz=d, detail=f"n={old[k].shape}"))
        except AssertionError as e:
            rows.append(dict(config=k, status="differs", max_abs_dz="", detail=str(e)))
            bad += 1
    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "status", "max_abs_dz", "detail"])
        w.writeheader()
        w.writerows(rows)
    print(f"[bit-identity] base={a.base} configs={len(cfgs)} identical={len(cfgs) - bad} failed={bad} -> {a.out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
