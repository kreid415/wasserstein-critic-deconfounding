"""Local end-to-end smoke test for every experiment harness.

# WHY: wave 2019132 burned 8,056 cpu-h on three defects that were all pure code bugs --
#   a ragged training_history dict that killed 45 tasks AFTER they finished training,
#   colliding --embed-out filenames that destroyed 76% of the embeddings, and flags/kwargs
#   that had to be verified by hand. Every one of those is reproducible on a laptop with
#   400-cell synthetic data in minutes. This script is the gate to run BEFORE any wave.
#
# WHAT IT CANNOT DO: synthetic data is easier than production. A local pass does NOT
#   establish that the adversary engages on real data, that runtimes fit a wall, or that
#   the allocation has headroom -- those need the real-data preflight gate and a
#   sacctmgr budget check. This catches plumbing, not science.

Usage:  python scripts/smoke_local.py [--keep]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import anndata as ad
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_dataset(path, n_cells, n_batches, n_types, seed=0, batch_strength=2.0):
    """Synthetic counts with a REAL batch effect and real cell-type structure.

    # WHY a real batch effect: a smoke test on batch-free data would let an inert
    #   adversary pass. The batch shift is applied in log-space to a gene block so the
    #   latent has something to remove.
    """
    rng = np.random.default_rng(seed)
    # WHY 2,600 genes: prep_data runs sc.pp.filter_cells(min_genes=300) then
    #   sc.pp.highly_variable_genes(n_top_genes=2000, batch_key=...). A fixture with
    #   fewer detected genes than min_genes drops EVERY cell, and HVG selection on an
    #   empty frame raises "No objects to concatenate" -- which looks like 13 harness
    #   failures but is one fixture defect. The fixture must clear the real thresholds.
    n_genes = 2600
    batch = rng.integers(0, n_batches, n_cells)
    ctype = rng.integers(0, n_types, n_cells)

    base = rng.gamma(2.0, 1.0, size=(n_types, n_genes))          # cell-type programme
    shift = rng.normal(0, batch_strength, size=(n_batches, n_genes))  # batch effect
    # a floor on the rate keeps ~all genes detected, clearing min_genes=300
    lam = base[ctype] * np.exp(shift[batch] * 0.3) + 0.8
    X = rng.poisson(np.clip(lam, 1e-3, 50.0)).astype("float32")

    obs = pd.DataFrame({
        "batch": [f"B{i}" for i in batch],
        "celltype": [f"T{i}" for i in ctype],
    }, index=[f"c{i}" for i in range(n_cells)])
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.layers["counts"] = X.copy()
    a.write_h5ad(path)
    return path


def build_fixtures(root):
    data = os.path.join(root, "data")
    os.makedirs(data, exist_ok=True)
    specs = {
        # name          cells  batches  types
        "smoke_small": (400, 3, 4),
        "smoke_multi": (600, 5, 5),   # >=4 batches so E8 has something to sweep
    }
    registry = {}
    for name, (n, nb, nt) in specs.items():
        p = os.path.join(data, f"{name}.h5ad")
        make_dataset(p, n, nb, nt, seed=abs(hash(name)) % 1000)
        registry[name] = {
            "path": p, "file": f"{name}.h5ad", "batch_key": "batch",
            "celltype_key": "celltype", "n_batches": nb, "n_obs": n,
            "prep": "rna", "role": "smoke", "modality": "rna",
        }
    reg_path = os.path.join(root, "registry.json")
    with open(reg_path, "w") as fh:
        json.dump(registry, fh, indent=2)
    return reg_path, registry


def paga_works():
    """Is PAGA functional in THIS environment?

    # WHY: scanpy 1.9.8 + igraph 1.0.0 (the local sandbox) crash inside
    #   scanpy's get_sparse_from_igraph, while the cluster env (scanpy 1.11.5) is fine.
    #   PAGA is a real metric we depend on, so a local crash is an ENVIRONMENT limit, not
    #   a code defect -- reporting it as a smoke failure would train us to ignore the
    #   smoke test, which is worse than not having one.
    """
    import warnings
    warnings.simplefilter("ignore")
    import numpy as np
    import scanpy as sc
    import anndata as _ad
    import pandas as _pd

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 12)).astype("float32")
    a = _ad.AnnData(X=X, obs=_pd.DataFrame(
        {"g": _pd.Categorical(rng.choice(["a", "b", "c"], n))},
        index=[f"c{i}" for i in range(n)]))
    try:
        sc.pp.neighbors(a, use_rep="X")
        sc.tl.paga(a, groups="g")
        return True
    except Exception:
        return False


def run(name, cmd, root):
    env = dict(os.environ)
    env.update(KMP_AFFINITY="disabled", OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2",
               MKL_THREADING_LAYER="SEQUENTIAL", PYTHONWARNINGS="ignore")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
    ok = r.returncode == 0
    if not ok:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-6:])
        print(f"FAIL {name}\n{tail}", flush=True)
    else:
        print(f"pass {name}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the temp workspace")
    args = ap.parse_args()

    root = tempfile.mkdtemp(prefix="wcd_smoke_")
    reg, registry = build_fixtures(root)
    out = os.path.join(root, "results")
    emb = os.path.join(root, "emb")
    os.makedirs(out, exist_ok=True)
    py = sys.executable
    E = ["--epochs", "20"]
    results = {}

    # E1 -- lambda sweep path, both heads (critic exercises the reference machinery)
    for head in ("critic", "discriminator"):
        results[f"E1_{head}"] = run(f"E1_{head}", [
            py, "scripts/run_experiment.py", "--experiment", "E1", "--dataset", "smoke_small",
            "--registry", reg, "--backbone", "NB_uncond", "--head", head,
            "--d-coef-only", "0.2", "--seed-only", "0", *E,
            "--embed-out", emb, "--out", f"{out}/E1_{head}.csv"], root)

    # E2 -- conditioned backbone is a DIFFERENT code path from unconditioned
    for bb in ("NB_uncond", "NB"):
        results[f"E2_{bb}"] = run(f"E2_{bb}", [
            py, "scripts/run_experiment.py", "--experiment", "E2", "--dataset", "smoke_small",
            "--registry", reg, "--backbone", bb, "--head", "discriminator", "--seed-only", "0",
            *E, "--embed-out", emb, "--out", f"{out}/E2_{bb}.csv"], root)

    # E4 -- reference designs, incl. the joint/rotating modes
    for design in ("fixed_ref0", "joint", "discriminator"):
        results[f"E4_{design}"] = run(f"E4_{design}", [
            py, "scripts/run_reference.py", "--dataset", "smoke_small", "--registry", reg,
            "--backbone", "NB_uncond", "--ref-design-only", design, *E,
            "--embed-out", emb, "--out", f"{out}/E4_{design}.csv"], root)

    # E9 -- formulation arms
    for arm in ("reference", "pooled", "barycenter"):
        results[f"E9_{arm}"] = run(f"E9_{arm}", [
            py, "scripts/run_formulations.py", "--dataset", "smoke_small", "--registry", reg,
            "--backbone", "NB_uncond", "--arms", arm, *E,
            "--embed-out", emb, "--out", f"{out}/E9_{arm}.csv"], root)

    # E5 -- biology; writes a DIRECTORY, not a csv
    results["E5"] = run("E5", [
        py, "scripts/run_biology.py", "--dataset", "smoke_small", "--registry", reg,
        "--backbone", "NB_uncond", *E, "--outdir", f"{out}/E5"], root)

    # E8 -- --batch-count path
    results["E8"] = run("E8", [
        py, "scripts/run_experiment.py", "--experiment", "E8", "--dataset", "smoke_multi",
        "--registry", reg, "--backbone", "NB_uncond", "--head", "critic", "--seed-only", "0",
        "--batch-count", "3", *E, "--out", f"{out}/E8.csv"], root)

    # nested CV -- the path that lost 45 tasks to the ragged-history bug.
    # Its outer-fold block calls calculate_additional_metrics -> sc.tl.paga, so it can
    # only run where PAGA works (see paga_works()).
    cv_possible = paga_works()
    if not cv_possible:
        print("skip CV (PAGA unavailable in this environment -- scanpy/igraph "
              "incompatibility; the nested-CV path must be smoke-tested on the cluster)",
              flush=True)
    results["CV"] = run("CV", [
        py, "scripts/hyperparameter_search.py", "--dataset", "smoke_small",
        "--output-dir", f"{out}/CV", "--registry", reg, "--backbone", "NB_uncond",
        "--criterion", "scib", "--reference-rule", "entropy",
        "--outer-fold-only", "0", "--head-only", "critic",
        "--lambda-grid", "0.0,0.2", "--epochs", "20", "--inner-epochs", "20",
        "--outer-folds", "3", "--inner-folds", "2"], root) if cv_possible else None

    print("\n=== POST-CONDITIONS ===")
    checks = {}

    # embeddings must be unique per (dataset, arm) -- the collision that lost 76% of them
    n_emb = 0
    if os.path.isdir(emb):
        for d in os.listdir(emb):
            sub = os.path.join(emb, d)
            if os.path.isdir(sub):
                n_emb += len([f for f in os.listdir(sub) if f.endswith(".npz")])
    n_emb_runs = sum(1 for k, v in results.items()
                     if v and (k.startswith(("E1_", "E2_", "E4_", "E9_"))))
    checks["embeddings_not_clobbered"] = n_emb >= n_emb_runs
    print(f"  embeddings: {n_emb} files for {n_emb_runs} embed-writing runs "
          f"-> {'OK' if checks['embeddings_not_clobbered'] else 'COLLISION'}")

    # every result csv must be non-empty and carry the core metrics
    core = ["ilisi", "clisi", "ari", "nmi", "linear_batch_lift", "linear_label_lift"]
    bad = []
    for f in sorted(os.listdir(out)):
        if not f.endswith(".csv"):
            continue
        p = os.path.join(out, f)
        if os.path.getsize(p) < 2:
            bad.append(f"{f}:empty")
            continue
        df = pd.read_csv(p)
        miss = [m for m in core if m not in df.columns]
        if len(df) == 0:
            bad.append(f"{f}:norows")
        elif miss:
            bad.append(f"{f}:missing{miss}")
    checks["csvs_well_formed"] = not bad
    print(f"  result csvs: {'all well-formed' if not bad else 'PROBLEMS ' + str(bad)}")

    # nested CV must have written its per-fold output (the ragged-DataFrame regression)
    cvdir = os.path.join(out, "CV")
    cvf = [f for f in os.listdir(cvdir)] if os.path.isdir(cvdir) else []
    checks["cv_wrote_output"] = (
        any("final_best_results" in f for f in cvf) if cv_possible else True)
    print(f"  nested CV output: {cvf if cvf else 'NONE'} -> "
          + ("SKIPPED (no PAGA here)" if not cv_possible
             else ("OK" if checks["cv_wrote_output"] else "MISSING")))

    skipped = [k for k, v in results.items() if v is None]
    graded = {k: v for k, v in results.items() if v is not None}
    n_pass = sum(1 for v in graded.values() if v)
    print(f"\n  harnesses: {n_pass}/{len(graded)} passed"
          + (f"  ({len(skipped)} skipped: {skipped})" if skipped else ""))
    for k, v in graded.items():
        if not v:
            print(f"    FAILED: {k}")
    all_ok = all(graded.values()) and all(checks.values())
    print(f"\n=== SMOKE {'PASS' if all_ok else 'FAIL'} ===")
    if args.keep:
        print(f"workspace kept: {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
