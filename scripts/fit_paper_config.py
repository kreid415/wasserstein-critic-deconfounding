#!/usr/bin/env python
"""Fit ONE manifest row and write its latent + provenance (replaces scvi_adv_fit.py; code review N9).

Every setting comes from the manifest row; nothing has a default here, and a row that is missing
a column, or still carries a lambda grid INDEX instead of a frozen value, is refused.

Env:  MANIFEST (tsv), TAG (row tag), PREPPED_DIR (prep_scib_task.py outputs), OUT_DIR, WCD_SRC
Out:  OUT_DIR/latents/<tag>.npz   z (posterior mean), batch, celltype, config json, history json
      OUT_DIR/models/<tag>/       trained scvi-tools model (decoder needed for marker/DE analyses)
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REQUIRED = ["tag", "experiment", "task", "counts", "arm", "lam", "n_critic", "adv_input", "zstd", "cond",
            "decoder", "n_latent", "n_layers", "n_hidden", "likelihood", "batch_size", "max_epochs",
            "train_size", "seed", "reference", "extra"]
ADVERSARIAL = {"discriminator", "discriminator_sn", "reference", "reference_fixed", "pooled", "pooled_sn",
               "barycenter", "barycenter_sn", "reference_sn", "mmd", "mmd_ref", "sinkhorn"}


def load_row(manifest, tag):
    m = pd.read_csv(manifest, sep="\t", comment="#", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in m.columns]
    if missing:
        raise KeyError(f"manifest lacks columns {missing}")
    r = m[m.tag == tag]
    if len(r) != 1:
        raise KeyError(f"tag {tag!r} matches {len(r)} rows")
    r = r.iloc[0].to_dict()
    empty = [c for c in REQUIRED if r[c] == ""]
    if empty:
        raise ValueError(f"{tag}: empty fields {empty}")
    return r


def subsample(adata, spec, seed):
    """Deterministic subsamples for X3 (composition shift) and X8 (number of batches)."""
    import anndata as ad  # noqa: F401
    # The subsample depends on the design cell (task, V, subset, dose) only, never on the training
    # seed, so every arm and seed of one design cell sees the same cells.
    obs = adata.obs
    if spec["kind"] == "batches":
        rng = np.random.default_rng(30_000 + 100 * spec["subset"] + spec["n_batches"])
        sizes = obs["batch"].value_counts()
        ok = sizes[sizes >= spec["n_cells"] // spec["n_batches"]].index.sort_values()
        if len(ok) < spec["n_batches"]:
            raise ValueError(f"only {len(ok)} batches have >= {spec['n_cells'] // spec['n_batches']} cells")
        pick = np.random.default_rng(20_000 + 100 * spec["subset"] + spec["n_batches"]).choice(
            np.asarray(ok), spec["n_batches"], replace=False)
        per = spec["n_cells"] // spec["n_batches"]
        idx = np.concatenate([rng.choice(np.where(obs["batch"].values == b)[0], per, replace=False) for b in sorted(pick)])
        return adata[np.sort(idx)].copy()
    if spec["kind"] == "composition":
        # deplete the pre-declared cell types in the pre-declared batch, then subsample to n_cells
        rng = np.random.default_rng(40_000 + spec["deplete_pct"])
        b, types, pct = spec["batch"], spec["types"], spec["deplete_pct"]
        keep = np.ones(adata.n_obs, dtype=bool)
        hit = np.where((obs["batch"].astype(str).values == b) & obs["celltype"].astype(str).isin(types).values)[0]
        drop = rng.choice(hit, int(round(len(hit) * pct / 100)), replace=False) if len(hit) else hit
        keep[drop] = False
        idx = np.where(keep)[0]
        if len(idx) > spec["n_cells"]:
            idx = np.sort(rng.choice(idx, spec["n_cells"], replace=False))
        return adata[idx].copy()
    raise ValueError(spec)


def provenance():
    import scvi
    import torch
    sha = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip() != ""
    return dict(git_sha=sha, git_dirty=dirty, scvi=scvi.__version__, torch=torch.__version__,
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))


def main():
    r = load_row(os.environ["MANIFEST"], os.environ["TAG"])
    out_dir = os.environ["OUT_DIR"]
    npz = os.path.join(out_dir, "latents", f"{r['tag']}.npz")
    if os.path.exists(npz):
        print(f"[skip] {r['tag']} exists", flush=True)
        return
    import anndata as ad
    import scvi
    import torch
    import importlib.util
    spec = importlib.util.spec_from_file_location("wcd_data", os.path.join(os.environ["WCD_SRC"], "wcd_vae", "wcd", "data.py"))
    wcd_data = importlib.util.module_from_spec(spec); spec.loader.exec_module(wcd_data)
    select_reference_batch = wcd_data.select_reference_batch   # the repo's rule, loaded without the package __init__

    seed = int(r["seed"])
    arm = r["arm"]
    extra = json.loads(r["extra"])
    try:
        lam = float(r["lam"])
    except ValueError:
        raise ValueError(f"{r['tag']}: lam={r['lam']!r} is a grid index; freeze the lambda grid first")
    if arm in ADVERSARIAL and lam <= 0:
        raise ValueError(f"{r['tag']}: adversarial arm with lam={lam}")

    a = ad.read_h5ad(os.path.join(os.environ["PREPPED_DIR"], f"{r['task']}__{r['counts']}.h5ad"))
    if "subsample" in extra:
        a = subsample(a, extra["subsample"], seed)
    a = a[:, a.var["highly_variable"].values].copy()
    a.obs["batch"] = a.obs["batch"].astype(str).astype("category")

    # reference batch: 'auto' = repo rule (max cell-type entropy, ties by size) on the cells actually used
    ref_name = None
    if r["reference"] == "auto":
        ref_name = str(select_reference_batch(a, "batch", "celltype"))
    else:
        ref_name = sorted(a.obs["batch"].cat.categories)[int(r["reference"])]

    common = dict(n_latent=int(r["n_latent"]), max_epochs=int(r["max_epochs"]), batch_size=int(r["batch_size"]),
                  seed=seed, conditioned=bool(int(r["cond"])))
    backbone = dict(n_layers=int(r["n_layers"]), n_hidden=int(r["n_hidden"]), gene_likelihood=r["likelihood"],
                    train_size=float(r["train_size"]))
    t0 = time.time()
    if arm in ("scanvi", "sysvi"):
        z, model = fit_baseline(a, arm, lam, r, common, backbone)
    else:
        from scvi_adversarial_plan import fit_adversarial_scvi
        n_critic = int(r["n_critic"])
        z, model = fit_adversarial_scvi(
            a, "batch", adversary=arm, d_coef=lam, n_critic=(n_critic if n_critic > 0 else None),
            reference_batch=ref_name, adv_input=r["adv_input"], zstd=bool(int(r["zstd"])),
            model_name=r["decoder"], adv_hidden=int(extra.get("adv_width", 128)),
            critic_lr=float(extra.get("adv_lr", 1e-4)), disc_lr=float(extra.get("adv_lr", 1e-3)),
            bary_iter=int(extra.get("bary_iter", 10)), bary_warm_iter=extra.get("bary_warm_iter"),
            **common, **backbone)
    secs = time.time() - t0
    hist = {k: v.iloc[:, 0].astype(float).tolist() for k, v in getattr(model, "history_", {}).items()
            if hasattr(v, "iloc")}
    cfg = dict(row=r, reference_name=ref_name, n_cells=int(a.n_obs), n_batches=int(a.obs["batch"].nunique()),
               fit_seconds=round(secs, 1), **provenance())
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    tmp = npz + ".tmp.npz"
    np.savez_compressed(tmp, z=np.asarray(z, dtype=np.float32), obs_names=a.obs_names.to_numpy(dtype="U128"),
                        batch=a.obs["batch"].astype(str).to_numpy(dtype="U64"),
                        celltype=a.obs["celltype"].astype(str).to_numpy(dtype="U64"),
                        config=json.dumps(cfg), history=json.dumps(hist))
    model.save(os.path.join(out_dir, "models", r["tag"]), overwrite=True, save_anndata=False)
    os.replace(tmp, npz)
    print(f"[fit] {r['tag']} {secs:.0f}s z{np.asarray(z).shape} finite={bool(np.isfinite(z).all())}", flush=True)


def fit_baseline(a, arm, lam, r, common, backbone):
    import scvi
    scvi.settings.seed = common["seed"]
    if arm == "scanvi":
        # scIB's scANVI protocol (scib 1.1.7 integration.scanvi): scVI with the same backbone, then
        # SCANVI.from_scvi_model trained min(10, max(2, round(epochs / 3))) epochs on all cells.
        s = a.copy(); s.X = s.layers["counts"].copy()
        scvi.model.SCVI.setup_anndata(s, batch_key="batch", labels_key="celltype")
        vae = scvi.model.SCVI(s, n_latent=common["n_latent"], n_layers=backbone["n_layers"],
                              n_hidden=backbone["n_hidden"], gene_likelihood=backbone["gene_likelihood"])
        vae.train(max_epochs=common["max_epochs"], batch_size=common["batch_size"], train_size=backbone["train_size"],
                  early_stopping=False, enable_progress_bar=False)
        m = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="UnknownUnknown")
        m.train(max_epochs=int(min(10, max(2, round(common["max_epochs"] / 3.0)))), batch_size=common["batch_size"],
                train_size=backbone["train_size"], early_stopping=False, enable_progress_bar=False)
        return m.get_latent_representation(), m
    # sysVI (scvi-tools 1.4.2 scvi.external.SysVI): Gaussian likelihood on scIB's normalised X,
    # VampPrior (default), strength knob = z_distance_cycle_weight (= lam; scvi-tools default 2.0).
    from scvi.external import SysVI
    s = a.copy()
    SysVI.setup_anndata(s, batch_key="batch")
    m = SysVI(s, n_latent=common["n_latent"], n_hidden=backbone["n_hidden"], n_layers=backbone["n_layers"])
    m.train(max_epochs=common["max_epochs"], batch_size=common["batch_size"], train_size=backbone["train_size"],
            early_stopping=False, enable_progress_bar=False,
            plan_kwargs=dict(z_distance_cycle_weight=lam))   # TrainingPlan forwards extra kwargs to SysVAE.loss
    return m.get_latent_representation(), m


if __name__ == "__main__":
    main()
