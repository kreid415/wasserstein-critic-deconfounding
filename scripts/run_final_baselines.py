#!/usr/bin/env python
"""Reproducible baseline latents: every classical method x all 8 datasets x 5 seeds.

Emits latents (z+batch+celltype+X_pca) to DURABLE storage in the SAME npz format the adversarial
sweep uses, so baselines score through the identical scripts/score_final_config.py path. No inline
scoring here (keeps latent-gen and scoring separate, as for the adversarial arm).

Methods: unintegrated(PCA), harmony, scanorama, scvi, scanvi. (LinearSCVI is covered by the
adversary=none control in the adversarial sweep; Seurat runs separately in R via seurat_score.)

Seeds: PCA/harmony/scanorama are near-deterministic but we still tag by seed and set numpy/scvi
seeds so the 5-seed CI machinery treats every method identically. scvi/scanvi are genuinely
stochastic and use scvi.settings.seed.

Tag convention matches the sweep: <dataset>_XZB_<method>_s<seed>  (XZB = final Baseline).
Resume-safe: skips a tag whose npz already exists. Env: WCD_EMBED_OUT, WCD_DATA (registry data root).
"""
import os, sys, argparse, numpy as np, scanpy as sc
sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import run_baselines` works from any cwd
from wcd_vae.wcd.experiment import load_task, save_embedding
import run_baselines as RB   # reuse the emb_* implementations verbatim

METHODS = ["unintegrated", "harmony", "scanorama", "scvi", "scanvi"]
DATASETS = ["pancreas", "immune", "lung", "sim1", "sim2", "atac_small",
            "immune_hum_mou", "atac_large"]
SEEDS = [0, 1, 2, 3, 4]

def seed_all(seed):
    import random; random.seed(seed); np.random.seed(seed)
    try:
        import scvi; scvi.settings.seed = seed
    except Exception:
        pass
    try:
        import torch; torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="configs/dataset_registry.json")
    ap.add_argument("--data-root", default=os.environ.get("WCD_DATA"))
    ap.add_argument("--embed-out", default=os.environ.get("WCD_EMBED_OUT",
        "/home/kendall/experiment_data/wasserstein-critic-deconfounding/embeddings_final"))
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()

    # ephemeral guard (same rule as the sweep driver)
    emb = args.embed_out
    import subprocess
    fst = subprocess.run(["df", "-PT", emb], capture_output=True, text=True).stdout.splitlines()
    fstype = fst[-1].split()[1] if len(fst) > 1 else "?"
    assert "workspaces/" not in emb and fstype not in ("tmpfs", "ramfs"), \
        f"FATAL: embed-out {emb} is ephemeral (fstype={fstype})"
    os.makedirs(emb, exist_ok=True)
    print(f"[guard] baseline latents -> {emb} (fstype={fstype}) OK", flush=True)

    import json
    registry = json.load(open(args.registry))
    for ds in args.datasets:
        adata, bk, ck, _ref = load_task(ds, data_root=args.data_root, registry=registry)
        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata, n_comps=50)
        for method in args.methods:
            for seed in args.seeds:
                tag = f"{ds}_XZB_{method}_s{seed}"
                out = os.path.join(emb, f"{tag}.npz")
                if os.path.exists(out):
                    print(f"[skip] {tag}", flush=True); continue
                try:
                    seed_all(seed)
                    a = adata.copy()
                    RB.METHODS[method](a, bk, ck)          # writes a.obsm["X_emb"]
                    a.obs["batch"] = a.obs[bk]; a.obs["celltype"] = a.obs[ck]
                    save_embedding(a, emb, tag, "batch", "celltype", embed_key="X_emb")
                    print(f"[ok] {tag} z={a.obsm['X_emb'].shape}", flush=True)
                except Exception as e:
                    print(f"[FAIL] {tag}: {type(e).__name__} {str(e)[:120]}", flush=True)

if __name__ == "__main__":
    main()
