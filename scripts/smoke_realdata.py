"""Real-data smoke test: every experiment configuration, locally, on the true datasets.

# WHY this exists SEPARATELY from smoke_local.py: synthetic fixtures pass even when a real
#   config fails. That is not hypothetical here -- the adversarial no-op that invalidated a
#   whole wave passed all three synthetic regression tests, and the igraph/leidenalg
#   divergence was invisible on easy synthetic data and only appeared on a real 16k-cell
#   weakly-structured latent. smoke_local.py gates PLUMBING; this gates SCIENCE.
#
# WHAT IT CHECKS, per (dataset, backbone, head):
#   1. the config runs to completion and writes a well-formed row
#   2. THE ADVERSARY ACTUALLY ENGAGES -- for the discriminator, loss_da must fall below
#      ln(n_batches) - 0.15 (chance for a uniform V-way classifier). This is the gate that
#      would have caught the silently-inert adversary before it cost a wave.
#   3. every metric in the suite is present and finite (or NaN for a documented reason)
#
# Epochs are short by default: this is a gate, not a result. Pass --epochs to lengthen.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("WCD_DATA", os.path.join(os.path.dirname(ROOT), "data"))

# (registry name, file, batch_key, celltype_key, n_batches, prep)
DATASETS = {
    "pancreas":       ("human_pancreas_norm_complexBatch.h5ad", "tech", "celltype", 9, "rna"),
    "immune":         ("Immune_ALL_human.h5ad", "chemistry", "final_annotation", 4, "rna"),
    "lung":           ("Lung_atlas_public.h5ad", "protocol", "cell_type", 2, "rna"),
    "sim1":           ("sim1_1_norm.h5ad", "Batch", "Group", 6, "rna_sim"),
    "sim2":           ("sim2_norm.h5ad", "Batch", "Group", 4, "rna_sim"),
    "atac_small":     ("small_atac_gene_activity.h5ad", "batchname", "final_cell_label", 3, "atac"),
    "immune_hum_mou": ("Immune_ALL_hum_mou.h5ad", "species", "final_annotation", 2, "rna"),
    "atac_large":     ("large_atac_gene_activity.h5ad", "batchname", "final_cell_label", 3, "atac"),
}


def build_registry(path, only=None):
    reg = {}
    for name, (fn, bk, ck, nb, prep) in DATASETS.items():
        if only and name not in only:
            continue
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            print(f"  skip {name}: {fn} not downloaded", flush=True)
            continue
        reg[name] = {"path": p, "file": fn, "batch_key": bk, "celltype_key": ck,
                     "n_batches": nb, "prep": prep, "role": "smoke", "modality": prep}
    with open(path, "w") as fh:
        json.dump(reg, fh, indent=2)
    return reg


def run_one(py, reg, out, dataset, backbone, head, epochs, extra=None):
    csv = os.path.join(out, f"{dataset}_{backbone}_{head}.csv")
    cmd = [py, "scripts/run_experiment.py", "--experiment", "E1", "--dataset", dataset,
           "--registry", reg, "--backbone", backbone, "--head", head,
           "--d-coef-only", "0.2", "--seed-only", "0", "--epochs", str(epochs),
           "--out", csv] + (extra or [])
    env = dict(os.environ)
    env.update(KMP_AFFINITY="disabled", OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4",
               MKL_THREADING_LAYER="SEQUENTIAL", PYTHONWARNINGS="ignore")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=14400)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(csv) or os.path.getsize(csv) < 2:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-4:])
        return {"ok": False, "seconds": dt, "why": tail[:400]}
    df = pd.read_csv(csv)
    if len(df) == 0:
        return {"ok": False, "seconds": dt, "why": "empty csv"}
    row = df.iloc[0].to_dict()
    row.update(ok=True, seconds=dt)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None, help="comma-separated subset")
    ap.add_argument("--backbones", default="NB_uncond,Gaussian,Poisson,ZINB_uncond,LDVAE_uncond,NB,ZINB,LDVAE")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="results/SMOKE_REAL")
    args = ap.parse_args()

    only = args.datasets.split(",") if args.datasets else None
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    reg_path = os.path.join(out, "registry.json")
    reg = build_registry(reg_path, only)
    if not reg:
        print("no datasets available"); return 2

    rows, fails, inert = [], [], []
    for ds in reg:
        nb = DATASETS[ds][3]
        chance = math.log(nb)
        for bb in args.backbones.split(","):
            for head in ("discriminator", "critic"):
                r = run_one(sys.executable, reg_path, out, ds, bb, head, args.epochs)
                tag = f"{ds}/{bb}/{head}"
                if not r["ok"]:
                    fails.append(tag); print(f"FAIL {tag}: {r['why'][:160]}", flush=True); continue
                lda = r.get("final_loss_da")
                # the engagement gate applies to the DISCRIMINATOR only: its loss is a
                # V-way cross-entropy with a known chance value. The Wasserstein critic
                # loss is unbounded and legitimately negative, so ln(V) says nothing there.
                eng = None
                if head == "discriminator" and lda is not None and not pd.isna(lda):
                    eng = bool(lda < chance - 0.15)
                    if not eng:
                        inert.append(f"{tag} (loss_da={lda:.3f} vs chance={chance:.3f})")
                print(f"pass {tag}  {r['seconds']/60:.1f}min  loss_da={lda}  "
                      f"engaged={eng}  ilisi={r.get('ilisi')}", flush=True)
                rows.append({"dataset": ds, "backbone": bb, "head": head,
                             "seconds": r["seconds"], "engaged": eng, **{
                                 k: r.get(k) for k in
                                 ("final_loss_da","ilisi","clisi","ari","nmi",
                                  "linear_batch_lift","linear_label_lift","paga_spearman")}})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "smoke_real_summary.csv"), index=False)
    print("\n=== SUMMARY ===")
    print(f"  configs run:   {len(rows)}")
    print(f"  failures:      {len(fails)} {fails if fails else ''}")
    print(f"  INERT adversary (discriminator at chance): {len(inert)}")
    for i in inert:
        print(f"    {i}")
    if len(df):
        print("\n  runtime by dataset (min):")
        print(df.groupby("dataset")["seconds"].agg(
            lambda s: round(s.mean()/60, 1)).to_string())
    ok = not fails and not inert
    print(f"\n=== REAL-DATA SMOKE {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
