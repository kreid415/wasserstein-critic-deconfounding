"""E4 — isolate the reference-based formulation (R2.5).

# WHY: R2.5 argues the critic's reference sensitivity / scaling issues may come from
#      forcing every batch to align to ONE reference, not from the Wasserstein objective.
#      To separate the two we vary ONLY the reference design and hold everything else fixed.
# HOW: For the critic we run three reference MODES --
#        fixed    : each batch index in turn as the sole reference (reference-coverage:
#                   how much does the choice of reference matter?),
#        rotating : reference cycles across batches per epoch (no privileged anchor),
#        joint    : reference drawn at random per epoch (approx all-pairs on average).
#      The discriminator (no reference concept) is included as the reference-free control.
#      Output: one row per (mode/reference, seed) with the full metric suite, so we can
#      report reference-coverage spread AND whether rotating/joint removes the pathology.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from wcd_vae.wcd.experiment import evaluate_config, load_task

SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-count", type=int, default=None)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--d-coef", type=float, default=0.2)
    args = ap.parse_args()

    with open(args.registry) as fh:
        registry = json.load(fh)

    adata, batch_key, celltype_key, largest = load_task(
        args.dataset, batch_count=args.batch_count, balance=args.balance,
        data_root=args.data_root, registry=registry,
    )
    n_batches = int(adata.obs[batch_key].nunique())
    print(f"[{args.dataset}] n_obs={adata.n_obs} n_batches={n_batches} largest={largest}")

    rows = []

    def run(tag, **cfg):
        for seed in SEEDS:
            try:
                row = evaluate_config(adata, batch_key, celltype_key, d_coef=args.d_coef,
                                      seed=seed, epochs=args.epochs, **cfg)
                row.update({"experiment": "E4", "dataset": args.dataset, "ref_design": tag,
                            "balanced": args.balance, "n_batches": n_batches})
                rows.append(row)
                print(f"  {tag:16s} seed={seed} | iLISI={row['ilisi']:.3f} cLISI={row['clisi']:.3f} "
                      f"ASWb={row['asw_batch']:.3f} ARI={row['ari']:.3f}", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
            except Exception as e:
                print(f"  {tag:16s} seed={seed} FAILED {type(e).__name__}: {e}", flush=True)

    # 1. Reference-coverage: fixed critic, each batch as the sole reference.
    for ref in range(n_batches):
        run(f"fixed_ref{ref}", critic=True, reference_batch=ref, reference_mode="fixed")

    # 2. Rotating and joint reference modes (reference_batch base = 0).
    run("rotating", critic=True, reference_batch=0, reference_mode="rotating")
    run("joint", critic=True, reference_batch=0, reference_mode="joint")

    # 3. Discriminator control (reference-free).
    run("discriminator", critic=False, reference_batch=None, reference_mode="fixed")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # Reference-coverage summary: spread of metrics across the fixed-reference choices.
    fixed = df[df["ref_design"].str.startswith("fixed_ref")]
    if len(fixed):
        for m in ("ilisi", "clisi", "asw_batch", "ari"):
            vals = fixed.groupby("ref_design")[m].mean()
            print(f"  coverage[{m}] range=[{vals.min():.3f},{vals.max():.3f}] "
                  f"spread={vals.max() - vals.min():.3f} cv={vals.std() / (abs(vals.mean()) + 1e-9):.3f}")
    print(f"\nWrote {len(df)} rows -> {args.out}")
    _ = np  # kept for potential coverage stats extension


if __name__ == "__main__":
    main()
