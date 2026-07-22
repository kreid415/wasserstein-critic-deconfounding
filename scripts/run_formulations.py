"""E9 -- critic formulation comparison (reference vs pooled vs barycenter).

# WHY: A reviewer can argue the critic's pathologies (reference sensitivity, multi-batch
#      scaling collapse) stem from the FIXED-REFERENCE design, not the Wasserstein
#      objective. To settle this we hold the backbone, adversarial weight, and evaluation
#      fixed and vary ONLY the critic's alignment target:
#        reference  - align each batch to a designated reference batch (original design),
#        pooled     - align every batch to the global pool (V-way joint, the fair
#                     critic counterpart to the discriminator),
#        barycenter - align all batches to a learnable virtual centre (Frechet mean).
#      The discriminator is the reference-free control.
# HOW: One row per (formulation, seed) with the full metric suite, incremental CSV, and
#      --resume on (formulation, seed) so wall-limited datasets (cross-species) can finish
#      in a follow-up job. --batch-count drives the scaling variant for the pooled critic.
"""
import argparse
import json
import os

import pandas as pd

from wcd_vae.wcd.experiment import evaluate_config, load_task

SEEDS = [0, 1, 2]
# The three critic formulations + the discriminator control.
ARMS = [
    ("reference", {"critic": True, "reference_batch": 0, "formulation": "reference"}),
    ("pooled", {"critic": True, "reference_batch": None, "formulation": "pooled"}),
    ("barycenter", {"critic": True, "reference_batch": None, "formulation": "barycenter"}),
    ("discriminator", {"critic": False, "reference_batch": None, "formulation": "reference"}),
]


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
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of arms to run (default: all)")
    ap.add_argument("--resume", action="store_true",
                    help="skip (formulation, seed) rows already present in --out")
    args = ap.parse_args()

    with open(args.registry) as fh:
        registry = json.load(fh)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    adata, batch_key, celltype_key, largest = load_task(
        args.dataset, batch_count=args.batch_count, balance=args.balance,
        data_root=args.data_root, registry=registry,
    )
    n_batches = int(adata.obs[batch_key].nunique())
    print(f"[{args.dataset}] n_obs={adata.n_obs} n_batches={n_batches} largest={largest}")

    rows = []
    done_keys = set()
    if args.resume and os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        rows.extend(prev.to_dict("records"))
        if "formulation" in prev.columns and "method" in prev.columns and "seed" in prev.columns:
            # tag = discriminator when method==discriminator else the formulation
            done_keys = {
                (("discriminator" if m == "discriminator" else f), int(s))
                for m, f, s in zip(prev["method"], prev["formulation"], prev["seed"])
            }
        print(f"[resume] loaded {len(rows)} rows; {len(done_keys)} (arm,seed) done", flush=True)

    selected = set(args.arms.split(",")) if args.arms else {a for a, _ in ARMS}

    for tag, cfg in ARMS:
        if tag not in selected:
            continue
        for seed in SEEDS:
            if (tag, int(seed)) in done_keys:
                print(f"  {tag:14s} seed={seed} | SKIP (resume)", flush=True)
                continue
            try:
                row = evaluate_config(adata, batch_key, celltype_key, d_coef=args.d_coef,
                                      seed=seed, epochs=args.epochs, **cfg)
                row.update({"experiment": "E9", "dataset": args.dataset, "arm": tag,
                            "balanced": args.balance, "n_batches": n_batches})
                rows.append(row)
                print(f"  {tag:14s} seed={seed} | iLISI={row['ilisi']:.3f} "
                      f"cLISI={row['clisi']:.3f} ASWb={row['asw_batch']:.3f} "
                      f"ARI={row['ari']:.3f}", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
            except Exception as e:
                print(f"  {tag:14s} seed={seed} FAILED {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
