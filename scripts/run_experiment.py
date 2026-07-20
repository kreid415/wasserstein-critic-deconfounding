"""Unified driver for the reviewer-response experiments E1/E2/E4/E8.

# WHY: E1 (lambda_adv Pareto), E2 (backbone sweep), E4 (reference design), E8 (multibatch
#      scaling) are the SAME loop over (config) -> evaluate_config -> row, differing only
#      in which axis is swept. One audited driver keeps the training/eval path identical
#      across experiments so cross-experiment comparisons are apples-to-apples.
#      (E3 external baselines and E5 biology use their own scripts; E6 has its own.)
# HOW: --experiment selects the config grid; each config is trained with the shared
#      harness.evaluate_config and appended to a tidy CSV (one row per fitted model).
"""
import argparse
import json
import os

import pandas as pd

from wcd_vae.wcd.experiment import evaluate_config, load_task

# lambda_adv grid for the Pareto front (R2.2 / R3.3). 0 == no adversarial term.
LAMBDA_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
BACKBONES = ["scCRAFT", "scVI_NB", "Gaussian", "ZINB"]
SEEDS = [0, 1, 2]


def configs_for(experiment, task_entry):
    """Yield config dicts (kwargs for evaluate_config) for the requested experiment."""
    if experiment == "E1":
        # Pareto front: both heads x full lambda grid x 3 seeds, default backbone.
        for critic in (False, True):
            for lam in LAMBDA_GRID:
                for seed in SEEDS:
                    yield {"critic": critic, "d_coef": lam, "seed": seed,
                           "reference_batch": 0 if critic else None}
    elif experiment == "E2":
        # Architecture generality: 4 backbones x both heads x 3 seeds at the paper's
        # operating point (d_coef=0.2). Capacity sweep handled by --zdim/--override.
        for backbone in BACKBONES:
            for critic in (False, True):
                for seed in SEEDS:
                    yield {"critic": critic, "d_coef": 0.2, "seed": seed, "backbone": backbone,
                           "reference_batch": 0 if critic else None}
    elif experiment == "E8":
        # Multibatch scaling handled by caller varying --batch-count; here just both
        # heads x 3 seeds at the operating point with the full metric suite.
        for critic in (False, True):
            for seed in SEEDS:
                yield {"critic": critic, "d_coef": 0.2, "seed": seed,
                       "reference_batch": 0 if critic else None}
    else:
        raise ValueError(f"run_experiment does not handle '{experiment}' (see dedicated script)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["E1", "E2", "E8"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-count", type=int, default=None)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--zdim", type=int, default=256)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    with open(args.registry) as fh:
        registry = json.load(fh)
    entry = registry[args.dataset]

    # create output dir upfront so the incremental per-config CSV writes below succeed.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    adata, batch_key, celltype_key, largest = load_task(
        args.dataset, batch_count=args.batch_count, balance=args.balance,
        data_root=args.data_root, registry=registry,
    )
    # reference-batch name -> index for the critic (largest batch, matching scripts).
    print(f"[{args.dataset}] n_obs={adata.n_obs} batch_key={batch_key} largest={largest} "
          f"n_batches={adata.obs[batch_key].nunique()}")

    rows = []
    for i, cfg in enumerate(configs_for(args.experiment, entry)):
        cfg = dict(cfg, epochs=args.epochs, z_dim=args.zdim)
        try:
            row = evaluate_config(adata, batch_key, celltype_key, **cfg)
            row.update({"experiment": args.experiment, "dataset": args.dataset,
                        "balanced": args.balance, "batch_count": adata.obs[batch_key].nunique()})
            rows.append(row)
            print(f"  [{i}] {row['method']:13s} bb={row['backbone']:8s} lam={row['d_coef']:<5} "
                  f"seed={row['seed']} | iLISI={row['ilisi']:.3f} cLISI={row['clisi']:.3f} "
                  f"ASWb={row['asw_batch']:.3f} ARI={row['ari']:.3f}", flush=True)
            # incremental save so a timeout still yields partial results
            pd.DataFrame(rows).to_csv(args.out, index=False)
        except Exception as e:
            print(f"  [{i}] cfg={cfg} FAILED {type(e).__name__}: {e}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
