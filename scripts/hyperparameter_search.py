"""Nested cross-validation with in-fold lambda_adv selection.

# WHY this script was rewritten: the previous version hardcoded three dataset paths under
#   /workspaces/data (a sandbox layout that no longer exists on either cluster), imported
#   `wcd_vae.data` / `wcd_vae.hyperparameter` (the modules live under `wcd_vae.wcd.`), and
#   exposed no --backbone, so it could not run the unconditioned backbones the revision
#   uses. It now goes through the same registry + load_task path as every other harness,
#   which also gives it the entropy-based reference batch for free.
"""

import argparse
import json
import os
import warnings

from wcd_vae.wcd.experiment import load_task
from wcd_vae.wcd.hyperparameter import run_comprehensive_nested_cv
from wcd_vae.wcd.primitives import seed_everything

warnings.filterwarnings("ignore")

LAMBDA_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Nested CV with in-fold lambda selection")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", dest="output_dir", required=True)
    ap.add_argument("--registry", default="configs/dataset_registry_cluster.json")
    ap.add_argument("--data-root", dest="data_root", default=None)
    ap.add_argument("--backbone", default="NB_uncond")
    ap.add_argument("--batch-count", dest="batch_count", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=500, help="outer-fold ceiling (early stopping applies)")
    ap.add_argument("--inner-epochs", dest="inner_epochs", type=int, default=500)
    ap.add_argument("--warmup-epoch", dest="warmup_epoch", type=int, default=5)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=1024)
    ap.add_argument("--outer-folds", dest="outer_folds", type=int, default=5)
    ap.add_argument("--inner-folds", dest="inner_folds", type=int, default=3)
    ap.add_argument("--criterion", default="scib", choices=["scib", "lisi"])
    ap.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    ap.add_argument("--reference-rule", dest="reference_rule", default="entropy",
                    choices=["entropy", "largest"])
    ap.add_argument("--skip-discr", dest="skip_discr", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    with open(args.registry) as fh:
        registry = json.load(fh)
    os.makedirs(args.output_dir, exist_ok=True)

    adata, batch_key, celltype_key, reference = load_task(
        args.dataset, batch_count=args.batch_count, data_root=args.data_root,
        registry=registry, reference_rule=args.reference_rule,
    )
    print(f"[{args.dataset}] n_obs={adata.n_obs} n_batches={adata.obs[batch_key].nunique()} "
          f"reference={reference} backbone={args.backbone} criterion={args.criterion}",
          flush=True)

    run_comprehensive_nested_cv(
        adata,
        batch_key=batch_key,
        celltype_key=celltype_key,
        output_dir=args.output_dir,
        output_prefix=args.dataset,
        d_coef_range=LAMBDA_GRID,
        n_outer_folds=args.outer_folds,
        n_inner_folds=args.inner_folds,
        epochs=args.epochs,
        inner_epochs=args.inner_epochs,
        warmup_epoch=args.warmup_epoch,
        batch_size=args.batch_size,
        # WHY by NAME: training resolves the name to the correct batch index; the integer
        #   reference_batch is the legacy positional fallback and is NOT the entropy pick.
        reference_batch=0,
        reference_batch_name_str=reference,
        backbone=args.backbone,
        registry=registry,
        criterion=args.criterion,
        early_stopping=args.early_stopping,
        skip_discr=args.skip_discr,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
