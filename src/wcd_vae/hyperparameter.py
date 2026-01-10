import os
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scib
from sklearn.model_selection import StratifiedKFold
import torch

from wcd_vae.metrics import clisi_graph, ilisi_graph
from wcd_vae.scCRAFT.model import obtain_embeddings, train_integration_model
from wcd_vae.scCRAFT.utils import set_seed


def calculate_additional_metrics(adata, batch_key, celltype_key, embed_key="X_scCRAFT"):
    """
    Helper function to calculate ASW_batch, ASW_celltype, and ARI.
    Optimized to use existing embeddings and silence noisy output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        asw_batch = scib.me.silhouette_batch(
            adata, batch_key=batch_key, group_key=celltype_key, embed=embed_key, verbose=False
        )
        asw_celltype = scib.me.silhouette(adata, group_key=celltype_key, embed=embed_key)

    sc.pp.neighbors(adata, use_rep=embed_key)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        scib.me.cluster_optimal_resolution(
            adata, label_key=celltype_key, cluster_key="louvain_opt", use_rep=embed_key
        )
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    ari = scib.me.ari(adata, celltype_key, "louvain_opt")
    return asw_batch, asw_celltype, ari


def run_comprehensive_nested_cv(
    adata,
    batch_key,
    celltype_key,
    d_coef_range=(0.01, 0.05, 0.1, 0.2, 0.5),
    n_outer_folds=5,
    n_inner_folds=3,
    z_dim=256,
    epochs=500,
    inner_epochs=100,
    warmup_epoch=10,
    disc_iter=10,
    reference_batch=None,
    reference_batch_name_str=None,
    output_dir=None,
    output_prefix=None,
    random_state=42,
    skip_discr=False,
):
    """
    Performs optimized nested cross-validation.

    Optimization: Expensive scib metrics (ASW, ARI) are ONLY calculated in the outer loop
    on the final test set. Inner loops use only iLISI/cLISI for speed.
    """

    set_seed(random_state)

    num_adversarias = 2 if not skip_discr else 1

    total_steps = n_outer_folds * num_adversarias * len(d_coef_range) * n_inner_folds
    current_step = 0
    print(f"Starting OPTIMIZED nested CV. Total inner steps: {total_steps}")

    cell_indices = np.arange(adata.n_obs)
    cell_labels = adata.obs[celltype_key]

    outer_kf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)

    # Output structure for final, best-model results (includes ALL metrics)
    metrics_list_final = ["ilisi", "clisi", "asw_batch", "asw_celltype", "ari", "best_d_coef"]
    outer_fold_results_dict = {
        "critic": {m: [] for m in metrics_list_final},
        "no_critic": {m: [] for m in metrics_list_final},
    }

    sensitivity_records = []

    for outer_fold_idx, (train_idx, test_idx) in enumerate(
        outer_kf.split(cell_indices, cell_labels)
    ):
        print(f"\n=== Starting Outer Fold {outer_fold_idx + 1}/{n_outer_folds} ===")
        adata_train = adata[train_idx].copy()
        adata_test = adata[test_idx].copy()

        for use_critic in [True, False] if not skip_discr else [False]:
            critic_label = "critic" if use_critic else "no_critic"
            iters = disc_iter if use_critic else 1
            print(f"  --- Processing method: {critic_label} ---")

            inner_kf = StratifiedKFold(
                n_splits=n_inner_folds, shuffle=True, random_state=random_state
            )

            inner_selection_scores = {}
            train_labels = cell_labels[train_idx]

            for d_coef in d_coef_range:
                # Only track iLISI and cLISI for inner loops
                temp_inner_ilisi = []
                temp_inner_clisi = []

                for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
                    inner_kf.split(train_idx, train_labels)
                ):
                    current_step += 1
                    print(
                        f"    [Step {current_step}/{total_steps}] Inner Fold {inner_fold_idx + 1} | d_coef={d_coef}"
                    )

                    # 1. Prepare Inner Data
                    actual_train_idx = train_idx[inner_train_idx]
                    actual_val_idx = train_idx[inner_val_idx]
                    adata_inner_train = adata[actual_train_idx].copy()
                    adata_inner_val = adata[actual_val_idx].copy()

                    # 2. Train Inner Model
                    model = train_integration_model(
                        adata_inner_train,
                        batch_key=batch_key,
                        z_dim=z_dim,
                        d_coef=d_coef,
                        epochs=inner_epochs,
                        warmup_epoch=warmup_epoch,
                        critic=use_critic,
                        disc_iter=iters,
                        reference_batch=reference_batch,
                        reference_batch_name_str=reference_batch_name_str,
                    )

                    # 3. Evaluate on Inner Validation Set (FAST METRICS ONLY)
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    model_on_device = model.to(device)
                    obtain_embeddings(adata_inner_train, model_on_device)
                    obtain_embeddings(adata_inner_val, model_on_device)

                    adata_inner_comb = adata_inner_train.concatenate(adata_inner_val)
                    inner_val_indices = np.arange(adata_inner_train.n_obs, adata_inner_comb.n_obs)

                    # Calculate only the fast LISI metrics
                    ilisi_val = ilisi_graph(
                        adata_inner_comb,
                        batch_key=batch_key,
                        type="embed",
                        use_rep="X_scCRAFT",
                        subset_indices=inner_val_indices,
                    )
                    clisi_val = clisi_graph(
                        adata_inner_comb,
                        label_key=celltype_key,
                        type="embed",
                        use_rep="X_scCRAFT",
                        subset_indices=inner_val_indices,
                    )

                    temp_inner_ilisi.append(ilisi_val)
                    temp_inner_clisi.append(clisi_val)

                    # Log inner fold result
                    sensitivity_records.append(
                        {
                            "outer_fold": outer_fold_idx + 1,
                            "inner_fold": inner_fold_idx + 1,
                            "data_type": "inner_validation_raw",
                            "method": critic_label,
                            "d_coef": d_coef,
                            "ilisi": ilisi_val,
                            "clisi": clisi_val,
                            "composite_score": ilisi_val - clisi_val,
                        }
                    )

                # --- End of Inner Folds for this d_coef ---
                avg_ilisi = np.mean(temp_inner_ilisi)
                avg_clisi = np.mean(temp_inner_clisi)
                avg_composite = avg_ilisi - avg_clisi

                inner_selection_scores[d_coef] = avg_composite

                # Log averaged result
                sensitivity_records.append(
                    {
                        "outer_fold": outer_fold_idx + 1,
                        "inner_fold": "average",
                        "data_type": "inner_validation_avg",
                        "method": critic_label,
                        "d_coef": d_coef,
                        "ilisi": avg_ilisi,
                        "clisi": avg_clisi,
                        "composite_score": avg_composite,
                    }
                )

            # --- SELECTION & FINAL TRAINING (Outer Fold) ---
            best_d_coef = max(inner_selection_scores, key=inner_selection_scores.get)
            print(f"  >>> Best d_coef selected for {critic_label}: {best_d_coef}")

            print(f"  --- Training FINAL {critic_label} model on full Outer Fold train data ---")
            final_model = train_integration_model(
                adata_train,
                batch_key=batch_key,
                z_dim=z_dim,
                d_coef=best_d_coef,
                epochs=epochs,
                critic=use_critic,
                disc_iter=iters,
                reference_batch=reference_batch,
                reference_batch_name_str=reference_batch_name_str,
            )

            # Evaluate on HELD-OUT TEST SET (FULL METRICS SUITE)
            model_on_device = final_model.to(device)
            obtain_embeddings(adata_train, model_on_device)
            obtain_embeddings(adata_test, model_on_device)

            adata_outer_comb = adata_train.concatenate(adata_test)
            test_indices = np.arange(adata_train.n_obs, adata_outer_comb.n_obs)
            adata_test_only = adata_outer_comb[test_indices].copy()

            # 1. Calculate LISI
            test_ilisi = ilisi_graph(
                adata_outer_comb,
                batch_key=batch_key,
                type="embed",
                use_rep="X_scCRAFT",
                subset_indices=test_indices,
            )
            test_clisi = clisi_graph(
                adata_outer_comb,
                label_key=celltype_key,
                type="embed",
                use_rep="X_scCRAFT",
                subset_indices=test_indices,
            )

            # 2. Calculate expensive scib metrics here
            test_asw_batch, test_asw_celltype, test_ari = calculate_additional_metrics(
                adata_test_only, batch_key, celltype_key, embed_key="X_scCRAFT"
            )

            print(
                f"  >>> Final Test Scores ({critic_label}): iLISI={test_ilisi:.3f}, cLISI={test_clisi:.3f}, ARI={test_ari:.3f}"
            )

            # Save all metrics to the final results dictionary
            outer_fold_results_dict[critic_label]["ilisi"].append(test_ilisi)
            outer_fold_results_dict[critic_label]["clisi"].append(test_clisi)
            outer_fold_results_dict[critic_label]["asw_batch"].append(test_asw_batch)
            outer_fold_results_dict[critic_label]["asw_celltype"].append(test_asw_celltype)
            outer_fold_results_dict[critic_label]["ari"].append(test_ari)
            outer_fold_results_dict[critic_label]["best_d_coef"].append(best_d_coef)

            # Log final test result to sensitivity DF (will contain NaNs for ASW/ARI in inner loop rows)
            sensitivity_records.append(
                {
                    "outer_fold": outer_fold_idx + 1,
                    "inner_fold": "final_test",
                    "data_type": "outer_test_final",
                    "method": critic_label,
                    "d_coef": best_d_coef,
                    "ilisi": test_ilisi,
                    "clisi": test_clisi,
                    "asw_batch": test_asw_batch,
                    "asw_celltype": test_asw_celltype,
                    "ari": test_ari,
                    "composite_score": test_ilisi - test_clisi,
                }
            )

    # --- FORMAT FINAL OUTPUTS ---

    # 1. Final Results DF (Includes all metrics for best models)
    final_results_data = []
    for fold in range(n_outer_folds):
        for m in ["critic", "no_critic"]:
            record = {
                "fold": fold + 1,
                "method": m,
            }
            for metric in metrics_list_final:
                record[metric] = outer_fold_results_dict[m][metric][fold]
            final_results_data.append(record)
    final_results_df = pd.DataFrame(final_results_data)

    # 2. Sensitivity DF (Inner rows have iLISI/cLISI, outer rows have all)
    sensitivity_df = pd.DataFrame(sensitivity_records)

    # --- SAVING ---
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        prefix = f"{output_dir}/{output_prefix}_" if output_prefix else f"{output_dir}/"

        final_results_df.to_csv(f"{prefix}final_best_results.csv", index=False)
        with open(f"{prefix}final_results_dict.pkl", "wb") as f:
            pickle.dump(outer_fold_results_dict, f)

        sensitivity_df.to_csv(f"{prefix}comprehensive_sensitivity_records.csv", index=False)

        print(f"\nResults saved to directory: {output_dir}")

    return final_results_df, outer_fold_results_dict, sensitivity_df
