from collections import defaultdict
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import scib  # Import scib for additional metrics
from sklearn.model_selection import StratifiedKFold
import torch

from wcd_vae.metrics import clisi_graph, ilisi_graph
from wcd_vae.scCRAFT.model import obtain_embeddings, train_integration_model
from wcd_vae.scCRAFT.utils import set_seed


def calculate_additional_metrics(adata, batch_key, celltype_key, embed_key="X_scCRAFT"):
    """Helper function to calculate ASW_batch, ASW_celltype, and ARI."""

    # Ensure site-packages/scib/metrics/silhouette.py:39: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated.
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    # ASW Batch (Integration Metric - lower is better for mixing, but scib returns 1-ASW so higher is better)
    # Note: scib.me.silhouette_batch returns 1 - abs(ASW), so higher is better mixing.
    asw_batch = scib.me.silhouette_batch(
        adata, batch_key=batch_key, group_key=celltype_key, embed=embed_key, verbose=False
    )

    # ASW Cell Type (Bio-conservation Metric - higher is better)
    asw_celltype = scib.me.silhouette(adata, group_key=celltype_key, embed=embed_key)

    # ARI (Bio-conservation Metric - higher is better)
    # ARI requires a clustering. We'll use Louvain clustering on the embedding.
    scib.pp.reduce_data(
        adata,
        n_top_genes=2000,
        batch_key=batch_key,
        pca=True,
        neighbors=True,
        use_rep=embed_key,
        umap=False,
    )
    scib.me.cluster_optimal_resolution(
        adata, label_key=celltype_key, cluster_key="louvain_opt", use_rep=embed_key
    )
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
    epochs=100,
    disc_iter=10,
    reference_batch=None,
    output_dir=None,
    output_prefix=None,
    random_state=42,
):
    """
    Performs nested cross-validation, calculating a comprehensive suite of metrics.

    Returns:
    1. final_results_df: The 'best' performance on held-out test sets (for method comparison).
    2. outer_fold_results_dict: Raw dictionary of the final results.
    3. sensitivity_df: A comprehensive log of ALL inner-loop runs (for sensitivity plotting).
    """

    set_seed(random_state)

    total_steps = n_outer_folds * 2 * len(d_coef_range) * n_inner_folds
    current_step = 0
    print(f"Starting comprehensive nested CV with full metrics. Total inner steps: {total_steps}")

    cell_indices = np.arange(adata.n_obs)
    cell_labels = adata.obs[celltype_key]

    outer_kf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)

    # --- DATA STRUCTURES FOR OUTPUT ---
    # Initialize dictionaries to hold lists for all metrics
    metrics_list = ["ilisi", "clisi", "asw_batch", "asw_celltype", "ari", "best_d_coef"]
    outer_fold_results_dict = {
        "critic": {m: [] for m in metrics_list},
        "no_critic": {m: [] for m in metrics_list},
    }

    sensitivity_records = []

    for outer_fold_idx, (train_idx, test_idx) in enumerate(
        outer_kf.split(cell_indices, cell_labels)
    ):
        print(f"\n=== Starting Outer Fold {outer_fold_idx + 1}/{n_outer_folds} ===")
        adata_train = adata[train_idx].copy()
        adata_test = adata[test_idx].copy()

        for use_critic in [True, False]:
            critic_label = "critic" if use_critic else "no_critic"
            iters = disc_iter if use_critic else 1
            print(f"  --- Processing method: {critic_label} ---")

            inner_kf = StratifiedKFold(
                n_splits=n_inner_folds, shuffle=True, random_state=random_state
            )

            # Dictionary to hold average composite scores for selection
            inner_selection_scores = {}

            train_labels = cell_labels[train_idx]

            for d_coef in d_coef_range:
                # Temp lists for averaging across inner folds
                temp_metrics = defaultdict(list)

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
                        epochs=epochs,
                        critic=use_critic,
                        disc_iter=iters,
                        reference_batch=reference_batch,
                    )

                    # 3. Evaluate on Inner Validation Set
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    model_on_device = model.to(device)
                    obtain_embeddings(adata_inner_train, model_on_device)
                    obtain_embeddings(adata_inner_val, model_on_device)

                    adata_inner_comb = adata_inner_train.concatenate(adata_inner_val)
                    inner_val_indices = np.arange(adata_inner_train.n_obs, adata_inner_comb.n_obs)

                    # Create a view/copy of the validation set for scib metrics
                    adata_val_only = adata_inner_comb[inner_val_indices].copy()

                    # --- CALCULATE ALL METRICS ---
                    # LISI metrics (on combined data, subsetted)
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

                    # scib metrics (on validation set only)
                    asw_batch_val, asw_celltype_val, ari_val = calculate_additional_metrics(
                        adata_val_only, batch_key, celltype_key, embed_key="X_scCRAFT"
                    )

                    # Store for averaging
                    temp_metrics["ilisi"].append(ilisi_val)
                    temp_metrics["clisi"].append(clisi_val)
                    temp_metrics["asw_batch"].append(asw_batch_val)
                    temp_metrics["asw_celltype"].append(asw_celltype_val)
                    temp_metrics["ari"].append(ari_val)

                    # --- LOGGING: Save individual inner fold result ---
                    sensitivity_records.append(
                        {
                            "outer_fold": outer_fold_idx + 1,
                            "inner_fold": inner_fold_idx + 1,
                            "data_type": "inner_validation_raw",
                            "method": critic_label,
                            "d_coef": d_coef,
                            "ilisi": ilisi_val,
                            "clisi": clisi_val,
                            "asw_batch": asw_batch_val,
                            "asw_celltype": asw_celltype_val,
                            "ari": ari_val,
                            "composite_score": ilisi_val - clisi_val,
                        }
                    )

                # --- End of Inner Folds for this d_coef ---
                # Calculate averages
                avg_metrics = {k: np.mean(v) for k, v in temp_metrics.items()}
                avg_composite = avg_metrics["ilisi"] - avg_metrics["clisi"]

                # Store for selection (still using only iLISI - cLISI for selection)
                inner_selection_scores[d_coef] = avg_composite

                # --- LOGGING: Save averaged result for this d_coef ---
                record = {
                    "outer_fold": outer_fold_idx + 1,
                    "inner_fold": "average",
                    "data_type": "inner_validation_avg",
                    "method": critic_label,
                    "d_coef": d_coef,
                    "composite_score": avg_composite,
                }
                # Add all averaged metrics to the record
                record.update(avg_metrics)
                sensitivity_records.append(record)

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
            )

            # Evaluate on HELD-OUT TEST SET
            model_on_device = final_model.to(device)
            obtain_embeddings(adata_train, model_on_device)
            obtain_embeddings(adata_test, model_on_device)

            adata_outer_comb = adata_train.concatenate(adata_test)
            test_indices = np.arange(adata_train.n_obs, adata_outer_comb.n_obs)
            adata_test_only = adata_outer_comb[test_indices].copy()

            # --- CALCULATE ALL FINAL TEST METRICS ---
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

            test_asw_batch, test_asw_celltype, test_ari = calculate_additional_metrics(
                adata_test_only, batch_key, celltype_key, embed_key="X_scCRAFT"
            )

            print(
                f"  >>> Final Test Scores ({critic_label}): iLISI={test_ilisi:.3f}, cLISI={test_clisi:.3f}, ARI={test_ari:.3f}"
            )

            # Save Final Results
            outer_fold_results_dict[critic_label]["ilisi"].append(test_ilisi)
            outer_fold_results_dict[critic_label]["clisi"].append(test_clisi)
            outer_fold_results_dict[critic_label]["asw_batch"].append(test_asw_batch)
            outer_fold_results_dict[critic_label]["asw_celltype"].append(test_asw_celltype)
            outer_fold_results_dict[critic_label]["ari"].append(test_ari)
            outer_fold_results_dict[critic_label]["best_d_coef"].append(best_d_coef)

            # --- LOGGING: Save final test result to sensitivity DF too ---
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

    # 1. Final Results DF
    final_results_data = []
    for fold in range(n_outer_folds):
        for m in ["critic", "no_critic"]:
            record = {
                "fold": fold + 1,
                "method": m,
            }
            # Add all metrics from the dictionary
            for metric in metrics_list:
                record[metric] = outer_fold_results_dict[m][metric][fold]
            final_results_data.append(record)

    final_results_df = pd.DataFrame(final_results_data)

    # 2. Sensitivity DF
    sensitivity_df = pd.DataFrame(sensitivity_records)

    # --- SAVING ---
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        prefix = f"{output_dir}/{output_prefix}_" if output_prefix else f"{output_dir}/"

        final_results_df.to_csv(f"{prefix}final_best_results_full_metrics.csv", index=False)
        with open(f"{prefix}final_results_dict_full_metrics.pkl", "wb") as f:
            pickle.dump(outer_fold_results_dict, f)

        sensitivity_df.to_csv(
            f"{prefix}comprehensive_sensitivity_records_full_metrics.csv", index=False
        )

        print(f"\nResults saved to directory: {output_dir}")
        print("- Final best results: *_final_best_results_full_metrics.csv")
        print("- Comprehensive records: *_comprehensive_sensitivity_records_full_metrics.csv")

    return final_results_df, outer_fold_results_dict, sensitivity_df
