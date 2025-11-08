from collections import defaultdict
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch

from wcd_vae.metrics import clisi_graph, ilisi_graph
from wcd_vae.scCRAFT.model import obtain_embeddings, train_integration_model
from wcd_vae.scCRAFT.utils import set_seed


def nested_cv_hyperparameter_tuning(
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
    Perform nested cross-validation for hyperparameter tuning of d_coef with and without critic.
    ... (rest of docstring) ...
    """

    set_seed(random_state)

    # Prepare data indices for stratified splitting by cell type
    cell_indices = np.arange(adata.n_obs)

    # Create stratified outer folds
    outer_kf = KFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)

    # Store results for each outer fold
    outer_fold_results = {
        "critic": {"ilisi": [], "clisi": [], "best_d_coef": []},
        "no_critic": {"ilisi": [], "clisi": [], "best_d_coef": []},
    }

    for _, (train_idx, test_idx) in enumerate(outer_kf.split(cell_indices)):
        # Split data for outer fold
        adata_train = adata[train_idx].copy()
        adata_test = adata[test_idx].copy()

        # Inner CV for hyperparameter selection (both critic and no_critic)
        for use_critic in [True, False]:
            critic_label = "critic" if use_critic else "no_critic"
            iters = disc_iter if use_critic else 1

            # Inner cross-validation for hyperparameter selection
            inner_kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
            inner_scores = defaultdict(list)

            for d_coef in d_coef_range:
                # Inner fold validation scores
                inner_ilisi_scores = []
                inner_clisi_scores = []

                for _, (inner_train_idx, inner_val_idx) in enumerate(inner_kf.split(train_idx)):
                    # Get actual indices
                    actual_train_idx = train_idx[inner_train_idx]
                    actual_val_idx = train_idx[inner_val_idx]

                    adata_inner_train = adata[actual_train_idx].copy()
                    adata_inner_val = adata[actual_val_idx].copy()

                    # Train model on inner training set
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

                    # --- START: MODIFIED INNER LOOP EVALUATION ---
                    # Get embeddings for BOTH inner train and validation sets
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    model_on_device = model.to(device)

                    obtain_embeddings(adata_inner_train, model_on_device)
                    obtain_embeddings(adata_inner_val, model_on_device)

                    # Combine them
                    adata_inner_combined = adata_inner_train.concatenate(adata_inner_val)

                    # Define the indices for the validation set
                    inner_val_indices = np.arange(
                        adata_inner_train.n_obs, adata_inner_combined.n_obs
                    )

                    # Compute validation scores on the combined graph, subsetting
                    ilisi_val = ilisi_graph(
                        adata_inner_combined,
                        batch_key=batch_key,
                        type="embed",
                        use_rep="X_scCRAFT",
                        subset_indices=inner_val_indices,
                    )
                    clisi_val = clisi_graph(
                        adata_inner_combined,
                        label_key=celltype_key,
                        type="embed",
                        use_rep="X_scCRAFT",
                        subset_indices=inner_val_indices,
                    )
                    # --- END: MODIFIED INNER LOOP EVALUATION ---

                    inner_ilisi_scores.append(ilisi_val)
                    inner_clisi_scores.append(clisi_val)

                # Average scores across inner folds
                avg_ilisi = np.mean(inner_ilisi_scores)
                avg_clisi = np.mean(inner_clisi_scores)

                # Composite score (higher iLisi is better, lower cLISI is better)
                composite_score = avg_ilisi - avg_clisi

                inner_scores[d_coef] = {
                    "ilisi": avg_ilisi,
                    "clisi": avg_clisi,
                    "composite": composite_score,
                }

            # Select best hyperparameter based on composite score
            best_d_coef = max(inner_scores.keys(), key=lambda k: inner_scores[k]["composite"])

            # Train final model on full training set with best hyperparameter
            final_model = train_integration_model(
                adata_train,
                batch_key=batch_key,
                z_dim=z_dim,
                d_coef=best_d_coef,
                epochs=epochs,
                critic=use_critic,
                disc_iter=disc_iter,
            )

            # --- START: MODIFIED OUTER LOOP EVALUATION (from before) ---
            # Evaluate on test set
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_on_device = final_model.to(device)

            # Get embeddings for BOTH train and test sets
            obtain_embeddings(adata_train, model_on_device)
            obtain_embeddings(adata_test, model_on_device)

            # Combine them
            adata_combined = adata_train.concatenate(adata_test)

            # Define the indices for the test set
            test_indices = np.arange(adata_train.n_obs, adata_combined.n_obs)

            # Compute test scores on the combined graph, subsetting to test cells
            test_ilisi = ilisi_graph(
                adata_combined,
                batch_key=batch_key,
                type="embed",
                use_rep="X_scCRAFT",
                subset_indices=test_indices,
            )
            test_clisi = clisi_graph(
                adata_combined,
                label_key=celltype_key,
                type="embed",
                use_rep="X_scCRAFT",
                subset_indices=test_indices,
            )
            # --- END: MODIFIED OUTER LOOP EVALUATION ---

            print(f"  Test scores: iLISI={test_ilisi:.4f}, cLISI={test_clisi:.4f}")

            # Store results
            outer_fold_results[critic_label]["ilisi"].append(test_ilisi)
            outer_fold_results[critic_label]["clisi"].append(test_clisi)
            outer_fold_results[critic_label]["best_d_coef"].append(best_d_coef)

    # Create results DataFrame
    results_data = []
    for fold in range(n_outer_folds):
        for critic_type in ["critic", "no_critic"]:
            results_data.append(
                {
                    "fold": fold + 1,
                    "method": critic_type,
                    "ilisi": outer_fold_results[critic_type]["ilisi"][fold],
                    "clisi": outer_fold_results[critic_type]["clisi"][fold],
                    "best_d_coef": outer_fold_results[critic_type]["best_d_coef"][fold],
                }
            )

    results_df = pd.DataFrame(results_data)

    if output_dir:
        prefix = f"{output_dir}/{output_prefix}_" if output_prefix else f"{output_dir}/"
        results_df.to_csv(f"{prefix}nested_cv_results.csv", index=False)
        print(f"\nResults saved to '{prefix}nested_cv_results.csv'")

        with Path(f"{prefix}outer_fold_results.pkl").open("wb") as f:
            pickle.dump(outer_fold_results, f)

    return results_df, outer_fold_results
