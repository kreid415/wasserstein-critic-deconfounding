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
    random_state=42,
    reference_batch=None,
    output_dir=None,
    output_prefix=None,
):
    """
    Perform nested cross-validation for hyperparameter tuning of d_coef with and without critic.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    batch_key : str
        Key for batch information
    celltype_key : str
        Key for cell type information
    d_coef_range : list
        Range of d_coef values to test
    n_outer_folds : int
        Number of outer CV folds
    n_inner_folds : int
        Number of inner CV folds for hyperparameter selection
    z_dim : int
        Latent dimension
    epochs : int
        Training epochs
    disc_iter : int
        Discriminator iterations
    random_state : int
        Random seed

    Returns:
    --------
    results_df : DataFrame
        Results with outer fold statistics
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

                    # Get embeddings for validation set
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    obtain_embeddings(adata_inner_val, model.to(device))

                    # Compute validation scores
                    ilisi_val = ilisi_graph(
                        adata_inner_val, batch_key=batch_key, type="embed", use_rep="X_scCRAFT"
                    )
                    clisi_val = clisi_graph(
                        adata_inner_val, label_key=celltype_key, type="embed", use_rep="X_scCRAFT"
                    )

                    inner_ilisi_scores.append(ilisi_val)
                    inner_clisi_scores.append(clisi_val)

                # Average scores across inner folds
                avg_ilisi = np.mean(inner_ilisi_scores)
                avg_clisi = np.mean(inner_clisi_scores)

                # Composite score (higher iLISI is better, lower cLISI is better)
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

            # Evaluate on test set
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            obtain_embeddings(adata_test, final_model.to(device))

            # Compute test scores
            test_ilisi = ilisi_graph(
                adata_test, batch_key=batch_key, type="embed", use_rep="X_scCRAFT"
            )
            test_clisi = clisi_graph(
                adata_test, label_key=celltype_key, type="embed", use_rep="X_scCRAFT"
            )

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
