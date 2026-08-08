import os
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scib
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
import torch

from wcd_vae.wcd.evaluation import clisi_graph, ilisi_graph
from wcd_vae.wcd.primitives import seed_everything
from wcd_vae.wcd.training import obtain_embeddings, train_integration_model


_PAGA_BASELINE_CACHE = {}


def _paga_connectivities(adata_sub, celltype_key):
    """PAGA connectivities as a DataFrame, or None when the graph is degenerate.

    WHY: when no k-NN edge joins two different cell types -- every cluster a separate
    connected component -- the contracted cluster graph has ZERO edges, and
    ``igraph.Graph.get_adjacency_sparse`` builds ``csr_matrix((weights, zip(*edges)))``
    from an empty edge list, which scipy rejects with

        ValueError: mismatching number of index arrays for shape; got 0, expected 2

    So ``sc.tl.paga`` RAISES on well-separated data rather than returning the all-zero
    matrix it logically implies. Reproduced on scanpy 1.11.5 / igraph 0.11.8 / scipy
    1.14.1 at cluster separations of 2.0 sigma and above; below ~1.5 sigma stray edges
    appear and PAGA succeeds, which is why real datasets have never hit this.

    Returning None is BEHAVIOUR-PRESERVING, not a new fallback: had scanpy returned the
    all-zero matrix, ``np.var(tech_edges) > 0`` downstream would have rejected that batch
    anyway, and an all-zero GLOBAL matrix rejects every batch and yields NaN. This turns
    a crash into the value the existing guards already produce.
    """
    try:
        sc.tl.paga(adata_sub, groups=celltype_key)
    except ValueError as exc:  # degenerate cluster graph -- see docstring
        if "mismatching number of index arrays" not in str(exc):
            raise
        return None
    cats = adata_sub.obs[celltype_key].cat.categories
    return pd.DataFrame(
        adata_sub.uns["paga"]["connectivities"].toarray(), index=cats, columns=cats
    )


def _paga_baseline(adata, tech_key, celltype_key, baseline_rep):
    """Per-batch PAGA connectivity matrices on the unintegrated representation.

    Returns {batch_name: DataFrame}. Memoised: the result depends only on the dataset
    (baseline coordinates, batch labels, cell-type labels), so it is identical for every
    configuration a sweep evaluates. See the call site for measured costs.

    Batches with fewer than 3 cell types are omitted, matching the original behaviour --
    a Spearman correlation over fewer than 3 points is not meaningful.
    """
    import hashlib

    import numpy as np

    # NOTE label arrays must be hashed via their CATEGORY CODES, not .tobytes().
    # `adata.obs[k].astype(str).to_numpy()` produces a dtype=object array of Python str
    # POINTERS, so .tobytes() hashes memory addresses: the key changed on every call
    # (cache never hit) and, worse, address reuse could produce a false HIT on different
    # labels. Codes are plain integers, so the hash reflects the actual assignment.
    rep = np.ascontiguousarray(adata.obsm[baseline_rep])

    def _label_hash(col):
        cat = adata.obs[col].astype("category")
        codes = np.ascontiguousarray(cat.cat.codes.to_numpy())
        names = "\x00".join(map(str, cat.cat.categories)).encode()
        return hashlib.blake2b(codes.tobytes() + names, digest_size=8).hexdigest()

    key = (
        hashlib.blake2b(rep.view(np.uint8), digest_size=16).hexdigest(),
        _label_hash(tech_key),
        _label_hash(celltype_key),
        baseline_rep,
    )
    hit = _PAGA_BASELINE_CACHE.get(key)
    if hit is not None:
        return hit

    out = {}
    for tech in adata.obs[tech_key].unique():
        adata_tech = adata[adata.obs[tech_key] == tech].copy()
        if len(adata_tech.obs[celltype_key].unique()) < 3:
            continue
        sc.pp.neighbors(adata_tech, use_rep=baseline_rep)
        df_tech = _paga_connectivities(adata_tech, celltype_key)
        if df_tech is None:
            # no inter-cell-type edges in this batch: nothing to correlate against
            continue
        out[tech] = df_tech

    # one dataset at a time: holding more would pin large graphs for no benefit
    _PAGA_BASELINE_CACHE.clear()
    _PAGA_BASELINE_CACHE[key] = out
    return out


def compute_mean_paga_spearman(
    adata, tech_key="tech", celltype_key="celltype", embed_key="X_latent", baseline_rep="X_pca"
):
    """
    Computes the mean Spearman correlation of PAGA connectivities
    across all technologies. Optimized for training loops.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Keep the training logs clean

        # 1. Global Integrated PAGA
        adata_global = adata.copy()
        sc.pp.neighbors(adata_global, use_rep=embed_key)
        df_global = _paga_connectivities(adata_global, celltype_key)
        if df_global is None:
            # Degenerate global graph -> every edge weight is 0 -> the np.var guard below
            # would reject every batch. NaN ("not measurable"), never 0.0.
            return float("nan")
        global_celltypes = df_global.index

        spearman_scores = []

        # 2. Loop through each technology
        #
        # WHY CACHED: the per-batch reference graphs are built on baseline_rep (X_pca),
        # the UNINTEGRATED representation. They therefore depend only on the dataset --
        # not on the embedding, the adversarial head, lambda, the seed or the backbone --
        # yet a sweep recomputes an identical result for every configuration it
        # evaluates. Measured cost per config: 2.36s on pancreas (9 batches) and 10.51s
        # on atac_large (84,813 cells), against a global (embedding-dependent) part of
        # 3.37s and 12.12s that genuinely must be recomputed.
        #
        # The key covers everything the result depends on: the baseline coordinates
        # themselves (content-hashed -- X_pca is recomputed per config and could differ),
        # the batch and cell-type assignments, and the representation name.
        baseline = _paga_baseline(adata, tech_key, celltype_key, baseline_rep)

        for df_tech in baseline.values():
            tech_cats = df_tech.index

            # 3. Align and Extract
            common_ct = list(set(global_celltypes) & set(tech_cats))
            common_ct.sort()

            if len(common_ct) < 3:
                continue

            aligned_global = df_global.loc[common_ct, common_ct].to_numpy()
            aligned_tech = df_tech.loc[common_ct, common_ct].to_numpy()

            upper_tri_idx = np.triu_indices_from(aligned_tech, k=1)
            global_edges = aligned_global[upper_tri_idx]
            tech_edges = aligned_tech[upper_tri_idx]

            # 4. Correlate
            if np.var(tech_edges) > 0 and np.var(global_edges) > 0:
                corr, _ = spearmanr(tech_edges, global_edges)
                if not np.isnan(corr):
                    spearman_scores.append(corr)

        # WHY: returning 0.0 when NO batch produced a usable correlation is
        #      indistinguishable from "topology fully destroyed" -- a silent fallback
        #      that reads as a real measurement. NaN means "not measurable" (e.g. every
        #      PAGA connectivity matrix was constant, as happens when clusters are
        #      fully disconnected), which callers can drop rather than average in.
        return float(np.mean(spearman_scores)) if spearman_scores else float("nan")


def calculate_additional_metrics(adata, batch_key, celltype_key, embed_key="X_latent"):
    """
    Helper function to calculate ASW_batch, ASW_celltype, ARI, and Graph Connectivity.
    Optimized to use existing embeddings and silence noisy output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        asw_batch = scib.me.silhouette_batch(
            adata, batch_key=batch_key, group_key=celltype_key, embed=embed_key, verbose=False
        )
        asw_celltype = scib.me.silhouette(adata, group_key=celltype_key, embed=embed_key)

    # Required for both optimal_resolution, PAGA, and Graph Connectivity
    sc.pp.neighbors(adata, use_rep=embed_key)

    # --- ADDED FOR PAGA & TOPOLOGY ---
    # Compute PAGA (stores connectivity matrix in adata.uns['paga']['connectivities']).
    # Guarded: a fully disconnected cluster graph makes sc.tl.paga raise rather than
    # return zeros, and this call exists only to populate .uns for downstream readers --
    # graph_connectivity and paga_spearman below do not depend on it.
    _paga_connectivities(adata, celltype_key)

    # Compute Graph Connectivity (Quantitative topological metric between 0 and 1)
    graph_conn = scib.me.graph_connectivity(adata, label_key=celltype_key)
    # ---------------------------------

    paga_spearman = compute_mean_paga_spearman(
        adata,
        tech_key=batch_key,
        celltype_key=celltype_key,
        embed_key=embed_key,
        baseline_rep="X_pca",
    )

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

    # Return graph_conn as well
    return asw_batch, asw_celltype, ari, graph_conn, paga_spearman


# scIB categories (Luecken et al. 2022). kBET is omitted -- it requires rpy2 + the R kBET
# package, which is not installed, and returns all-NaN here.
_SCIB_BATCH = {"ilisi": +1, "asw_batch": +1, "graph_conn": +1, "pcr": +1}
_SCIB_BIO = {"clisi": -1, "ari": +1, "nmi": +1, "asw_celltype": +1,
             "isolated_asw": +1, "isolated_f1": +1, "cell_cycle": +1,
             "paga_spearman": +1}


def _scib_overall(suite_rows, w_batch=0.4):
    """0.4*batch + 0.6*bio over a list of metric dicts, scaled within this comparison.

    # WHY: scIB ships no scalar aggregate (verified in scib 1.1.7), so the overall score
    #   must be computed. We follow the paper: min-max scale each metric across the
    #   candidates being compared, average within category, then weight 0.4/0.6.
    # HOW: sign-corrects lower-is-better metrics (cLISI) before scaling; metrics that are
    #   all-NaN or constant across candidates contribute nothing rather than 0 or 0.5.
    """
    import numpy as np

    def cat_score(spec):
        vals = []
        for m, sign in spec.items():
            xs = np.array([r.get(m, np.nan) for r in suite_rows], dtype=float) * sign
            if not np.isfinite(xs).any():
                continue
            lo, hi = np.nanmin(xs), np.nanmax(xs)
            if not np.isfinite(lo) or hi - lo < 1e-12:
                continue
            vals.append(np.nanmean((xs - lo) / (hi - lo)))
        return float(np.mean(vals)) if vals else np.nan

    b, o = cat_score(_SCIB_BATCH), cat_score(_SCIB_BIO)
    if not np.isfinite(b) and not np.isfinite(o):
        return float("nan")
    if not np.isfinite(b):
        return o
    if not np.isfinite(o):
        return b
    return w_batch * b + (1.0 - w_batch) * o


def _scib_overall_per_candidate(suite_rows, w_batch=0.4):
    """Per-candidate scIB overall score: one value PER row in ``suite_rows``.

    # WHY: selection compares candidates, so each candidate needs its own score. The
    #   scaling is still done ACROSS candidates (that is what makes the categories
    #   commensurable), but the result is a vector, not a single number.
    """
    import numpy as np

    n = len(suite_rows)

    def cat_scores(spec):
        acc = np.zeros(n, dtype=float)
        used = 0
        for m, sign in spec.items():
            xs = np.array([r.get(m, np.nan) for r in suite_rows], dtype=float) * sign
            if not np.isfinite(xs).all():
                continue
            lo, hi = xs.min(), xs.max()
            if hi - lo < 1e-12:
                continue
            acc += (xs - lo) / (hi - lo)
            used += 1
        return acc / used if used else np.full(n, np.nan)

    b, o = cat_scores(_SCIB_BATCH), cat_scores(_SCIB_BIO)
    out = np.where(
        np.isfinite(b) & np.isfinite(o), w_batch * b + (1 - w_batch) * o,
        np.where(np.isfinite(o), o, b),
    )
    return out


def run_comprehensive_nested_cv(
    adata,
    batch_key,
    celltype_key,
    output_dir,
    d_coef_range=(0.01, 0.05, 0.1, 0.2, 0.5),
    n_outer_folds=5,
    n_inner_folds=3,
    z_dim=256,
    epochs=500,
    inner_epochs=100,
    warmup_epoch=10,
    disc_iter=10,
    batch_size=1024,
    reference_batch=None,
    reference_batch_name_str=None,
    output_prefix=None,
    random_state=42,
    skip_discr=False,
    clisi_weight=1.0,
    backbone="NB_uncond",
    registry=None,
    criterion="scib",
    early_stopping=True,
    outer_fold_only=None,
    head_only=None,
):
    """
    Performs optimized nested cross-validation.

    Optimization: Expensive scib metrics (ASW, ARI) are ONLY calculated in the outer loop
    on the final test set. Inner loops use only iLISI/cLISI for speed.
    """

    seed_everything(random_state)

    num_adversarias = 2 if not skip_discr else 1

    total_steps = n_outer_folds * num_adversarias * len(d_coef_range) * n_inner_folds
    current_step = 0
    print(f"Starting OPTIMIZED nested CV. Total inner steps: {total_steps}")

    cell_indices = np.arange(adata.n_obs)
    cell_labels = adata.obs[celltype_key]

    outer_kf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)

    # Output structure for final, best-model results (includes ALL metrics)
    metrics_list_final = [
        "ilisi",
        "clisi",
        "asw_batch",
        "asw_celltype",
        "ari",
        "graph_conn",
        "paga_spearman",
        "best_d_coef",
    ]
    outer_fold_results_dict = {
        "critic": {m: [] for m in metrics_list_final},
        "no_critic": {m: [] for m in metrics_list_final},
    }

    sensitivity_records = []

    # WHY outer_fold_only / head_only: a full nested run is 5 outer folds x (3 inner x 10
    #   lambda + 1) x 2 heads = 310 fits, which is 120 h on pancreas and 1,506 h on the
    #   cross-species dataset -- far past any wall (max 72 h). Splitting to one
    #   (outer_fold, head) task keeps every task inside the wall. The StratifiedKFold split
    #   is deterministic given random_state, so a task computing only fold k sees exactly
    #   the same train/test partition it would in a full run -- the results are identical,
    #   just distributed. Output files are per-(fold, head) and merged afterwards.
    for outer_fold_idx, (train_idx, test_idx) in enumerate(
        outer_kf.split(cell_indices, cell_labels)
    ):
        if outer_fold_only is not None and outer_fold_idx != int(outer_fold_only):
            continue
        print(f"\n=== Starting Outer Fold {outer_fold_idx + 1}/{n_outer_folds} ===")
        adata_train = adata[train_idx].copy()
        adata_test = adata[test_idx].copy()

        heads = [True, False] if not skip_discr else [True]
        if head_only is not None:
            if head_only not in ("critic", "discriminator"):
                raise ValueError(f"head_only must be 'critic' or 'discriminator', got {head_only!r}")
            heads = [head_only == "critic"]
        for use_critic in heads:
            critic_label = "critic" if use_critic else "no_critic"
            iters = disc_iter if use_critic else 1
            print(f"  --- Processing method: {critic_label} ---")

            inner_kf = StratifiedKFold(
                n_splits=n_inner_folds, shuffle=True, random_state=random_state
            )

            inner_selection_scores = {}
            inner_suite = {}
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
                    model, _ = train_integration_model(
                        adata_inner_train,
                        batch_key=batch_key,
                        z_dim=z_dim,
                        d_coef=d_coef,
                        epochs=inner_epochs,
                        warmup_epoch=warmup_epoch,
                        critic=use_critic,
                        disc_iter=iters,
                        batch_size=batch_size,
                        reference_batch=reference_batch,
                        reference_batch_name_str=reference_batch_name_str,
                        backbone=backbone,
                        early_stopping=early_stopping,
                        es_celltype_key=celltype_key,
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
                        use_rep="X_latent",
                        subset_indices=inner_val_indices,
                    )
                    clisi_val = clisi_graph(
                        adata_inner_comb,
                        label_key=celltype_key,
                        type="embed",
                        use_rep="X_latent",
                        subset_indices=inner_val_indices,
                    )

                    temp_inner_ilisi.append(ilisi_val)
                    temp_inner_clisi.append(clisi_val)

                    # WHY: selection now maximises the scIB overall score rather than
                    #   (iLISI - 10*cLISI). The old form was effectively single-metric --
                    #   over the lambda sweep iLISI swings 0.021 while cLISI swings 0.036,
                    #   so a x10 penalty let cLISI dominate by 17.7x. We collect the full
                    #   suite on the inner validation cells and combine 0.4*batch +
                    #   0.6*bio, the published scIB weighting.
                    if criterion == "scib":
                        try:
                            from wcd_vae.wcd.experiment import full_metric_suite

                            inner_suite.setdefault(d_coef, []).append(
                                full_metric_suite(
                                    adata_inner_comb, batch_key, celltype_key,
                                    embed_key="X_latent",
                                )
                            )
                        except Exception as exc:  # keep the fold, fall back to LISI
                            print(f"    [warn] scIB suite failed on inner fold: {exc}")

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
                            "batch_size": batch_size,
                            "composite_score": ilisi_val - (clisi_val * clisi_weight),
                        }
                    )

                # --- End of Inner Folds for this d_coef ---
                avg_ilisi = np.mean(temp_inner_ilisi)
                avg_clisi = np.mean(temp_inner_clisi)
                avg_composite = avg_ilisi - (avg_clisi * clisi_weight)

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
            # WHY: the scIB overall score min-max scales each metric ACROSS THE
            #   CANDIDATES BEING COMPARED, so it must be computed over all lambda values
            #   together -- scoring each lambda's folds in isolation would scale every
            #   candidate onto the same [0,1] range and make selection arbitrary.
            if criterion == "scib" and inner_suite:
                lams = [lam for lam in d_coef_range if inner_suite.get(lam)]
                if lams:
                    # mean metric dict per lambda, then one scaling pass over lambdas
                    per_lam = []
                    for lam in lams:
                        keys = {k for r in inner_suite[lam] for k in r}
                        per_lam.append({
                            k: float(np.nanmean([r.get(k, np.nan) for r in inner_suite[lam]]))
                            for k in keys
                        })
                    scored = _scib_overall_per_candidate(per_lam)
                    for lam, sc in zip(lams, scored):
                        if np.isfinite(sc):
                            inner_selection_scores[lam] = sc

            best_d_coef = max(inner_selection_scores, key=inner_selection_scores.get)
            print(f"  >>> Best d_coef selected for {critic_label}: {best_d_coef}")

            print(f"  --- Training FINAL {critic_label} model on full Outer Fold train data ---")
            final_model, training_history = train_integration_model(
                adata_train,
                batch_key=batch_key,
                z_dim=z_dim,
                d_coef=best_d_coef,
                epochs=epochs,
                critic=use_critic,
                disc_iter=iters,
                reference_batch=reference_batch,
                reference_batch_name_str=reference_batch_name_str,
                backbone=backbone,
                early_stopping=early_stopping,
                es_celltype_key=celltype_key,
                batch_size=batch_size,
            )

            if output_dir:
                # Create a specific filename for this fold/method
                # WHY 0-based: --outer-fold-only is 0-indexed and the results suffix uses
                #   f"_fold{outer_fold_only}", so a 1-based history name meant the SAME
                #   fold appeared as fold0 in one file and fold1 in another -- a merge trap.
                hist_filename = (
                    f"{output_prefix}_fold{outer_fold_idx}_{critic_label}_history.csv"
                )
                full_hist_path = Path(output_dir) / hist_filename
                full_hist_path.parent.mkdir(parents=True, exist_ok=True)
                # Convert to DataFrame and save
                pd.DataFrame(training_history).to_csv(full_hist_path, index_label="epoch")
                print(f"  >>> Saved training history to: {full_hist_path}")

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
                use_rep="X_latent",
                subset_indices=test_indices,
            )
            test_clisi = clisi_graph(
                adata_outer_comb,
                label_key=celltype_key,
                type="embed",
                use_rep="X_latent",
                subset_indices=test_indices,
            )

            # 2. Calculate expensive scib metrics here
            test_asw_batch, test_asw_celltype, test_ari, test_graph_conn, test_paga_correlation = (
                calculate_additional_metrics(
                    adata_test_only, batch_key, celltype_key, embed_key="X_latent"
                )
            )

            print(
                f"  >>> Final Test Scores ({critic_label}): iLISI={test_ilisi:.3f}, cLISI={test_clisi:.3f}, ARI={test_ari:.3f}, Graph Connectivity={test_graph_conn:.3f}, PAGA Spearman={test_paga_correlation:.3f}"
            )

            # Save all metrics to the final results dictionary
            outer_fold_results_dict[critic_label]["ilisi"].append(test_ilisi)
            outer_fold_results_dict[critic_label]["clisi"].append(test_clisi)
            outer_fold_results_dict[critic_label]["asw_batch"].append(test_asw_batch)
            outer_fold_results_dict[critic_label]["asw_celltype"].append(test_asw_celltype)
            outer_fold_results_dict[critic_label]["ari"].append(test_ari)
            outer_fold_results_dict[critic_label]["graph_conn"].append(test_graph_conn)
            outer_fold_results_dict[critic_label]["best_d_coef"].append(best_d_coef)
            outer_fold_results_dict[critic_label]["paga_spearman"].append(test_paga_correlation)

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
                    "graph_conn": test_graph_conn,
                    "paga_spearman": test_paga_correlation,
                    "composite_score": ilisi_val - (clisi_val * clisi_weight),
                    "batch_size": batch_size,
                }
            )

    # --- FORMAT FINAL OUTPUTS ---

    # Define methods list based on the flag
    methods_to_save = ["critic"] if skip_discr else ["critic", "no_critic"]

    # 1. Final Results DF (Includes all metrics for best models)
    final_results_data = []
    for fold in range(n_outer_folds):
        for m in methods_to_save:  # <--- Use the dynamic list here
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
        prefix = (
            Path(f"{output_dir}/{output_prefix}_") if output_prefix else Path(f"{output_dir}/")
        )
        prefix.mkdir(parents=True, exist_ok=True)

        # WHY the suffix: parallel per-(fold, head) tasks must not write the same path --
        #   truncate-mode writes from concurrent jobs silently interleave.
        suffix = ""
        if outer_fold_only is not None:
            suffix += f"_fold{int(outer_fold_only)}"
        if head_only is not None:
            suffix += f"_{head_only}"
        final_results_df.to_csv(f"{prefix}final_best_results{suffix}.csv", index=False)
        with open(f"{prefix}final_results_dict.pkl", "wb") as f:
            pickle.dump(outer_fold_results_dict, f)

        sensitivity_df.to_csv(
            f"{prefix}comprehensive_sensitivity_records{suffix}.csv", index=False
        )

        print(f"\nResults saved to directory: {output_dir}")

    return final_results_df, outer_fold_results_dict, sensitivity_df
