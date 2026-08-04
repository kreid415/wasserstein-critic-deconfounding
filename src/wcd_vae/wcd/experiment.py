"""Shared experiment harness \u2014 AUTHORED CONTRIBUTION (K. Reid), revision.

# WHY: Every reviewer-response experiment (E1 Pareto, E2 backbones, E3 baselines,
#      E4 reference design, E5 biology, E6 local/global, E8 multibatch) needs the
#      SAME three things: load a registered task, train one (backbone, head, lambda_adv,
#      seed) configuration, and score it with the SAME full metric suite. Reviewers
#      R1.minor.1 and R1.major.1 specifically asked for the complete local+global suite
#      instead of iLISI alone, so a single audited harness (rather than per-experiment
#      copies) is the correctness-critical piece.
# HOW: `load_task` reads configs/dataset_registry.json and calls prep_data with the
#      right modality; `train_one` trains a single config and writes embeddings;
#      `full_metric_suite` returns every metric tagged LOCAL (neighbourhood-scale) or
#      GLOBAL (embedding-scale) so E6 can decompose the iLISI-vs-ASW_batch mismatch.
"""
import json
import os
from pathlib import Path
import warnings

import numpy as np
import scanpy as sc
import scib

from wcd_vae.wcd.data import prep_data, select_reference_batch
from wcd_vae.wcd.evaluation import clisi_graph, ilisi_graph, probe_metrics
from wcd_vae.wcd.primitives import seed_everything
from wcd_vae.wcd.training import obtain_embeddings, train_integration_model

# Metric taxonomy for the local-vs-global decomposition (E6 / R1.major.1).
LOCAL_METRICS = ("ilisi", "clisi", "graph_conn", "kbet")
GLOBAL_METRICS = ("asw_batch", "asw_celltype", "ari", "nmi", "pcr", "isolated_asw",
                  "isolated_f1", "cell_cycle")
# Probe-based conservation/residual-batch metrics (geometry-free; see evaluation.probe_metrics).
PROBE_METRICS = ("knn_label_lift", "linear_label_lift", "knn_batch_lift", "linear_batch_lift")
# Continuous-topology conservation, as reported in the submitted manuscript.
TOPOLOGY_METRICS = ("paga_spearman",)


def _registry_path():
    # WHY: the registry ships in the repo; HOW: resolve relative to this file's package root.
    here = Path(__file__).resolve()
    for cand in (
        here.parents[3] / "configs" / "dataset_registry.json",  # repo_root/configs
        Path.cwd() / "configs" / "dataset_registry.json",
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError("dataset_registry.json not found in configs/")


def load_registry():
    with open(_registry_path()) as fh:
        return json.load(fh)


def load_task(name, batch_count=None, balance=False, data_root=None, registry=None,
              reference_rule="entropy"):
    """Load and preprocess a registered task by name.

    Returns ``(adata, batch_key, celltype_key, reference_batch_name)``.

    # WHY reference_rule: the 4th return value is the batch every caller uses as the
    #   critic's alignment target. It used to be the LARGEST batch, while the harnesses
    #   separately passed ``reference_batch=0`` -- the ALPHABETICALLY first batch, which on
    #   4 of 6 datasets was one of the smallest. Neither is a principled target. The
    #   default is now ``"entropy"``: the batch whose cell-type distribution has maximum
    #   Shannon entropy (ties broken by size), i.e. the batch that best represents the
    #   biology. Pass ``reference_rule="largest"`` to reproduce pre-2026-08 results.
    """
    registry = registry or load_registry()
    if name not in registry:
        raise KeyError(f"Unknown task '{name}'. Known: {sorted(registry)}")
    entry = registry[name]
    path = entry["path"]
    if data_root is not None:
        path = os.path.join(data_root, entry["file"])
    path = os.path.expandvars(os.path.expanduser(path))
    bc = batch_count if batch_count is not None else entry.get("n_batches", 2)
    adata, largest = prep_data(
        path,
        batch_key=entry["batch_key"],
        celltype_key=entry["celltype_key"],
        batch_count=bc,
        balance=balance,
        modality=entry.get("prep", "rna"),
    )
    if reference_rule == "entropy":
        reference = select_reference_batch(adata, entry["batch_key"], entry["celltype_key"])
    elif reference_rule == "largest":
        reference = largest
    else:
        raise ValueError(f"reference_rule must be 'entropy' or 'largest', got {reference_rule!r}")
    if reference != largest:
        print(f"[reference] entropy rule selects '{reference}' (largest batch was '{largest}')")
    return adata, entry["batch_key"], entry["celltype_key"], reference


def train_one(
    adata,
    batch_key,
    *,
    critic,
    d_coef,
    seed,
    reference_batch=None,
    reference_batch_name_str=None,
    disc_iter=None,
    z_dim=256,
    epochs=500,
    warmup_epoch=5,
    batch_size=1024,
    backbone=None,
    reference_mode="fixed",
    formulation="reference",
    early_stopping=True,
    es_celltype_key=None,
):
    """Train ONE configuration and return the fitted VAE + history.

    `critic` selects the adversarial head (True=reference-Wasserstein, False=JS
    discriminator). `d_coef` is lambda_adv. `backbone` (optional) selects an
    native VAE backbone name (None -> NB default inside train_integration_model).
    """
    seed_everything(seed)
    iters = disc_iter if disc_iter is not None else (10 if critic else 1)
    # WHY epochs=500 + early stopping: 500 matches the original nested-CV protocol, and
    #      early stopping on a held-out cell-type probe prevents the conservation decay
    #      measured over 150->450 epochs from silently degrading every result.
    kwargs = {
        "batch_key": batch_key,
        "critic": critic,
        "d_coef": d_coef,
        "disc_iter": iters,
        "early_stopping": early_stopping,
        "es_celltype_key": es_celltype_key,
        "z_dim": z_dim,
        "epochs": epochs,
        "warmup_epoch": warmup_epoch,
        "batch_size": batch_size,
        "reference_batch": reference_batch,
        "reference_batch_name_str": reference_batch_name_str,
        "reference_mode": reference_mode,
        "formulation": formulation,
    }
    if backbone is not None:
        kwargs["backbone"] = backbone
    vae, history = train_integration_model(adata, **kwargs)
    return vae, history


def _resolution_cache(adata, embed_key, resolutions=None, flavor=None):
    """Leiden labels at each resolution, computed ONCE and cached.

    # WHY: scib's cluster_optimal_resolution runs a full resolution sweep PER CALLER, and
    #      isolated_labels_f1 calls it once PER ISOLATED LABEL. Measured on pancreas
    #      (16,382 cells, 16 cores): the ari/nmi sweep costs 767s and isolated_labels_f1
    #      costs 773s -- together 9.4x the 164s training cost, and both re-cluster the
    #      SAME graph at the SAME resolutions. The two sweeps optimise DIFFERENT
    #      objectives (NMI vs per-label F1), so a single "optimal" clustering cannot be
    #      shared -- but the underlying clusterings can. We compute each resolution once
    #      and let every metric pick its own best from the cache: identical numbers,
    #      one sweep instead of N.
    # HOW: neighbours are built once here rather than inside each sweep. The Leiden
    #      FLAVOUR is deliberately left at scanpy's default (leidenalg), matching
    #      scib.cluster_optimal_resolution exactly: a matched-resolution A/B showed
    #      igraph and leidenalg give IDENTICAL ARI/NMI on structured, weakly-structured
    #      and degenerate synthetic latents (|delta| = 0.00e+00 in all three) and igraph
    #      was NOT faster (0.8-1.0x), so there is nothing to gain by diverging.
    """
    import scanpy as sc

    if resolutions is None:
        resolutions = [round(0.2 * i, 2) for i in range(1, 11)]  # 0.2..2.0, matches scib's n=10
    if "neighbors" not in adata.uns or adata.uns["neighbors"].get("params", {}).get(
        "use_rep"
    ) != embed_key:
        sc.pp.neighbors(adata, use_rep=embed_key)

    cache = {}
    for res in resolutions:
        key = f"_leiden_r{res}"
        kwargs = {"resolution": res, "key_added": key}
        if flavor is None:
            sc.tl.leiden(adata, **kwargs)  # scanpy/scib default -- keeps parity
        else:
            try:
                sc.tl.leiden(adata, flavor=flavor, n_iterations=2, directed=False, **kwargs)
            except TypeError:
                sc.tl.leiden(adata, **kwargs)
        cache[res] = adata.obs[key].astype(str).to_numpy()
    return cache


def _best_from_cache(cache, score_fn):
    """Pick the (resolution, labels, score) maximising ``score_fn(labels)``."""
    best = (None, None, -np.inf)
    for res, labels in cache.items():
        try:
            sc = float(score_fn(labels))
        except Exception:
            continue
        if np.isfinite(sc) and sc > best[2]:
            best = (res, labels, sc)
    return best


def full_metric_suite(
    adata,
    batch_key,
    celltype_key,
    embed_key="X_latent",
    perplexity=30,
    include=("kbet", "pcr", "isolated_asw"),
):
    """Compute the complete local+global integration metric suite for one embedding.

    Returns a flat dict of floats. Local: iLISI, cLISI, graph_conn, (kBET).
    Global: ASW_batch, ASW_celltype, ARI, NMI, (PCR), (isolated-label ASW).
    Higher-is-better convention noted per key in METRIC_DIRECTION.
    """
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ---- LOCAL (neighbourhood-scale) ----
        out["ilisi"] = float(
            ilisi_graph(adata, batch_key=batch_key, type="embed", use_rep=embed_key, perplexity=perplexity)
        )
        out["clisi"] = float(
            clisi_graph(adata, label_key=celltype_key, type="embed", use_rep=embed_key, perplexity=perplexity)
        )
        sc.pp.neighbors(adata, use_rep=embed_key)
        out["graph_conn"] = float(scib.me.graph_connectivity(adata, label_key=celltype_key))
        if "kbet" in include:
            try:
                out["kbet"] = float(
                    scib.me.kBET(adata, batch_key=batch_key, label_key=celltype_key, embed=embed_key, type_="embed")
                )
            except Exception:
                out["kbet"] = np.nan

        # ---- GLOBAL (embedding-scale) ----
        out["asw_batch"] = float(
            scib.me.silhouette_batch(adata, batch_key=batch_key, group_key=celltype_key, embed=embed_key, verbose=False)
        )
        out["asw_celltype"] = float(scib.me.silhouette(adata, group_key=celltype_key, embed=embed_key))
        if "pcr" in include:
            try:
                out["pcr"] = float(scib.me.pcr(adata, covariate=batch_key, embed=embed_key))
            except Exception:
                out["pcr"] = np.nan
        if "isolated_asw" in include:
            try:
                out["isolated_asw"] = float(
                    scib.me.isolated_labels(
                        adata, label_key=celltype_key, batch_key=batch_key, embed=embed_key, cluster=False
                    )
                )
            except Exception:
                out["isolated_asw"] = np.nan

        # ---- ONE resolution sweep, shared by every clustering-based metric ----
        from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

        truth = adata.obs[celltype_key].astype(str).to_numpy()
        cache = _resolution_cache(adata, embed_key)

        # ARI/NMI: scib selects the resolution maximising NMI, then reports both there.
        _res, best_labels, _sc = _best_from_cache(
            cache, lambda lab: normalized_mutual_info_score(truth, lab)
        )
        if best_labels is None:
            out["ari"] = out["nmi"] = np.nan
        else:
            adata.obs["louvain_opt"] = best_labels
            out["ari"] = float(adjusted_rand_score(truth, best_labels))
            out["nmi"] = float(normalized_mutual_info_score(truth, best_labels))

        # isolated-label F1: scib optimises max-F1 SEPARATELY PER ISOLATED LABEL, so each
        # label picks its own resolution from the same cache.
        try:
            # NOTE scib.metrics.isolated_labels resolves to the FUNCTION, not the module,
            #      so the helper must be imported from the module path explicitly.
            from scib.metrics.isolated_labels import get_isolated_labels

            iso = get_isolated_labels(
                adata, celltype_key, batch_key, iso_threshold=None, verbose=False
            )
            f1s = []
            for label in iso:
                y_true = truth == label

                def _maxf1(lab, _yt=y_true):
                    return max(
                        f1_score(lab == c, _yt) for c in np.unique(lab)
                    )

                _r, _l, sc_f1 = _best_from_cache(cache, _maxf1)
                if np.isfinite(sc_f1):
                    f1s.append(sc_f1)
            out["isolated_f1"] = float(np.mean(f1s)) if f1s else np.nan
        except Exception:
            out["isolated_f1"] = np.nan

        # cell-cycle conservation: the remaining scIB bio metric; measured at ~7s, so it
        # completes the published suite essentially for free.
        try:
            out["cell_cycle"] = float(
                scib.me.cell_cycle(adata, adata, batch_key=batch_key, embed=embed_key,
                                   organism="human")
            )
        except Exception:
            out["cell_cycle"] = np.nan

        # WHY: ARI/NMI/ASW_celltype all need compact clusters; probes measure the
        #      cell-type information actually present, and the batch probe doubles as
        #      a guard against a silently inert adversary.
        try:
            out.update(probe_metrics(adata, celltype_key, batch_key, embed_key=embed_key))
        except Exception:
            for k in PROBE_METRICS:
                out[k] = np.nan

        # WHY: PAGA Spearman is a headline CONSERVATION metric in the submitted
        #      manuscript (reported per dataset with p-values), but it lived only in the
        #      nested-CV hyperparameter path and was never wired into this suite, so no
        #      revision experiment reported it. Baseline topology comes from per-batch
        #      PAGA on raw PCA; NaN means "not measurable", never 0.0.
        try:
            from wcd_vae.wcd.hyperparameter import compute_mean_paga_spearman

            out["paga_spearman"] = float(
                compute_mean_paga_spearman(
                    adata, tech_key=batch_key, celltype_key=celltype_key,
                    embed_key=embed_key, baseline_rep="X_pca",
                )
            )
        except Exception:
            out["paga_spearman"] = np.nan

    return out


# Higher-is-better (+1) or lower-is-better (-1) for building composite / Pareto scores.
METRIC_DIRECTION = {
    "ilisi": +1, "clisi": -1, "graph_conn": +1, "kbet": +1,
    "asw_batch": +1, "asw_celltype": +1, "ari": +1, "nmi": +1,
    "pcr": +1, "isolated_asw": +1,
}


def evaluate_config(
    adata,
    batch_key,
    celltype_key,
    *,
    critic,
    d_coef,
    seed,
    reference_batch=None,
    reference_batch_name_str=None,
    backbone=None,
    reference_mode="fixed",
    formulation="reference",
    z_dim=256,
    epochs=500,
    warmup_epoch=5,
    batch_size=1024,
    metric_kwargs=None,
    embed_out=None,
    early_stopping=True,
):
    """Train one config and return {**metrics, config columns}. The atomic unit
    every experiment loops over.

    ``embed_out``: directory to persist the trained latent to (one .npz per config).
    # WHY: metrics-only CSVs cannot be re-analysed -- adding any new embedding-derived
    #      metric (PAGA, probes, trajectory) otherwise costs a full retraining wave.
    #      Persisting z lets future metrics be computed from disk in seconds. Belongs on
    #      SCRATCH, never $HOME: these are large and regenerable.
    """
    import torch

    ad = adata.copy()
    vae, hist = train_one(
        ad, batch_key, critic=critic, d_coef=d_coef, seed=seed,
        reference_batch=reference_batch, reference_batch_name_str=reference_batch_name_str,
        backbone=backbone, reference_mode=reference_mode, formulation=formulation,
        z_dim=z_dim, epochs=epochs,
        warmup_epoch=warmup_epoch, batch_size=batch_size,
        # WHY: without es_celltype_key the early-stopping check is skipped silently, so
        #      the label key MUST be threaded here or epochs=500 runs unguarded.
        early_stopping=early_stopping, es_celltype_key=celltype_key,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obtain_embeddings(ad, vae.to(device))
    if embed_out:
        import os

        os.makedirs(embed_out, exist_ok=True)
        tag = (f"{'critic' if critic else 'discriminator'}_{backbone or 'NB'}"
               f"_lam{str(d_coef).replace('.', 'p')}_s{seed}")
        np.savez_compressed(
            os.path.join(embed_out, f"{tag}.npz"),
            z=np.asarray(ad.obsm["X_latent"], dtype=np.float32),
            # WHY: numpy stores <U dtype string arrays fine, but object-dtype arrays
            #      require allow_pickle on load. Cast to a fixed-width unicode dtype so
            #      np.load(...) works with the safe default.
            batch=ad.obs[batch_key].astype(str).to_numpy(dtype="U64"),
            celltype=ad.obs[celltype_key].astype(str).to_numpy(dtype="U64"),
            X_pca=np.asarray(ad.obsm["X_pca"], dtype=np.float32) if "X_pca" in ad.obsm else np.empty(0),
        )
    metrics = full_metric_suite(ad, batch_key, celltype_key, embed_key="X_latent", **(metric_kwargs or {}))
    row = {
        "method": "critic" if critic else "discriminator",
        "backbone": backbone or "NB",
        "d_coef": d_coef,
        "seed": seed,
        "reference_batch": reference_batch,
        "reference_mode": reference_mode,
        "formulation": formulation,
        "final_loss": float(hist["all_loss"][-1]),
        "final_loss_da": float(hist["loss_da"][-1]),
        **metrics,
    }
    return row
