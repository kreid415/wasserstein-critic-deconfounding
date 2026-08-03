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
import sys
import warnings

import numpy as np
import scanpy as sc
import scib

from wcd_vae.wcd.data import prep_data
from wcd_vae.wcd.evaluation import clisi_graph, ilisi_graph, probe_metrics
from wcd_vae.wcd.primitives import seed_everything
from wcd_vae.wcd.training import obtain_embeddings, train_integration_model

# Metric taxonomy for the local-vs-global decomposition (E6 / R1.major.1).
LOCAL_METRICS = ("ilisi", "clisi", "graph_conn", "kbet")
GLOBAL_METRICS = ("asw_batch", "asw_celltype", "ari", "nmi", "pcr", "isolated_asw")
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


def load_task(name, batch_count=None, balance=False, data_root=None, registry=None):
    """Load and preprocess a registered task by name.

    Returns (adata, batch_key, celltype_key, largest_batch_name).
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
    return adata, entry["batch_key"], entry["celltype_key"], largest


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
    epochs=150,
    warmup_epoch=5,
    batch_size=1024,
    backbone=None,
    reference_mode="fixed",
    formulation="reference",
):
    """Train ONE configuration and return the fitted VAE + history.

    `critic` selects the adversarial head (True=reference-Wasserstein, False=JS
    discriminator). `d_coef` is lambda_adv. `backbone` (optional) selects an
    native VAE backbone name (None -> NB default inside train_integration_model).
    """
    seed_everything(seed)
    iters = disc_iter if disc_iter is not None else (10 if critic else 1)
    kwargs = {
        "batch_key": batch_key,
        "critic": critic,
        "d_coef": d_coef,
        "disc_iter": iters,
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

        # ARI / NMI need a clustering; use scib optimal-resolution louvain (silenced)
        old = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            scib.me.cluster_optimal_resolution(adata, label_key=celltype_key, cluster_key="louvain_opt", use_rep=embed_key)
        finally:
            sys.stdout.close()
            sys.stdout = old
        out["ari"] = float(scib.me.ari(adata, celltype_key, "louvain_opt"))
        out["nmi"] = float(scib.me.nmi(adata, celltype_key, "louvain_opt"))

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
    epochs=150,
    warmup_epoch=5,
    batch_size=1024,
    metric_kwargs=None,
    embed_out=None,
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
