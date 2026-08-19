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
import contextlib
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
    lr_g=1e-3,
    lr_d=1e-3,
    backbone=None,
    reference_mode="fixed",
    formulation="reference",
    early_stopping=True,
    es_celltype_key=None,
    es_patience=5,
    es_check_every=10,
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
        # WHY THREADED: the ES criterion (held-out cell-type probe accuracy) is sampled
        # every es_check_every epochs. At high lambda the selected checkpoint clustered at
        # the FIRST check (epoch 10 = warmup_epoch 5 + one interval), i.e. the criterion is
        # LEFT-censored: es_check_every=10 is too coarse to locate the optimum there.
        # These were pinned at train_integration_model's defaults because evaluate_config
        # never forwarded them; now a high-lambda run can tighten the cadence without
        # editing training defaults (which would silently change every other experiment).
        "es_patience": es_patience,
        "es_check_every": es_check_every,
        "z_dim": z_dim,
        "epochs": epochs,
        "warmup_epoch": warmup_epoch,
        "batch_size": batch_size,
        "lr_g": lr_g,
        "lr_d": lr_d,
        "reference_batch": reference_batch,
        "reference_batch_name_str": reference_batch_name_str,
        "reference_mode": reference_mode,
        "formulation": formulation,
    }
    if backbone is not None:
        kwargs["backbone"] = backbone
    vae, history = train_integration_model(adata, **kwargs)
    return vae, history


@contextlib.contextmanager
def _gpu_silhouette_backend():
    """Route sklearn's silhouette functions through the GPU kernel for this block.

    scib's silhouette wrappers import ``silhouette_samples`` / ``silhouette_score``
    into their own module namespace at import time, so patching ``sklearn.metrics``
    alone would not reach them -- both binding sites are replaced here and restored
    on exit, including on exception. A no-op when CUDA is unavailable.
    """
    import sklearn.metrics as skm

    try:
        import torch

        if not torch.cuda.is_available():
            yield
            return
    except ImportError:
        yield
        return

    # NOTE `from scib.metrics import silhouette` binds the FUNCTION, not the module --
    # scib.metrics re-exports its symbols. The modules that actually hold the sklearn
    # bindings are scib.metrics.silhouette / scib.metrics.isolated_labels, which must be
    # imported via importlib. Getting this wrong is silent: setattr on the wrong object
    # succeeds, the patch does nothing, and the suite runs at CPU speed with correct
    # numbers -- which is exactly how it first shipped.
    import importlib

    from wcd_vae.wcd.evaluation import gpu_silhouette_samples, gpu_silhouette_score

    _sil = importlib.import_module("scib.metrics.silhouette")
    _iso = importlib.import_module("scib.metrics.isolated_labels")

    targets = [
        (skm, "silhouette_samples", gpu_silhouette_samples),
        (skm, "silhouette_score", gpu_silhouette_score),
        (_sil, "silhouette_samples", gpu_silhouette_samples),
        (_sil, "silhouette_score", gpu_silhouette_score),
        (_iso, "silhouette_samples", gpu_silhouette_samples),
    ]
    # Fail loudly if a binding site disappears in a future scib: a missing target means
    # the metric silently reverts to CPU, which is a performance regression no test
    # would catch.
    missing = [f"{m.__name__}.{n}" for m, n, _ in targets if not hasattr(m, n)]
    if missing:
        raise AttributeError(f"GPU silhouette patch targets not found: {missing}")

    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in targets]
    try:
        for mod, name, repl in targets:
            setattr(mod, name, repl)
        yield
    finally:
        for mod, name, orig in saved:
            setattr(mod, name, orig)


def _resolution_cache(adata, embed_key, resolutions=None, flavor="igraph"):
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
    #      FLAVOUR defaults to igraph with n_iterations=2.
    #
    #      CORRECTION to an earlier note here, which claimed igraph "was NOT faster
    #      (0.8-1.0x)" and broke parity so leidenalg had to be pinned. That A/B was
    #      wrong: it left n_iterations at scanpy's igraph default of -1 (iterate to
    #      convergence) while leidenalg uses 2, so it compared different amounts of
    #      work. Matching n_iterations=2 restores parity AND the speed. Measured on
    #      REAL latents: pancreas 62.2s -> 4.1s (15.1x) with |dARI| 1.1e-05 and
    #      |dNMI| 4.7e-05; sim2 65.2s -> 5.4s (12.1x) with |dARI| 1.7e-05.
    #
    #      Across synthetic structure regimes the agreement is exact where structure
    #      exists (strong and moderate separation both give |dARI| = 0.00e+00) but is
    #      only approximate in the degenerate regime (|dARI| = 9.2e-03 at 9.1x). That
    #      residual is 0.1x the between-backbone ARI spread (0.088) and touches only
    #      ari/nmi/isolated_f1 -- ASW, LISI, probe and PAGA metrics involve no
    #      clustering. Pass flavor=None to force strict leidenalg parity.
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
        # WHY the backend swap: asw_celltype, asw_batch and isolated_asw are 35% of the
        #   warm suite on pancreas and 40% of an entire config on atac_large (353.05s of
        #   ~875s) -- the largest GPU-addressable block, while the GPU otherwise idles
        #   near 20% because the rest of the suite is CPU-bound. The share GROWS with n,
        #   so this pays most on the datasets that dominate the programme.
        # WHY a patch rather than a rewrite: all three bottom out in sklearn's
        #   silhouette_samples / silhouette_score. Swapping only the BACKEND lets scib's
        #   own aggregation run unchanged -- the abs(), the 1-x rescaling, the per-group
        #   means, the singleton and single-batch skips. Reimplementing those wrappers
        #   would risk silent drift in exactly the conventions that keep these numbers
        #   comparable to the published scIB benchmark.
        # SAFETY: gpu_silhouette_samples runs in float64 and falls back to sklearn when
        #   CUDA is absent, so results never depend on which backend ran. Measured
        #   agreement: 6.4e-09 (pancreas) and 1.3e-11 (atac_large) -- machine precision.
        with _gpu_silhouette_backend():
            out["asw_batch"] = float(
                scib.me.silhouette_batch(
                    adata, batch_key=batch_key, group_key=celltype_key, embed=embed_key, verbose=False
                )
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


def save_embedding(ad, embed_out, tag, batch_key, celltype_key, embed_key="X_latent"):
    """Persist one latent embedding as ``<embed_out>/<tag>.npz``.

    # WHY THIS IS A SHARED HELPER: metrics-only CSVs cannot be re-analysed. Adding any
    #   new embedding-derived metric (kBET, PAGA, probes, trajectory) to a finished wave
    #   otherwise costs a full retraining run. This project has paid that bill twice --
    #   once for PAGA, then again when kBET was requested after a 918-config / 24 h wave
    #   that persisted nothing. Every harness that trains a model must call this.
    # NOTE the DATASET is not part of ``tag``: callers MUST pass a dataset-specific
    #   ``embed_out`` directory. Wave 2019132 wrote every dataset into one flat directory
    #   and 7 datasets clobbered each other down to 96 files for 408 configs.
    # ``tag`` must encode EVERY field the calling experiment's grid varies, or configs
    #   silently overwrite each other with a complete-looking CSV.
    """
    import os

    # embed_key is a parameter because the baseline harness (scVI/Harmony/Scanorama)
    # writes its embedding to obsm["X_emb"] rather than obsm["X_latent"]. Persisting
    # baseline latents matters for the same reason as our own: a new embedding-derived
    # metric must not require re-running the baselines either.
    os.makedirs(embed_out, exist_ok=True)
    np.savez_compressed(
        os.path.join(embed_out, f"{tag}.npz"),
        z=np.asarray(ad.obsm[embed_key], dtype=np.float32),
        # WHY the dtype cast: numpy stores <U dtype string arrays fine, but object-dtype
        #      arrays require allow_pickle on load. Fixed-width unicode keeps np.load
        #      working with the safe default.
        batch=ad.obs[batch_key].astype(str).to_numpy(dtype="U64"),
        celltype=ad.obs[celltype_key].astype(str).to_numpy(dtype="U64"),
        X_pca=np.asarray(ad.obsm["X_pca"], dtype=np.float32) if "X_pca" in ad.obsm else np.empty(0),
    )
    return os.path.join(embed_out, f"{tag}.npz")


# Higher-is-better (+1) or lower-is-better (-1) for building composite / Pareto scores.
# clisi is -1 because THIS MODULE imports the LOCAL clisi_graph (see the import above,
# ``from wcd_vae.wcd.evaluation import clisi_graph``), which normalises
# (lisi - 1)/(n_celltypes - 1) so 1.0 = cell types fully MIXED -> lower is better.
# scib's own clisi_graph is scaled the OTHER way (1.0 = separated, higher is better); if
# the import is ever switched to scib's, flip this to +1 in the same commit. Measured
# side by side: local -0.0000 (separated) / 0.7909 (mixed); scib 1.0000 / 0.2085.
METRIC_DIRECTION = {
    "ilisi": +1, "clisi": -1, "graph_conn": +1, "kbet": +1,
    "asw_batch": +1, "asw_celltype": +1, "ari": +1, "nmi": +1,
    "pcr": +1, "isolated_asw": +1, "isolated_f1": +1, "cell_cycle": +1,
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
    lr_g=1e-3,
    lr_d=1e-3,
    metric_kwargs=None,
    embed_out=None,
    early_stopping=True,
    es_patience=5,
    es_check_every=10,
    full_n_batches=None,
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
        warmup_epoch=warmup_epoch, batch_size=batch_size, lr_g=lr_g, lr_d=lr_d,
        # WHY: without es_celltype_key the early-stopping check is skipped silently, so
        #      the label key MUST be threaded here or epochs=500 runs unguarded.
        early_stopping=early_stopping, es_celltype_key=celltype_key,
        es_patience=es_patience, es_check_every=es_check_every,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obtain_embeddings(ad, vae.to(device))
    if embed_out:
        # WHY reference_mode/formulation in the tag: E4 sweeps reference designs and E9
        #   sweeps formulations at the SAME (head, backbone, lambda, seed), so without
        #   them those arms silently overwrite each other.
        # WHY batch_size/lr are in the tag: the filename must encode EVERY field any
        #   experiment grid varies, or configs silently overwrite each other. E10 holds
        #   method/backbone/lambda/seed fixed and varies exactly these two, so without
        #   them its four cells per head collapse to one file -- invisible in the CSV.
        #   Suffixes are omitted at the production defaults so existing filenames are
        #   unchanged.
        opt = ""
        if z_dim != 256:
            # WHY z_dim IS IN THE TAG: a capacity sweep varies z_dim while holding
            # method/backbone/lambda/seed fixed. Without this suffix every z_dim writes the
            # SAME filename and silently overwrites the production 256-dim latent -- the
            # identity-string bug this repo has hit three times. Omitted at the 256 default
            # so existing filenames are unchanged.
            opt += f"_zd{int(z_dim)}"
        if batch_size != 1024:
            opt += f"_bs{int(batch_size)}"
        if abs(lr_g - 1e-3) > 1e-12 or abs(lr_d - 1e-3) > 1e-12:
            opt += f"_lr{str(lr_g).replace('.', 'p').replace('-', 'm')}"
        # WHY THE BATCH COUNT IS IN THE TAG: E8 sweeps batch_count, subsetting the data to
        #   k batches while holding (head, backbone, lambda, seed) FIXED -- so without this
        #   every level of the sweep writes to ONE file. Measured on the 2026-08-12 wave:
        #   26 E8 rows collapsed onto 6 filenames, silently overwriting 20 embeddings, and
        #   the surviving .npz for each tag was whichever level finished last. The CSVs
        #   looked complete throughout (bc=2 ARI 0.360 vs bc=8 ARI 0.020, both present).
        # The caller passes full_n_batches (the dataset's UNSUBSET batch count) because
        #   load_task subsets the data BEFORE evaluate_config sees it -- so the batch count
        #   cannot be recovered here by comparison; both adata and ad are already subset.
        #   The suffix is omitted when the two agree, keeping non-E8 filenames unchanged.
        n_batches_here = int(ad.obs[batch_key].nunique())
        if full_n_batches is not None and n_batches_here != int(full_n_batches):
            opt += f"_bc{n_batches_here}"
        # WHY THE REFERENCE BATCH IS IN THE TAG: E4 sweeps ref_design over
        #   fixed_ref0 .. fixed_ref{N-1} plus rotating/joint, and ALL the fixed_refN arms
        #   carry reference_mode="fixed" -- so reference_mode alone does not distinguish
        #   them and all N would write to one file. Which batch the critic aligns to
        #   changes the model: on pancreas, fixed_ref0 gave ARI 0.480 where the same
        #   (backbone, lambda, seed) at the E1/E2 grid point gave 0.049.
        #   Caught before any E4 latent was lost, unlike the E8 case (8ea5ea4).
        # WHY reference_batch==0 IS NOT SAFE TO OMIT: the two harnesses interpret the
        #   index differently. E1/E2/E8/E10/E5/E9 pass reference_batch_name_str (the
        #   ENTROPY-selected batch, per experiment_protocol.md S9 item 2) and training
        #   resolves by NAME, ignoring the index -- on pancreas that is inDrop1, index 3.
        #   E4 passes no name and resolves POSITIONALLY, so its fixed_ref0 aligns to
        #   celseq, index 0. Both correctly record reference_batch=0 yet train different
        #   models: measured ARI 0.072 (E1) vs 0.480 (E4 fixed_ref0) at the same
        #   (backbone, lambda, seed). Keying the suffix on the index alone therefore let
        #   E4's fixed_ref0 overwrite E1's latent. The tag must record WHICH RESOLUTION
        #   was used, not just the index value.
        if reference_mode == "fixed" and reference_batch is not None:
            if reference_batch_name_str is None:
                # positional resolution (E4's reference sweep): always tag the index
                opt += f"_refidx{int(reference_batch)}"
            elif int(reference_batch) != 0:
                opt += f"_ref{int(reference_batch)}"
        tag = (f"{'critic' if critic else 'discriminator'}_{backbone or 'NB'}"
               f"_lam{str(d_coef).replace('.', 'p')}_s{seed}"
               f"_{reference_mode}_{formulation}{opt}")
        save_embedding(ad, embed_out, tag, batch_key, celltype_key)
    metrics = full_metric_suite(ad, batch_key, celltype_key, embed_key="X_latent", **(metric_kwargs or {}))
    row = {
        "method": "critic" if critic else "discriminator",
        "backbone": backbone or "NB",
        "d_coef": d_coef,
        "seed": seed,
        "batch_size": batch_size,
        "lr_g": lr_g,
        "lr_d": lr_d,
        # WHY these are recorded: a results CSV has to be self-describing. Six training
        # parameters used to be settable here but absent from the row, so a reader could
        # not reconstruct the protocol from the results file and had to read code at the
        # right commit. disc_iter is the important one -- it defaults to 10 for the critic
        # and 1 for the discriminator, a 10x asymmetry in adversary updates BETWEEN THE
        # TWO ARMS BEING COMPARED. It is deliberate (standard WGAN practice), but an
        # undisclosed 10x difference in a controlled ablation is exactly what a reviewer
        # should be able to see without reading source.
        # kl_coef is deliberately NOT here: evaluate_config does not expose it, so it is
        # always train_integration_model's default (0.005). Recording a value this
        # function cannot vary would imply a knob that does not exist.
        # evaluate_config does not pass disc_iter, so train_one applies its head default.
        # Mirror that SAME expression here rather than hardcoding a number, so the two
        # cannot drift apart silently.
        "disc_iter": 10 if critic else 1,
        "z_dim": z_dim,
        "warmup_epoch": warmup_epoch,
        "epochs_budget": epochs,
        "early_stopping": bool(early_stopping),
        # Recorded so a checkpoint selected at the first check is auditable: es_best_epoch
        # == warmup_epoch + es_check_every is the signature of a left-censored selection.
        "es_patience": es_patience,
        "es_check_every": es_check_every,
        "reference_batch": reference_batch,
        "reference_mode": reference_mode,
        # WHY reference_resolution IS RECORDED: reference_batch ALONE IS AMBIGUOUS.
        #   E1/E2/E8/E10/E5/E9 pass reference_batch_name_str (the entropy-selected batch)
        #   and training resolves BY NAME, leaving the index at its legacy 0; E4 passes no
        #   name and resolves POSITIONALLY. Both then record reference_batch=0 while
        #   training against different batches -- on pancreas inDrop1 (index 3) versus
        #   celseq (index 0), measured ARI 0.072 vs 0.480 at the same
        #   (backbone, lambda, seed). Without this field no consumer -- reader, reviewer,
        #   or the wave monitor rebuilding embedding filenames -- can tell which model a
        #   row describes, and the embed tag itself has to encode the same distinction.
        "reference_resolution": (
            None if reference_batch is None
            else ("name" if reference_batch_name_str is not None else "index")
        ),
        "reference_batch_name": reference_batch_name_str,
        "formulation": formulation,
        # WHY: with early_stopping=True the "epochs" column records the requested BUDGET
        # (500), not where training actually stopped. Recording the selected epoch makes
        # each row self-describing and lets a wave be audited for runs that stopped early.
        "epochs_run": len(hist["all_loss"]),
        "es_best_epoch": (hist.get("_es_trace") or {}).get("es_best_epoch"),
        "final_loss": float(hist["all_loss"][-1]),
        "final_loss_da": float(hist["loss_da"][-1]),
        **metrics,
    }
    return row
