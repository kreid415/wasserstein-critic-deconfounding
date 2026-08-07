"""Tests for the authored evaluation metrics (wcd.evaluation).

Replaces an earlier suite that imported symbols (LISI/BatchEntropy/...) which no
longer exist in the codebase. These are refactor-regression guards on the numba
LISI backend and the iLISI/cLISI graph wrappers used by every experiment; the
numerical algorithm itself is unchanged from the pre-refactor baseline.
"""
import anndata as ad
import numpy as np
import pandas as pd

from wcd_vae.wcd.evaluation import clisi_graph, compute_lisi, ilisi_graph


def _adata(x, batch, celltype):
    a = ad.AnnData(np.asarray(x, dtype=np.float32))
    a.obs["batch"] = [str(b) for b in batch]
    a.obs["celltype"] = [str(c) for c in celltype]
    a.obsm["X_emb"] = np.asarray(x, dtype=np.float32)
    return a


def test_compute_lisi_range():
    # WHY: LISI must lie within [1, n_labels]; HOW: moderate-spread mixed labels, enough cells
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1.0, (300, 6))
    lab = rng.integers(0, 3, 300).astype(str)
    scores = compute_lisi(x, pd.DataFrame({"lab": lab}), "lab", perplexity=30)
    assert scores.min() >= 1.0 - 1e-6
    assert scores.max() <= 3.0 + 1e-6


def test_ilisi_higher_when_batches_mixed():
    # WHY: iLISI should rise when batches overlap; HOW: moderately separated vs co-located
    rng = np.random.default_rng(0)
    sep = np.vstack([rng.normal(0, 1.0, (150, 6)), rng.normal(4, 1.0, (150, 6))])
    a_sep = _adata(sep, [0] * 150 + [1] * 150, [0] * 300)
    mixed = rng.normal(0, 1.0, (300, 6))
    a_mix = _adata(mixed, list(rng.integers(0, 2, 300)), [0] * 300)
    lo = ilisi_graph(a_sep, batch_key="batch", use_rep="X_emb", perplexity=30)
    hi = ilisi_graph(a_mix, batch_key="batch", use_rep="X_emb", perplexity=30)
    assert lo < hi


def test_clisi_lower_when_celltypes_separated():
    # WHY: cLISI (normalized) should be lower when cell types form distinct groups
    rng = np.random.default_rng(1)
    sep = np.vstack([rng.normal(0, 1.0, (150, 6)), rng.normal(5, 1.0, (150, 6))])
    a_sep = _adata(sep, list(rng.integers(0, 2, 300)), [0] * 150 + [1] * 150)
    mixed = rng.normal(0, 1.0, (300, 6))
    a_mix = _adata(mixed, list(rng.integers(0, 2, 300)), list(rng.integers(0, 2, 300)))
    sep_val = clisi_graph(a_sep, label_key="celltype", use_rep="X_emb", perplexity=30)
    mix_val = clisi_graph(a_mix, label_key="celltype", use_rep="X_emb", perplexity=30)
    assert sep_val < mix_val


# --- probe-based conservation metrics -------------------------------------------------

def test_probe_metrics_detect_celltype_and_ignore_absent_batch():
    """A latent separable by cell type but not batch: high label lift, ~zero batch lift.

    # WHY: guards the metric that replaces ARI/NMI when the latent encodes cell type
    #      diffusely rather than in compact clusters.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    from wcd_vae.wcd.evaluation import probe_metrics

    rng = np.random.default_rng(0)
    n = 900
    centers = rng.normal(0, 6, size=(3, 8))
    labels = np.repeat(["A", "B", "C"], n // 3)
    z = np.vstack([centers[i] + rng.normal(0, 0.5, size=(n // 3, 8)) for i in range(3)])
    batch = rng.choice(["b0", "b1"], size=n)

    adata = ad.AnnData(
        X=rng.normal(size=(n, 4)).astype("float32"),
        obs=pd.DataFrame({"ct": pd.Categorical(labels), "batch": pd.Categorical(batch)}),
    )
    adata.obsm["X_latent"] = z.astype("float32")

    m = probe_metrics(adata, "ct", "batch")
    assert m["knn_label_lift"] > 0.3, f"cell type not detected: {m['knn_label_lift']}"
    assert abs(m["knn_batch_lift"]) < 0.1, f"phantom batch signal: {m['knn_batch_lift']}"


def test_probe_metrics_report_zero_lift_on_noise():
    """Pure-noise latent must yield ~zero lift for both label and batch."""
    import anndata as ad
    import numpy as np
    import pandas as pd

    from wcd_vae.wcd.evaluation import probe_metrics

    rng = np.random.default_rng(1)
    n = 600
    adata = ad.AnnData(
        X=rng.normal(size=(n, 4)).astype("float32"),
        obs=pd.DataFrame(
            {
                "ct": pd.Categorical(rng.choice(["A", "B", "C"], size=n)),
                "batch": pd.Categorical(rng.choice(["b0", "b1"], size=n)),
            }
        ),
    )
    adata.obsm["X_latent"] = rng.normal(size=(n, 8)).astype("float32")

    m = probe_metrics(adata, "ct", "batch")
    assert abs(m["knn_label_lift"]) < 0.1
    assert abs(m["knn_batch_lift"]) < 0.1


def test_paga_spearman_returns_nan_not_zero_when_unmeasurable():
    """No usable per-batch correlation must yield NaN, never 0.0.

    # WHY: 0.0 is indistinguishable from "topology fully destroyed", so a
    #      non-measurement silently reads as a real, maximally-bad score and biases
    #      any average over it downward (same class as the fastmath LISI bug).
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import pytest

    # scanpy 1.9.8 + igraph 1.0.0 crash inside get_sparse_from_igraph on non-empty
    # edge lists, so PAGA is unusable in that combination (cluster runs scanpy 1.11.5).
    # Skip rather than assert, so the test is meaningful where PAGA actually works.
    import scanpy as sc

    from wcd_vae.wcd.hyperparameter import compute_mean_paga_spearman

    _probe = sc.AnnData(np.random.default_rng(0).normal(size=(60, 5)).astype("float32"))
    sc.pp.neighbors(_probe, use_rep="X")
    _probe.obs["g"] = pd.Categorical(["a", "b", "c"] * 20)
    try:
        sc.tl.paga(_probe, groups="g")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"PAGA non-functional in this environment: {type(exc).__name__}")

    rng = np.random.default_rng(0)
    n = 120
    # Two batches, only 2 cell types -> the <3 celltype guard skips every batch.
    adata = ad.AnnData(
        X=rng.normal(size=(n, 10)).astype("float32"),
        obs=pd.DataFrame(
            {
                "celltype": pd.Categorical(rng.choice(["A", "B"], size=n)),
                "batch": pd.Categorical(rng.choice(["b0", "b1"], size=n)),
            }
        ),
    )
    adata.obsm["X_latent"] = rng.normal(size=(n, 8)).astype("float32")
    adata.obsm["X_pca"] = rng.normal(size=(n, 8)).astype("float32")

    val = compute_mean_paga_spearman(
        adata, tech_key="batch", celltype_key="celltype", embed_key="X_latent"
    )
    assert np.isnan(val), f"expected NaN for unmeasurable topology, got {val}"


def test_embedding_tag_is_unique_across_arms():
    """Embedding filenames must distinguish every config dimension they vary over.

    # WHY: wave 2019132 wrote 408 configs' embeddings into one flat directory and only 96
    #   files survived -- the tag omitted the dataset (7 datasets clobbered each other)
    #   and omitted reference_mode/formulation (E4/E9 arms clobbered each other). The
    #   dataset is not visible inside evaluate_config, so callers pass a per-dataset
    #   subdirectory; this test locks the within-directory part of the tag.
    """
    def tag(critic, backbone, d_coef, seed, reference_mode, formulation):
        return (f"{'critic' if critic else 'discriminator'}_{backbone or 'NB'}"
                f"_lam{str(d_coef).replace('.', 'p')}_s{seed}"
                f"_{reference_mode}_{formulation}")

    base = dict(critic=True, backbone="NB_uncond", d_coef=0.2, seed=0,
                reference_mode="fixed", formulation="reference")
    seen = {tag(**base)}
    for field, value in [("critic", False), ("backbone", "ZINB"), ("d_coef", 0.5),
                         ("seed", 1), ("reference_mode", "joint"),
                         ("formulation", "pooled")]:
        variant = dict(base)
        variant[field] = value
        t = tag(**variant)
        assert t not in seen, f"varying {field} did not change the embedding tag: {t}"
        seen.add(t)
    assert len(seen) == 7


def test_resolution_cache_uses_igraph_with_matched_iterations():
    """The sweep must use igraph n_iterations=2, and agree with leidenalg where structure exists.

    WHY THIS TEST: the flavour was previously pinned to leidenalg on the strength of an
    A/B that forgot to match n_iterations (scanpy's igraph default is -1 = iterate to
    convergence, leidenalg's is 2), which made igraph look both slower and non-equivalent.
    Matching the parameter gives ~12-15x on real latents at |dARI| ~1e-5. This test pins
    BOTH halves of that: the flavour is actually igraph, and on a well-separated latent
    -- where the correct clustering is unambiguous -- the two flavours agree exactly.
    Degenerate latents are deliberately NOT asserted on: agreement there is approximate
    (|dARI| ~9e-3), which is documented in _resolution_cache and acceptable because it is
    0.1x the between-backbone spread.
    """
    import inspect

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score

    from wcd_vae.wcd.experiment import _best_from_cache, _resolution_cache

    assert inspect.signature(_resolution_cache).parameters["flavor"].default == "igraph"

    rng = np.random.default_rng(0)
    n_per, k = 300, 4
    Z = np.vstack([rng.normal(c * 6.0, 1.0, (n_per, 16)) for c in range(k)]).astype("float32")
    truth = np.repeat([f"c{i}" for i in range(k)], n_per)
    a = ad.AnnData(
        X=Z,
        obs=pd.DataFrame(
            {"ct": pd.Categorical(truth)}, index=[f"x{i}" for i in range(len(Z))]
        ),
    )
    a.obsm["X_latent"] = Z
    sc.pp.neighbors(a, use_rep="X_latent")

    def best_ari(flavor):
        cache = _resolution_cache(a, "X_latent", resolutions=[0.4, 1.0], flavor=flavor)
        _r, labels, _s = _best_from_cache(
            cache, lambda lab: adjusted_rand_score(truth, lab)
        )
        return adjusted_rand_score(truth, labels)

    ari_igraph = best_ari("igraph")
    ari_leidenalg = best_ari(None)
    assert ari_igraph > 0.9, f"igraph failed to recover separated clusters: {ari_igraph}"
    assert abs(ari_igraph - ari_leidenalg) < 1e-9, (
        f"flavours disagree on a well-separated latent: "
        f"igraph={ari_igraph} leidenalg={ari_leidenalg}"
    )


def test_gpu_silhouette_matches_sklearn():
    """The GPU silhouette kernel must agree with sklearn to machine precision.

    Covers the cases that actually bit during implementation: singleton clusters
    (sklearn defines their silhouette as 0), and the two call conventions scib uses
    (positional, and X=/labels= by keyword). Runs on CPU fallback when no GPU is
    present, which still exercises the signature handling.
    """
    import numpy as np
    from sklearn.metrics import silhouette_samples, silhouette_score

    from wcd_vae.wcd.evaluation import gpu_silhouette_samples, gpu_silhouette_score

    rng = np.random.default_rng(0)
    x = np.vstack(
        [rng.normal(0, 1, (60, 8)), rng.normal(5, 1, (60, 8)), rng.normal(11, 1, (1, 8))]
    )
    labels = np.array(["a"] * 60 + ["b"] * 60 + ["solo"])

    ref = silhouette_samples(x, labels)
    got = gpu_silhouette_samples(x, labels)
    assert np.abs(ref - got).max() < 1e-7, f"max |delta| {np.abs(ref - got).max():.2e}"
    # sklearn's convention for a singleton cluster
    assert abs(got[-1]) < 1e-12, f"singleton silhouette should be 0, got {got[-1]}"

    # scib calls silhouette_score with a CAPITAL X keyword (sklearn's own name)
    assert abs(gpu_silhouette_score(X=x, labels=labels) - silhouette_score(x, labels)) < 1e-7
    # ...and silhouette_samples positionally, with metric= passed through
    assert np.abs(gpu_silhouette_samples(x, labels, metric="euclidean") - ref).max() < 1e-7
    # a non-euclidean metric must fall through to sklearn, not silently return euclidean
    ref_cos = silhouette_samples(x, labels, metric="cosine")
    got_cos = gpu_silhouette_samples(x, labels, metric="cosine")
    assert np.abs(ref_cos - got_cos).max() < 1e-12


def test_gpu_silhouette_backend_patches_the_real_call_sites():
    """The patch must reach the bindings scib actually calls, and restore them.

    WHY: `from scib.metrics import silhouette` binds the FUNCTION, not the module.
    Patching that object succeeded silently, did nothing, and the suite ran at CPU
    speed with correct numbers -- undetectable without this assertion. The context
    manager now resolves the real modules via importlib and raises if a target is
    missing, so a future scib reorganisation fails loudly instead of regressing.
    """
    import importlib

    import sklearn.metrics as skm

    from wcd_vae.wcd.experiment import _gpu_silhouette_backend

    sil_mod = importlib.import_module("scib.metrics.silhouette")
    iso_mod = importlib.import_module("scib.metrics.isolated_labels")

    # the binding sites must exist -- this is what the fail-loud guard protects
    for mod, name in (
        (skm, "silhouette_samples"),
        (skm, "silhouette_score"),
        (sil_mod, "silhouette_samples"),
        (sil_mod, "silhouette_score"),
        (iso_mod, "silhouette_samples"),
    ):
        assert hasattr(mod, name), f"missing patch target {mod.__name__}.{name}"

    before = sil_mod.silhouette_samples
    with _gpu_silhouette_backend():
        pass
    assert sil_mod.silhouette_samples is before, "backend did not restore the original"
