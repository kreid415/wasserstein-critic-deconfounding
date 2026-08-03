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

    from wcd_vae.wcd.hyperparameter import compute_mean_paga_spearman

    # scanpy 1.9.8 + igraph 1.0.0 crash inside get_sparse_from_igraph on non-empty
    # edge lists, so PAGA is unusable in that combination (cluster runs scanpy 1.11.5).
    # Skip rather than assert, so the test is meaningful where PAGA actually works.
    import scanpy as sc

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
