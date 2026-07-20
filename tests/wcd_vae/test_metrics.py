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
