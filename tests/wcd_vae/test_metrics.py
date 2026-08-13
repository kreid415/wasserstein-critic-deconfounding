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


def test_knn_helper_is_exact_and_cached():
    """_knn must match sklearn exactly, and memoise across the two LISI calls.

    WHY: at scale the kNN search IS the LISI cost (atac_large: 92.53s search vs 1.34s
    numba kernel). iLISI and cLISI differ only in the label they score, so the search
    was being run twice on an identical embedding. Both properties are asserted here
    because a silent cache miss is a pure performance regression no other test sees.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    from wcd_vae.wcd import evaluation as ev

    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 24))
    k = 15

    ev._KNN_CACHE.clear()
    dist, idx = ev._knn(x, k)
    ref_d, ref_i = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(x).kneighbors(x)
    assert idx.shape == ref_i.shape
    assert (idx == ref_i).mean() == 1.0, "neighbour indices must be exact, not approximate"
    assert np.abs(dist - ref_d).max() < 1e-6

    # a second call on the same embedding must hit the cache, not search again
    assert len(ev._KNN_CACHE) == 1
    again = ev._knn(x, k)
    assert again[1] is idx, "identical query should return the cached arrays"
    assert len(ev._KNN_CACHE) == 1

    # a DIFFERENT embedding of the same shape must NOT collide (content-hash key)
    y = rng.normal(size=(400, 24))
    _, idx_y = ev._knn(y, k)
    assert not np.array_equal(idx_y, idx), "different data returned cached neighbours"


def test_paga_baseline_cache_keys_on_content_not_pointers():
    """The PAGA baseline cache must hit on identical input and miss on changed labels.

    WHY: the per-batch reference graphs are built on the UNINTEGRATED representation, so
    they are invariant across every configuration a sweep evaluates (measured 9.3s/config
    on atac_large). The first implementation hashed `obs[key].astype(str).to_numpy()`
    via .tobytes() -- a dtype=object array of str POINTERS. The key changed on every call
    so the cache never hit, and address reuse could equally have produced a false HIT on
    different labels. Both directions are asserted here.
    """
    import anndata as ad_mod
    import numpy as np
    import pandas as pd

    from wcd_vae.wcd import hyperparameter as hp

    rng = np.random.default_rng(0)
    # PAGA needs genuine neighbourhood structure: give each cell type a separated
    # centroid so the per-batch graphs are well defined. 300 cells x 3 batches x 5 types.
    n_per, n_types, n_batches = 20, 5, 3
    n = n_per * n_types * n_batches
    types = np.tile(np.repeat(np.arange(n_types), n_per), n_batches)
    coords = rng.normal(size=(n, 4)).astype(np.float32) * 0.35
    coords += (types[:, None] * 6.0).astype(np.float32)
    obs = pd.DataFrame(
        {
            "batch": np.repeat([f"b{i}" for i in range(n_batches)], n_per * n_types),
            "celltype": np.array([f"t{i}" for i in types]),
        }
    )
    adata = ad_mod.AnnData(X=coords.copy(), obs=obs)
    adata.obsm["X_pca"] = coords

    hp._PAGA_BASELINE_CACHE.clear()
    hp._paga_baseline(adata, "batch", "celltype", "X_pca")
    key1 = next(iter(hp._PAGA_BASELINE_CACHE))
    hp._paga_baseline(adata, "batch", "celltype", "X_pca")
    key2 = next(iter(hp._PAGA_BASELINE_CACHE))
    assert key1 == key2, "key must be stable across identical calls, or the cache never hits"

    # changed labels must MISS -- a false hit would silently reuse the wrong graphs
    adata.obs["batch"] = np.asarray(rng.permutation(adata.obs["batch"].to_numpy()))
    hp._paga_baseline(adata, "batch", "celltype", "X_pca")
    assert next(iter(hp._PAGA_BASELINE_CACHE)) != key1, "permuted labels reused the cache"

    # changed baseline coordinates must also MISS
    hp._PAGA_BASELINE_CACHE.clear()
    hp._paga_baseline(adata, "batch", "celltype", "X_pca")
    key3 = next(iter(hp._PAGA_BASELINE_CACHE))
    adata.obsm["X_pca"] = (coords + rng.normal(size=coords.shape).astype(np.float32)).copy()
    hp._paga_baseline(adata, "batch", "celltype", "X_pca")
    assert next(iter(hp._PAGA_BASELINE_CACHE)) != key3, "changed X_pca reused the cache"


def test_optimiser_settings_thread_through_the_whole_call_chain():
    """lr_g/lr_d/batch_size must survive every layer down to the optimizer.

    WHY: the chain is evaluate_config -> train_one -> train_integration_model ->
    train_model -> optim.Adam, and a parameter dropped at ANY layer fails silently --
    the run completes, the row records the requested value, and the model trains at the
    default. This asserts the signature chain rather than the behaviour, so it fails at
    import time rather than after a multi-day wave produces mislabelled results.
    """
    import inspect

    from wcd_vae.wcd.experiment import evaluate_config, train_one
    from wcd_vae.wcd.training import SCIntegrationModel, train_integration_model

    chain = [evaluate_config, train_one, train_integration_model, SCIntegrationModel.train_model]
    for fn in chain:
        params = inspect.signature(fn).parameters
        for name in ("lr_g", "lr_d", "batch_size"):
            assert name in params, f"{fn.__qualname__} does not accept {name}"

    # ...and the optimizers must actually READ them, not use a literal
    src = inspect.getsource(SCIntegrationModel.train_model)
    assert "lr=lr_d" in src and "lr=lr_g" in src, "Adam is not using the lr parameters"


def test_e10_grid_is_balanced_and_has_a_matched_baseline():
    """E10 must contain the production cell for BOTH heads, or deltas are unanchored."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_experiment import BASE_LR, configs_for

    cfgs = list(configs_for("E10", None, "refA"))
    cells = {(c["critic"], c["batch_size"], round(c["lr_g"] / BASE_LR, 3)) for c in cfgs}
    for crit in (False, True):
        assert (crit, 1024, 1.0) in cells, "missing the production baseline cell"
        for mult in (1.0, 2.0, 4.0):
            assert (crit, 4096, mult) in cells, f"missing bs=4096 lr={mult}x"
    # every cell must carry the same number of seeds, or arms are not comparable
    from collections import Counter

    counts = Counter((c["critic"], c["batch_size"], round(c["lr_g"] / BASE_LR, 3)) for c in cfgs)
    assert len(set(counts.values())) == 1, f"unbalanced seed counts: {counts}"


def test_resume_key_covers_every_field_each_grid_varies():
    """--resume must not collapse configs that differ only in a newer grid dimension.

    WHY: the key was (method, backbone, d_coef, seed). E10 holds all four fixed and varies
    batch_size and lr, so resuming an interrupted E10 wave would have skipped 18 of its 24
    configs as 'already done' and produced a silently truncated experiment. Asserted
    generically: for every experiment, distinct configs must yield distinct resume keys.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_experiment import _resume_key, configs_for

    for exp in ("E1", "E2", "E8", "E10"):
        cfgs = list(configs_for(exp, None, "refA"))
        keys = {
            _resume_key(
                "critic" if c.get("critic") else "discriminator",
                c.get("backbone", "NB"),
                c["d_coef"],
                c["seed"],
                c.get("batch_size"),
                c.get("lr_g"),
            )
            for c in cfgs
        }
        assert len(keys) == len(cfgs), (
            f"{exp}: {len(cfgs)} configs collapse to {len(keys)} resume keys -- "
            "a resumed wave would skip the difference"
        )

    # rows written before batch_size/lr_g columns existed must still match a default config
    assert _resume_key("critic", "NB", 0.2, 0, None, None) == _resume_key(
        "critic", "NB", 0.2, 0, 1024, 1e-3
    ), "legacy rows would be re-run instead of resumed"


def test_gpu_chunk_size_does_not_change_results():
    """Chunk size is a memory knob, never a numerical one.

    WHY: _vram_safe_chunk picks the row-block from FREE VRAM at call time, so two runs of
    the same config on a differently-loaded GPU use different chunks. If that changed any
    value, results would depend on what else was running -- irreproducible in the worst
    possible way. Both kernels must be exactly block-invariant.
    """
    import numpy as np

    from wcd_vae.wcd import evaluation as ev

    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(i * 4, 1, (300, 16)) for i in range(4)]).astype(np.float64)
    labels = np.repeat([f"t{i}" for i in range(4)], 300)

    big = ev.gpu_silhouette_samples(x, labels, chunk=2048)
    small = ev.gpu_silhouette_samples(x, labels, chunk=256)
    assert np.abs(big - small).max() < 1e-12, "silhouette depends on chunk size"

    ev._KNN_CACHE.clear()
    d1, i1 = ev._knn(x, 10)
    ev._KNN_CACHE.clear()
    orig = ev._vram_safe_chunk
    ev._vram_safe_chunk = lambda *a, **k: 128
    try:
        d2, i2 = ev._knn(x, 10)
    finally:
        ev._vram_safe_chunk = orig
        ev._KNN_CACHE.clear()
    assert (i1 == i2).all(), "kNN neighbours depend on chunk size"
    assert np.abs(d1 - d2).max() < 1e-12


def test_embedding_filename_encodes_every_varied_field():
    """Two configs that differ in ANY grid dimension must not share an embedding file.

    WHY: the tag was method/backbone/lambda/seed/mode/formulation. E10 holds all of those
    fixed and varies batch_size and lr, so its four cells per head would have written to
    one filename -- three embeddings silently lost, with a complete-looking CSV. This is
    the same failure mode as the historical flat-directory collision.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_experiment import configs_for

    def tag(c):
        bs = c.get("batch_size", 1024)
        lr = c.get("lr_g", 1e-3)
        opt = ""
        if bs != 1024:
            opt += f"_bs{int(bs)}"
        if abs(lr - 1e-3) > 1e-12:
            opt += f"_lr{str(lr).replace('.', 'p').replace('-', 'm')}"
        return (
            f"{'critic' if c.get('critic') else 'discriminator'}_{c.get('backbone', 'NB')}"
            f"_lam{str(c['d_coef']).replace('.', 'p')}_s{c['seed']}"
            f"_{c.get('reference_mode', 'fixed')}_{c.get('formulation', 'reference')}{opt}"
        )

    for exp in ("E1", "E2", "E8", "E10"):
        cfgs = list(configs_for(exp, None, "refA"))
        tags = {tag(c) for c in cfgs}
        assert len(tags) == len(cfgs), (
            f"{exp}: {len(cfgs)} configs collapse to {len(tags)} embedding filenames"
        )


def test_paga_survives_a_fully_disconnected_cluster_graph():
    """Well-separated cell types must yield NaN, not a crash.

    WHY: when no k-NN edge joins two different cell types, the contracted cluster graph
    has zero edges and igraph's get_adjacency_sparse hands scipy an empty index array --
    `sc.tl.paga` raises ValueError instead of returning the all-zero matrix it implies.
    Separation makes PAGA fail, which is the opposite of the intuition that clean data is
    the easy case, and evaluate_config's blanket `except Exception` would have swallowed
    it into a silent NaN column for an entire wave.

    NaN is the correct value here (an all-zero matrix has zero variance, so every batch
    is already rejected by the existing guard) -- this asserts it arrives WITHOUT an
    exception, and that the same input does not crash the cached baseline path either.
    """
    import anndata as ad_mod
    import numpy as np
    import pandas as pd

    from wcd_vae.wcd import hyperparameter as hp

    rng = np.random.default_rng(0)
    n_per, n_types, n_batches = 20, 5, 3
    n = n_per * n_types * n_batches
    types = np.tile(np.repeat(np.arange(n_types), n_per), n_batches)
    # 6.0 sigma apart: verified to leave the contracted graph with exactly 0 edges.
    coords = rng.normal(size=(n, 4)).astype(np.float32) * 0.35
    coords += (types[:, None] * 6.0).astype(np.float32)
    obs = pd.DataFrame(
        {
            "batch": np.repeat([f"b{i}" for i in range(n_batches)], n_per * n_types),
            "celltype": np.array([f"t{i}" for i in types]),
        }
    )
    adata = ad_mod.AnnData(X=coords.copy(), obs=obs)
    adata.obsm["X_pca"] = coords
    adata.obsm["X_latent"] = coords

    hp._PAGA_BASELINE_CACHE.clear()
    assert hp._paga_baseline(adata, "batch", "celltype", "X_pca") == {}

    val = hp.compute_mean_paga_spearman(
        adata, tech_key="batch", celltype_key="celltype", embed_key="X_latent"
    )
    assert np.isnan(val), f"expected NaN for an unmeasurable topology, got {val}"


def test_clisi_sign_matches_the_clisi_implementation_actually_imported():
    """The cLISI sign must match the FUNCTION THIS CODE CALLS, not scib's docs.

    WHY THIS TEST EXISTS: there are two cLISI implementations with OPPOSITE orientation.
    scib.metrics.clisi_graph(scale=True) returns (nlabs - clisi)/(nlabs - 1), so 1.0 =
    cell types separated and higher is better. wcd_vae.wcd.evaluation.clisi_graph -- the
    one experiment.py imports, and therefore the one behind every ``clisi`` column in
    every results CSV -- normalises (lisi - 1)/(n_celltypes - 1), the iLISI direction, so
    1.0 = cell types fully MIXED and LOWER is better.

    On 2026-08-12 the sign was "corrected" from -1 to +1 after reading scib's source,
    which would have inverted the bio-conservation category for all existing results. It
    was caught by measuring the local function instead of trusting the package. This test
    measures rather than asserts a constant, so it fails if the import is ever switched to
    scib's implementation without flipping the sign in the same commit.
    """
    import numpy as np
    import pandas as pd
    import anndata as ad_mod

    from wcd_vae.wcd.evaluation import clisi_graph
    from wcd_vae.wcd.experiment import METRIC_DIRECTION
    from wcd_vae.wcd import hyperparameter as hp

    rng = np.random.default_rng(0)
    n_per, n_ct = 120, 4
    ct = np.repeat([f"t{i}" for i in range(n_ct)], n_per)
    obs = pd.DataFrame({"celltype": pd.Categorical(ct)})

    separated = np.zeros((len(ct), 8), dtype="float32")
    for i in range(n_ct):
        separated[ct == f"t{i}", i] = 8.0
    separated += rng.normal(size=separated.shape).astype("float32") * 0.3
    mixed = rng.normal(size=(len(ct), 8)).astype("float32")

    vals = {}
    for name, X in (("separated", separated), ("mixed", mixed)):
        a = ad_mod.AnnData(X=X.copy(), obs=obs.copy())
        a.obsm["X_e"] = X
        vals[name] = float(clisi_graph(a, label_key="celltype", type="embed", use_rep="X_e"))

    assert vals["separated"] < vals["mixed"], (
        f"local clisi_graph is expected LOWER for good bio; got {vals}"
    )
    assert METRIC_DIRECTION["clisi"] == -1
    assert hp._SCIB_BIO["clisi"] == -1

    # paga_spearman is this project's own metric, not scIB -- it must not silently
    # redefine the published bio category.
    assert "paga_spearman" not in hp._SCIB_BIO
    assert "paga_spearman" not in hp._SCIB_BATCH


def test_scib_categories_match_the_published_metric_set():
    """The criterion's categories must be the scIB ones, not a house blend.

    WHY: a score labelled "scIB" that quietly swaps in a custom metric is not comparable
    to published scIB numbers, which is the entire reason for using it in a resubmission.
    hvg_score (gene-space, undefined for an embedding) and trajectory (not computed here)
    are legitimately absent; the scorer skips absent metrics rather than imputing them.
    """
    from wcd_vae.wcd import hyperparameter as hp

    assert set(hp._SCIB_BATCH) == {"pcr", "asw_batch", "ilisi", "graph_conn", "kbet"}
    assert set(hp._SCIB_BIO) <= {
        "nmi", "ari", "asw_celltype", "isolated_f1", "isolated_asw",
        "clisi", "hvg_score", "cell_cycle", "trajectory",
    }, "bio category contains a non-scIB metric"


def test_lambda_zero_is_not_a_clean_control_across_heads():
    """DOCUMENTS A KNOWN LIMITATION: lambda=0 does NOT make the two heads identical.

    At d_coef=0 the generator's adversarial term is multiplied by zero, so the naive
    expectation is that critic and discriminator produce the SAME model for a given seed.
    They do not, and this test pins the reason so nobody "fixes" the symptom by reseeding.

    CAUSE: the adversary still trains at lambda=0 -- opt_d.step() is gated on `warmup`,
    not on d_coef -- and the critic's gradient penalty draws from the GLOBAL torch RNG
    (torch.rand / torch.randint in multi_class_gradient_penalty). The critic therefore
    consumes a different amount of randomness per minibatch than the discriminator, which
    desynchronises every subsequent shuffle. Verified: VAE weights at INITIALISATION are
    bit-identical across heads (max|delta| = 0), so the divergence is entirely RNG
    consumed during training.

    MEASURED IMPACT on the completed wave: mean |dARI| at lambda=0 is 0.0153, BELOW the
    seed-to-seed noise floor of 0.0230, signed mean -0.0032 with wilcoxon p=0.77 -- i.e.
    noisy but UNBIASED, and the head effect at lambda>0 survives subtracting each
    dataset's lambda=0 offset (+0.0835 -> +0.0867). So this is a reporting caveat, not a
    confound. If a clean control is ever required, either skip the adversary update
    entirely when d_coef == 0 or give the gradient penalty its own torch.Generator.
    """
    import torch

    from wcd_vae.wcd.critic import multi_class_gradient_penalty
    from wcd_vae.wcd.adversarial import Discriminator

    torch.manual_seed(0)
    V, N, Dm = 3, 60, 8
    z = torch.randn(N, Dm, requires_grad=True)
    bid = torch.arange(N) % V
    head = Discriminator(Dm, V, critic=True, reference_batch=0)

    # The gradient penalty must consume global RNG -- that is the mechanism above.
    torch.manual_seed(123)
    before = torch.rand(1).item()
    torch.manual_seed(123)
    multi_class_gradient_penalty(head, z, bid, reference_batch=0, formulation="reference")
    after = torch.rand(1).item()
    assert before != after, (
        "gradient penalty no longer consumes global RNG -- if it was given its own "
        "Generator, lambda=0 may now be a clean control and the docstring above, plus "
        "the manuscript caveat, should be updated"
    )


def test_disc_iter_differs_by_head_and_must_be_recorded():
    """The heads get DIFFERENT adversary budgets, so the value has to reach the row.

    WHY: evaluate_config defaults disc_iter to 10 for the critic and 1 for the
    discriminator -- a 10x asymmetry in adversary updates that is part of the protocol,
    not an accident (standard WGAN practice). But it was absent from the result row, so
    no results CSV recorded it and a reader could not tell the arms apart on that axis
    without reading code at the right commit. Six training parameters were unrecorded;
    this is the one that DIFFERS BETWEEN THE ARMS BEING COMPARED.
    """
    import inspect

    from wcd_vae.wcd import experiment as ex

    src = inspect.getsource(ex.evaluate_config)
    assert '"disc_iter"' in src, "disc_iter must be written into the result row"

    # The recorded value is a MIRROR of train_one's default expression, so pin the two
    # together: if train_one's default ever changes, this fails rather than silently
    # recording a number the run did not use.
    train_src = inspect.getsource(ex.train_one)
    assert "10 if critic else 1" in train_src, (
        "train_one's disc_iter default changed; update the mirrored expression in "
        "evaluate_config's result row to match"
    )
    assert "10 if critic else 1" in src


def _build_manifest(tmp_path, scope="light"):
    import subprocess
    import sys
    out = tmp_path / "m.tsv"
    subprocess.run(
        [sys.executable, "scripts/build_manifest.py", "--scope", scope, "-o", str(out)],
        check=True, capture_output=True,
    )
    rows = [dict(zip(("phase", "worker", "tag", "est", "cmd"), ln.split("\t")))
            for ln in out.read_text().splitlines()[1:]]
    return rows


def test_every_training_shard_persists_embeddings(tmp_path):
    """No shard may train a model without --embed-out.

    WHY: metrics-only CSVs cannot be re-analysed. Adding any embedding-derived metric
    (kBET, PAGA, probes, trajectory) to a finished wave otherwise costs a full retraining
    run -- this project paid that bill twice, most recently a 918-config / 24 h wave that
    persisted nothing and made a kBET request a 14-37 h re-run instead of a 30 s backfill.
    The flag defaults to None, so its omission is silent; only a test makes it loud.
    """
    for row in _build_manifest(tmp_path):
        if "support_overlap" in row["cmd"]:
            continue  # E6 trains no model -- there is no latent to persist
        assert "--embed-out" in row["cmd"], f"{row['tag']} does not persist embeddings"


def test_e10_high_batch_shards_are_serial_and_low_batch_are_not(tmp_path):
    """The bs=4096 arm must run alone; bs=1024 must not waste the lanes.

    WHY: E10 sweeps batch_size in {1024, 4096} and the 4096 arm exhausts the 8 GiB card
    whenever ~6 workers share it. Observed three times in ONE wave and on BOTH heads
    (immune/critic lost 9 of 12 configs, sim1/critic 9 of 12, lung/DISCRIMINATOR 4 of 12),
    which is why the mitigation cannot be scoped to the critic. bs=1024 is the production
    setting and has never OOM'd, so serialising it too would idle five lanes for hours.
    """
    rows = _build_manifest(tmp_path)
    e10 = [r for r in rows if "_E10_" in r["tag"]]
    assert e10, "no E10 shards emitted"
    for r in e10:
        assert "--batch-size-only" in r["cmd"], f"{r['tag']} must split E10 by batch size"
        expect = "serial" if "bs4096" in r["tag"] else "parallel"
        assert r["phase"] == expect, f"{r['tag']} should be {expect}, got {r['phase']}"


def test_manifest_covers_every_experiment_and_scales_with_scope(tmp_path):
    """All eight experiments must be present, and --scope must actually change the set.

    WHY: the previous manifest silently covered only E1/E2/E8/E10 -- half the programme
    (E3 baselines, E4 reference designs, E5 biology, E6 support overlap, E9 formulations)
    lives in separate harnesses and was simply absent, so a "full relaunch" would have
    quietly skipped it.
    """
    light = _build_manifest(tmp_path, "light")
    allds = _build_manifest(tmp_path, "all")
    for exp in ("E1", "E2", "E3", "E4", "E5", "E6", "E8", "E9", "E10"):
        assert any(exp in r["tag"] for r in light), f"{exp} missing from the light manifest"
    assert len(allds) > len(light), "scope=all must add the two heavy datasets"
    heavy_tags = [r for r in allds if r["tag"].startswith(("atac_large", "immune_hum_mou"))]
    assert heavy_tags, "scope=all emitted no heavy-dataset shards"
    assert not [r for r in light if r["tag"].startswith(("atac_large", "immune_hum_mou"))]


def test_e8_never_emitted_for_two_batch_datasets(tmp_path):
    """E8 sweeps batch_count 2..n_batches, which is degenerate when n_batches == 2."""
    import json

    reg = json.load(open("configs/dataset_registry.json"))
    for r in _build_manifest(tmp_path, "all"):
        if "_E8_" in r["tag"]:
            ds = r["tag"].split("_E8_")[0]
            assert reg[ds]["n_batches"] > 2, f"E8 emitted for 2-batch dataset {ds}"
            assert "--batch-count" in r["cmd"], f"{r['tag']} missing --batch-count"


def test_every_resumable_shard_requests_resume(tmp_path):
    """A failed shard must cost only its missing configs on the next pass, not all of them."""
    for r in _build_manifest(tmp_path):
        if any(k in r["tag"] for k in ("_E1_", "_E2_", "_E8_", "_E10_", "_E4_", "_E9_")):
            assert "--resume" in r["cmd"], f"{r['tag']} is not resumable"


def test_embed_tag_encodes_batch_count_subset(tmp_path):
    """A batch_count subset must not overwrite the full-dataset embedding.

    WHY: E8 sweeps batch_count while holding (head, backbone, lambda, seed) FIXED, so
    without the subset level in the filename every level of the sweep writes to ONE
    .npz. Measured on the 2026-08-12 wave before the fix: 26 E8 rows collapsed onto 6
    filenames, silently overwriting 20 embeddings, and the survivor for each tag was
    whichever level finished last. The CSVs looked complete the whole time (bc=2 gave
    ARI 0.360 and bc=8 gave 0.020 -- both recorded, only one latent kept).

    The suffix must be OMITTED when the data is not subset, so that all pre-existing
    non-E8 filenames are unchanged.
    """
    import numpy as np
    import pandas as pd
    import anndata as ad
    from wcd_vae.wcd.experiment import save_embedding

    def tag_for(n_batches_here, full_n_batches):
        """Mirror the tag construction in evaluate_config for the batch-count field."""
        opt = ""
        if full_n_batches is not None and n_batches_here != int(full_n_batches):
            opt += f"_bc{n_batches_here}"
        return f"critic_NB_lam0p2_s0_fixed_reference{opt}"

    assert tag_for(3, 3) == "critic_NB_lam0p2_s0_fixed_reference", "full run must not gain a suffix"
    assert tag_for(2, 3) == "critic_NB_lam0p2_s0_fixed_reference_bc2"
    assert tag_for(8, 9) == "critic_NB_lam0p2_s0_fixed_reference_bc8"
    assert tag_for(2, 3) != tag_for(3, 3), "subset and full run must not collide"

    # and the writes must land in distinct files
    rng = np.random.default_rng(0)
    a = ad.AnnData(
        X=rng.normal(size=(20, 4)).astype("float32"),
        obs=pd.DataFrame({"batch": pd.Categorical(["b0", "b1"] * 10),
                          "celltype": pd.Categorical(["t0", "t1"] * 10)}),
    )
    a.obsm["X_latent"] = a.X
    d = tmp_path / "emb"
    p_full = save_embedding(a, str(d), tag_for(3, 3), "batch", "celltype")
    p_sub = save_embedding(a, str(d), tag_for(2, 3), "batch", "celltype")
    assert p_full != p_sub
    assert len(list(d.glob("*.npz"))) == 2


def test_embed_tag_encodes_fixed_reference_batch():
    """E4's fixed_refN arms must not share one embedding filename.

    WHY: E4 sweeps ref_design over fixed_ref0 .. fixed_ref{N-1} plus rotating/joint, and
    EVERY fixed_refN arm carries reference_mode="fixed" -- so reference_mode alone does
    not distinguish them and all N would write to one .npz. Which batch the critic aligns
    to changes the model: on pancreas, fixed_ref0 gave ARI 0.480 where the same
    (backbone, lambda, seed) at the E1/E2 grid point gave 0.049.

    Caught before any E4 latent was lost, unlike the E8 batch_count case. The suffix must
    be omitted for reference_batch=0 (the default) and for the discriminator (which has
    no reference batch), so pre-existing filenames are unchanged.
    """
    def suffix(reference_mode, reference_batch):
        return (f"_ref{int(reference_batch)}"
                if reference_mode == "fixed" and reference_batch not in (None, 0) else "")

    assert suffix("fixed", 0) == "", "default reference must not gain a suffix"
    assert suffix("fixed", None) == "", "discriminator has no reference batch"
    assert suffix("fixed", 1) == "_ref1"
    assert suffix("fixed", 8) == "_ref8"
    # rotating/joint are already separated by reference_mode in the tag body
    assert suffix("rotating", 0) == ""
    assert suffix("joint", 0) == ""
    # the distinguishing property: no two fixed_refN arms collide
    tags = {f"critic_NB_lam0p2_s0_fixed_reference{suffix('fixed', b)}" for b in range(9)}
    assert len(tags) == 9, "all nine fixed_refN designs must map to distinct filenames"


def test_every_training_shard_sets_the_epoch_ceiling(tmp_path):
    """500 is a CEILING that early stopping decides against; 150 truncates the critic.

    WHY: every harness DEFAULTS to --epochs 150, and at 150 the critic head never
    early-stopped on pancreas, lung or sim2 -- it was cut off mid-improvement. The user
    directed 500 precisely so early stopping ends runs instead of the budget. Measured on
    the run this test was written for: 41% of rows hit the 150 ceiling with a median
    es_best_epoch of 140, i.e. still improving at truncation.

    The regression that made this necessary: the hand-made manifest passed --epochs 500 on
    all 90 shards, and the build_manifest.py rewrite dropped the flag entirely, silently
    reverting the whole programme to 150.
    """
    rows = _build_manifest(tmp_path)
    for r in rows:
        if r["tag"].endswith("_E3") or "support_overlap" in r["cmd"]:
            continue  # E3's baselines own their schedules; E6 trains nothing
        assert "--epochs 500" in r["cmd"], f"{r['tag']} does not set the 500-epoch ceiling"


def test_every_manifest_flag_is_accepted_by_its_harness(tmp_path):
    """A flag the target script does not define kills the shard instantly at argparse.

    WHY: run_baselines.py has no --epochs (scVI/scANVI own their training schedules and
    harmony/scanorama/combat do not train), so adding the epoch ceiling to the shared flag
    string made all six E3 shards fail on 'unrecognized arguments' -- caught here rather
    than at launch.
    """
    import shlex
    import subprocess
    import sys

    checked = set()
    for r in _build_manifest(tmp_path):
        args = shlex.split(r["cmd"].replace("$PY", "").replace('"$R"', "/tmp")
                           .replace('"$EMB"', "/tmp/emb"))
        script = args[0]
        flags = tuple(a for a in args if a.startswith("--"))
        if (script, flags) in checked:
            continue
        checked.add((script, flags))
        help_text = subprocess.run([sys.executable, script, "--help"],
                                   capture_output=True, text=True).stdout
        for a in flags:
            assert a in help_text, f"{script} does not accept {a} (shard {r['tag']})"


def test_wave_status_tag_matches_evaluate_config_tag():
    """The monitor's tag() must reproduce evaluate_config's embed tag EXACTLY.

    WHY THIS TEST EXISTS: scripts/wave_status.py rebuilds each result row's expected
    embedding filename to check none is missing. When the tag builder in experiment.py
    gains a suffix and the monitor does not, every affected latent is silently reported as
    an orphan and its row as missing -- so the check that exists to catch lost embeddings
    starts crying wolf, and a REAL loss hides among the false ones. That happened twice in
    one wave: first with E8's _bc suffix (the monitor read `n_batches` off the row, which
    the harness never records, so it looked for the plain filename and passed by
    coincidence), then with E4's _ref suffix (absent from the monitor entirely, reporting
    10 healthy E4 latents as orphans and masking one genuinely deleted file).

    Pinning them together means a future suffix breaks this test rather than the wave.
    """
    import pandas as pd

    def monitor_tag(row):
        import importlib
        import sys
        sys.path.insert(0, "scripts")
        import wave_status
        importlib.reload(wave_status)
        return wave_status.tag(row)

    # E4 fixed_ref3: the suffix must appear
    row = pd.Series({"dataset": "pancreas", "method": "critic", "backbone": "NB",
                     "d_coef": 0.2, "seed": 0, "reference_mode": "fixed",
                     "formulation": "reference", "reference_batch": 3,
                     "batch_count": None, "batch_size": 1024, "lr_g": 1e-3})
    assert monitor_tag(row).endswith("_ref3"), monitor_tag(row)

    # reference_batch=0 is the default and must NOT gain a suffix
    row0 = row.copy()
    row0["reference_batch"] = 0
    assert monitor_tag(row0) == "critic_NB_lam0p2_s0_fixed_reference", monitor_tag(row0)

    # the discriminator has no reference batch
    rowd = row.copy()
    rowd["method"], rowd["reference_batch"] = "discriminator", float("nan")
    assert monitor_tag(rowd) == "discriminator_NB_lam0p2_s0_fixed_reference", monitor_tag(rowd)

    # E8 subset: _bc comes from the REGISTRY's full count, not a row field
    row8 = row.copy()
    row8["reference_batch"], row8["batch_count"] = 0, 2
    assert monitor_tag(row8).endswith("_bc2"), monitor_tag(row8)


def test_embed_tag_separates_positional_from_named_reference():
    """E4's positional reference sweep must not collide with the entropy-named reference.

    THE BUG THIS PINS: the two harnesses interpret reference_batch differently.
    E1/E2/E8/E10/E5/E9 pass reference_batch_name_str -- the ENTROPY-selected batch, per
    experiment_protocol.md S9 item 2 -- and training resolves by NAME, ignoring the index.
    E4 passes no name and resolves POSITIONALLY, because sweeping references is its purpose
    (also settled in that protocol item, so E4 being non-comparable is BY DESIGN, not a
    defect). Both record reference_batch=0, so keying the filename suffix on the index
    alone made E4's fixed_ref0 overwrite E1's latent: measured ARI 0.072 (E1, entropy ref
    inDrop1 = index 3) vs 0.480 (E4 fixed_ref0, positional celseq = index 0) at the same
    (backbone, lambda, seed).

    The tag must encode WHICH RESOLUTION was used, not just the index value.
    """
    def suffix(reference_batch, reference_batch_name_str, reference_mode="fixed"):
        if reference_mode != "fixed" or reference_batch is None:
            return ""
        if reference_batch_name_str is None:
            return f"_refidx{int(reference_batch)}"
        return f"_ref{int(reference_batch)}" if int(reference_batch) != 0 else ""

    # named (entropy) resolution: index 0 is the legacy placeholder and must stay bare,
    # so every latent written before this change keeps its filename
    assert suffix(0, "inDrop1") == ""
    # positional resolution: ALWAYS tagged, including index 0 -- that is the collision fix
    assert suffix(0, None) == "_refidx0"
    assert suffix(3, None) == "_refidx3"
    # discriminator has no reference batch at all
    assert suffix(None, None) == ""
    # rotating/joint are separated by reference_mode in the tag body
    assert suffix(0, None, reference_mode="rotating") == ""
    # the property that matters: named-ref and positional-ref-0 differ
    assert suffix(0, "inDrop1") != suffix(0, None)
    # and every positional index maps to its own filename
    assert len({suffix(b, None) for b in range(9)}) == 9
