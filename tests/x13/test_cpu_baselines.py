"""Checks of scripts/run_cpu_baselines.py (X13). Run in env wcd-kbet (scib 1.1.7, harmony-pytorch 0.1.7, scanorama):
    KMP_AFFINITY=disabled python -m pytest -q tests/x13

1. The runner's Harmony statements reproduce scib.integration.harmony at the default setting to harmonize's own
   rerun noise (<= 5.7e-06 measured; tolerance 1e-4), and a different theta differs by > 100x that.
2. Scanorama's output (concatenated by batch) is returned in the input cell order.
3. Each method returns the declared number of dimensions at both settings.
4. With PREPPED_DIR set: the runner on atac_small writes the npz format of fit_paper_config.py (z, obs_names,
   batch, celltype, config, history) in the prepped cell order, and refuses to overwrite an existing output.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

scib = pytest.importorskip("scib")
pytest.importorskip("harmony")
import anndata as ad  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_cpu_baselines as rcb  # noqa: E402


def _toy(n=900, g=200, k=3, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.integers(0, k, n)
    ct = rng.integers(0, 4, n)
    mu = np.exp(rng.normal(0, 1, (4, g)))[ct] * np.exp(rng.normal(0, 0.5, (k, g)))[b]
    X = np.log1p(rng.poisson(mu)).astype(np.float32)
    a = ad.AnnData(sp.csr_matrix(X))
    a.obs_names = [f"cell{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["batch"] = pd.Categorical([f"b{i}" for i in b])
    a.obs["celltype"] = [f"t{i}" for i in ct]
    return a


HARMONY_RERUN_TOL = 1e-4   # harmonize reruns on identical input differ by <= 5.7e-06 (measured 2026-10-02)


def test_harmony_statements_reproduce_scib_harmony():
    a = _toy()
    z_scib = np.asarray(scib.integration.harmony(a.copy(), "batch").obsm["X_emb"])
    z_run = np.asarray(rcb.harmony_scib_statements(a.copy(), "batch", None).obsm["X_emb"])
    assert z_run.shape == z_scib.shape and float(np.abs(z_scib - z_run).max()) <= HARMONY_RERUN_TOL
    # the comparison has detection power: other settings differ far beyond the rerun tolerance
    z_other = np.asarray(rcb.harmony_scib_statements(a.copy(), "batch", 49).obsm["X_emb"])
    assert z_other.shape[1] == 49
    from harmony import harmonize
    import scanpy as sc
    b = a.copy()
    sc.tl.pca(b)
    z_theta = np.asarray(harmonize(b.obsm["X_pca"], b.obs, batch_key="batch", theta=2.5))
    assert float(np.abs(z_scib - z_theta).max()) > 100 * HARMONY_RERUN_TOL


def test_scanorama_output_is_in_input_cell_order():
    a = _toy()
    z, kw = rcb.embed(a, "scanorama", "primary")
    out = scib.integration.scanorama(a.copy(), "batch", dimred=10)
    assert kw == {"dimred": 10}
    assert not (out.obs_names == a.obs_names).all()          # scanorama returns cells grouped by batch
    assert np.array_equal(z, np.asarray(out.obsm["X_emb"])[out.obs_names.get_indexer(a.obs_names)])


@pytest.mark.parametrize("method", rcb.METHODS)
def test_declared_dimensions(method):
    a = _toy()
    for setting in rcb.SETTINGS:
        z, _ = rcb.embed(a, method, setting)
        dims = rcb.PRIMARY_DIMS if setting == "primary" else rcb.TOOL_DEFAULT_DIMS[method]
        assert z.shape == (a.n_obs, dims) and np.isfinite(z).all()


@pytest.mark.skipif(not os.environ.get("PREPPED_DIR"), reason="needs PREPPED_DIR (prepped scIB h5ad files)")
def test_runner_on_atac_small_writes_the_fit_npz_format(tmp_path):
    env = dict(os.environ, KMP_AFFINITY="disabled")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_cpu_baselines.py"), "--task", "atac_small",
           "--prepped-dir", os.environ["PREPPED_DIR"], "--out-dir", str(tmp_path)]
    subprocess.run(cmd, check=True, env=env)
    pre = ad.read_h5ad(os.path.join(os.environ["PREPPED_DIR"], "atac_small__scib.h5ad"), backed="r")
    n_files = 0
    for method in rcb.METHODS:
        for setting in rcb.SETTINGS:
            dims = rcb.PRIMARY_DIMS if setting == "primary" else rcb.TOOL_DEFAULT_DIMS[method]
            d = np.load(tmp_path / "latents" / f"X13_atac_small_{method}_d{dims}.npz", allow_pickle=False)
            assert set(d.files) == {"z", "obs_names", "batch", "celltype", "config", "history"}
            assert d["z"].shape == (pre.n_obs, dims) and np.isfinite(d["z"]).all()
            assert (d["obs_names"] == pre.obs_names.to_numpy()).all()
            assert (d["batch"] == pre.obs["batch"].astype(str).to_numpy()).all()
            cfg = json.loads(str(d["config"]))
            assert (cfg["row"]["arm"], cfg["row"]["setting"], cfg["row"]["n_dims"]) == (method, setting, dims)
            n_files += 1
    assert n_files == len(rcb.METHODS) * len(rcb.SETTINGS)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "exists" in r.stderr
