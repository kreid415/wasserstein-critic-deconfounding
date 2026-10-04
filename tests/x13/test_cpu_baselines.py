"""Checks of scripts/run_cpu_baselines.py (X13). Run in env wcd-kbet (scib 1.1.7, harmony-pytorch 0.1.7, scanorama):
    KMP_AFFINITY=disabled PREPPED_DIR=<prepped_scib> python -m pytest -q tests/x13

1. The runner's Harmony statements reproduce scib.integration.harmony at the default setting within tolerance
   1e-4 (harmonize reruns on this toy differed by up to 5.7e-06 in a long-running process or with
   MKL_DYNAMIC/OMP_DYNAMIC=FALSE, and by 0 in a fresh process with default settings); a shuffled batch column or
   the uncorrected PCA differs by more than 100x the tolerance (theta 2.5 instead of 2 does not: max|dz| 1e-05 on
   this toy, so it is not used).
2. The knobs reach the tools: theta 0 vs 2 changes Harmony's output far beyond the tolerance, knn 5 vs 20 changes
   Scanorama's; the fresh-process Harmony equals the in-process statements on the toy within the tolerance.
3. Scanorama's output (concatenated by batch) is returned in the input cell order.
4. Each arm returns the declared number of dimensions; pca refuses a knob value, harmony a negative theta,
   scanorama a non-integer knn; row selection refuses non-CPU tags and rows with the wrong knob or seed.
5. With PREPPED_DIR set: two fresh-process Harmony fits on atac_small at 50 PCs (theta 2, 1 thread) are
   bit-identical; the runner writes the npz format of fit_paper_config.py (z, obs_names, batch, celltype, config,
   history) in the prepped cell order for one row of each arm, and refuses to overwrite an existing output.
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
import build_paper_manifest as bpm  # noqa: E402
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


HARMONY_RERUN_TOL = 1e-4   # largest harmonize rerun difference seen on this toy: 5.7e-06 (2026-10-02; 0 in a fresh process)


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
    shuffled = b.obs.copy()                      # a wrong batch column (the realistic bug)
    shuffled["batch"] = pd.Categorical(np.random.default_rng(0).permutation(shuffled["batch"].astype(str).values))
    z_wrong = np.asarray(harmonize(b.obsm["X_pca"], shuffled, batch_key="batch"))
    assert float(np.abs(z_scib - z_wrong).max()) > 100 * HARMONY_RERUN_TOL
    assert float(np.abs(z_scib - np.asarray(b.obsm["X_pca"])).max()) > 100 * HARMONY_RERUN_TOL   # uncorrected PCA


def test_knobs_reach_the_tools():
    # on this toy theta changes Harmony's output at 50 PCs (max|dz| 1.8 for theta 0 vs 2) but hardly at 10 PCs
    # (2.7e-05); on atac_small at 10 PCs theta 0 / 2 / 8 gave kNN batch entropy 0.56 / 0.71 / 0.73 (2026-10-03)
    a = _toy()
    z2, kw2 = rcb.embed(a, "harmony", 50, "2")
    z0, kw0 = rcb.embed(a, "harmony", 50, "0")
    assert kw2 == {"n_comps": 50, "theta": 2.0, "n_jobs": 1, "fresh_process": True} and kw0["theta"] == 0.0
    assert float(np.abs(z2 - z0).max()) > 100 * HARMONY_RERUN_TOL
    z_in = np.asarray(rcb.harmony_scib_statements(a.copy(), "batch", 50, theta=2.0).obsm["X_emb"])
    assert float(np.abs(z2 - z_in).max()) <= HARMONY_RERUN_TOL           # fresh process = the same statements
    s20, kws = rcb.embed(a, "scanorama", 10, "20")
    s5, _ = rcb.embed(a, "scanorama", 10, "5")
    assert kws == {"dimred": 10, "knn": 20} and float(np.abs(s20 - s5).max()) > 1e-3


def test_scanorama_output_is_in_input_cell_order():
    a = _toy()
    z, kw = rcb.embed(a, "scanorama", 10, "20")
    out = scib.integration.scanorama(a.copy(), "batch", dimred=10, knn=20)
    assert kw == {"dimred": 10, "knn": 20}
    assert not (out.obs_names == a.obs_names).all()          # scanorama returns cells grouped by batch
    assert np.array_equal(z, np.asarray(out.obsm["X_emb"])[out.obs_names.get_indexer(a.obs_names)])


@pytest.mark.parametrize("arm,dims,value", [("pca", 10, "0"), ("pca", 50, "0"), ("harmony", 10, "0.5"),
                                            ("harmony", 50, "2"), ("scanorama", 10, "40"), ("scanorama", 100, "20")])
def test_declared_dimensions(arm, dims, value):
    a = _toy()
    z, _ = rcb.embed(a, arm, dims, value)
    assert z.shape == (a.n_obs, dims) and np.isfinite(z).all()


def test_invalid_knob_values_are_refused():
    a = _toy(n=300)
    for arm, value in [("pca", "1"), ("harmony", "-1"), ("scanorama", "2.5"), ("scanorama", "0"), ("bbknn", "1")]:
        with pytest.raises(ValueError):
            rcb.embed(a, arm, 10, value)


def _manifest(tmp_path):
    R = bpm.finalize(bpm.build(bpm.BACKBONES["stock"], "pilot", 3, 5, 8))
    m = pd.DataFrame(R)[bpm.COLS]
    p = tmp_path / "m.tsv"
    m.to_csv(p, sep="\t", index=False)
    return p, m


def test_row_selection_refuses_wrong_rows(tmp_path):
    p, m = _manifest(tmp_path)
    cpu = rcb.select_rows(p)
    assert len(cpu) == 16 * len(bpm.TASKS) and set(cpu.arm) == set(rcb.CPU_ARMS)
    with pytest.raises(ValueError):
        rcb.select_rows(p, tags=[m[m.arm == "sysvi"].tag.iloc[0]])
    with pytest.raises(KeyError):
        rcb.select_rows(p, tags=["no_such_tag"])
    bad = m.copy()
    i = bad.index[bad.arm == "harmony"][0]
    bad.loc[i, "extra"] = json.dumps({"knob": "knn"})
    pb = tmp_path / "bad.tsv"
    bad.to_csv(pb, sep="\t", index=False)
    with pytest.raises(ValueError):
        rcb.select_rows(pb, tags=[bad.loc[i, "tag"]])


@pytest.mark.skipif(not os.environ.get("PREPPED_DIR"), reason="needs PREPPED_DIR (prepped scIB h5ad files)")
def test_fresh_process_harmony_is_bit_identical_on_atac_small_d50():
    pre = ad.read_h5ad(os.path.join(os.environ["PREPPED_DIR"], "atac_small__scib.h5ad"))
    x = pre[:, pre.var["highly_variable"].values].copy()
    x.obs["batch"] = x.obs["batch"].astype(str).astype("category")
    z1, _ = rcb.harmony_fresh_process(x, 2.0, 50, threads=1)
    z2, _ = rcb.harmony_fresh_process(x, 2.0, 50, threads=1)
    assert z1.shape == (x.n_obs, 50) and np.isfinite(z1).all()
    assert np.array_equal(z1, z2), float(np.abs(z1 - z2).max())


@pytest.mark.skipif(not os.environ.get("PREPPED_DIR"), reason="needs PREPPED_DIR (prepped scIB h5ad files)")
def test_runner_on_atac_small_writes_the_fit_npz_format(tmp_path):
    p, m = _manifest(tmp_path)
    cpu = m[(m.task == "atac_small") & m.arm.isin(list(rcb.CPU_ARMS))]
    pick = {("harmony", "2", "10"), ("scanorama", "20", "10"), ("pca", "0", "10")}
    rows = cpu[[(r.arm, str(r.lam), str(r.n_latent)) in pick for r in cpu.itertuples()]]
    assert len(rows) == 3
    env = dict(os.environ, KMP_AFFINITY="disabled")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_cpu_baselines.py"), "--manifest", str(p),
           "--prepped-dir", os.environ["PREPPED_DIR"], "--out-dir", str(tmp_path / "out"), "--tag", *rows.tag]
    subprocess.run(cmd, check=True, env=env)
    pre = ad.read_h5ad(os.path.join(os.environ["PREPPED_DIR"], "atac_small__scib.h5ad"), backed="r")
    for r in rows.itertuples():
        d = np.load(tmp_path / "out" / "latents" / f"{r.tag}.npz", allow_pickle=False)
        assert set(d.files) == {"z", "obs_names", "batch", "celltype", "config", "history"}
        assert d["z"].shape == (pre.n_obs, 10) and np.isfinite(d["z"]).all()
        assert (d["obs_names"] == pre.obs_names.to_numpy()).all()
        assert (d["batch"] == pre.obs["batch"].astype(str).to_numpy()).all()
        cfg = json.loads(str(d["config"]))
        assert (cfg["row"]["tag"], cfg["row"]["arm"], cfg["knob"]) == (r.tag, r.arm, rcb.CPU_ARMS[r.arm])
        assert cfg["knob_value"] == float(r.lam)
        import host_info                                   # CR-02: CPU latents record the device and CPU model
        assert cfg["device"] == "cpu" and cfg["cpu_model"] == host_info.cpu_info()[0] and "gpu" not in cfg
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "exists" in r.stderr


# ---- non-finite CPU latents are a recorded outcome (lead decision 2026-10-03, PREREG sec 1) --------------------
def _cpu_toy_run(tmp_path, monkeypatch, fake_embed, extra_args=()):
    """Toy prepped task 'toy' with a pca and a scanorama X13 row; runs rcb.main in-process with embed replaced for
    scanorama by fake_embed. Returns (rows, out_dir)."""
    a = _toy(n=300, g=60)
    a.var["highly_variable"] = True
    a.write_h5ad(tmp_path / "toy__scib.h5ad")
    rows = [bpm.cpu_row("toy", "pca", None, 0, 10), bpm.cpu_row("toy", "scanorama", "knn", 20, 10)]
    for i, r in enumerate(rows):
        r["tag"] = f"X13_toy_{r['arm']}_{i}"
    man = tmp_path / "m.tsv"
    pd.DataFrame(rows)[bpm.COLS].to_csv(man, sep="\t", index=False)
    real = rcb.embed

    def embed(x, arm, dims, lam, threads=1):
        return fake_embed(x, dims) if arm == "scanorama" else real(x, arm, dims, lam, threads=threads)

    monkeypatch.setattr(rcb, "embed", embed)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run_cpu_baselines.py", "--manifest", str(man), "--prepped-dir", str(tmp_path),
                                      "--out-dir", str(out), *extra_args])
    rcb.main()
    return rows, out


def _nan_latent(x, dims):
    z = np.ones((x.n_obs, dims))
    z[:7, 0] = np.nan
    z[3, 1] = np.inf
    return z, {"note": "test"}


def nonfinite_contract_violations(rows, out):
    """The PREREG sec 1 contract for a non-finite CPU latent, as run_stage.py records it for GPU rows."""
    import hashlib
    import fit_outcome
    bad = []
    pca, scan = rows
    if (out / "status" / f"{pca['tag']}.json").exists():
        bad.append("finite row has a status record")
    p = out / "latents" / f"{scan['tag']}.npz"
    st_path = out / "status" / f"{scan['tag']}.json"
    if not p.exists() or not st_path.exists():
        return bad + [f"latent {p.exists()} / status {st_path.exists()} for the non-finite row"]
    st = fit_outcome.read_status(str(st_path), scan["tag"])
    z = np.load(p)["z"]
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    want = dict(status="nonfinite_latent", detail="8 of 3000 posterior-mean values are non-finite (7 cells)",
                latent_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), device="cpu", git_sha=head)
    bad += [f"{k}: {st.get(k)!r} != {v!r}" for k, v in want.items() if st.get(k) != v]
    if st["row"] != {k: str(v) for k, v in scan.items()}:
        bad.append(f"row {st['row']}")
    if np.isfinite(z).all():
        bad.append("saved latent is finite")
    return bad


def test_nonfinite_cpu_latent_is_recorded_not_an_error(tmp_path, monkeypatch):
    rows, out = _cpu_toy_run(tmp_path, monkeypatch, _nan_latent)       # returns normally: exit 0
    assert nonfinite_contract_violations(rows, out) == []
    assert np.isfinite(np.load(out / "latents" / f"{rows[0]['tag']}.npz")["z"]).all()
    with pytest.raises(FileExistsError, match="recorded outcome"):     # no silent refit over a recorded outcome
        _cpu_toy_run(tmp_path, monkeypatch, _nan_latent, ["--tag", rows[1]["tag"]])
    before = (out / "status" / f"{rows[1]['tag']}.json").read_bytes()
    _cpu_toy_run(tmp_path, monkeypatch, _nan_latent, ["--skip-existing"])
    assert (out / "status" / f"{rows[1]['tag']}.json").read_bytes() == before


def test_checker_detects_the_old_behaviour(tmp_path, monkeypatch):
    """Mutation check (fail-loud R11): the tagged runner raised on a non-finite latent and wrote no record."""
    src = open(rcb.__file__).read()
    old = "            if not np.isfinite(z.astype(np.float32)).all():       # the saved values, as the runner reads them\n"
    assert src.count(old) == 1
    import types
    mod = types.ModuleType("rcb_mutant")
    mod.__file__ = rcb.__file__
    exec(compile(src.replace(old, "            if False:\n"), rcb.__file__ + "<mutant>", "exec"), mod.__dict__)
    monkeypatch.setattr(sys.modules[__name__], "rcb", mod)
    rows, out = _cpu_toy_run(tmp_path, monkeypatch, _nan_latent)
    assert nonfinite_contract_violations(rows, out)


def test_wrong_latent_shape_stays_an_infrastructure_error(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="latent shape"):
        _cpu_toy_run(tmp_path, monkeypatch, lambda x, dims: (np.ones((x.n_obs, dims + 1)), {}))
    assert not (tmp_path / "out" / "status").exists()


def test_runner_runs_as_a_script_under_safe_path(tmp_path):
    """The runner is started as a script; with PYTHONSAFEPATH=1 the script directory is not on sys.path, so it must add
    it itself (it imports scripts/host_info.py and scripts/fit_outcome.py). One pca row on a toy task, exit 0."""
    a = _toy(n=300, g=60)
    a.var["highly_variable"] = True
    a.write_h5ad(tmp_path / "toy__scib.h5ad")
    r = bpm.cpu_row("toy", "pca", None, 0, 10)
    r["tag"] = "X13_toy_pca_script"
    man = tmp_path / "m.tsv"
    pd.DataFrame([r])[bpm.COLS].to_csv(man, sep="\t", index=False)
    env = dict(os.environ, PYTHONSAFEPATH="1", KMP_AFFINITY="disabled", OMP_NUM_THREADS="1")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_cpu_baselines.py"), "--manifest", str(man),
                        "--prepped-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")], env=env,
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert p.returncode == 0, p.stderr[-2000:]
    cfg = json.loads(str(np.load(tmp_path / "out" / "latents" / "X13_toy_pca_script.npz")["config"]))
    assert cfg["device"] == "cpu" and cfg["cpu_model"]
