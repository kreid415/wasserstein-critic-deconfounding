"""score_scib_native.py after the code check of prereg-tier12-v1 (CR-01, CR-05, CR-08, CR-14). Runs in the wcd-kbet
env with R_HOME and R_LIBS as in scripts/run_wave.sh; skipped, with the reason, where rpy2 or R kBET is missing.

The toy prepped file follows the production convention (prep_scib_task.py): obsm X_pca from the training (HVG)
features and NO uns['pca']; 60 HVG features carry the cell types, 60 non-HVG features carry a strong batch effect,
so an HVG-only and an all-feature PCR reference give different PCR_batch values.
1. Full data: PCR_batch equals pcr_comparison with the all-feature PCA of the cells (scIB's own convention).
2. Subsample (X3/X8 path): PCR_batch equals the all-feature PCA reference of the SUBSET's cells and differs from the
   HVG reference the tagged scorer used; a mutant with the tagged HVG sc.pp.pca line fails this check.
3. Subset refusals: an unknown cell name, a repeated cell name, a prepped file with uns['pca'] (CR-14, CR-01).
4. Provenance columns (pinned 2026-10-03) and diagnostic flags on every row: git SHA = HEAD, versions JSON keys,
   CPU fields; kBET labels skipped = 2 by construction (one label of 5 cells, one single-batch label), forced = 0;
   traj_root_fallback '' without trajectory, 0 with a root cell, 1 when get_root raises RootCellError (scib then
   reports 0; forced by a stub, and naturally on separated types whose start type is not in the largest kNN
   component); kbet_label_flags known answers; META may not set scorer columns.
"""
import importlib
import json
import os
import subprocess
import sys
import types

os.environ.setdefault("KMP_AFFINITY", "disabled")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ro = pytest.importorskip("rpy2.robjects", reason="rpy2 is not installed (kBET runs in the wcd-kbet env)")
from rpy2.rinterface_lib.embedded import RRuntimeError  # noqa: E402

try:
    ro.r("suppressMessages(library(kBET))")
except RRuntimeError as exc:
    pytest.skip(f"R kBET package not loadable (set R_LIBS as in scripts/run_wave.sh): {exc}", allow_module_level=True)
ad = pytest.importorskip("anndata")
sc = pytest.importorskip("scanpy")
scib = pytest.importorskip("scib")
import score_scib_native as SN  # noqa: E402


def _toy(n_per=60, trajectory=False, seed=0, sep=3.0):
    rng = np.random.default_rng(seed)
    b = np.repeat(np.arange(3), 3 * n_per)
    ct = np.tile(np.repeat(np.arange(3), n_per), 3)
    labels = [f"c{i}" for i in ct]
    bats = [f"b{i}" for i in b]
    centers, shift = rng.normal(0, sep, (3, 60)), rng.normal(0, 4, (3, 60))
    hvg = centers[ct] + rng.normal(0, 1, (len(b), 60))
    other = shift[b] + rng.normal(0, 1, (len(b), 60))
    extra_x, extra_l, extra_b = [], [], []
    for lab, bat, k in (("c_small", "b0", 5), ("c_single", "b1", 30)):   # skipped by scib's kBET by construction
        extra_x.append(np.hstack([rng.normal(6, 1, (k, 60)), rng.normal(0, 1, (k, 60)) + shift[int(bat[1])]]))
        extra_l += [lab] * k
        extra_b += [bat] * k
    X = np.vstack([np.hstack([hvg, other])] + extra_x).astype(np.float32)
    a = ad.AnnData(X)
    a.obs_names = [f"cell{i}" for i in range(a.n_obs)]
    a.var_names = [f"g{i}" for i in range(120)]
    a.obs["batch"] = pd.Categorical(bats + extra_b)
    a.obs["celltype"] = pd.Categorical(labels + extra_l)
    a.var["highly_variable"] = np.arange(120) < 60
    a.obsm["X_pca"] = sc.pp.pca(X[:, :60], n_comps=10, random_state=0)   # prep convention: HVG PCA, no uns['pca']
    a.uns.update(batch_key="batch", celltype_key="celltype", organism="none", modality="rna")
    if trajectory:
        code = a.obs["celltype"].map({"c0": 0.0, "c1": 1.0, "c2": 2.0, "c_small": 0.5, "c_single": 1.5}).astype(float)
        a.obs["dpt_pseudotime"] = (code + rng.uniform(0, 0.5, a.n_obs)).to_numpy() / 3.0
    z = np.hstack([hvg[:, :6], 0.5 * other[:, :2]])
    z = np.vstack([z] + [np.hstack([e[:, :6], 0.5 * e[:, 60:62]]) for e in extra_x]).astype(np.float32)
    return a, z


def _write(d, a, z, name, cells=None):
    pre = str(d / f"{name}__scib.h5ad")
    a.write_h5ad(pre)
    idx = np.arange(a.n_obs) if cells is None else np.asarray(cells)
    npz = str(d / f"{name}.npz")
    np.savez(npz, z=z[idx], batch=a.obs["batch"].astype(str).to_numpy()[idx],
             celltype=a.obs["celltype"].astype(str).to_numpy()[idx], obs_names=a.obs_names.to_numpy()[idx])
    return pre, npz


def _score(monkeypatch, d, pre, npz, name, mod=SN, meta=None):
    out = str(d / f"{name}.csv")
    for k, v in dict(PREPPED=pre, NPZ=npz, TAG=name, OUT_CSV=out, META=json.dumps(meta or {})).items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("KBET_SEED", raising=False)
    mod.main()
    return pd.read_csv(out, keep_default_na=False, na_values=[""]).iloc[0]


def _expected_pcr(a, z, cells, hvg_reference=False):
    """PCR_batch of scib.metrics.metrics on these cells with the all-feature (or the tagged HVG) reference."""
    pre = a[np.asarray(cells)].copy()
    integ = pre.copy()
    integ.obsm["X_emb"] = z[np.asarray(cells)]
    if hvg_reference:
        sc.pp.pca(pre, n_comps=50, mask_var="highly_variable")
    return float(scib.metrics.pcr_comparison(pre, integ, covariate="batch", embed="X_emb"))


@pytest.fixture(scope="module")
def toy():
    return _toy()


def test_full_data_pcr_uses_the_all_feature_reference_and_rows_carry_provenance(monkeypatch, tmp_path, toy):
    a, z = toy
    pre, npz = _write(tmp_path, a, z, "full")
    r = _score(monkeypatch, tmp_path, pre, npz, "full")
    assert r["PCR_batch"] == _expected_pcr(a, z, np.arange(a.n_obs))
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert r["scorer_git_sha"] == head and r["scorer_dirty"] in (0, 1)
    v = json.loads(r["scorer_versions"])
    assert set(v) == set(SN.VERSION_PACKAGES) | {"R_kBET"} and all(isinstance(x, str) and x for x in v.values())
    import importlib.metadata as md
    assert v["scib"] == md.version("scib") == "1.1.7"
    flags = next(line for line in open("/proc/cpuinfo") if line.startswith("flags")).split(":", 1)[1].split()
    assert r["cpu_simd"] == next((x for x in ("avx512f", "avx2", "sse4_2") if x in flags), "none")
    assert isinstance(r["cpu_model"], str) and r["cpu_model"] and r["score_host"]
    assert (r["numba_cpu_name"] if isinstance(r["numba_cpu_name"], str) else "") == os.environ.get("NUMBA_CPU_NAME", "")
    assert pd.isna(r["traj_root_fallback"])                      # not computed: written as ''
    assert int(r["kbet_labels_skipped"]) == 2 and int(r["kbet_labels_forced_one"]) == 0 and r["kbet_r_calls"] == 3
    cols = list(pd.read_csv(tmp_path / "full.csv").columns)
    assert all(c in cols for c in SN.PROVENANCE_COLS + SN.DIAGNOSTIC_COLS)


def test_subsample_pcr_reference_is_the_all_feature_pca_of_the_subset(monkeypatch, tmp_path, toy):
    a, z = toy
    cells = np.sort(np.random.default_rng(1).choice(a.n_obs, 400, replace=False))
    pre, npz = _write(tmp_path, a, z, "sub", cells)
    r = _score(monkeypatch, tmp_path, pre, npz, "sub")
    allgene, hvg = _expected_pcr(a, z, cells), _expected_pcr(a, z, cells, hvg_reference=True)
    assert r["PCR_batch"] == allgene
    assert abs(allgene - hvg) > 0.05, (allgene, hvg)            # the two conventions differ on this toy
    # mutation check (fail-loud R11): the tagged scorer's HVG PCA line makes this test fail
    src = open(SN.__file__).read()
    anchor = "        pre = pre[idx].copy()\n"
    assert src.count(anchor) == 1
    mod = types.ModuleType("score_scib_native_hvg_reference")
    mod.__file__ = SN.__file__
    exec(compile(src.replace(anchor, anchor + '        sc.pp.pca(pre, n_comps=50, mask_var="highly_variable")\n'),
                 SN.__file__ + "<mutant>", "exec"), mod.__dict__)
    try:
        rm = _score(monkeypatch, tmp_path, pre, npz, "sub_mutant", mod=mod)
    finally:
        SN.seed_kbet(SN.KBET_SEED_DEFAULT)
        SN.count_kbet_neighbours()
        SN.watch_trajectory_root()
    assert rm["PCR_batch"] == hvg and rm["PCR_batch"] != allgene


def test_subsample_refuses_unknown_repeated_names_and_uns_pca(monkeypatch, tmp_path, toy):
    a, z = toy
    cells = np.arange(0, a.n_obs, 2)
    pre, npz = _write(tmp_path, a, z, "bad", cells)
    d = dict(np.load(npz, allow_pickle=True))
    for name, change, msg in (("unknown", lambda o: np.concatenate([o[:-1], ["no_such_cell"]]), "not in"),
                              ("repeat", lambda o: np.concatenate([o[:-1], o[:1]]), "more than once")):
        p = str(tmp_path / f"{name}.npz")
        np.savez(p, **{**d, "obs_names": change(d["obs_names"])})
        with pytest.raises(ValueError, match=msg):
            _score(monkeypatch, tmp_path, pre, p, name)
    b = a.copy()
    sc.pp.pca(b, n_comps=10)                                     # writes uns['pca']
    pre_pca, _ = _write(tmp_path, b, z, "with_pca", cells)
    with pytest.raises(ValueError, match="uns\\['pca'\\]"):
        _score(monkeypatch, tmp_path, pre_pca, npz, "with_pca")


def test_meta_may_not_set_scorer_columns(monkeypatch, tmp_path, toy):
    a, z = toy
    pre, npz = _write(tmp_path, a, z, "meta")
    with pytest.raises(ValueError, match="scorer columns"):
        _score(monkeypatch, tmp_path, pre, npz, "meta", meta={"cpu_model": "x"})


def test_kbet_label_flags_known_answers():
    obs = pd.DataFrame({"celltype": ["a"] * 12 + ["b"] * 12 + ["c"] * 12 + ["d"] * 4 + ["e"] * 15,
                        "batch": ["x", "y"] * 6 * 3 + ["x"] * 4 + ["y"] * 15})
    obs["celltype"] = pd.Categorical(obs["celltype"], categories=list("abcdez"))   # 'z': unused category
    s = types.SimpleNamespace(calls=1, nan_returns=0)
    nn = types.SimpleNamespace(calls=2, neighbors_errors=1)
    # tested a, b, c; one failed the 75% rule (no diffusion_nn call), one NeighborsError; d small, e single batch
    assert SN.kbet_label_flags(obs, "celltype", "batch", s, nn) == (2, 2)
    assert SN.kbet_label_flags(obs, "celltype", "batch", types.SimpleNamespace(calls=3, nan_returns=1),
                               types.SimpleNamespace(calls=3, neighbors_errors=0)) == (0, 3)
    with pytest.raises(RuntimeError, match="accounting"):
        SN.kbet_label_flags(obs, "celltype", "batch", types.SimpleNamespace(calls=3, nan_returns=0),
                            types.SimpleNamespace(calls=2, neighbors_errors=0))


def test_trajectory_root_flag(monkeypatch, tmp_path):
    a, z = _toy(trajectory=True, seed=2, sep=0.3)       # overlapping types: one kNN component, a root exists
    pre, npz = _write(tmp_path, a, z, "traj")
    r = _score(monkeypatch, tmp_path, pre, npz, "traj")
    assert int(r["traj_root_fallback"]) == 0 and r["trajectory"] > 0
    traj = importlib.import_module("scib.metrics.trajectory")
    real = getattr(traj.get_root, "__wrapped__", traj.get_root)

    def no_root(*args, **kwargs):
        raise traj.RootCellError("no root cell (test)")

    monkeypatch.setattr(traj, "get_root", no_root)
    try:
        r = _score(monkeypatch, tmp_path, pre, npz, "traj_noroot")
    finally:
        traj.get_root = real
    assert int(r["traj_root_fallback"]) == 1 and r["trajectory"] == 0.0    # scib's silent fallback, now flagged


def test_trajectory_root_flag_natural_fallback(monkeypatch, tmp_path):
    """Separated cell types: the start type c0 is not in the largest kNN component, scib 1.1.7 raises RootCellError
    and silently reports trajectory 0; the row now says so."""
    a, z = _toy(trajectory=True, seed=2)
    pre, npz = _write(tmp_path, a, z, "traj_sep")
    r = _score(monkeypatch, tmp_path, pre, npz, "traj_sep")
    assert int(r["traj_root_fallback"]) == 1 and r["trajectory"] == 0.0


def test_scorer_runs_as_a_script_under_safe_path(tmp_path, toy):
    """run_stage.py starts the scorer as a script; with PYTHONSAFEPATH=1 (set in this sandbox) the script directory is
    not on sys.path, so the scorer must add it itself (it imports scripts/host_info.py). Found by the stage e2e test."""
    a, z = toy
    pre, npz = _write(tmp_path, a, z, "script")
    out = tmp_path / "script.csv"
    env = dict(os.environ, PREPPED=pre, NPZ=npz, TAG="script", OUT_CSV=str(out), META="{}", PYTHONSAFEPATH="1",
               KMP_AFFINITY="disabled", OMP_NUM_THREADS="1")
    env.pop("KBET_SEED", None)
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "score_scib_native.py")], env=env,
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert p.returncode == 0, p.stderr[-2000:]
    r = pd.read_csv(out).iloc[0]
    assert r["cpu_model"] and r["scorer_git_sha"] and int(r["kbet_labels_skipped"]) == 2
