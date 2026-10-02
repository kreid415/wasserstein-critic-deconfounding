"""Runner (scripts/fit_paper_config.py) and manifest (scripts/build_paper_manifest.py) checks for the missing arms
(docs/SPECS_missing_arms.md). Run in the scvi env: WCD_SRC=src python -m pytest -q <this file>

1. `extra` keys: unknown keys and inconsistent options are refused before any data is read.
2. X3 oracle weights: w = 1 / keep-fraction for kept depleted-type cells of the depleted batch, 1 elsewhere;
   their sum restores the depleted count exactly; undefined (refused) at doses 0 and 100.
3. subsample(): the cells selected for every X3 and X8 design cell are unchanged from the base commit
   (real prepped obs; needs PREPPED_DIR, otherwise skipped with that reason).
4. End to end through main(): one row per new option (r1_gamma, discriminator_ref, sampler, iw) on a toy prepped
   file writes a finite latent whose config records the row and the keep fraction.
5. Manifest: base rows unchanged; added rows exactly X3 288 (IW), X6 27 (R1), X7 54 (reference JS),
   X8 144 (stratified); no duplicate tags; tags (= output files) unique.
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fit_paper_config as fpc  # noqa: E402
import build_paper_manifest as bpm  # noqa: E402

BASE = "8cda6de"
ad = pytest.importorskip("anndata")


def _base_module(path, name):
    src = subprocess.run(["git", "-C", ROOT, "show", f"{BASE}:{path}"], check=True, capture_output=True, text=True).stdout
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = os.path.join(ROOT, path)      # the base module resolves its own directory
    exec(compile(src, f"{BASE}:{path}", "exec"), mod.__dict__)
    return mod


def _obs_adata(sizes):
    """obs-only AnnData: sizes[(batch, celltype)] = count."""
    rows = [(b, c) for (b, c), n in sizes.items() for _ in range(n)]
    obs = pd.DataFrame(rows, columns=["batch", "celltype"], index=[f"c{i}" for i in range(len(rows))])
    return ad.AnnData(obs=obs)


def test_extra_keys_are_consumed_or_refused():
    fpc.check_extra("t", "pooled", {"subsample": {"kind": "batches"}, "sampler": "stratified"})
    fpc.check_extra("t", "discriminator_r1", {"r1_gamma": 10})
    fpc.check_extra("t", "pooled", {"subsample": {"kind": "composition"}, "iw": "depletion_oracle"})
    bad = [("pooled", {"lambda_typo": 1}, KeyError), ("discriminator_r1", {}, ValueError),
           ("discriminator", {"r1_gamma": 10}, ValueError), ("pooled", {"sampler": "balanced"}, ValueError),
           ("pooled", {"iw": "estimated"}, ValueError), ("pooled", {"iw": "depletion_oracle"}, ValueError)]
    for arm, extra, exc in bad:
        with pytest.raises(exc):
            fpc.check_extra("t", arm, extra)


def test_depletion_oracle_weights_known_answers():
    a0 = _obs_adata({("b0", "t0"): 100, ("b0", "t1"): 60, ("b1", "t0"): 80, ("b1", "t1"): 40, ("b2", "t1"): 50})
    for pct, kappa in [(50, 0.5), (80, 0.2), (95, 0.05)]:
        spec = dict(kind="composition", batch="b0", types=["t0"], deplete_pct=pct, n_cells=10_000)
        a, info = fpc.subsample(a0, spec, 0, return_info=True)
        w, k = fpc.depletion_oracle_weights(a, spec, info)
        assert info == dict(n_hit=100, n_drop=int(round(100 * pct / 100))) and abs(k - kappa) < 1e-12
        n_kept = int(((a.obs.batch == "b0") & (a.obs.celltype == "t0")).sum())
        assert abs(w["b0"]["t0"] * n_kept - 100) < 1e-9                  # depleted count restored exactly
        assert w["b0"]["t1"] == w["b1"]["t0"] == w["b1"]["t1"] == w["b2"]["t1"] == 1.0
    for pct in (0, 100):
        spec = dict(kind="composition", batch="b0", types=["t0"], deplete_pct=pct, n_cells=10_000)
        a, info = fpc.subsample(a0, spec, 0, return_info=True)
        with pytest.raises(ValueError, match="keep fraction"):
            fpc.depletion_oracle_weights(a, spec, info)


@pytest.mark.skipif(not os.environ.get("PREPPED_DIR"), reason="needs PREPPED_DIR (prepped scIB h5ad files)")
def test_subsample_cells_unchanged_from_base():
    import h5py
    old = _base_module("scripts/fit_paper_config.py", "fpc_base")
    n_checked = 0
    for task, (b, types) in bpm.X3_TARGETS.items():
        with h5py.File(os.path.join(os.environ["PREPPED_DIR"], f"{task}__scib.h5ad"), "r") as f:
            obs = ad.io.read_elem(f["obs"])
        a0 = ad.AnnData(obs=obs[["batch", "celltype"]].copy())
        for pct in [0, 50, 80, 95, 100]:
            spec = dict(kind="composition", batch=b, types=types, deplete_pct=pct, n_cells=min(20000, bpm.TASKS[task][0]))
            assert list(old.subsample(a0, spec, 0).obs_names) == list(fpc.subsample(a0, spec, 0).obs_names)
            n_checked += 1
    for task, n in bpm.X8_TOTAL.items():
        with h5py.File(os.path.join(os.environ["PREPPED_DIR"], f"{task}__scib.h5ad"), "r") as f:
            obs = ad.io.read_elem(f["obs"])
        a0 = ad.AnnData(obs=obs[["batch", "celltype"]].copy())
        for V in [2, 4, 8, 16]:
            for sub in [0, 1]:
                spec = dict(kind="batches", n_batches=V, subset=sub, n_cells=n)
                assert list(old.subsample(a0, spec, 0).obs_names) == list(fpc.subsample(a0, spec, 0).obs_names)
                n_checked += 1
    assert n_checked == 4 * 5 + 3 * 4 * 2


def _toy_prepped(path, n=600, g=60, k=3, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.integers(0, k, n)
    ct = rng.integers(0, 3, n)
    mu = np.exp(rng.normal(0, 1, (3, g)))[ct] * np.exp(rng.normal(0, 0.5, (k, g)))[b]
    X = rng.poisson(mu).astype(np.float32)
    a = ad.AnnData(np.log1p(X))
    a.layers["counts"] = X
    a.obs["batch"] = pd.Categorical([f"b{i}" for i in b])
    a.obs["celltype"] = pd.Categorical([f"t{i}" for i in ct])
    a.var["highly_variable"] = True
    a.write_h5ad(path)


def test_runner_end_to_end_new_options(tmp_path):
    _toy_prepped(tmp_path / "toy__scib.h5ad")
    base = dict(experiment="T", task="toy", counts="scib", lam="1.0", adv_input="mean", zstd="0", cond="1",
                decoder="SCVI", n_latent="4", n_layers="1", n_hidden="128", likelihood="zinb", batch_size="128",
                max_epochs="1", train_size="0.9", seed="0", reference="auto")
    comp = dict(kind="composition", batch="b1", types=["t0"], deplete_pct=50, n_cells=600)
    rows = [dict(base, tag="r1", arm="discriminator_r1", n_critic="1", extra=json.dumps({"r1_gamma": 10})),
            dict(base, tag="refjs", arm="discriminator_ref", n_critic="1", reference="0", extra="{}"),
            dict(base, tag="strat", arm="pooled", n_critic="5", extra=json.dumps({"sampler": "stratified"})),
            dict(base, tag="iw", arm="pooled", n_critic="5", extra=json.dumps({"subsample": comp, "iw": "depletion_oracle"}))]
    man = tmp_path / "m.tsv"
    pd.DataFrame(rows)[fpc.REQUIRED].to_csv(man, sep="\t", index=False)
    env = dict(os.environ, MANIFEST=str(man), PREPPED_DIR=str(tmp_path), OUT_DIR=str(tmp_path / "out"),
               WCD_SRC=os.path.join(ROOT, "src"), CUDA_VISIBLE_DEVICES="", KMP_AFFINITY="disabled", OMP_NUM_THREADS="1")
    for r in rows:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fit_paper_config.py")], check=True,
                       env=dict(env, TAG=r["tag"]))
        d = np.load(tmp_path / "out" / "latents" / f"{r['tag']}.npz", allow_pickle=False)
        cfg = json.loads(str(d["config"]))
        assert np.isfinite(d["z"]).all() and d["z"].shape[1] == 4 and cfg["row"]["arm"] == r["arm"]
        assert json.loads(cfg["row"]["extra"]) == json.loads(r["extra"])
        if r["tag"] == "iw":
            assert abs(cfg["iw_keep_fraction"] - 0.5) < 0.01, cfg["iw_keep_fraction"]
        else:
            assert cfg["iw_keep_fraction"] is None


def test_manifest_adds_exactly_the_signed_off_rows(tmp_path):
    old_src = tmp_path / "old_bpm.py"
    old_src.write_text(subprocess.run(["git", "-C", ROOT, "show", f"{BASE}:scripts/build_paper_manifest.py"],
                                      check=True, capture_output=True, text=True).stdout)
    args = ["--backbone", "stock", "--design", "pilot", "--uncond-seeds", "5", "--bary-iter", "10"]
    subprocess.run([sys.executable, str(old_src), *args, "--out", str(tmp_path / "old.tsv")], check=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_paper_manifest.py"), *args,
                    "--out", str(tmp_path / "new.tsv")], check=True)
    rd = lambda p: pd.read_csv(p, sep="\t", comment="#", dtype=str, keep_default_na=False)  # noqa: E731
    o, n = rd(tmp_path / "old.tsv"), rd(tmp_path / "new.tsv")
    assert not n.tag.duplicated().any()
    m = o.merge(n, on="tag", how="left", suffixes=("", "_new"), indicator=True)
    assert (m["_merge"] == "both").all()
    for c in o.columns:
        if c != "tag":
            assert (m[c] == m[c + "_new"]).all(), c
    add = n[~n.tag.isin(o.tag)].copy()
    add["opt"] = add.extra.map(lambda s: ",".join(sorted(set(json.loads(s)) - {"subsample"})))
    got = add.groupby(["experiment", "arm", "opt"]).size().to_dict()
    exp = {("X3", a, "iw"): 72 for a in ["discriminator", "reference", "pooled", "mmd"]}
    exp.update({("X6", "discriminator_r1", "r1_gamma"): 27, ("X7", "discriminator_ref", ""): 54,
                ("X8", "reference", "sampler"): 72, ("X8", "pooled", "sampler"): 72})
    assert got == exp, got
    assert set(add[add.experiment == "X3"].extra.map(lambda s: json.loads(s)["subsample"]["deplete_pct"])) == {50, 80, 95}
    assert (add[add.arm.isin(["discriminator_r1", "discriminator_ref"])].n_critic == "1").all()
