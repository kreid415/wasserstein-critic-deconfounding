"""Runner (scripts/fit_paper_config.py) and manifest (scripts/build_paper_manifest.py) checks for the missing arms
(docs/SPECS_missing_arms.md). Run in the scvi env: WCD_SRC=src python -m pytest -q <this file>

1. `extra` keys: unknown keys and inconsistent options are refused before any data is read.
2. X3 oracle weights and the X3 draw: superseded by SI-41 (equal cell count per dose, draw 'nested_v1', exact
   (batch, cell type) oracle weights); tested in tests/scvi/test_x3_si41.py.
3. subsample(): the cells selected for every X8 design cell equal the base commit's function (real prepped obs; needs
   PREPPED_DIR, otherwise skipped with that reason). X3 cells changed by design (SI-41; test_x3_si41.py).
4. End to end through main(): one row per new option (r1_gamma, discriminator_ref, sampler, iw) on a toy prepped
   file writes a finite latent whose config records the row and the keep fraction.
5. Manifest: base rows unchanged except the signed-off changes (SI-26 follow-up seeds, SI-27 unconditioned A2/A3,
   SI-31..SI-33 X3 reference and targets, SI-41 X3 n_cells = X3_N and draw 'nested_v1'); added rows exactly X3 288 (IW), X6 27 (R1), X7 54 (reference JS),
   X8 144 (stratified), X13 128 CPU baselines (SI-34..SI-36); no duplicate tags; tags (= output files) unique.
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

BASE = "8cda6de"            # code before the missing arms: existing-arm bit identity is checked against it
MANIFEST_BASE = "2e4f24d"   # prereg-rules tip: the manifest the missing-arm rows are added to (merged 2026-10-03)
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


@pytest.mark.skipif(not os.environ.get("PREPPED_DIR"), reason="needs PREPPED_DIR (prepped scIB h5ad files)")
def test_subsample_cells_unchanged_from_base():
    import h5py
    old = _base_module("scripts/fit_paper_config.py", "fpc_base")
    n_checked = 0
    for task, n in bpm.X8_TOTAL.items():
        with h5py.File(os.path.join(os.environ["PREPPED_DIR"], f"{task}__scib.h5ad"), "r") as f:
            obs = ad.io.read_elem(f["obs"])
        a0 = ad.AnnData(obs=obs[["batch", "celltype"]].copy())
        for V in [2, 4, 8, 16]:
            for sub in [0, 1]:
                spec = dict(kind="batches", n_batches=V, subset=sub, n_cells=n)
                assert list(old.subsample(a0, spec, 0).obs_names) == list(fpc.subsample(a0, spec, 0).obs_names)
                n_checked += 1
    assert n_checked == 3 * 4 * 2


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
    obs = ad.read_h5ad(tmp_path / "toy__scib.h5ad").obs.astype(str)
    n100 = int(((obs.batch != "b1") | (obs.celltype != "t0")).sum())       # SI-41: the dose-100 size
    comp = dict(kind="composition", batch="b1", types=["t0"], deplete_pct=50, n_cells=n100, draw="nested_v1")
    rows = [dict(base, tag="r1", arm="discriminator_r1", n_critic="1", extra=json.dumps({"r1_gamma": 10})),
            dict(base, tag="refjs", arm="discriminator_ref", n_critic="1", reference="0", extra="{}"),
            dict(base, tag="strat", arm="pooled", n_critic="5", extra=json.dumps({"sampler": "stratified"})),
            dict(base, tag="iw", arm="pooled", n_critic="5", reference="b0",      # X3 rows name a non-depleted reference (SI-31)
                 extra=json.dumps({"subsample": comp, "iw": "depletion_oracle"}))]
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
            assert set(cfg["iw_keep_fraction"]) == {"t0"} and abs(cfg["iw_keep_fraction"]["t0"] - 0.5) < 0.02
            assert cfg["subsample_info"]["draw"] == "nested_v1" and len(d["z"]) == n100
            kd = pd.Series(list(zip(d["batch"], d["celltype"]))).value_counts()
            k0 = cfg["subsample_info"]["k0_counts"]
            assert all(abs(cfg["iw_weights"][b][c] * n - k0.get(b, {}).get(c, 0)) < 1e-9 for (b, c), n in kd.items())
        else:
            assert cfg["iw_keep_fraction"] is None and cfg["iw_weights"] is None and cfg["subsample_info"] is None


def test_manifest_adds_exactly_the_signed_off_rows(tmp_path):
    old_src = tmp_path / "old_bpm.py"
    old_src.write_text(subprocess.run(["git", "-C", ROOT, "show", f"{MANIFEST_BASE}:scripts/build_paper_manifest.py"],
                                      check=True, capture_output=True, text=True).stdout)
    args = ["--backbone", "stock", "--design", "pilot", "--uncond-seeds", "5", "--bary-iter", "10"]
    subprocess.run([sys.executable, str(old_src), *args, "--out", str(tmp_path / "old.tsv")], check=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_paper_manifest.py"), *args,
                    "--out", str(tmp_path / "new.tsv")], check=True)
    rd = lambda p: pd.read_csv(p, sep="\t", comment="#", dtype=str, keep_default_na=False)  # noqa: E731
    o, n = rd(tmp_path / "old.tsv"), rd(tmp_path / "new.tsv")
    assert not n.tag.duplicated().any()
    # Signed-off changes since MANIFEST_BASE (CONSTRAINTS.md): SI-26 moves the follow-ups X3/X6/X7/X8/X12 from
    # seeds 0-2 to bpm.FOLLOWUP_SEEDS (field seed and hence tag change, nothing else); SI-27 adds A2/A3 on the
    # unconditioned decoder, one row per conditioned row. Every other base row must be unchanged, tag included.
    key = [c for c in o.columns if c != "tag"]
    fu = o.experiment.isin(["X3", "X6", "X7", "X8", "X12"])
    assert set(o.seed[fu]) == {"0", "1", "2"}
    o2 = o.copy()
    o2.loc[fu, "seed"] = o2.seed[fu].map(lambda s: str(bpm.FOLLOWUP_SEEDS[int(s)]))
    # SI-31..SI-33 (2026-10-03): every X3 row names the fixed reference; atac_small and sim2 changed target
    x3 = (o2.experiment == "X3").to_numpy()
    def _x3_new(r):
        b, types, ref = bpm.X3_TARGETS[r.task]
        ex = json.loads(r.extra)
        ex["subsample"]["batch"], ex["subsample"]["types"] = b, types
        ex["subsample"]["n_cells"], ex["subsample"]["draw"] = bpm.X3_N[r.task], bpm.X3_DRAW   # SI-41
        return pd.Series({"reference": ref, "extra": json.dumps(ex)})
    o2.loc[x3, ["reference", "extra"]] = o2[x3].apply(_x3_new, axis=1).to_numpy()
    m = o2.merge(n, on=key, how="left", suffixes=("", "_new"), indicator=True, validate="one_to_one")
    assert (m["_merge"] == "both").all()
    assert (m.tag == m.tag_new)[~fu.to_numpy()].all() and (m.tag != m.tag_new)[fu.to_numpy()].all()
    add = n.merge(o2[key], on=key, how="left", indicator=True)
    add = add[add["_merge"] == "left_only"].drop(columns="_merge")
    a23 = o[o.experiment.isin(["A2", "A3"])].assign(cond="0")
    assert len(a23) == 144 and set(o.cond[o.experiment.isin(["A2", "A3"])]) == {"1"}
    mirror = add[add.experiment.isin(["A2", "A3"])]
    assert sorted(map(tuple, mirror[key].to_numpy())) == sorted(map(tuple, a23[key].to_numpy()))
    add = add[~add.experiment.isin(["A2", "A3"])].copy()
    # SI-34..SI-36: X13 CPU baselines, 16 rows per task (harmony 6 + 1, scanorama 6 + 1, pca 2), seed 0
    cpu = add[add.arm.isin(list(bpm.CPU_BASELINES))]
    assert len(cpu) == 16 * len(bpm.TASKS) and set(cpu.experiment) == {"X13"} and set(cpu.seed) == {"0"}
    assert cpu.groupby("arm").size().to_dict() == {"harmony": 7 * 8, "scanorama": 7 * 8, "pca": 2 * 8}
    add = add[~add.arm.isin(list(bpm.CPU_BASELINES))].copy()
    assert set(add.seed) == {str(s) for s in bpm.FOLLOWUP_SEEDS}
    add["opt"] = add.extra.map(lambda s: ",".join(sorted(set(json.loads(s)) - {"subsample"})))
    got = add.groupby(["experiment", "arm", "opt"]).size().to_dict()
    exp = {("X3", a, "iw"): 72 for a in ["discriminator", "reference", "pooled", "mmd"]}
    exp.update({("X6", "discriminator_r1", "r1_gamma"): 27, ("X7", "discriminator_ref", ""): 54,
                ("X8", "reference", "sampler"): 72, ("X8", "pooled", "sampler"): 72})
    assert got == exp, got
    assert set(add[add.experiment == "X3"].extra.map(lambda s: json.loads(s)["subsample"]["deplete_pct"])) == {50, 80, 95}
    assert (add[add.arm.isin(["discriminator_r1", "discriminator_ref"])].n_critic == "1").all()
