"""X15 KL warm-up setting (CONSTRAINTS.md SI-44): extra {"kl_warmup": "stock" | "complete"} in fit_paper_config.py.
Run in the scvi env, CPU only: CUDA_VISIBLE_DEVICES= WCD_SRC=src python -m pytest -q tests/scvi/test_kl_warmup.py

1. kl_warmup_epochs: 'complete' -> the row's max_epochs; 'stock' and no key -> None (the training plan keeps
   scvi-tools' default); other values are refused, and scANVI / sysVI rows with the key are refused.
2. Byte identity (end to end through fit_paper_config.py on a toy prepped file, arms none and discriminator): the
   latent of a row without the key equals the previous commit's fitter (eb444fe) exactly, and a 'stock' row equals
   it too; the resolved plan records n_epochs_kl_warmup 400 for both.
3. 'complete' is live: the resolved plan records n_epochs_kl_warmup = max_epochs (3) and the latent differs.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

scvi = pytest.importorskip("scvi")
import anndata as ad  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fit_paper_config as fpc  # noqa: E402

PREVIOUS = "eb444fe"      # integrate-fixes tip: the fitter before the X15 setting
EPOCHS = 3


def test_kl_warmup_epochs():
    assert fpc.kl_warmup_epochs({}, 239) is None
    assert fpc.kl_warmup_epochs({"kl_warmup": "stock"}, 239) is None
    assert fpc.kl_warmup_epochs({"kl_warmup": "complete"}, 239) == 239
    assert fpc.kl_warmup_epochs({"kl_warmup": "complete"}, "94") == 94
    for bad in ("Complete", "full", 400, None):
        with pytest.raises(ValueError):
            fpc.kl_warmup_epochs({"kl_warmup": bad}, 239)
    with pytest.raises(ValueError):
        fpc.check_extra("t", "none", {"kl_warmup": "400"})
    fpc.check_extra("t", "none", {"kl_warmup": "complete"})


def _toy(n=600, g=60, k=3, seed=0):
    rng = np.random.default_rng(seed)
    b, ct = rng.integers(0, k, n), rng.integers(0, 3, n)
    mu = np.exp(rng.normal(0, 1, (3, g)))[ct] * np.exp(rng.normal(0, 0.5, (k, g)))[b]
    X = rng.poisson(mu).astype(np.float32)
    a = ad.AnnData(np.log1p(X))
    a.layers["counts"] = X
    a.obs["batch"] = pd.Categorical([f"b{i}" for i in b])
    a.obs["celltype"] = pd.Categorical([f"t{i}" for i in ct])
    a.var["highly_variable"] = True
    a.obs_names = [f"c{i}" for i in range(n)]
    return a


def _row(arm, extra):
    lam, nc = (0, 0) if arm == "none" else (1, 1)
    tag = f"X15_toy_{arm}_{(extra.get('kl_warmup') or 'nokey')}"
    return dict(tag=tag, experiment="X15", task="toy", counts="scib", arm=arm, lam=lam, n_critic=nc, adv_input="mean",
                zstd=0, cond=1, decoder="SCVI", n_latent=4, n_layers=1, n_hidden=128, likelihood="zinb", batch_size=128,
                max_epochs=EPOCHS, train_size=0.9, seed=10, reference="auto", extra=json.dumps(extra))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("klw")
    prep = tmp / "prepped"
    prep.mkdir()
    _toy().write_h5ad(prep / "toy__scib.h5ad")
    rows = [_row(a, e) for a in ("none", "discriminator") for e in ({}, {"kl_warmup": "stock"}, {"kl_warmup": "complete"})]
    man = tmp / "m.tsv"
    pd.DataFrame(rows)[fpc.REQUIRED].to_csv(man, sep="\t", index=False)
    return tmp, prep, man, rows


def _env(prep, man, tag, out, src):
    return dict(os.environ, MANIFEST=str(man), TAG=tag, PREPPED_DIR=str(prep), OUT_DIR=str(out), WCD_SRC=src,
                KMP_AFFINITY="disabled", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")


def _fit(scripts, src, prep, man, tag, out):
    r = subprocess.run([sys.executable, os.path.join(scripts, "fit_paper_config.py")],
                       env=_env(prep, man, tag, out, src), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    f = np.load(os.path.join(out, "latents", f"{tag}.npz"))
    return f["z"], json.loads(str(f["config"]))


@pytest.fixture(scope="module")
def fits(setup):
    tmp, prep, man, rows = setup
    new = {r["tag"]: _fit(os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src"), prep, man, r["tag"], tmp / "new")
           for r in rows}
    old_root = tmp / "old"                     # the previous commit's tree, as a git repository (provenance reads its SHA)
    old_root.mkdir()
    tar = subprocess.run(["git", "-C", ROOT, "archive", PREVIOUS, "scripts", "src"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(old_root)], input=tar, check=True)
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", PREVIOUS]):
        subprocess.run(["git", "-C", str(old_root), *cmd], check=True, capture_output=True)
    old = {r["tag"]: _fit(str(old_root / "scripts"), str(old_root / "src"), prep, man, r["tag"], tmp / "old_out")
           for r in rows if r["extra"] == "{}"}
    return new, old


@pytest.mark.parametrize("arm", ["none", "discriminator"])
def test_stock_and_no_key_are_byte_identical_to_the_previous_fitter(fits, arm):
    new, old = fits
    z_prev, _ = old[f"X15_toy_{arm}_nokey"]
    z_nokey, c_nokey = new[f"X15_toy_{arm}_nokey"]
    z_stock, c_stock = new[f"X15_toy_{arm}_stock"]
    assert z_prev.dtype == z_nokey.dtype == z_stock.dtype == np.float32
    assert z_nokey.tobytes() == z_prev.tobytes() and z_stock.tobytes() == z_prev.tobytes()
    assert c_nokey["plan"]["fit"]["n_epochs_kl_warmup"] == c_stock["plan"]["fit"]["n_epochs_kl_warmup"] == 400


@pytest.mark.parametrize("arm", ["none", "discriminator"])
def test_complete_sets_the_warmup_to_max_epochs_and_changes_the_fit(fits, arm):
    new, _ = fits
    z_complete, c = new[f"X15_toy_{arm}_complete"]
    z_stock, _ = new[f"X15_toy_{arm}_stock"]
    assert c["plan"]["fit"]["n_epochs_kl_warmup"] == EPOCHS == int(c["row"]["max_epochs"])
    assert np.isfinite(z_complete).all() and not np.array_equal(z_complete, z_stock)


def test_baseline_rows_refuse_the_key(setup):
    tmp, prep, man, rows = setup
    row = dict(_row("none", {"kl_warmup": "complete"}), tag="X13_toy_scanvi_complete", arm="scanvi", experiment="X13")
    m2 = tmp / "m_scanvi.tsv"
    pd.DataFrame([row])[fpc.REQUIRED].to_csv(m2, sep="\t", index=False)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fit_paper_config.py")],
                       env=_env(tmp / "absent", m2, row["tag"], tmp / "out_scanvi", os.path.join(ROOT, "src")),
                       capture_output=True, text=True)
    assert r.returncode != 0 and "kl_warmup applies to the scVI-backbone arms" in r.stderr
