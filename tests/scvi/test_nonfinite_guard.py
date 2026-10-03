"""Non-finite-loss guard (docs/PREREG.md section 1: 'diverged' = a training loss term became non-finite).
Run in the scvi env, CPU only: CUDA_VISIBLE_DEVICES= WCD_SRC=src python -m pytest -q tests/scvi/test_nonfinite_guard.py

1. A loss term that overflows (lambda = 1e39 is inf in float32, so gen_loss = loss_vae + lambda * adv_term = inf)
   raises NonFiniteLossError at epoch 0, step 0, in every branch of training_step (JS, critic, barycenter,
   critic-free), naming gen_loss as the non-finite term while loss_vae is finite.
2. A NaN decoder weight makes the scVI loss NaN: raised at the injected step for 'none' and 'scvi_adv' (the
   branches that run scvi-tools' own training_step).
3. A NaN encoder weight makes torch.distributions refuse q_m in the next forward pass: raised as NonFiniteLossError
   from the original ValueError, at the injected step, listing the poisoned parameter.
4. Parameters made non-finite by the last optimizer step are caught at the end of training.
5. Any other error in the forward pass propagates unchanged (it is an infrastructure / code error, not a divergence).
6. scripts/fit_paper_config.py end to end: a diverging row writes OUT_DIR/status/<tag>.json (status diverged, epoch,
   step, the manifest row) and exits with EXIT_DIVERGED, writing no latent; a finite row writes its latent and no
   status; a row that fails for another reason exits 1 and writes nothing; a row with a status file is refused.
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
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WCD_SRC", os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scvi_adversarial_plan as plan  # noqa: E402
import fit_outcome  # noqa: E402
import fit_paper_config as fpc  # noqa: E402

PLAN = plan.WassersteinAdversarialTrainingPlan


def _toy(n=600, g=60, k=3, seed=0):
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
    return a


def _fit(arm, d_coef=1.0, cond=True, max_epochs=1, **kw):
    base = arm[:-3] if arm.endswith("_sn") else arm
    n_critic = 5 if base in ("reference", "reference_fixed", "pooled", "barycenter") else None
    return plan.fit_adversarial_scvi(_toy(), "batch", adversary=arm, d_coef=d_coef, n_critic=n_critic,
                                     reference_batch="b0", adv_input="mean", zstd=False, n_latent=4,
                                     max_epochs=max_epochs, batch_size=128, seed=0, conditioned=cond, **kw)


def _poison_at(monkeypatch, step, pick, after=False):
    """Fill the parameter chosen by pick(plan) with NaN before (or after) training step `step`."""
    orig = PLAN.training_step

    def wrapped(self, batch, batch_idx):
        if not after and self._nf_step + 1 == step:
            with torch.no_grad():
                pick(self).fill_(float("nan"))
        out = orig(self, batch, batch_idx)
        if after and step == "last" and self.trainer.is_last_batch and self.current_epoch == self.trainer.max_epochs - 1:
            with torch.no_grad():
                pick(self).fill_(float("nan"))
        return out

    monkeypatch.setattr(PLAN, "training_step", wrapped)


def _first_weight(module):
    return next(p for n, p in module.named_parameters() if n.endswith("weight") and p.dim() == 2)


@pytest.mark.parametrize("arm", ["discriminator", "pooled", "barycenter", "mmd"])
def test_overflowing_loss_raises_at_first_step(arm):
    with pytest.raises(fit_outcome.NonFiniteLossError) as ei:
        _fit(arm, d_coef=1e39)
    e = ei.value
    assert (e.epoch, e.step) == (0, 0), (e.epoch, e.step)
    assert e.terms["gen_loss"] == float("inf") and np.isfinite(e.terms["loss_vae"]), e.terms
    assert "gen_loss=inf" in str(e)


@pytest.mark.parametrize("arm", ["none", "scvi_adv"])
def test_nan_decoder_weight_raises_scvi_loss(arm, monkeypatch):
    _poison_at(monkeypatch, 3, lambda p: _first_weight(p.module.decoder))
    with pytest.raises(fit_outcome.NonFiniteLossError) as ei:
        _fit(arm, max_epochs=2)
    e = ei.value
    assert e.step == 3 and e.epoch == 0, (e.epoch, e.step)
    assert np.isnan(e.terms["train_loss"]), e.terms
    if arm == "scvi_adv":
        assert {"classifier_loss", "fool_loss"} <= set(e.terms), e.terms


def test_nan_encoder_weight_raises_from_forward(monkeypatch):
    _poison_at(monkeypatch, 2, lambda p: _first_weight(p.module.z_encoder))
    with pytest.raises(fit_outcome.NonFiniteLossError) as ei:
        _fit("discriminator", max_epochs=2)
    e = ei.value
    assert e.step == 2 and e.terms == {}, (e.step, e.terms)
    assert isinstance(e.__cause__, ValueError) and str(e.__cause__).startswith("Expected parameter loc")
    assert "forward pass refused" in e.detail and "z_encoder" in e.detail, e.detail


def test_nonfinite_parameters_after_last_step_raise(monkeypatch):
    _poison_at(monkeypatch, "last", lambda p: _first_weight(p.module.decoder), after=True)
    with pytest.raises(fit_outcome.NonFiniteLossError) as ei:
        _fit("pooled", max_epochs=1)
    assert "after the last optimizer step" in ei.value.detail and "decoder" in ei.value.detail


def test_other_forward_errors_propagate_unchanged(monkeypatch):
    orig = plan.AdversarialTrainingPlan.forward

    def boom(self, *a, **k):
        if self._nf_step == 1:
            raise ValueError("Expected parameter: not raised by torch.distributions")
        return orig(self, *a, **k)

    monkeypatch.setattr(plan.AdversarialTrainingPlan, "forward", boom)
    with pytest.raises(ValueError, match="not raised by torch.distributions") as ei:
        _fit("discriminator")
    assert not isinstance(ei.value, fit_outcome.NonFiniteLossError)


def test_distribution_validation_error_is_recognised():
    with pytest.raises(ValueError) as real:
        torch.distributions.Normal(torch.tensor([float("nan")]), torch.tensor([1.0]))
    assert plan._is_distribution_validation_error(real.value)
    try:
        raise ValueError(str(real.value))       # same text, raised elsewhere
    except ValueError as fake:
        assert not plan._is_distribution_validation_error(fake)


def test_finite_fit_is_unaffected():
    z, _ = _fit("discriminator")
    assert np.isfinite(z).all() and z.shape == (600, 4)


# ---- 6. fit_paper_config.py end to end ------------------------------------------------------------------------
def _row(tag, **over):
    r = dict(tag=tag, experiment="T", task="toy", counts="scib", arm="discriminator", lam="1.0", n_critic="1",
             adv_input="mean", zstd="0", cond="1", decoder="SCVI", n_latent="4", n_layers="1", n_hidden="128",
             likelihood="zinb", batch_size="128", max_epochs="1", train_size="0.9", seed="0", reference="auto",
             extra="{}")
    r.update(over)
    return r


def run_fitter_rows(code_root, tmp_path):
    """Fit the three probe rows with the fitter at code_root; returns {tag: (returncode, status record or None,
    latent exists)}. Used by the test below and by the mutation check against the base commit."""
    _toy().write_h5ad(tmp_path / "toy__scib.h5ad")
    rows = [_row("div", lam="1e39"), _row("ok"), _row("err", task="absent")]
    man = tmp_path / "m.tsv"
    pd.DataFrame(rows)[fpc.REQUIRED].to_csv(man, sep="\t", index=False)
    out = tmp_path / "out"
    env = dict(os.environ, MANIFEST=str(man), PREPPED_DIR=str(tmp_path), OUT_DIR=str(out),
               WCD_SRC=os.path.join(code_root, "src"), CUDA_VISIBLE_DEVICES="", KMP_AFFINITY="disabled",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    res = {}
    for r in rows:
        p = subprocess.run([sys.executable, os.path.join(code_root, "scripts", "fit_paper_config.py")],
                           env=dict(env, TAG=r["tag"]), capture_output=True, text=True)
        st = out / "status" / f"{r['tag']}.json"
        res[r["tag"]] = (p.returncode, json.loads(st.read_text()) if st.exists() else None,
                         (out / "latents" / f"{r['tag']}.npz").exists(), p.stderr[-2000:])
    return res, rows, env, out


def check_fitter_contract(res, rows):
    """The fitter contract of docs/PREREG.md section 1 (raises AssertionError on a violation)."""
    rc, st, lat, err = res["div"]
    assert rc == fit_outcome.EXIT_DIVERGED, f"diverging row exit {rc}: {err}"
    assert st is not None and st["status"] == "diverged" and not lat
    assert (st["epoch"], st["step"]) == (0, 0) and st["terms"]["gen_loss"] == "inf", st
    assert st["row"] == rows[0] and st["tag"] == "div" and st["detail"]
    rc, st, lat, err = res["ok"]
    assert rc == 0 and st is None and lat, f"finite row: exit {rc}, status {st}, latent {lat}: {err}"
    rc, st, lat, err = res["err"]
    assert rc == 1 and st is None and not lat, f"failing row: exit {rc}, status {st}, latent {lat}"


def test_fitter_records_divergence_and_nothing_else(tmp_path):
    res, rows, env, out = run_fitter_rows(ROOT, tmp_path)
    check_fitter_contract(res, rows)
    before = (out / "status" / "div.json").read_bytes()
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fit_paper_config.py")],
                       env=dict(env, TAG="div"), capture_output=True, text=True)
    assert p.returncode == 1 and "already has a recorded outcome" in p.stderr
    assert (out / "status" / "div.json").read_bytes() == before
    rec = fit_outcome.read_status(str(out / "status" / "div.json"), tag="div")
    assert rec["git_sha"] and "gpu" in rec
