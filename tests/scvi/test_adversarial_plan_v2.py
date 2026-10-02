"""End-to-end checks of scripts/scvi_adversarial_plan.py (2026-10-02 rewrite). Run in the scvi env:
    WCD_SRC=src python -m pytest -q tests/scvi/test_adversarial_plan_v2.py

1. adversary='none' reproduces stock scvi-tools SCVI bit-for-bit (same seed).
2. Every arm trains and returns a finite latent, conditioned and unconditioned.
3. The discriminator refuses n_critic != 1; critics refuse a missing n_critic; reference arms
   refuse a missing reference batch; adv_input must be explicit.
4. The adversary takes the documented number of optimizer steps per generator step.
5. reference_fixed: the generator step sends no adversarial gradient to reference cells.
"""
import os
import sys

import numpy as np
import pytest

scvi = pytest.importorskip("scvi")
import anndata as ad  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("WCD_SRC", os.path.join(ROOT, "src"))
import scvi_adversarial_plan as plan  # noqa: E402


def _toy(n=600, g=60, k=3, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.integers(0, k, n)
    ct = rng.integers(0, 3, n)
    mu = np.exp(rng.normal(0, 1, (3, g)))[ct] * np.exp(rng.normal(0, 0.5, (k, g)))[b]
    X = rng.poisson(mu).astype(np.float32)
    a = ad.AnnData(X)
    a.layers["counts"] = X.copy()
    a.obs["batch"] = [f"b{i}" for i in b]
    a.obs["batch"] = a.obs["batch"].astype("category")
    return a


COMMON = dict(n_latent=4, max_epochs=1, batch_size=128, seed=0, zstd=False)


def _kw(arm):
    n_critic = 5 if arm.split("_sn")[0] in plan._CRITIC_FORMULATIONS else None
    return dict(adversary=arm, d_coef=1.0, n_critic=n_critic, reference_batch="b0", adv_input="mean")


def test_none_is_bit_identical_to_stock_scvi():
    a = _toy()
    scvi.settings.seed = 0
    s = a.copy(); s.X = s.layers["counts"].copy()
    scvi.model.SCVI.setup_anndata(s, batch_key="batch")
    m = scvi.model.SCVI(s, n_latent=4)
    m.train(max_epochs=2, batch_size=128, early_stopping=False, enable_progress_bar=False)
    z_stock = m.get_latent_representation()
    z_plan, _ = plan.fit_adversarial_scvi(a, "batch", conditioned=True, **{**_kw("none"), **COMMON, "max_epochs": 2})
    assert np.array_equal(z_stock, z_plan), float(np.abs(z_stock - z_plan).max())


@pytest.mark.parametrize("arm", ["discriminator", "discriminator_sn", "reference", "reference_fixed", "pooled",
                                 "barycenter", "mmd", "mmd_ref", "sinkhorn", "pooled_sn", "scvi_adv"])
@pytest.mark.parametrize("conditioned", [True, False])
def test_every_arm_trains_finite(arm, conditioned):
    if arm == "scvi_adv" and not conditioned:
        pytest.skip("scvi-tools disables its classifier without batch conditioning")
    z, model = plan.fit_adversarial_scvi(_toy(), "batch", conditioned=conditioned, **_kw(arm), **COMMON)
    assert z.shape == (600, 4) and np.isfinite(z).all()


def test_argument_guards():
    a = _toy()
    with pytest.raises(ValueError, match="exactly one"):
        plan.fit_adversarial_scvi(a, "batch", conditioned=True, **{**_kw("discriminator"), "n_critic": 10}, **COMMON)
    with pytest.raises(ValueError, match="n_critic"):
        plan.fit_adversarial_scvi(a, "batch", conditioned=True, **{**_kw("pooled"), "n_critic": None}, **COMMON)
    with pytest.raises(ValueError, match="reference_batch"):
        plan.fit_adversarial_scvi(a, "batch", conditioned=True, **{**_kw("reference"), "reference_batch": None}, **COMMON)
    with pytest.raises(ValueError, match="adv_input"):
        plan.fit_adversarial_scvi(a, "batch", conditioned=True, **{**_kw("pooled"), "adv_input": None}, **COMMON)


@pytest.mark.parametrize("arm,expected", [("discriminator", 1), ("pooled", 5)])
def test_adversary_steps_per_generator_step(arm, expected, monkeypatch):
    calls = {"d": 0, "g": 0}
    orig = torch.optim.Adam.step

    def counting_step(self, *args, **kw):
        role = getattr(self, "_wcd_role", None)
        if role in calls:
            calls[role] += 1
        return orig(self, *args, **kw)

    orig_conf = plan.WassersteinAdversarialTrainingPlan.configure_optimizers

    def tagged(self):
        opts = orig_conf(self)
        opts[0]._wcd_role, opts[1]._wcd_role = "g", "d"
        return opts

    monkeypatch.setattr(torch.optim.Adam, "step", counting_step)
    monkeypatch.setattr(plan.WassersteinAdversarialTrainingPlan, "configure_optimizers", tagged)
    plan.fit_adversarial_scvi(_toy(), "batch", conditioned=True, **_kw(arm), **COMMON)
    assert calls["g"] > 0 and calls["d"] == expected * calls["g"], calls


def test_reference_fixed_sends_no_gradient_to_reference_cells():
    Discriminator, _c, _a, _b = plan._load_wcd_heads(os.environ["WCD_SRC"])
    head = Discriminator(n_input=4, domain_number=3, critic=True, reference_batch=0, formulation="reference")
    torch.manual_seed(0)
    z = torch.randn(90, 4, requires_grad=True)
    b = torch.arange(90) % 3
    z_g = torch.where((b == 0)[:, None], z.detach(), z)
    loss, _ = head(z_g, b, reference_batch=0, with_gp=False)
    (g,) = torch.autograd.grad(loss, z)
    assert torch.all(g[b == 0] == 0) and g[b != 0].abs().sum() > 0
