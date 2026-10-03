"""End-to-end checks of the X6 / X7 / X8 / X3 additions to scripts/scvi_adversarial_plan.py
(docs/SPECS_missing_arms.md, signed off 2026-10-02). Run in the scvi env:
    WCD_SRC=src python -m pytest -q tests/scvi/test_missing_arms_plan.py

1. discriminator_r1, discriminator_ref, the stratified sampler and importance-weighted arms train to finite
   latents (conditioned and, where defined, unconditioned).
2. Guards: r1_gamma required for *_r1 and refused elsewhere; one adversary step only; reference required;
   importance weights refused for unconditioned models, for arms outside X3, and for incomplete tables;
   unknown samplers refused.
3. discriminator_r1 and discriminator_ref take one adversary step per generator step.
4. The stratified sampler feeds the training step exactly 128 / V cells per batch, ceil(n_train / 128) times
   per epoch (batch in the batch slot or, unconditioned, in the labels slot).
5. Registering cell types in the labels slot changes no latent (stock SCVI, same seed).
6. Importance weights equal to 1 reproduce the unweighted fit (pooled and mmd bit for bit on CPU; the
   discriminator and reference critic to float tolerance, their weighted means use a different summation).
"""
import math
import os
import sys
import tempfile

import numpy as np
import pytest

scvi = pytest.importorskip("scvi")
import anndata as ad  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("WCD_SRC", os.path.join(ROOT, "src"))
import scvi_adversarial_plan as plan  # noqa: E402
import check_arm_bitidentity as bi  # noqa: E402


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
    a.obs["celltype"] = [f"t{i}" for i in ct]
    return a


COMMON = dict(n_latent=4, max_epochs=1, batch_size=128, seed=0, zstd=False)


def _iw(k=3, val=2.0):
    return {f"b{i}": {f"t{j}": (val if (i, j) == (1, 0) else 1.0) for j in range(3)} for i in range(k)}


def _kw(arm, **over):
    base = arm[:-3] if arm.endswith(("_sn", "_r1")) else arm
    n_critic = 5 if base in plan._CRITIC_FORMULATIONS else None
    kw = dict(adversary=arm, d_coef=1.0, n_critic=n_critic, reference_batch="b0", adv_input="mean")
    if arm.endswith("_r1"):
        kw["r1_gamma"] = 10.0
    kw.update(over)
    return kw


@pytest.mark.parametrize("arm,conditioned,opts", [
    ("discriminator_r1", True, {}), ("discriminator_r1", False, {}),
    ("discriminator_ref", True, {}), ("discriminator_ref", False, {}),
    ("reference", True, {"sampler": "stratified"}), ("pooled", True, {"sampler": "stratified"}),
    ("pooled", False, {"sampler": "stratified"}),
    ("discriminator", True, {"iw_weights": _iw()}), ("reference", True, {"iw_weights": _iw()}),
    ("pooled", True, {"iw_weights": _iw()}), ("mmd", True, {"iw_weights": _iw()}),
])
def test_new_arms_train_finite(arm, conditioned, opts):
    z, model = plan.fit_adversarial_scvi(_toy(), "batch", conditioned=conditioned, **_kw(arm, **opts), **COMMON)
    assert z.shape == (600, 4) and np.isfinite(z).all()


def test_new_arm_guards():
    a = _toy()
    cases = [
        (dict(_kw("discriminator_r1"), r1_gamma=None), "r1_gamma"),
        (dict(_kw("discriminator"), r1_gamma=10.0), "r1_gamma"),
        (dict(_kw("discriminator_ref"), n_critic=10), "exactly one"),
        (dict(_kw("discriminator_ref"), reference_batch=None), "reference_batch"),
        (dict(_kw("pooled_r1")), "_r1 is only valid"),
        (dict(_kw("discriminator_ref_sn")), "_sn is only valid"),
        (dict(_kw("sinkhorn"), iw_weights=_iw()), "importance weights are defined"),
        (dict(_kw("pooled"), sampler="balanced"), "sampler"),
    ]
    for kw, msg in cases:
        with pytest.raises(ValueError, match=msg):
            plan.fit_adversarial_scvi(a, "batch", conditioned=True, **kw, **COMMON)
    with pytest.raises(ValueError, match="conditioned"):
        plan.fit_adversarial_scvi(a, "batch", conditioned=False, **_kw("pooled", iw_weights=_iw()), **COMMON)
    incomplete = _iw()
    del incomplete["b2"]["t1"]
    with pytest.raises(ValueError, match="lacks"):
        plan.fit_adversarial_scvi(a, "batch", conditioned=True, **_kw("pooled", iw_weights=incomplete), **COMMON)


@pytest.mark.parametrize("arm", ["discriminator_r1", "discriminator_ref"])
def test_new_js_arms_take_one_adversary_step(arm, monkeypatch):
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
    assert calls["g"] > 0 and calls["d"] == calls["g"], calls


@pytest.mark.parametrize("conditioned", [True, False])
def test_stratified_sampler_feeds_equal_counts(conditioned, monkeypatch):
    k, n = 4, 800
    seen = []
    orig = plan.WassersteinAdversarialTrainingPlan.training_step
    slot = "batch" if conditioned else "labels"

    def recording(self, batch, batch_idx):
        seen.append(torch.bincount(batch[slot].long().squeeze(-1).cpu(), minlength=k).tolist())
        return orig(self, batch, batch_idx)

    monkeypatch.setattr(plan.WassersteinAdversarialTrainingPlan, "training_step", recording)
    plan.fit_adversarial_scvi(_toy(n=n, k=k), "batch", conditioned=conditioned,
                              **_kw("pooled", sampler="stratified"), **{**COMMON, "max_epochs": 2})
    n_train = math.ceil(0.9 * n)
    assert len(seen) == 2 * math.ceil(n_train / 128), len(seen)
    assert all(c == [32] * k for c in seen), seen[:3]


def test_labels_slot_changes_no_latent():
    a = _toy()
    zs = []
    for labels in (None, "celltype"):
        scvi.settings.seed = 0
        s = a.copy(); s.X = s.layers["counts"].copy()
        scvi.model.SCVI.setup_anndata(s, batch_key="batch", labels_key=labels)
        m = scvi.model.SCVI(s, n_latent=4)
        m.train(max_epochs=2, batch_size=128, early_stopping=False, enable_progress_bar=False)
        zs.append(m.get_latent_representation())
    assert np.array_equal(zs[0], zs[1]), float(np.abs(zs[0] - zs[1]).max())


def test_unit_importance_weights_reproduce_the_unweighted_fit():
    cfgs = []
    for arm in ["pooled", "mmd", "discriminator", "reference"]:
        for iw in (None, "ones"):
            c = dict(arm=arm, cond=True, adv_input="mean", zstd=False, model="SCVI")
            if iw:
                c["iw"] = iw
            c["key"] = f"{arm}|{iw}"
            cfgs.append(c)
    with tempfile.TemporaryDirectory() as tmp:
        z = bi.run_version(ROOT, cfgs, os.path.join(tmp, "z.npz"))
    errors = [k for k in z if k.endswith("::error")]
    assert not errors, {k: str(z[k]) for k in errors}
    for arm in ["pooled", "mmd"]:
        bi.compare(z[f"{arm}|None"], z[f"{arm}|ones"])                    # bit for bit
    for arm in ["discriminator", "reference"]:
        assert np.allclose(z[f"{arm}|None"], z[f"{arm}|ones"], rtol=0, atol=1e-4), \
            float(np.abs(z[f"{arm}|None"] - z[f"{arm}|ones"]).max())
