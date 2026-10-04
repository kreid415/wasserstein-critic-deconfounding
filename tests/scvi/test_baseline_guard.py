"""X13 neural baselines under the non-finite-loss guard (code check CR-06) and resolved plan settings in the latent
config (CR-09). Run in the scvi env, CPU only: CUDA_VISIBLE_DEVICES= WCD_SRC=src python -m pytest -q <this file>

1. Bit identity: scANVI and sysVI latents of the guarded fitter equal the tagged fitter's (prereg-tier12-v1) exactly.
2. A NaN decoder weight makes the loss NaN: NonFiniteLossError at the injected step, in the scVI pretraining phase
   and in the scANVI phase (step counted over the whole fit), naming train_loss; a NaN encoder weight makes
   torch.distributions refuse the forward pass: raised as NonFiniteLossError from the ValueError; parameters made
   non-finite by the last step are caught at the end; any other forward error propagates unchanged. Mutants without
   the per-step check or without the forward mapping fail these checks.
3. End to end through fit_paper_config.py: a sysVI row with z_distance_cycle_weight 1e39 writes status 'diverged' and
   exits EXIT_DIVERGED with no latent; a finite sysVI row writes its latent with config 'plan'.
4. resolved_plan known answers: scvi-tools' generator optimiser (Adam, lr 1e-3, eps 0.01, weight decay 1e-6) and KL
   warm-up (400 epochs; 0 for sysVI, which scvi-tools forces), WGAN-GP critic Adam(1e-4, betas (0, 0.9)) with
   n_critic 5 and lambda_GP 10, discriminator Adam lr 1e-3 with one step and no gradient penalty.
"""
import json
import os
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

scvi = pytest.importorskip("scvi")
import anndata as ad  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WCD_SRC", os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fit_outcome  # noqa: E402
import fit_paper_config as fpc  # noqa: E402
import scvi_adversarial_plan as plan  # noqa: E402
from scvi.train import SemiSupervisedTrainingPlan, TrainingPlan  # noqa: E402

TAG = "prereg-tier12-v1"


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
    return a


COMMON = dict(n_latent=4, max_epochs=2, batch_size=128, seed=0, conditioned=True)
BACKBONE = dict(n_layers=1, n_hidden=128, gene_likelihood="zinb", train_size=0.9)


def _fit(arm, lam, mod=fpc):
    return mod.fit_baseline(_toy(), arm, lam, {}, COMMON, BACKBONE)


def _tagged_module():
    src = subprocess.run(["git", "-C", ROOT, "show", f"{TAG}:scripts/fit_paper_config.py"], check=True,
                         capture_output=True, text=True).stdout
    mod = types.ModuleType("fpc_tagged")
    mod.__file__ = os.path.join(ROOT, "scripts", "fit_paper_config.py")
    exec(compile(src, f"{TAG}:scripts/fit_paper_config.py", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("arm,lam", [("scanvi", 0.0), ("sysvi", 2.0)])
def test_guarded_baselines_are_bit_identical_to_the_tagged_fitter(arm, lam):
    z_old, _ = _fit(arm, lam, mod=_tagged_module())
    z_new, _, rec = _fit(arm, lam)
    assert z_new.dtype == z_old.dtype and np.array_equal(z_new, z_old)
    assert set(rec) == ({"scvi_pretraining", "scanvi"} if arm == "scanvi" else {"fit"})


def _poison(monkeypatch, base, at, pick):
    """Fill pick(plan) with NaN before the at-th call (0-based) of base.training_step."""
    orig, calls = base.training_step, {"n": -1}

    def wrapped(self, batch, batch_idx):
        calls["n"] += 1
        if calls["n"] == at:
            with torch.no_grad():
                pick(self).fill_(float("nan"))
        return orig(self, batch, batch_idx)

    monkeypatch.setattr(base, "training_step", wrapped)


def _first_weight(module):
    return next(p for n, p in module.named_parameters() if n.endswith("weight") and p.dim() == 2)


def nan_decoder_violations(monkeypatch, mod, base, at, phase):
    """Reasons why a NaN decoder weight at step `at` of `base` was not recorded as the expected divergence."""
    _poison(monkeypatch, base, at, lambda p: _first_weight(p.module.decoder))
    try:
        _fit("scanvi", 0.0, mod=mod)
    except fit_outcome.NonFiniteLossError as e:
        bad = []
        n_pre = 2 * int(np.ceil(600 * 0.9 / 128))          # pretraining steps (2 epochs x 5 minibatches)
        want = at if phase == "scvi pretraining" else n_pre + at
        if e.step != want:
            bad.append(f"step {e.step}, expected {want}")
        if not (set(e.terms) == {"train_loss"} and np.isnan(e.terms["train_loss"])):
            bad.append(f"terms {e.terms}")
        if not e.detail.startswith(phase):
            bad.append(f"detail {e.detail!r}")
        return bad
    finally:
        monkeypatch.undo()
    return ["no NonFiniteLossError"]


@pytest.mark.parametrize("base,at,phase", [(TrainingPlan, 3, "scvi pretraining"), (SemiSupervisedTrainingPlan, 2, "scanvi")])
def test_nan_decoder_weight_raises_at_the_injected_step(monkeypatch, base, at, phase):
    assert nan_decoder_violations(monkeypatch, fpc, base, at, phase) == []


def test_mutant_without_the_step_check_fails(monkeypatch):
    """Mutation check (fail-loud R11): without the per-step finiteness check the divergence is caught late (or not
    at all), so the check above reports it."""
    src = open(fpc.__file__).read()
    old = "            if not bool(torch.isfinite(value)):\n"
    assert src.count(old) == 1
    mod = types.ModuleType("fpc_mutant")
    mod.__file__ = fpc.__file__
    exec(compile(src.replace(old, "            if False:\n"), fpc.__file__ + "<mutant>", "exec"), mod.__dict__)
    assert nan_decoder_violations(monkeypatch, mod, TrainingPlan, 3, "scvi pretraining")


def test_nan_encoder_weight_raises_from_forward(monkeypatch):
    _poison(monkeypatch, TrainingPlan, 2, lambda p: _first_weight(p.module.z_encoder))
    with pytest.raises(fit_outcome.NonFiniteLossError) as ei:
        _fit("scanvi", 0.0)
    e = ei.value
    assert e.step == 2 and e.terms == {}, (e.step, e.terms)
    assert isinstance(e.__cause__, ValueError) and str(e.__cause__).startswith("Expected parameter")
    assert e.detail.startswith("scvi pretraining: forward pass refused") and "z_encoder" in e.detail, e.detail


def test_mutant_without_forward_mapping_fails(monkeypatch):
    src = open(fpc.__file__).read()
    old = "                if not _is_distribution_validation_error(e):\n                    raise\n"
    assert src.count(old) == 1
    mod = types.ModuleType("fpc_mutant_fwd")
    mod.__file__ = fpc.__file__
    exec(compile(src.replace(old, "                raise\n"), fpc.__file__ + "<mutant>", "exec"), mod.__dict__)
    _poison(monkeypatch, TrainingPlan, 2, lambda p: _first_weight(p.module.z_encoder))
    with pytest.raises(ValueError) as ei:
        _fit("scanvi", 0.0, mod=mod)
    assert not isinstance(ei.value, fit_outcome.NonFiniteLossError)


def test_nonfinite_parameters_after_the_last_step_raise(monkeypatch):
    """Poison a decoder weight after the last optimizer step (on_train_batch_end runs after the automatic step)."""
    def poison(self, outputs, batch, batch_idx):
        if self.trainer.is_last_batch and self.current_epoch == self.trainer.max_epochs - 1:
            with torch.no_grad():
                _first_weight(self.module.decoder).fill_(float("nan"))

    monkeypatch.setattr(SemiSupervisedTrainingPlan, "on_train_batch_end", poison, raising=False)
    with pytest.raises(fit_outcome.NonFiniteLossError) as ei:
        _fit("scanvi", 0.0)
    assert ei.value.detail.startswith("scanvi: non-finite parameters after the last optimizer step"), ei.value.detail


def test_other_forward_errors_propagate_unchanged(monkeypatch):
    orig = TrainingPlan.forward

    def boom(self, *a, **k):
        raise ValueError("Expected parameter: not raised by torch.distributions")

    monkeypatch.setattr(TrainingPlan, "forward", boom)
    with pytest.raises(ValueError, match="not raised by torch.distributions") as ei:
        _fit("sysvi", 2.0)
    assert not isinstance(ei.value, fit_outcome.NonFiniteLossError)
    monkeypatch.setattr(TrainingPlan, "forward", orig)


def test_wrong_base_plan_is_refused():
    with pytest.raises(TypeError, match="expected TrainingPlan"):
        fpc.guarded_plan_class(SemiSupervisedTrainingPlan, TrainingPlan, {"step": -1}, "x")


# ---- 3. end to end ------------------------------------------------------------------------------------------------
def _row(tag, **over):
    r = dict(tag=tag, experiment="T", task="toy", counts="scib", arm="sysvi", lam="2.0", n_critic="0", adv_input="mean",
             zstd="0", cond="1", decoder="SCVI", n_latent="4", n_layers="1", n_hidden="128", likelihood="zinb",
             batch_size="128", max_epochs="1", train_size="0.9", seed="0", reference="auto", extra="{}")
    r.update(over)
    return r


def test_fitter_records_baseline_divergence(tmp_path):
    _toy().write_h5ad(tmp_path / "toy__scib.h5ad")
    rows = [_row("sysvi_div", lam="1e39"), _row("sysvi_ok")]
    man = tmp_path / "m.tsv"
    pd.DataFrame(rows)[fpc.REQUIRED].to_csv(man, sep="\t", index=False)
    out = tmp_path / "out"
    env = dict(os.environ, MANIFEST=str(man), PREPPED_DIR=str(tmp_path), OUT_DIR=str(out), WCD_SRC=os.path.join(ROOT, "src"),
               CUDA_VISIBLE_DEVICES="", KMP_AFFINITY="disabled", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    res = {}
    for r in rows:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fit_paper_config.py")],
                           env=dict(env, TAG=r["tag"]), capture_output=True, text=True)
        res[r["tag"]] = p
    p = res["sysvi_div"]
    assert p.returncode == fit_outcome.EXIT_DIVERGED, p.stderr[-2000:]
    st = json.loads((out / "status" / "sysvi_div.json").read_text())
    assert st["status"] == "diverged" and st["row"] == rows[0] and st["detail"].count("sysvi") >= 1
    assert st["terms"]["train_loss"] in ("inf", "nan"), st["terms"]
    assert not (out / "latents" / "sysvi_div.npz").exists()
    assert res["sysvi_ok"].returncode == 0, res["sysvi_ok"].stderr[-2000:]
    cfg = json.loads(str(np.load(out / "latents" / "sysvi_ok.npz")["config"]))
    assert cfg["plan"]["fit"]["plan_class"] == "GuardedTrainingPlan" and cfg["plan"]["fit"]["n_epochs_kl_warmup"] == 0
    assert not (out / "status" / "sysvi_ok.json").exists()


# ---- 4. resolved plan settings ---------------------------------------------------------------------------------
def test_resolved_plan_known_answers():
    _, _, rec = _fit("scanvi", 0.0)
    pre = rec["scvi_pretraining"]
    assert (pre["plan_class"], pre["optimizer_name"], pre["lr"], pre["eps"], pre["weight_decay"]) == \
        ("GuardedTrainingPlan", "Adam", 1e-3, 0.01, 1e-6)
    assert pre["n_epochs_kl_warmup"] == 400 and rec["scanvi"]["plan_class"] == "GuardedSemiSupervisedTrainingPlan"
    _, _, rec = _fit("sysvi", 2.0)
    assert rec["fit"]["n_epochs_kl_warmup"] == 0                   # scvi-tools SysVI.train forces no KL warm-up
    for arm, n_critic in (("pooled", 5), ("discriminator", None)):
        _, m = plan.fit_adversarial_scvi(_toy(), "batch", adversary=arm, d_coef=1.0, n_critic=n_critic,
                                         reference_batch="b0", adv_input="mean", zstd=False, n_latent=4, max_epochs=1,
                                         batch_size=128, seed=0, conditioned=True)
        r = fpc.resolved_plan(m)
        gen, adv = r["optimizers"][0]["param_groups"][0], r["optimizers"][1]["param_groups"][0]
        assert (gen["lr"], gen["eps"], gen["weight_decay"]) == (1e-3, 0.01, 1e-6)
        assert r["adv_hidden"] == 128 and r["n_epochs_kl_warmup"] == 400
        if arm == "pooled":
            assert (r["adv_steps"], r["critic_lr"], r["critic_betas"], r["lambda_gp"]) == (5, 1e-4, [0.0, 0.9], 10.0)
            assert (adv["lr"], adv["betas"]) == (1e-4, [0.0, 0.9]) and r["gradient_penalty"] is True
        else:
            assert (r["adv_steps"], r["disc_lr"], r["lambda_gp"], r["gradient_penalty"]) == (1, 1e-3, None, False)
            assert (adv["lr"], adv["eps"]) == (1e-3, 0.01)
    json.dumps(r)                                                  # JSON-serialisable (stored in the npz config)


def test_the_head_uses_the_default_gradient_penalty_weight():
    """resolved_plan records the default of critic.multi_class_gradient_penalty: valid only while no caller passes
    lambda_gp (static check of the head's calls)."""
    src = open(os.path.join(ROOT, "src", "wcd_vae", "wcd", "adversarial.py")).read()
    assert src.count("multi_class_gradient_penalty(") == 2 and "lambda_gp" not in src


def test_fitter_writes_the_plan_record_for_every_plan_kind(tmp_path):
    """fit_paper_config.py as a script (PYTHONSAFEPATH=1, as run_stage starts it) for arms without a wcd adversary
    (none, scvi_adv), a JS and a critic-free arm: exit 0 and cfg['plan'] with the fields of that plan kind. The stage
    e2e test found resolved_plan reading adv_steps on 'none', which the plan never sets."""
    _toy().write_h5ad(tmp_path / "toy__scib.h5ad")
    base = dict(arm="none", lam="0", n_critic="0", decoder="SCVI", likelihood="zinb")
    rows = [_row("none", **base), _row("scvi_adv", **dict(base, arm="scvi_adv")),
            _row("disc", arm="discriminator", lam="1.0", n_critic="1"), _row("mmd", arm="mmd", lam="1.0", n_critic="0")]
    man = tmp_path / "m.tsv"
    pd.DataFrame(rows)[fpc.REQUIRED].to_csv(man, sep="\t", index=False)
    out = tmp_path / "out"
    env = dict(os.environ, MANIFEST=str(man), PREPPED_DIR=str(tmp_path), OUT_DIR=str(out), WCD_SRC=os.path.join(ROOT, "src"),
               CUDA_VISIBLE_DEVICES="", KMP_AFFINITY="disabled", OMP_NUM_THREADS="1", PYTHONSAFEPATH="1")
    for r in rows:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fit_paper_config.py")],
                           env=dict(env, TAG=r["tag"]), capture_output=True, text=True)
        assert p.returncode == 0, (r["tag"], p.stderr[-2000:])
    plans = {r["tag"]: json.loads(str(np.load(out / "latents" / f"{r['tag']}.npz")["config"]))["plan"]["fit"] for r in rows}
    assert plans["none"]["adversary"] == "none" and plans["none"]["adversarial_classifier"] is False
    assert "adv_steps" not in plans["none"] and len(plans["none"]["optimizers"]) == 1
    assert plans["scvi_adv"]["adversarial_classifier"] is True and len(plans["scvi_adv"]["optimizers"]) == 2
    assert plans["disc"]["adv_steps"] == 1 and plans["disc"]["lambda_gp"] is None
    assert plans["mmd"]["adv_steps"] == 0 and plans["mmd"]["gradient_penalty"] is False
    assert all(p["n_epochs_kl_warmup"] == 400 for p in plans.values())
