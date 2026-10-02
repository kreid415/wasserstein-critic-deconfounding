"""WassersteinAdversarialTrainingPlan -- a drop-in scvi-tools TrainingPlan that adds a
SWAPPABLE adversary on the latent of any scvi module (SCVI / LinearSCVI). Arms:

    none            stock scvi training (seeded => bit-identical to stock SCVI / LinearSCVI)
    scvi_adv        scvi-tools' own adversary, unchanged (AdversarialTrainingPlan with
                    adversarial_classifier=True: Classifier(n_hidden=32), kappa = 1 - kl_weight,
                    one classifier step per generator step). Conditioned models only.
    discriminator   V-way classifier head (JS-type adversary), bounded scvi-tools fool loss,
                    ONE adversary step per generator step (as discriminators are trained).
    reference       per-batch W1 critic heads, target = a designated reference batch (the
                    reference cells receive the pull of every head: a star of pairwise terms)
    reference_fixed as reference, but reference cells are detached in the generator step, so the
                    reference distribution is a fixed target (code review N5 control)
    pooled          per-batch W1 critic heads, target of head k = all OTHER batches
    barycenter      per-batch W1 critic heads, target = free-support W2 barycenter of the batch
                    distributions, recomputed each step (wcd.barycenter)
    mmd, sinkhorn   critic-free divergences, each batch vs the other batches (wcd.alignment)
    mmd_ref         MMD of each batch to the reference batch (reference anchoring without W1; X7)
    <critic>_sn     spectral-norm Lipschitz variant of a critic (gradient penalty dropped)
    discriminator_sn  spectral-normalised discriminator (Lipschitz control on the JS arm; X6)

Settings that change the science are REQUIRED arguments (no silent defaults): adv_input
('mean' = posterior mean, 'sample' = posterior sample z), n_critic (critic steps per generator
step; the discriminator is always 1), zstd (standardise the adversary input per latent dimension
with posterior-mean statistics, gradients kept, so no arm can lower its loss by rescaling z).

The adversary heads are the authored wcd modules, loaded by file path to bypass the wcd package
__init__ (which imports scib, absent from the scvi environment).
"""
import os
import sys
import types
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from scvi.train import AdversarialTrainingPlan
from scvi import REGISTRY_KEYS

_CRITIC_FORMULATIONS = ("reference", "reference_fixed", "pooled", "barycenter")
_CRITIC_FREE = ("mmd", "sinkhorn", "mmd_ref")
_ARMS = ("none", "scvi_adv", "discriminator") + _CRITIC_FORMULATIONS + _CRITIC_FREE


def _parse_adversary(adversary):
    """Return (base, spectral_norm). 'pooled_sn' -> ('pooled', True)."""
    if adversary.endswith("_sn"):
        base = adversary[:-3]
        if base not in _CRITIC_FORMULATIONS + ("discriminator",):
            raise ValueError(f"_sn is only valid on a critic formulation or the discriminator, got {adversary!r}")
        return base, True
    if adversary not in _ARMS:
        raise ValueError(f"unknown adversary {adversary!r}; valid: {_ARMS}")
    return adversary, False


# ---------------------------------------------------------------------------------------------
# Load the authored adversary heads by FILE PATH (bypass the wcd package __init__, which imports
# scib). We stub the one primitive they need (MultiClassCrossEntropy == V-way cross-entropy) and
# register fake parent packages so the modules' absolute imports resolve.
# ---------------------------------------------------------------------------------------------
def _load_wcd_heads(src_root):
    """Import wcd.critic and wcd.adversarial in isolation. Returns (Discriminator, critic_mod)."""
    # stub wcd_vae.wcd.primitives with just MultiClassCrossEntropy
    prim = types.ModuleType("wcd_vae.wcd.primitives")

    class MultiClassCrossEntropy(nn.Module):
        def __init__(self, reduction="mean"):
            super().__init__()
            self.reduction = reduction

        def forward(self, logits, target):
            return F.cross_entropy(logits, target, reduction=self.reduction)

    prim.MultiClassCrossEntropy = MultiClassCrossEntropy
    # register the parent packages so `from wcd_vae.wcd.critic import ...` resolves
    for name in ("wcd_vae", "wcd_vae.wcd"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
    sys.modules["wcd_vae.wcd.primitives"] = prim

    def _load(modname, filename):
        path = os.path.join(src_root, "wcd_vae", "wcd", filename)
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    critic_mod = _load("wcd_vae.wcd.critic", "critic.py")
    adv_mod = _load("wcd_vae.wcd.adversarial", "adversarial.py")
    align_mod = _load("wcd_vae.wcd.alignment", "alignment.py")   # critic-free MMD/Sinkhorn
    bary_mod = _load("wcd_vae.wcd.barycenter", "barycenter.py")  # barycenter target (POT)
    return adv_mod.Discriminator, critic_mod, align_mod, bary_mod


class WassersteinAdversarialTrainingPlan(AdversarialTrainingPlan):
    """AdversarialTrainingPlan with a swappable adversary on the latent (see module docstring)."""

    def __init__(self, module, *, adversary="none", d_coef=0.0, n_critic=None, reference_batch=None,
                 adv_input=None, zstd=False, bary_support=None, bary_iter=10, bary_weights="equal",
                 wcd_src_root=None, n_domains=None, adv_batch_slot="batch",
                 critic_lr=1e-4, critic_betas=(0.0, 0.9), adv_hidden=128, disc_lr=1e-3,
                 bary_warm_iter=None, **kwargs):
        base, sn = _parse_adversary(adversary)
        if base == "scvi_adv":
            kwargs["adversarial_classifier"] = True
            kwargs.setdefault("scale_adversarial_loss", "auto")
        else:
            kwargs.setdefault("adversarial_classifier", False)
        super().__init__(module, **kwargs)
        self.adversary, self.adversary_base, self.spectral_norm = adversary, base, sn
        self.is_critic = base in _CRITIC_FORMULATIONS
        self.is_critic_free = base in _CRITIC_FREE
        self.d_coef = float(d_coef)
        self.adv_batch_slot = adv_batch_slot
        self.critic_lr, self.critic_betas = float(critic_lr), tuple(critic_betas)
        self.disc_lr, self.adv_hidden = float(disc_lr), int(adv_hidden)
        self.bary_warm_iter = None if bary_warm_iter is None else int(bary_warm_iter)
        self._bary_prev = None
        self._wcd_head = self._align_fn = self._bary = None
        if base == "scvi_adv" and self.adversarial_classifier is False:
            raise ValueError("scvi_adv needs a batch-conditioned model (scvi disables it when n_batch == 1)")
        if base in ("none", "scvi_adv"):
            return
        if adv_input not in ("mean", "sample"):
            raise ValueError(f"adv_input must be 'mean' or 'sample', got {adv_input!r}")
        self.adv_input, self.zstd = adv_input, bool(zstd)
        if base == "discriminator":
            if n_critic not in (None, 1):
                raise ValueError("the discriminator takes exactly one adversary step per generator step")
            self.adv_steps = 1
        elif self.is_critic:
            if n_critic is None or int(n_critic) < 1:
                raise ValueError("critic arms need an explicit n_critic >= 1")
            self.adv_steps = int(n_critic)
        else:
            self.adv_steps = 0
        if base in ("reference", "reference_fixed", "mmd_ref"):
            if reference_batch is None:
                raise ValueError("reference arms need reference_batch")
            self.reference_batch = int(reference_batch)
        else:
            self.reference_batch = None
        self.bary_support, self.bary_iter, self.bary_weights = bary_support, int(bary_iter), bary_weights
        src_root = wcd_src_root or os.environ.get("WCD_SRC")
        Discriminator, _critic, align_mod, bary_mod = _load_wcd_heads(src_root)
        self._bary_fn = bary_mod.batch_barycenter_support
        n_batch = int(n_domains) if n_domains is not None else int(self.module.n_batch)
        if self.is_critic_free:
            fn = align_mod.CRITIC_FREE_LOSSES[base]
            if base in align_mod.NEEDS_REFERENCE:
                ref = self.reference_batch
                self._align_fn = lambda z, b: fn(z, b, reference_batch=ref)
            else:
                self._align_fn = fn
        else:
            head_form = {"reference_fixed": "reference"}.get(base, base) if self.is_critic else "reference"
            self._wcd_head = Discriminator(
                n_input=int(self.module.n_latent), domain_number=n_batch, critic=self.is_critic,
                reference_batch=(self.reference_batch if head_form == "reference" and self.is_critic else None),
                formulation=head_form, spectral_norm=sn, n_hidden=self.adv_hidden,
            )
            self.register_module("wcd_adversary", self._wcd_head)

    # ---------------------------------------------------------------------------------------
    def configure_optimizers(self):
        """Generator: scvi-tools' own optimizer for every arm (Adam lr=1e-3, eps=0.01, wd=1e-6).
        Discriminator: scvi-tools' adversarial-classifier optimizer (Adam lr=1e-3, eps=0.01,
        wd=self.weight_decay; AdversarialTrainingPlan.configure_optimizers, scvi-tools 1.4.2).
        Critics: WGAN-GP Algorithm 1 defaults (Adam lr=1e-4, beta1=0, beta2=0.9; Gulrajani et al.
        2017, arXiv 1704.00028v3), with n_critic=5 and lambda_GP=10 set by the manifest/head."""
        if self.adversary_base in ("none", "scvi_adv"):
            return super().configure_optimizers()
        params_g = filter(lambda p: p.requires_grad, self.module.parameters())
        opt_g = self.get_optimizer_creator()(params_g)
        if self.is_critic_free:
            return [opt_g]
        params_d = filter(lambda p: p.requires_grad, self._wcd_head.parameters())
        if self.is_critic:
            opt_d = torch.optim.Adam(params_d, lr=self.critic_lr, betas=self.critic_betas)
        else:
            opt_d = torch.optim.Adam(params_d, lr=self.disc_lr, eps=0.01, weight_decay=self.weight_decay)
        return [opt_g, opt_d]

    def _adv_view(self, z, mu):
        """The adversary's input: posterior mean or sample, optionally standardised per dimension
        by the minibatch posterior-mean mean/SD (gradients kept: rescaling z cannot lower a loss)."""
        if not self.zstd:
            return z
        m = mu.mean(0, keepdim=True)
        s = mu.std(0, unbiased=False, keepdim=True).clamp_min(1e-4)
        return (z - m) / s

    def _head(self, z, batch_index, target, with_gp):
        ref = self.reference_batch if self.adversary_base in ("reference", "reference_fixed") else None
        out = self._wcd_head(z, batch_index, reference_batch=ref, target_samples=target, with_gp=with_gp)
        loss, gp = out
        if self.spectral_norm or not with_gp or not torch.is_tensor(gp):
            gp = z.new_zeros(())
        return loss, gp

    def _fool_loss(self, z, batch_index):
        """scvi-tools fool loss: CE between the head's prediction and uniform-over-other-batches."""
        logits = self._wcd_head(z, None)
        k = logits.shape[1]
        logp = F.log_softmax(logits, dim=1)
        other = (~F.one_hot(batch_index, k).bool()).float() / (k - 1)
        return -(logp * other).sum(dim=1).mean()

    def _log(self, name, value):
        self.log(name, value.detach() if torch.is_tensor(value) else value, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        if self.adversary_base in ("none", "scvi_adv"):
            return super().training_step(batch, batch_idx)
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
            self.log("kl_weight", self.kl_weight, on_step=True, on_epoch=False)
        slot = REGISTRY_KEYS.LABELS_KEY if self.adv_batch_slot == "labels" else REGISTRY_KEYS.BATCH_KEY
        batch_index = batch[slot].long().squeeze(-1)
        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        mu = inference_outputs["qz"].loc
        z_used = mu if self.adv_input == "mean" else inference_outputs["z"]
        z_adv = self._adv_view(z_used, mu)
        loss_vae = scvi_loss.loss
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        self._log("z_rms", mu.pow(2).mean().sqrt())

        if self.is_critic_free:
            (opt_g,) = [self.optimizers()]
            adv_term = self._align_fn(z_adv, batch_index)
            g = torch.autograd.grad(adv_term, z_adv, retain_graph=True)[0]
            gen_loss = loss_vae + self.d_coef * adv_term
            opt_g.zero_grad()
            self.manual_backward(gen_loss)
            opt_g.step()
            self._log("adv_div", adv_term)
            self._log("adv_grad_norm", g.norm(dim=1).mean())
            self.log("train_loss", loss_vae, on_step=self.on_step, on_epoch=self.on_epoch, prog_bar=True)
            return loss_vae

        opt_g, opt_d = self.optimizers()
        target = None
        if self.adversary_base == "barycenter":
            warm = self.bary_warm_iter is not None and self._bary_prev is not None
            target = self._bary_fn(z_adv.detach(), batch_index, n_support=self.bary_support,
                                   n_iter=(self.bary_warm_iter if warm else self.bary_iter), weights=self.bary_weights,
                                   init=(self._bary_prev if warm else None))
            self._bary_prev = target
        z_d = z_adv.detach()
        for _ in range(self.adv_steps):
            if self.is_critic:
                loss_d, gp = self._head(z_d, batch_index, target, with_gp=True)
            else:
                loss_d, gp = self._head(z_d, batch_index, None, with_gp=False)
            opt_d.zero_grad()
            self.manual_backward(loss_d + gp, retain_graph=False)
            opt_d.step()
        self._log("adv_loss", loss_d)
        self._log("adv_gp", gp)
        if self.is_critic:
            z_g = z_adv
            if self.adversary_base == "reference_fixed":
                is_ref = (batch_index == self.reference_batch)[:, None]
                z_g = torch.where(is_ref, z_adv.detach(), z_adv)
            loss_g, _ = self._head(z_g, batch_index, target, with_gp=False)
            adv_term = -loss_g                       # the critic's W1 estimate (mean over heads)
            self._log("adv_w1", adv_term)
        else:
            adv_term = self._fool_loss(z_adv, batch_index)
            with torch.no_grad():
                acc = (self._wcd_head(z_d, None).argmax(1) == batch_index).float().mean()
            self._log("adv_acc", acc)
        g = torch.autograd.grad(adv_term, z_adv, retain_graph=True, allow_unused=True)[0]
        if g is not None:
            self._log("adv_grad_norm", g.norm(dim=1).mean())
        gen_loss = loss_vae + self.d_coef * adv_term
        opt_g.zero_grad()
        self.manual_backward(gen_loss)
        opt_g.step()
        self.log("train_loss", loss_vae, on_step=self.on_step, on_epoch=self.on_epoch, prog_bar=True)
        return loss_vae


def fit_adversarial_scvi(adata, batch_key, *, adversary, d_coef, n_critic, reference_batch, adv_input,
                         zstd, n_latent, max_epochs, batch_size, seed, conditioned, model_name="SCVI",
                         n_layers=1, n_hidden=128, gene_likelihood="zinb", train_size=0.9,
                         bary_support=None, bary_iter=10, bary_weights="equal", wcd_src_root=None,
                         critic_lr=1e-4, critic_betas=(0.0, 0.9), adv_hidden=128, disc_lr=1e-3,
                         bary_warm_iter=None):
    """Fit one scvi model + adversary. Every design setting is a required keyword; the backbone
    settings default to scvi-tools' own defaults (n_layers=1, n_hidden=128, zinb, train_size=0.9).
    Returns (posterior-mean latent [n, n_latent], trained model)."""
    import scvi
    if model_name == "SCVI":
        from scvi.model import SCVI as Model
    elif model_name == "LinearSCVI":
        from scvi.model import LinearSCVI as Model
    else:
        raise ValueError(f"model_name must be 'SCVI' or 'LinearSCVI', got {model_name!r}")
    scvi.settings.seed = seed
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    n_domains = int(a.obs[batch_key].nunique())
    if conditioned:
        Model.setup_anndata(a, batch_key=batch_key)
        slot = "batch"
    else:
        Model.setup_anndata(a, labels_key=batch_key)
        slot = "labels"
    if model_name == "SCVI":
        model = Model(a, n_latent=n_latent, n_layers=n_layers, n_hidden=n_hidden, gene_likelihood=gene_likelihood)
    else:   # LinearSCVI: n_layers applies to the encoder only
        model = Model(a, n_latent=n_latent, n_layers=n_layers, n_hidden=n_hidden, gene_likelihood=gene_likelihood)
    if isinstance(reference_batch, str):
        mapping = list(model.adata_manager.get_state_registry(slot).categorical_mapping)
        reference_batch = mapping.index(reference_batch)
    model._training_plan_cls = WassersteinAdversarialTrainingPlan
    plan_kwargs = dict(adversary=adversary, d_coef=d_coef, n_critic=n_critic, reference_batch=reference_batch,
                       adv_input=adv_input, zstd=zstd, n_domains=n_domains, adv_batch_slot=slot,
                       bary_support=bary_support, bary_iter=bary_iter, bary_weights=bary_weights,
                       wcd_src_root=(wcd_src_root or os.environ.get("WCD_SRC")),
                       critic_lr=critic_lr, critic_betas=critic_betas, adv_hidden=adv_hidden, disc_lr=disc_lr,
                       bary_warm_iter=bary_warm_iter)
    model.train(max_epochs=max_epochs, batch_size=batch_size, early_stopping=False, train_size=train_size,
                enable_progress_bar=False, plan_kwargs=plan_kwargs)
    return model.get_latent_representation(), model


fit_adversarial_linearscvi = fit_adversarial_scvi   # old name, kept for imports
