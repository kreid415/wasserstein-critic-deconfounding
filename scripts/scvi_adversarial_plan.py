"""WassersteinAdversarialTrainingPlan -- a drop-in scvi-tools TrainingPlan that adds a
SWAPPABLE adversary on the latent z of any scvi module (LinearSCVI/SCVI), so the SAME
scvi-tools generative model can be trained with:

    adversary = "none"           -> stock scvi training (seeded => BIT-IDENTICAL to LinearSCVI)
              | "discriminator"  -> V-way cross-entropy domain classifier (JS adversary)
              | "reference"      -> reference Wasserstein critic (align to a designated batch)
              | "pooled"         -> pooled Wasserstein critic (align to the global pool; the
                                    fair W1 counterpart to the discriminator's JS objective)
              | "barycenter"     -> barycenter Wasserstein critic (align to a learnable centre)

WHY THIS DESIGN (upstreamable):
  scvi-tools already ships AdversarialTrainingPlan (a two-optimizer, manual-optimization
  TrainingPlan that adds kappa*fool_loss to the VAE loss and trains the adversary on z.detach()).
  We subclass it and REPLACE the built-in Classifier + its loss with our discriminator / Wasserstein
  critic heads. Because the generative model IS scvi's own module and the adversary contributes
  EXACTLY ZERO when adversary="none", a seeded run reproduces stock LinearSCVI bit-for-bit -- which
  is both the correctness guarantee and what makes this a genuine scvi-tools extension.

  The adversary heads (Discriminator, ReferenceWassersteinLoss, multi_class_gradient_penalty) are
  the authored wcd heads, loaded here by file path to bypass the wcd package __init__ (which pulls
  in scib, absent from scvi-env). They are pure torch and depend only on a trivial cross-entropy
  primitive, stubbed below.

USAGE:
    from scvi.model import LinearSCVI
    LinearSCVI.setup_anndata(adata, batch_key="batch")   # or omit batch_key for unconditioned
    model = LinearSCVI(adata, n_latent=30)
    model.train(max_epochs=239, batch_size=512,
                plan_kwargs=dict(adversary="barycenter", d_coef=50.0, disc_iter=10),
                # inject the custom plan:
                )  # see run helper below for the plan_class wiring
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


# Wasserstein-critic formulations (all use the critic branch + gradient penalty + disc_iter inner
# loop). "discriminator" (JS classifier) and "none" are NOT critics. Keep this the single source of
# truth so the head-construction and inner-loop gates cannot drift apart. The suffix "_sn" selects
# the spectral-norm Lipschitz variant of the same critic (GP is then skipped).
_CRITIC_FORMULATIONS = ("reference", "pooled", "barycenter")

# Critic-FREE alignment divergences: no adversary network, no inner loop, no gradient penalty --
# a closed-form/fixed-iteration divergence between each batch and the pool, added straight to the
# generator objective. Isolates the IPM GEOMETRY from the ADVERSARIAL ESTIMATOR.
_CRITIC_FREE = ("mmd", "sinkhorn")


def _parse_adversary(adversary):
    """Return (base, spectral_norm). 'reference_sn' -> ('reference', True); else (adversary, False).
    Only critic formulations accept the _sn suffix."""
    if adversary.endswith("_sn"):
        base = adversary[:-3]
        if base not in _CRITIC_FORMULATIONS:
            raise ValueError(f"spectral-norm suffix _sn only valid on a critic formulation, got {adversary!r}")
        return base, True
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
    return adv_mod.Discriminator, critic_mod, align_mod


class WassersteinAdversarialTrainingPlan(AdversarialTrainingPlan):
    """AdversarialTrainingPlan with a swappable adversary on z (see module docstring).

    Extra plan_kwargs beyond AdversarialTrainingPlan:
      adversary   : "none" | "discriminator" | "reference" | "barycenter"  (default "none")
      d_coef      : adversarial weight lambda applied to the generator fool-loss (default 0.0)
      disc_iter   : critic inner-loop steps per generator step for the Wasserstein arms (default 10;
                    1 for the discriminator, whose JS objective needs no inner loop)
      reference_batch : reference batch index for the "reference" formulation (default 0)
      wcd_src_root    : path to the wcd `src/` dir holding the head modules (default from env WCD_SRC)

    Adversary weight: we use a FIXED d_coef (lambda) rather than scvi's kappa=1-kl_weight, so the
    sweep's lambda has a stable meaning across kl-warmup. Set scale_adversarial_loss=d_coef upstream
    if kappa-style annealing is wanted instead. When adversary="none", NO adversary is built and the
    training_step is stock scvi -> seeded bit-identity with LinearSCVI.
    """

    def __init__(self, module, *, adversary="none", d_coef=0.0, disc_iter=10,
                 reference_batch=0, wcd_src_root=None, n_domains=None, adv_batch_slot="batch",
                 **kwargs):
        # Build stock plan first WITHOUT scvi's own adversarial_classifier (we supply our own head).
        kwargs.setdefault("adversarial_classifier", False)
        super().__init__(module, **kwargs)
        self.adversary = adversary
        self.d_coef = float(d_coef)
        self.disc_iter = int(disc_iter)
        self.reference_batch = reference_batch
        # WHICH registry slot carries the adversary's TRUE batch labels, and how many domains.
        # CONDITIONED backbone: decoder is batch-conditioned (setup batch_key), n_batch>1, labels in
        #   REGISTRY_KEYS.BATCH_KEY. adv_batch_slot="batch", n_domains=module.n_batch.
        # UNCONDITIONED backbone: decoder NOT conditioned (setup labels_key=batch), n_batch==1, real
        #   batch labels live in REGISTRY_KEYS.LABELS_KEY. adv_batch_slot="labels", n_domains passed
        #   in explicitly (the true #batches). The adversary is then the SOLE integrator.
        self.adv_batch_slot = adv_batch_slot
        self._wcd_head = None
        self._align_fn = None
        # split e.g. "reference_sn" -> base="reference", spectral_norm=True
        base, self.spectral_norm = _parse_adversary(adversary)
        self.adversary_base = base
        self.is_critic = base in _CRITIC_FORMULATIONS
        self.is_critic_free = base in _CRITIC_FREE

        if adversary != "none":
            src_root = wcd_src_root or os.environ.get("WCD_SRC")
            Discriminator, _critic, align_mod = _load_wcd_heads(src_root)
            n_batch = int(n_domains) if n_domains is not None else int(self.module.n_batch)
            if self.is_critic_free:
                # No adversary network at all -- a differentiable divergence on z directly.
                self._align_fn = align_mod.CRITIC_FREE_LOSSES[base]
            else:
                self._wcd_head = Discriminator(
                    n_input=int(self.module.n_latent),
                    domain_number=n_batch,
                    critic=self.is_critic,
                    reference_batch=(reference_batch if base == "reference" else None),
                    formulation=(base if self.is_critic else "reference"),
                    spectral_norm=self.spectral_norm,
                )
                self.register_module("wcd_adversary", self._wcd_head)
            # the barycenter anchors are trained by the GENERATOR optimizer (they define the target
            # the encoder aligns to); everything else in the head is the adversary optimizer's.

    # ---- optimizers: VAE(+anchors) vs adversary -------------------------------------------
    def configure_optimizers(self):
        if self.adversary == "none":
            return super().configure_optimizers()
        if self.is_critic_free:
            # No adversary network -> ONE optimizer (the generator). The alignment divergence is
            # differentiable and added straight to the generator loss; nothing to train adversarially.
            opt_g = torch.optim.Adam(self.module.parameters(), lr=self.lr, eps=self.eps,
                                     weight_decay=self.weight_decay, betas=(0.9, 0.999))
            return [opt_g]
        # generator params: the scvi module + (for barycenter) the learnable anchors
        gen_params = list(self.module.parameters())
        adv_params = []
        for n, p in self._wcd_head.named_parameters():
            if n == "anchors":
                gen_params.append(p)          # anchors move WITH the generator (Frechet mean target)
            else:
                adv_params.append(p)
        opt_g = torch.optim.Adam(gen_params, lr=self.lr, eps=self.eps,
                                 weight_decay=self.weight_decay, betas=(0.9, 0.999))
        opt_d = torch.optim.Adam(adv_params, lr=1e-3, betas=(0.5, 0.9))
        return [opt_g, opt_d]

    # ---- training step: stock scvi loss - lambda*loss_da (generator), adversary on z.detach() --
    def training_step(self, batch, batch_idx):
        if self.adversary == "none":
            return super().training_step(batch, batch_idx)   # stock -> seeded bit-identity

        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
            self.log("kl_weight", self.kl_weight, on_step=True, on_epoch=False)
        # adversary's batch labels: BATCH_KEY (conditioned) or LABELS_KEY (unconditioned, where the
        # decoder's BATCH_KEY is all-zeros but the real batches live in the labels slot).
        _slot = REGISTRY_KEYS.LABELS_KEY if self.adv_batch_slot == "labels" else REGISTRY_KEYS.BATCH_KEY
        batch_index = batch[_slot].long().squeeze(-1)

        if self.is_critic_free:
            # CRITIC-FREE: one optimizer, no inner loop, no GP. The alignment divergence D(z, batch)
            # is a differentiable measure of how separable the batches are; the generator minimises
            # loss_vae + lambda * D (batches more mixed => D smaller). Sign is + here (D>=0, and we
            # DRIVE IT DOWN), matching "loss_vae - lambda*loss_da" with loss_da = -D.
            opt_g = self.optimizers()   # single optimizer (Lightning returns it unwrapped)
            inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
            z = inference_outputs["z"]
            loss_vae = scvi_loss.loss
            align = self._align_fn(z, batch_index)
            gen_loss = loss_vae + self.d_coef * align
            opt_g.zero_grad()
            self.manual_backward(gen_loss)
            opt_g.step()
            self.log("train_loss", loss_vae, on_step=self.on_step, on_epoch=self.on_epoch, prog_bar=True)
            return loss_vae

        opt_g, opt_d = self.optimizers()
        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        z = inference_outputs["z"]
        loss_vae = scvi_loss.loss

        # 1) adversary (critic/discriminator) update on detached z, disc_iter times
        for _ in range(self.disc_iter if self.is_critic else 1):
            loss_da_d, gp = self._adv_loss(z.detach(), batch_index)
            loss_d = loss_da_d + gp
            opt_d.zero_grad()
            self.manual_backward(loss_d, retain_graph=False)
            opt_d.step()

        # 2) generator update: minimise loss_vae - lambda * loss_da
        loss_da_g, _gp = self._adv_loss(z, batch_index)
        gen_loss = loss_vae - self.d_coef * loss_da_g
        opt_g.zero_grad()
        self.manual_backward(gen_loss)
        opt_g.step()

        self.log("train_loss", loss_vae, on_step=self.on_step, on_epoch=self.on_epoch, prog_bar=True)
        return loss_vae

    def _adv_loss(self, z, batch_index):
        """Call the wcd head; returns (adversarial_loss, gradient_penalty). When spectral_norm is on,
        the head's Lipschitz constraint is enforced per-layer, so the sampled gradient penalty is
        DROPPED (the two must not be stacked)."""
        ref = self.reference_batch if self.adversary_base == "reference" else None
        out = self._wcd_head(z, batch_index, reference_batch=ref)
        if isinstance(out, tuple):
            gp = out[1] if (len(out) > 1 and out[1] is not None) else z.new_zeros(())
            if self.spectral_norm:
                gp = z.new_zeros(())   # SN already constrains Lipschitz; do not add GP on top
            return out[0], gp
        return out, z.new_zeros(())


# ================================================================================================
# Run helper: fit a LinearSCVI (conditioned or unconditioned) with a chosen adversary, and return
# the latent. Injects the custom plan by setting model._training_plan_cls (scvi's train() reads it).
# ================================================================================================
def fit_adversarial_linearscvi(
    adata, batch_key, *, adversary="none", d_coef=0.0, disc_iter=10, reference_batch=0,
    n_latent=30, max_epochs=239, batch_size=512, seed=0, conditioned=True, wcd_src_root=None,
    model_name="LinearSCVI", max_kl_weight=None, n_epochs_kl_warmup=None,
):
    """Fit an scvi module (LinearSCVI = linear decoder; SCVI = nonlinear decoder) + swappable
    adversary. conditioned=False omits batch_key from setup so the decoder is NOT batch-conditioned
    (adversary is then the sole integrator). The plan reads inference_outputs["z"], which both
    modules provide identically, so the adversary is model-agnostic. Returns the latent Z."""
    import functools
    import scvi
    if model_name == "SCVI":
        from scvi.model import SCVI as Model   # nonlinear DecoderSCVI
    elif model_name == "LinearSCVI":
        from scvi.model import LinearSCVI as Model   # LinearDecoderSCVI
    else:
        raise ValueError(f"model_name must be 'LinearSCVI' or 'SCVI', got {model_name!r}")

    scvi.settings.seed = seed
    a = adata.copy()
    if "counts" in a.layers:
        a.X = a.layers["counts"].copy()
    n_domains = int(a.obs[batch_key].nunique())
    if conditioned:
        # decoder IS batch-conditioned; adversary reads BATCH_KEY.
        Model.setup_anndata(a, batch_key=batch_key)
        adv_batch_slot = "batch"
    else:
        # decoder NOT conditioned (no batch_key); real batches go to the LABELS slot so the
        # adversary still sees them and is the SOLE integrator.
        Model.setup_anndata(a, labels_key=batch_key)
        adv_batch_slot = "labels"
    model = Model(a, n_latent=n_latent)

    # inject the custom plan class + its extra kwargs
    model._training_plan_cls = WassersteinAdversarialTrainingPlan
    plan_kwargs = dict(adversary=adversary, d_coef=d_coef, disc_iter=disc_iter,
                       reference_batch=reference_batch, n_domains=n_domains,
                       adv_batch_slot=adv_batch_slot,
                       wcd_src_root=(wcd_src_root or os.environ.get("WCD_SRC")))
    # KL controls (scvi anneals kl_weight from min to max_kl_weight over n_epochs_kl_warmup).
    # max_kl_weight is the scvi-native equivalent of the wcd harness's fixed kl_coef -- sweep it to
    # test whether looser KL lifts the critics (default scvi max_kl_weight=1.0). Flows through the
    # plan's **kwargs to the base TrainingPlan.
    if max_kl_weight is not None:
        plan_kwargs["max_kl_weight"] = float(max_kl_weight)
    if n_epochs_kl_warmup is not None:
        plan_kwargs["n_epochs_kl_warmup"] = int(n_epochs_kl_warmup)
    model.train(max_epochs=max_epochs, batch_size=batch_size, early_stopping=False,
                enable_progress_bar=False, plan_kwargs=plan_kwargs)
    return model.get_latent_representation()
