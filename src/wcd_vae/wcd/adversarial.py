"""Adversarial head with discriminator/critic dispatch — AUTHORED (K. Reid).

# WHY: Hold one network architecture fixed while switching only the adversarial
#      objective, so the discriminator-vs-critic comparison is controlled.
# HOW: A single MLP head (fc1->fc2->fc3) whose loss is either V-way CrossEntropy
#      (JS-divergence discriminator) or the reference Wasserstein critic, selected
#      by the ``critic`` flag; the WGAN gradient penalty is applied only in critic mode.
Provenance: the discriminator MLP scaffold was originally derived by modifying scCRAFT's
discriminator (renamed ``Discriminator``); the critic branch, formulation dispatch,
reference/anchor plumbing, and gradient-penalty call are authored additions, and the
cross-entropy primitive it calls is the clean-room ``wcd.primitives.MultiClassCrossEntropy``.
Not a clean-room reimplementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from wcd_vae.wcd.critic import ReferenceWassersteinLoss, multi_class_gradient_penalty
from wcd_vae.wcd.primitives import MultiClassCrossEntropy


class Discriminator(nn.Module):
    def __init__(
        self,
        n_input,
        domain_number,
        critic=False,
        reference_batch=None,
        reference_batch_name_str=None,
        formulation="reference",
        n_anchors=64,
        spectral_norm=False,
        n_hidden=128,
    ):
        super().__init__()
        n_hidden = int(n_hidden)
        self.critic = critic
        self.domain_number = int(domain_number)
        self.formulation = formulation
        # WHY spectral_norm: an ALTERNATIVE way to enforce the critic's 1-Lipschitz constraint.
        #      WGAN-GP penalises the gradient norm (a soft, sampled constraint); spectral
        #      normalisation bounds each layer's spectral norm directly (a hard, per-layer
        #      constraint). Switching GP->SN isolates whether the critic's underperformance comes
        #      from the Lipschitz-enforcement mechanism. When True, the PLAN must skip the gradient
        #      penalty (SN already constrains Lipschitz); the two must not be stacked.
        self.spectral_norm = spectral_norm
        _sn = (lambda m: nn.utils.parametrizations.spectral_norm(m)) if spectral_norm else (lambda m: m)
        # Define layers
        self.fc1 = _sn(nn.Linear(n_input, n_hidden))
        self.fc2 = _sn(nn.Linear(n_hidden, n_hidden))
        self.fc3 = _sn(nn.Linear(n_hidden, domain_number))

        # Barycenter formulation: the target is the free-support Wasserstein barycenter of the
        # batch distributions, computed by the CALLER (wcd.barycenter) from the current latents
        # and passed to forward() as `target_samples`. It is not a learnable parameter
        # (2026-10-02, code review N3: the old 64 learnable anchors started at ~0 and were moved by
        # the generator optimiser, giving a near-origin fixed point / contracting quantiser).
        self.anchors = None   # kept only so legacy callers that test `anchors is None` still work

        if self.critic:
            # If using critic, use Wasserstein loss with the chosen alignment target.
            if formulation == "reference" and reference_batch is None:
                raise ValueError("Reference batch must be provided for reference formulation.")
            self.loss = ReferenceWassersteinLoss(
                reference_class=reference_batch if reference_batch is not None else -1,
                formulation=formulation,
            )
        else:
            # If not using critic, use cross-entropy loss
            self.loss = MultiClassCrossEntropy()

    def forward(self, x, batch_ids, generator=False, reference_batch=None, target_samples=None,
                with_gp=True, weights=None):
        # Forward pass through layers
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        output = self.fc3(h)

        if batch_ids is None:
            # If batch_ids is None, return the output directly
            return output

        if weights is not None:
            # Importance weights (X3, SI-21) are defined for the reference / pooled critic losses only; the
            # weighted discriminator cross-entropy is computed by the caller from the logits.
            if not isinstance(self.loss, ReferenceWassersteinLoss) or self.formulation == "barycenter":
                raise NotImplementedError("weights are supported for the reference and pooled critic losses only")
            loss = self.loss(output, batch_ids, reference_batch, weights=weights)
            gp_loss = 0.0
            if with_gp:   # the gradient penalty stays unweighted (docs/SPECS_missing_arms.md section 4)
                gp_loss = multi_class_gradient_penalty(
                    self, x, batch_ids, reference_batch=reference_batch, formulation=self.formulation,
                    target_samples=None, num_domains=self.domain_number,
                )
            return loss, gp_loss

        if isinstance(self.loss, ReferenceWassersteinLoss):
            # Compute anchor scores once for the barycenter formulation.
            target_output = None
            if self.formulation == "barycenter":
                if target_samples is None:
                    raise ValueError("barycenter critic needs target_samples (wcd.barycenter support)")
                target_samples = target_samples.detach()
                a = F.relu(self.fc1(target_samples))
                a = F.relu(self.fc2(a))
                target_output = self.fc3(a)
            discriminator_loss = self.loss(
                output, batch_ids, reference_batch, target_output=target_output
            )
        else:
            discriminator_loss = self.loss(output, batch_ids)

        gp_loss = 0.0

        if self.loss.reduction == "mean":
            discriminator_loss = discriminator_loss.mean()
        elif self.loss.reduction == "sum":
            discriminator_loss = discriminator_loss.sum()
        if self.critic and with_gp:   # the generator step does not need the penalty
            gp_loss = multi_class_gradient_penalty(
                self,
                x,
                batch_ids,
                reference_batch=reference_batch,
                formulation=self.formulation,
                target_samples=target_samples if self.formulation == "barycenter" else None,
                num_domains=self.domain_number,
            )

        return discriminator_loss, gp_loss
