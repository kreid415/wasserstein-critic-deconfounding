"""Adversarial head with discriminator/critic dispatch — AUTHORED (K. Reid).

# WHY: Hold one network architecture fixed while switching only the adversarial
#      objective, so the discriminator-vs-critic comparison is controlled.
# HOW: A single MLP head (fc1->fc2->fc3) whose loss is either V-way CrossEntropy
#      (JS-divergence discriminator) or the reference Wasserstein critic, selected
#      by the ``critic`` flag; the WGAN gradient penalty is applied only in critic mode.
Modified from scCRAFT's `discriminator` (renamed `Discriminator`); the critic branch,
reference plumbing, and gradient-penalty call are authored additions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from wcd_vae.scCRAFT.networks import CrossEntropy
from wcd_vae.wcd.critic import ReferenceWassersteinLoss, multi_class_gradient_penalty


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
    ):
        super().__init__()
        n_hidden = 128
        self.critic = critic
        self.formulation = formulation
        # Define layers
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, domain_number)

        # WHY (formulation study): the barycenter critic aligns every batch to a LEARNABLE
        #      virtual center rather than an existing batch, testing whether the critic's
        #      pathologies stem from the fixed-reference design or the Wasserstein objective.
        # HOW: M anchor points in latent space, updated by the generator optimiser toward
        #      the Frechet mean of the batch distributions.
        self.anchors = None
        if self.critic and formulation == "barycenter":
            self.anchors = nn.Parameter(torch.randn(n_anchors, n_input) * 0.01)

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
            self.loss = CrossEntropy()

    def forward(self, x, batch_ids, generator=False, reference_batch=None):
        # Forward pass through layers
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        output = self.fc3(h)

        if batch_ids is None:
            # If batch_ids is None, return the output directly
            return output

        if isinstance(self.loss, ReferenceWassersteinLoss):
            # Compute anchor scores once for the barycenter formulation.
            target_output = None
            if self.formulation == "barycenter":
                a = F.relu(self.fc1(self.anchors))
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
        if self.critic:
            gp_loss = multi_class_gradient_penalty(
                self,
                x,
                batch_ids,
                reference_batch=reference_batch,
                formulation=self.formulation,
                target_samples=self.anchors if self.formulation == "barycenter" else None,
            )

        return discriminator_loss, gp_loss
