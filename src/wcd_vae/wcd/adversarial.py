"""Adversarial head with discriminator/critic dispatch — AUTHORED (K. Reid).

# WHY: Hold one network architecture fixed while switching only the adversarial
#      objective, so the discriminator-vs-critic comparison is controlled.
# HOW: A single MLP head (fc1->fc2->fc3) whose loss is either V-way CrossEntropy
#      (JS-divergence discriminator) or the reference Wasserstein critic, selected
#      by the ``critic`` flag; the WGAN gradient penalty is applied only in critic mode.
Modified from scCRAFT's `discriminator` (renamed `Discriminator`); the critic branch,
reference plumbing, and gradient-penalty call are authored additions.
"""

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
    ):
        super().__init__()
        n_hidden = 128
        self.critic = critic
        # Define layers
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, domain_number)

        if self.critic:
            # If using critic, use Wasserstein loss
            if reference_batch is not None:
                self.loss = ReferenceWassersteinLoss(reference_class=reference_batch)
            else:
                raise ValueError("Reference batch must be provided for Wasserstein loss.")
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

        if isinstance(self.loss, ReferenceWassersteinLoss) and reference_batch is not None:
            discriminator_loss = self.loss(output, batch_ids, reference_batch)
        else:
            discriminator_loss = self.loss(output, batch_ids)

        gp_loss = 0.0

        if self.loss.reduction == "mean":
            discriminator_loss = discriminator_loss.mean()
        elif self.loss.reduction == "sum":
            discriminator_loss = discriminator_loss.sum()
        if self.critic:
            gp_loss = multi_class_gradient_penalty(
                self, x, batch_ids, reference_batch=reference_batch
            )

        return discriminator_loss, gp_loss
