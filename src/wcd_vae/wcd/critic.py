"""Reference-based Wasserstein critic — AUTHORED CONTRIBUTION (K. Reid).

# WHY: Approximate the Earth-Mover (Wasserstein-1) distance between a designated
#      reference batch and every other batch in the latent space, as an alternative
#      to the JS-divergence discriminator whose gradients vanish under disjoint support.
# HOW: A multi-headed critic scores each non-reference batch against the reference via
#      Kantorovich potentials; the min-max game is regularised with a per-pair
#      gradient penalty (WGAN-GP) interpolating each batch toward the reference.
This module has NO upstream scCRAFT counterpart. It operates only on latent codes z,
so it is backbone-agnostic and attaches unchanged to any encoder in ``wcd.backbones``.
"""

import torch
from torch.autograd import grad
import torch.nn as nn


class ReferenceWassersteinLoss(nn.Module):
    """
    Calculates a Wasserstein-style loss between a designated reference class
    and all other classes present in a batch.
    """

    def __init__(self, reference_class: int = -1, reduction: str = "mean"):
        """
        Args:
            reference_class (int): Default reference class. Set to -1 if reference
                                   is determined dynamically per batch.
        """
        super().__init__()
        self.reference_class = reference_class
        self.reduction = reduction

    def forward(
        self, output: torch.Tensor, batch_ids: torch.Tensor, reference_batch=None
    ) -> torch.Tensor:
        num_domains = output.shape[1]

        # 1. Determine Active Reference
        # If a dynamic batch is passed, use it. Otherwise use the static init.
        active_ref_idx = reference_batch if reference_batch is not None else self.reference_class

        # Safety Check: Ensure we have a valid reference index (0 to K-1)
        if not (0 <= active_ref_idx < num_domains):
            # This catches the case where init is -1 but no dynamic batch was passed
            raise ValueError(
                f"Invalid reference index: {active_ref_idx}. "
                "Ensure reference_batch is passed if reference_class is -1."
            )

        # 2. Isolate Reference Samples
        mask_ref = batch_ids == active_ref_idx
        if mask_ref.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)

        output_ref = output[mask_ref]
        total_loss = 0.0
        pairs_calculated = 0

        for k in range(num_domains):
            if k == active_ref_idx:
                continue

            mask_k = batch_ids == k
            if mask_k.sum() == 0:
                continue

            # Critic head k on samples from domain k
            scores_k_on_head_k = output[mask_k, k]

            # Critic head k on samples from reference domain
            scores_ref_on_head_k = output_ref[:, k]

            # Maximize distance: E[Critic(Source)] - E[Critic(Ref)]
            # We return negative because optimizers minimize.
            diff = scores_k_on_head_k.mean() - scores_ref_on_head_k.mean()

            total_loss += diff
            pairs_calculated += 1

        if pairs_calculated == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)

        # 3. Reduction
        # Now dividing by the correct number of pairs (K-1)
        if self.reduction == "mean":
            return -1.0 * total_loss / pairs_calculated
        elif self.reduction == "sum":
            return -1.0 * total_loss
        else:
            return -1.0 * total_loss / pairs_calculated


def gradient_penalty(discriminator, real_samples, fake_samples, device="cpu"):
    """Computes gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, device=device)
    epsilon = epsilon.expand_as(real_samples)

    # if there is a mismatch in shape, subsample the larger tensor
    if fake_samples.shape != real_samples.shape:
        if fake_samples.shape[0] > real_samples.shape[0]:
            perm = torch.randperm(fake_samples.shape[0], device=device)
            fake_samples = fake_samples[perm[: real_samples.shape[0]]]
        elif real_samples.shape[0] > fake_samples.shape[0]:
            perm = torch.randperm(real_samples.shape[0], device=device)
            real_samples = real_samples[perm[: fake_samples.shape[0]]]

    # Interpolate between real and fake samples
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated.requires_grad_(True)

    # Forward pass
    d_interpolated = discriminator(interpolated, None, generator=True)

    # Forcing scalar output if necessary
    if d_interpolated.dim() > 1:
        d_interpolated = d_interpolated.view(-1)

    # Compute gradients w.r.t. interpolated
    gradients = grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Compute the gradient norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)

    # Compute the penalty
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


def multi_class_gradient_penalty(critic, z, batch_ids, lambda_gp=10.0, reference_batch=None):
    """
    Computes GP interpolating between specific Batch K and the Reference Batch.
    """
    _b, _latent_dim = z.shape
    critic_out = critic(z, batch_ids=None).shape[1]
    gp_total = 0.0
    device = z.device
    total_classes = 0

    # 1. Identify Reference Samples
    if reference_batch is None:
        # Fallback to random sampling if no reference provided (Original behavior)
        # Or raise error if strict compliance is needed
        ref_samples = z
    else:
        mask_ref = batch_ids == reference_batch
        ref_samples = z[mask_ref]

        # If no reference samples in this mini-batch, we can't compute valid GP
        if ref_samples.size(0) == 0:
            return torch.tensor(0.0, device=device)

    for k in range(critic_out):
        # Skip if k is the reference batch (no need to separate ref from ref)
        if reference_batch is not None and k == reference_batch:
            continue

        # Get samples from domain k
        mask_k = batch_ids == k
        if mask_k.sum() == 0:
            continue

        z_k = z[mask_k]

        # 2. Sample Correct "Real" Data (From Reference Batch Only)
        # Randomly select samples from the available reference samples to match z_k size
        rand_ref_indices = torch.randint(0, ref_samples.size(0), (z_k.size(0),), device=device)
        z_ref_subset = ref_samples[rand_ref_indices]

        # 3. Interpolate
        epsilon = torch.rand(z_k.size(0), 1, device=device)
        epsilon = epsilon.expand_as(z_k)

        # Interpolate between Batch K (z_k) and Reference (z_ref_subset)
        z_hat = epsilon * z_k + (1 - epsilon) * z_ref_subset
        z_hat.requires_grad_(True)

        # Forward pass through critic
        out = critic(z_hat, batch_ids=None)

        # We only care about the gradient of the k-th head output
        out_k = out[:, k].sum()

        # Compute gradients
        grad_k = torch.autograd.grad(
            outputs=out_k, inputs=z_hat, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        grad_norm = grad_k.view(grad_k.size(0), -1).norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()

        gp_total += gp
        total_classes += 1

    if total_classes == 0:
        return torch.tensor(0.0, device=z.device)

    return lambda_gp * gp_total / total_classes
