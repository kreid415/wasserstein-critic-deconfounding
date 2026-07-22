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

    def __init__(
        self, reference_class: int = -1, reduction: str = "mean", formulation: str = "reference"
    ):
        """
        Args:
            reference_class (int): Default reference class. Set to -1 if reference
                                   is determined dynamically per batch.
            formulation (str): Which target distribution each critic head aligns to.
                "reference"  - a single designated reference batch (original design).
                "pooled"     - the global pool of all cells (no privileged batch); the
                               V-way joint counterpart to the discriminator.
                "barycenter" - a learnable set of anchor points approximating the
                               Wasserstein (Fréchet) barycenter of all batches.
        """
        super().__init__()
        self.reference_class = reference_class
        self.reduction = reduction
        if formulation not in ("reference", "pooled", "barycenter"):
            raise ValueError(f"Unknown formulation: {formulation}")
        self.formulation = formulation

    def forward(
        self,
        output: torch.Tensor,
        batch_ids: torch.Tensor,
        reference_batch=None,
        target_output: torch.Tensor = None,
    ) -> torch.Tensor:
        """Wasserstein-1 dual objective; the alignment TARGET depends on ``formulation``.

        For every critic head ``k`` we maximise ``E[C_k(P_k)] - E[C_k(T)]`` where the
        target distribution ``T`` is:
          * ``reference``  - samples from the designated reference batch (excludes head=ref);
          * ``pooled``     - all cells in the mini-batch (the global pool);
          * ``barycenter`` - the learnable anchor points, whose critic scores are supplied
                             in ``target_output`` (shape [M, num_domains]).
        The negative is returned because optimisers minimise.
        """
        num_domains = output.shape[1]

        # 1. Resolve the target scores (shape [*, num_domains]) and the head to skip.
        skip_idx = -1  # no head skipped unless a reference is used
        if self.formulation == "reference":
            active_ref_idx = (
                reference_batch if reference_batch is not None else self.reference_class
            )
            if not (0 <= active_ref_idx < num_domains):
                raise ValueError(
                    f"Invalid reference index: {active_ref_idx}. "
                    "Ensure reference_batch is passed if reference_class is -1."
                )
            mask_ref = batch_ids == active_ref_idx
            if mask_ref.sum() == 0:
                return torch.tensor(0.0, device=output.device, requires_grad=True)
            target_scores = output[mask_ref]  # [N_ref, V]
            skip_idx = active_ref_idx
        elif self.formulation == "pooled":
            target_scores = output  # the whole mini-batch is the pooled target
        else:  # barycenter
            if target_output is None:
                raise ValueError(
                    "formulation='barycenter' requires target_output (anchor scores)."
                )
            target_scores = target_output  # [M, V]

        total_loss = 0.0
        pairs_calculated = 0

        for k in range(num_domains):
            if k == skip_idx:
                continue

            mask_k = batch_ids == k
            if mask_k.sum() == 0:
                continue

            # Critic head k on samples from domain k
            scores_k_on_head_k = output[mask_k, k]

            # Critic head k on the target distribution
            scores_target_on_head_k = target_scores[:, k]

            # Maximize distance: E[Critic(Source)] - E[Critic(Target)]
            # We return negative because optimizers minimize.
            diff = scores_k_on_head_k.mean() - scores_target_on_head_k.mean()

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


def multi_class_gradient_penalty(
    critic,
    z,
    batch_ids,
    lambda_gp=10.0,
    reference_batch=None,
    formulation="reference",
    target_samples=None,
):
    """
    Computes GP interpolating between each batch K and the alignment TARGET.

    The target latent samples depend on ``formulation``:
      * ``reference``  - the designated reference batch's samples (head=ref skipped);
      * ``pooled``     - the whole mini-batch z (the global pool);
      * ``barycenter`` - the learnable anchor points passed in ``target_samples``.
    """
    _b, _latent_dim = z.shape
    critic_out = critic(z, batch_ids=None).shape[1]
    gp_total = 0.0
    device = z.device
    total_classes = 0

    # 1. Identify the target samples to interpolate toward, and the head to skip.
    skip_idx = -1
    if formulation == "barycenter":
        if target_samples is None:
            raise ValueError("formulation='barycenter' requires target_samples (anchors).")
        ref_samples = target_samples
    elif formulation == "pooled" or reference_batch is None:
        # pooled: interpolate every batch toward the global pool
        ref_samples = z
    else:
        mask_ref = batch_ids == reference_batch
        ref_samples = z[mask_ref]
        skip_idx = reference_batch
        # If no reference samples in this mini-batch, we can't compute valid GP
        if ref_samples.size(0) == 0:
            return torch.tensor(0.0, device=device)

    for k in range(critic_out):
        # Skip if k is the reference batch (no need to separate ref from ref)
        if k == skip_idx:
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
