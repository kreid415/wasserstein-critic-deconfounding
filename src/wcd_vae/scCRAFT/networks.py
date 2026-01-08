from typing import Union

import torch
from torch.autograd import grad
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

torch.backends.cudnn.benchmark = True

# Net + Loss function


def log_nb_positive(
    x: Union[torch.Tensor],
    mu: Union[torch.Tensor],
    theta: Union[torch.Tensor],
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
):
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
        lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


class Encoder(nn.Module):
    def __init__(self, p_dim, latent_dim):
        super().__init__()
        # Define the architecture
        self.fc1 = nn.Linear(p_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mean = nn.Linear(512, latent_dim)  # Output layer for mean
        self.fc_var = nn.Linear(512, latent_dim)  # Output layer for variance
        # self.fc_library = nn.Linear(512, 1)        # Output layer for library size
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x, warmup):
        # Forward pass through the network
        x = self.fc1(x)
        x = self.relu(self.bn1(x))

        x = self.fc2(x)
        x = self.relu(self.bn2(x))

        # Separate paths for mean, variance, and library size
        q_m = self.fc_mean(x)
        q_v = torch.exp(self.fc_var(x)) + 1e-4
        # library = self.fc_library(x)  # Predicted log library size

        z = reparameterize_gaussian(q_m, q_v)

        return q_m, q_v, z


class Decoder(nn.Module):
    def __init__(self, p_dim, v_dim, latent_dim=256):
        super().__init__()
        self.relu = nn.ReLU()

        # Main decoder pathway
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + v_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, p_dim),
        )

        # Additional pathway for the batch effect (ec)
        self.decoder_ec = nn.Sequential(
            nn.Linear(v_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, p_dim),
        )

        # Parameters for ZINB distribution
        self.px_scale_decoder = nn.Linear(p_dim, p_dim)  # mean (rate) of ZINB
        self.px_r_decoder = nn.Linear(p_dim, p_dim)  # dispersion

    def forward(self, z, ec):
        # Main decoding
        z_ec = torch.cat((z, ec), dim=-1)
        decoded = self.decoder(z_ec)
        decoded_ec = self.decoder_ec(ec)

        # Combining outputs
        combined = self.relu(decoded + decoded_ec)

        # NB parameters with safe exponential

        px_scale = torch.exp(self.px_scale_decoder(combined))
        px_r = torch.exp(self.px_r_decoder(combined))

        # Scale the mean (px_scale) with the predicted library size
        px_rate = px_scale

        return px_rate, px_r


class VAE(nn.Module):
    def __init__(self, p_dim, v_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(p_dim, latent_dim)
        self.decoder = Decoder(p_dim, v_dim, latent_dim)

    def forward(self, x, ec, warmup):
        # Encoding
        q_m, q_v, z = self.encoder(x, warmup)

        # Decoding
        px_scale, px_r = self.decoder(z, ec)

        # Reconstruction Loss
        # reconst_loss = F.mse_loss(px_scale, x)
        reconst_loss = -log_nb_positive(x, px_scale, px_r)
        # KL Divergence
        mean = torch.zeros_like(q_m)
        scale = torch.ones_like(q_v)
        kl_divergence = kl(Normal(q_m, torch.sqrt(q_v)), Normal(mean, scale)).sum(dim=1)

        return reconst_loss, kl_divergence, z, px_scale


class CrossEntropy(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, output, target):
        # Apply log softmax to the output
        log_preds = F.log_softmax(output, dim=-1)

        # Compute the negative log likelihood loss
        loss = F.nll_loss(log_preds, target, reduction=self.reduction)

        return loss


class WassersteinLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, output, batch_ids):
        source = output[batch_ids != 0]
        target = output[batch_ids == 0]

        # Compute the Wasserstein loss
        loss = -1 * target.mean() + source.mean()

        return loss


class MultiClassWassersteinLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, output, batch_ids):
        """
        Args:
            output: Tensor of shape [B, K] - critic scores for each class.
            batch_ids: Tensor of shape [B] - true domain IDs (0 to K-1).
        Returns:
            Wasserstein loss encouraging domain confusion.
        """
        num_domains = output.shape[1]  # number of classes/domains
        loss = 0.0
        total = 0

        for k in range(num_domains):
            mask_k = batch_ids == k
            if mask_k.sum() == 0:
                continue

            # Scores for domain k samples from the k-th output head
            d_kk = output[mask_k, k]  # true class head for samples from domain k

            # Scores for domain k samples from all other heads
            d_kj = output[mask_k]  # [n_k, K]
            mask_other = torch.ones(num_domains, dtype=torch.bool, device=output.device)
            mask_other[k] = False
            d_kj_others = d_kj[:, mask_other]  # [n_k, K-1]

            # Wasserstein-style loss: true score - mean of other scores
            diff = d_kk.mean() - d_kj_others.mean()
            loss += diff
            total += 1

        loss = loss / total

        return loss


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
    b, latent_dim = z.shape
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
                self.loss = MultiClassWassersteinLoss()
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
