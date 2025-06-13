from dataclasses import dataclass
from typing import List

from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from wcd_vae.loss import wasserstein_loss


# -- Config Dataclass -----------------------------------------------------
@dataclass
class VAEConfig:
    input_dim: int
    latent_dim: int
    encoder_hidden_dims: List[int]
    decoder_hidden_dims: List[int]
    lr: float
    batchsize: int
    num_epochs: int
    weight_decay: float
    kl_anneal_end: int
    kl_anneal_max: float
    zinb: bool
    variational: bool
    linear_decoder: bool
    activation: str = "relu"
    use_batchnorm: bool = True
    dropout: float = 0.0
    decon_weight: float = 1.0
    recon_weight: float = 1.0
    kl_anneal_start: int = 0


# -- Utility --------------------------------------------------------------
def get_activation(name):
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }.get(name.lower(), nn.ReLU())


# -- Encoder Module -------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        dims = [config.input_dim] + list(config.encoder_hidden_dims)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if config.use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(dims[-1], config.latent_dim)
        self.logvar = nn.Linear(dims[-1], config.latent_dim)

    def forward(self, x):
        log_x = torch.log(x + 1)
        h = self.net(log_x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        logvar = torch.clamp(logvar, min=-4, max=4)  # Clamp for stability
        return mu, logvar


# -- Decoder Module -------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        dims = [config.latent_dim] + list(config.decoder_hidden_dims)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if config.use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
        self.net = nn.Sequential(*layers)
        # Output layers
        self.fc_mu = nn.Linear(dims[-1], config.input_dim)
        self.fc_pi = nn.Linear(dims[-1], config.input_dim)

    def forward(self, z):
        h = self.net(z)
        log_mu = self.fc_mu(h)
        pi_logit = self.fc_pi(h)
        return log_mu, pi_logit


class LinearDecoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.fc_mu = nn.Linear(config.latent_dim, config.input_dim)
        self.fc_pi = nn.Linear(config.latent_dim, config.input_dim)

    def forward(self, z):
        log_mu = self.fc_mu(z)
        pi_logit = self.fc_pi(z)
        return log_mu, pi_logit


# -- Base Class for VAE ---------------------------------------------------
class BaseVAE(pl.LightningModule):
    """
    Base class for Variational Autoencoder (VAE) models.
    Provides common functionality for encoding, decoding, and training steps.
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.encoder = Encoder(config)
        self.zinb = config.zinb
        if config.linear_decoder:
            self.decoder = LinearDecoder(config)
        else:
            self.decoder = Decoder(config)

        self.log_theta = torch.nn.Parameter(torch.randn(config.input_dim))

    def zinb_loss(self, x, log_mu, pi_logits):
        """
        x: input data
        mu: output of decoder
        dropout_logits: dropout logits of zinb distribution
        """
        theta = self.log_theta.exp()

        nb_logits = log_mu - self.log_theta

        distribution = ZeroInflatedNegativeBinomial(
            total_count=theta, logits=nb_logits, gate_logits=pi_logits, validate_args=False
        )

        loss = -distribution.log_prob(x).sum(-1).mean()

        x_hat = distribution.mean

        rmse = torch.sqrt(F.mse_loss(x_hat, x, reduction="mean"))

        return loss, rmse

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.config.variational else mu
        mu_zimb, pi_logit = self.decode(z)
        return mu_zimb, pi_logit, z, mu, logvar

    def kl_weight(self):
        # Linear annealing from kl_anneal_start to kl_anneal_end
        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        if current_epoch < self.config.kl_anneal_start or not self.config.variational:
            return 0.0
        elif current_epoch >= self.config.kl_anneal_end:
            return self.config.kl_anneal_max
        else:
            progress = (current_epoch - self.config.kl_anneal_start) / max(
                1, self.config.kl_anneal_end - self.config.kl_anneal_start
            )
            return progress * self.config.kl_anneal_max

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)


# -- VAE Lightning Module -------------------------------------------------
class VAE(BaseVAE):
    def __init__(self, config: VAEConfig):
        super().__init__(config)

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        x = x.view(x.size(0), -1)
        mu_zimb, pi_logit, _, mu, logvar = self.forward(x)
        if self.zinb:
            # Use Zero-Inflated Negative Binomial loss
            recon_loss, rmse = self.zinb_loss(x, mu_zimb, pi_logit)
        else:
            # Use standard reconstruction loss (e.g., MSE)
            recon_loss = F.mse_loss(mu_zimb, x, reduction="mean")

        recon_loss *= self.config.recon_weight / x.size(0)
        kl_weight = self.kl_weight()
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0) * kl_weight
        loss = recon_loss + kl_div

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0)
        )
        self.log(
            "recon_loss",
            recon_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "kl_div",
            kl_div,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "kl_weight",
            kl_weight,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        if self.zinb:
            self.log(
                "rmse",
                rmse,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)

        mu_zimb, pi_logit, _, mu, logvar = self.forward(x)
        if self.zinb:
            # Use Zero-Inflated Negative Binomial loss
            recon_loss, rmse = self.zinb_loss(x, mu_zimb, pi_logit)
        else:
            # Use standard reconstruction loss (e.g., MSE)
            recon_loss = F.mse_loss(mu_zimb, x, reduction="mean")

        recon_loss *= self.config.recon_weight / x.size(0)
        kl_div = (
            -0.5
            * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            / x.size(0)
            * self.config.kl_anneal_max
        )
        loss = recon_loss + kl_div

        self.log_dict(
            {
                "val_loss": loss,
                "val_recon_loss": recon_loss,
                "val_kl_div": kl_div,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        if self.zinb:
            self.log(
                "val_rmse",
                rmse,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        return loss


# -- VAE with Unbalanced OT Module ----------------------------------------
class VAE_OT(BaseVAE):
    """
    VAE with unbalanced optimal transport (unbalanced_ot) regularization on the latent space.
    This encourages the latent representations of different groups (e.g., batches) to be similar.
    """

    def __init__(self, config: VAEConfig, domain_loss, ot_lambda=1.0):
        super().__init__(config)
        self.domain_loss = domain_loss

    def training_step(self, batch, batch_idx):
        x, batch_label, _ = batch
        x = x.view(x.size(0), -1)
        mu_zimb, mu_theta, mu_pi, _, mu, logvar = self.forward(x)
        recon_loss = (
            self.zinb_loss(x, mu_zimb, mu_theta, mu_pi, eps=1e-8)
            * self.config.recon_weight
            / x.size(0)
        )
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        kl_weight = self.kl_weight()
        vae_loss = recon_loss + kl_weight * kl_div

        # --- Unbalanced OT regularization ---
        # Assume batch_label is 0 or 1 (two groups)
        labels = batch_label.argmax(dim=1)
        mask_0 = labels == 0
        mask_1 = labels == 1
        mu0, logvar0 = mu[mask_0], logvar[mask_0]
        mu1, logvar1 = mu[mask_1], logvar[mask_1]

        ot_loss = 0.0
        if mu0.shape[0] > 0 and mu1.shape[0] > 0:
            ot_loss, _ = self.domain_loss(mu0, logvar0.exp(), mu1, logvar1.exp(), device=x.device)

        total_loss = vae_loss + self.ot_lambda * ot_loss

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        self.log(
            "recon_loss",
            recon_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
        )
        self.log(
            "kl_div", kl_div, on_step=True, on_epoch=True, prog_bar=False, batch_size=x.size(0)
        )
        self.log(
            "kl_weight",
            kl_weight,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
        )
        self.log(
            "ot_loss", ot_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0)
        )
        return total_loss

    # Validation step for VAE_OT
    def validation_step(self, batch, batch_idx):
        x, batch_label, _ = batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)

        x_hat, z, mu, logvar = self.forward(x)
        mu_zimb, mu_theta, mu_pi, _, mu, logvar = self.forward(x)
        recon_loss = (
            self.zinb_loss(x, mu_zimb, mu_theta, mu_pi, eps=1e-8)
            * self.config.recon_weight
            / x.size(0)
        )
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Unbalanced OT regularization
        labels = batch_label.argmax(dim=1)
        mask_0 = labels == 0
        mask_1 = labels == 1
        mu0, logvar0 = mu[mask_0], logvar[mask_0]
        mu1, logvar1 = mu[mask_1], logvar[mask_1]

        ot_loss = 0.0
        if mu0.shape[0] > 0 and mu1.shape[0] > 0:
            ot_loss, _ = self.domain_loss(mu0, logvar0.exp(), mu1, logvar1.exp(), device=x.device)

        loss = recon_loss + kl_div + self.ot_lambda * ot_loss

        self.log_dict(
            {
                "val_loss": loss,
                "val_recon_loss": recon_loss,
                "val_kl_div": kl_div,
                "val_ot_loss": ot_loss,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return loss


# -- Discriminator Module -------------------------------------------------
class Discriminator(nn.Module):
    """
    Discriminator network for adversarial regularization of VAE latent space.
    Predicts confounding variable(s) from latent representation.
    """

    def __init__(self, latent_dim, hidden_dims=[64, 32], dropout=0.1, critic=False):
        """
        Args:
            latent_dim (int): Dimension of VAE latent space (input to discriminator)
            confounder_dim (int): Number of confounder classes (output dimension)
            hidden_dims (list): List of hidden layer sizes
            dropout (float): Dropout rate
        """
        super().__init__()
        dims = [latent_dim] + hidden_dims
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        if not critic:
            layers.append(nn.Sigmoid())  # Output logits for binary classification
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        """
        Args:
            z (Tensor): Latent representation (batch_size, latent_dim)
        Returns:
            logits (Tensor): Predicted confounder logits (batch_size, confounder_dim)
        """
        return self.net(z)


# -- Base adversarial class for VAE with critic/discriminator ------
class VAEAdvBase(pl.LightningModule):
    """
    Base class for VAE with adversarial regularization using a critic or discriminator.
    Implements 3-stage training: VAE-only, discriminator-only, then alternating.
    """

    def __init__(
        self,
        config,
        vae,
        critic,
        lr_vae=1e-3,
        lr_critic=1e-4,
        lambda_critic=1.0,
        lambda_gp=10.0,
        critic_steps=5,
        vae_stage_epochs=10,
        disc_stage_epochs=10,
        total_epochs=100,
        adv_weight=1.0,
    ):
        super().__init__()
        self.config = config
        self.vae = vae
        self.critic = critic
        self.lr_vae = lr_vae
        self.lr_critic = lr_critic
        self.lambda_critic = lambda_critic
        self.lambda_gp = lambda_gp
        self.critic_steps = critic_steps
        self.vae_stage_epochs = vae_stage_epochs
        self.disc_stage_epochs = disc_stage_epochs
        self.total_epochs = total_epochs
        self.adv_weight = adv_weight
        self.automatic_optimization = False

    def current_stage(self):
        epoch = self.current_epoch
        if epoch < self.vae_stage_epochs:
            return "vae"
        elif epoch < self.vae_stage_epochs + self.disc_stage_epochs:
            return "disc"
        else:
            return "alternate"

    def configure_optimizers(self):
        opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_critic, betas=(0.5, 0.9)
        )
        opt_vae = torch.optim.Adam(self.vae.parameters(), lr=self.lr_vae)
        return [opt_critic, opt_vae], []

    # The actual training_step must be implemented in subclasses
    # using self.current_stage() to control the logic


# Example: Discriminator-based adversarial VAE
class VAEDiscriminator(VAEAdvBase):
    def training_step(self, batch, batch_idx):
        x, batch_label, _ = batch
        x = x.view(x.size(0), -1)
        mu, logvar = self.vae.encoder(x)
        z = self.vae.reparameterize(mu, logvar)
        confounder_labels = batch_label.argmax(dim=1).float()

        opt_disc, opt_vae = self.optimizers()
        stage = self.current_stage()

        if stage == "vae":
            # Only train VAE
            opt_vae.zero_grad()
            x_hat = self.vae.decode(z)
            recon_loss = F.mse_loss(x_hat, x, reduction="mean")
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            kl_weight = self.vae.kl_weight() if hasattr(self.vae, "kl_weight") else 1.0
            vae_loss = recon_loss + kl_weight * kl_div
            self.manual_backward(vae_loss)
            opt_vae.step()
            self.log("train_vae_loss", vae_loss, prog_bar=True, on_step=True, on_epoch=True)
            return vae_loss

        elif stage == "disc":
            # Only train discriminator
            opt_disc.zero_grad()
            z_detached = z.detach()
            pred = self.critic(z_detached).squeeze()
            loss_disc = F.binary_cross_entropy(pred, confounder_labels)
            self.manual_backward(loss_disc)
            opt_disc.step()
            self.log("disc_loss", loss_disc, prog_bar=True, on_step=True, on_epoch=True)
            return loss_disc

        else:  # "alternate"
            if (batch_idx % (self.critic_steps + 1)) < self.critic_steps:
                opt_disc.zero_grad()
                z_detached = z.detach()
                pred = self.critic(z_detached).squeeze()
                loss_disc = F.binary_cross_entropy(pred, confounder_labels)
                self.manual_backward(loss_disc)
                opt_disc.step()
                self.log("disc_loss", loss_disc, prog_bar=True, on_step=True, on_epoch=True)
                return loss_disc
            else:
                opt_vae.zero_grad()
                x_hat = self.vae.decode(z)
                recon_loss = F.mse_loss(x_hat, x, reduction="mean")
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                kl_weight = self.vae.kl_weight() if hasattr(self.vae, "kl_weight") else 1.0
                vae_loss = recon_loss + kl_weight * kl_div
                pred = self.critic(z).squeeze()
                adv_loss = F.binary_cross_entropy(pred, 1.0 - confounder_labels)
                total_loss = vae_loss + self.adv_weight * adv_loss
                self.manual_backward(total_loss)
                opt_vae.step()
                self.log("train_vae_loss", vae_loss, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train_adv_loss", adv_loss, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
                return total_loss


# Example: Wasserstein-based adversarial VAE
class VAEWasserstein(VAEAdvBase):
    def gradient_penalty(self, real_z, fake_z):
        batch_size = min(real_z.size(0), fake_z.size(0))
        if batch_size == 0:
            return torch.tensor(0.0, device=real_z.device)
        # Subsample to match sizes
        idx_real = torch.randperm(real_z.size(0), device=real_z.device)[:batch_size]
        idx_fake = torch.randperm(fake_z.size(0), device=fake_z.device)[:batch_size]
        real_z = real_z[idx_real]
        fake_z = fake_z[idx_fake]
        alpha = torch.rand(batch_size, 1, device=real_z.device)
        alpha = alpha.expand_as(real_z)
        interpolates = alpha * real_z + (1 - alpha) * fake_z
        interpolates.requires_grad_(True)
        critic_interpolates = self.critic(interpolates)
        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = self.lambda_gp * ((grad_norm - 1) ** 2).mean()
        return gp

    def training_step(self, batch, batch_idx):
        x, batch_label, _ = batch
        x = x.view(x.size(0), -1)
        mu, logvar = self.vae.encoder(x)
        z = self.vae.reparameterize(mu, logvar)
        labels = batch_label.argmax(dim=1)
        mask_0 = labels == 0
        mask_1 = labels == 1
        z_real = z[mask_0]
        z_fake = z[mask_1]

        opt_critic, opt_vae = self.optimizers()
        stage = self.current_stage()

        if stage == "vae":
            opt_vae.zero_grad()
            x_hat = self.vae.decode(z)
            recon_loss = F.mse_loss(x_hat, x, reduction="mean")
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            kl_weight = self.vae.kl_weight() if hasattr(self.vae, "kl_weight") else 1.0
            vae_loss = recon_loss + kl_weight * kl_div
            self.manual_backward(vae_loss)
            opt_vae.step()
            self.log("train_vae_loss", vae_loss, prog_bar=True, on_step=True, on_epoch=True)
            return vae_loss

        elif stage == "disc":
            opt_critic.zero_grad()
            z_real_detach = z_real.detach()
            z_fake_detach = z_fake.detach()
            real_score = self.critic(z_real_detach)
            fake_score = self.critic(z_fake_detach)
            y_real = torch.ones_like(real_score)
            y_fake = -torch.ones_like(fake_score)
            w_loss_real = wasserstein_loss(real_score, y_real)
            w_loss_fake = wasserstein_loss(fake_score, y_fake)
            wasserstein = w_loss_real + w_loss_fake
            gp = self.gradient_penalty(z_real_detach, z_fake_detach)
            critic_loss = wasserstein + gp
            self.manual_backward(critic_loss)
            opt_critic.step()
            self.log("train_critic_loss", critic_loss, prog_bar=True, on_step=True, on_epoch=True)
            return critic_loss

        else:  # alternate
            if (batch_idx % (self.critic_steps + 1)) < self.critic_steps:
                opt_critic.zero_grad()
                z_real_detach = z_real.detach()
                z_fake_detach = z_fake.detach()
                real_score = self.critic(z_real_detach)
                fake_score = self.critic(z_fake_detach)
                y_real = torch.ones_like(real_score)
                y_fake = -torch.ones_like(fake_score)
                w_loss_real = wasserstein_loss(real_score, y_real)
                w_loss_fake = wasserstein_loss(fake_score, y_fake)
                wasserstein = w_loss_real + w_loss_fake
                gp = self.gradient_penalty(z_real_detach, z_fake_detach)
                critic_loss = wasserstein + gp
                self.manual_backward(critic_loss)
                opt_critic.step()
                self.log(
                    "train_critic_loss", critic_loss, prog_bar=True, on_step=True, on_epoch=True
                )
                return critic_loss
            else:
                opt_vae.zero_grad()
                real_score = self.critic(z_real)
                fake_score = self.critic(z_fake)
                y_real = torch.ones_like(real_score)
                y_fake = -torch.ones_like(fake_score)
                w_loss_real = wasserstein_loss(real_score, y_real)
                w_loss_fake = wasserstein_loss(fake_score, y_fake)
                wasserstein = w_loss_real + w_loss_fake
                x_hat = self.vae.decode(z)
                recon_loss = F.mse_loss(x_hat, x, reduction="mean")
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                kl_weight = self.vae.kl_weight() if hasattr(self.vae, "kl_weight") else 1.0
                vae_loss = recon_loss + kl_weight * kl_div
                total_loss = vae_loss + wasserstein
                self.manual_backward(total_loss)
                opt_vae.step()
                self.log("train_vae_loss", vae_loss, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train_adv_loss", wasserstein, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
                return total_loss


# -- VAE with Adversarial Confounder Module ----------------------------------------
class VAEDiscriminator(VAEAdvBase):
    """
    PyTorch Lightning module for VAE with adversarial regularization using a discriminator
    to remove confounder information from the latent space.
    """

    def __init__(
        self,
        vae,
        critic,
        lr_vae=1e-3,
        lr_critic=1e-4,
        lambda_critic=1.0,
        lambda_gp=10.0,
        critic_steps=1,
    ):
        super().__init__(vae, critic, lr_vae, lr_critic, lambda_critic, lambda_gp, critic_steps)
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, batch_label, _ = batch
        x = x.view(x.size(0), -1)
        mu, logvar = self.vae.encoder(x)
        z = self.vae.reparameterize(mu, logvar)
        confounder_labels = batch_label.argmax(dim=1).long()

        opt_disc, opt_vae = self.optimizers()

        # Alternate updates: update disc more often than vae
        if (batch_idx % (self.critic_steps + 1)) < self.critic_steps:
            # --- Discriminator update ---
            opt_disc.zero_grad()
            z_detached = z.detach()
            pred = self.critic(z_detached).squeeze()
            loss_disc = (
                F.binary_cross_entropy(pred, confounder_labels.float()) * self.lambda_critic
            )
            self.manual_backward(loss_disc)
            opt_disc.step()
            self.log("disc_loss", loss_disc, prog_bar=True, on_step=True, on_epoch=True)
        else:
            # --- VAE (generator) update ---
            opt_vae.zero_grad()
            x_hat = self.vae.decode(z)
            recon_loss = F.mse_loss(x_hat, x, reduction="mean") * self.config.recon_weight
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            kl_weight = self.vae.kl_weight() if hasattr(self.vae, "kl_weight") else 1.0
            vae_loss = recon_loss + kl_weight * kl_div
            pred = self.critic(z).squeeze()
            adv_loss = (
                F.binary_cross_entropy(pred, confounder_labels.float()) * self.lambda_critic * -1
            )
            total_loss = vae_loss + adv_loss
            self.manual_backward(total_loss)
            opt_vae.step()
            self.log("train_vae_loss", vae_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_adv_loss", adv_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
            return total_loss

    def validation_step(self, batch, batch_idx):
        x, batch_label, _ = batch
        x = x.view(x.size(0), -1)
        mu, logvar = self.vae.encoder(x)
        z = self.vae.reparameterize(mu, logvar)

        # Convert one-hot to class indices for confounder prediction
        confounder_labels = batch_label.argmax(dim=1).long()

        pred = self.critic(z).squeeze()
        # For binary confounder, use BCEWithLogitsLoss
        loss_disc = F.binary_cross_entropy(pred, confounder_labels.float()) * self.lambda_critic
        self.log("disc_loss", loss_disc, prog_bar=True, on_step=True, on_epoch=True)

        x_hat = self.vae.decode(z)
        recon_loss = F.mse_loss(x_hat, x, reduction="mean") * self.config.recon_weight
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        kl_weight = self.vae.kl_weight() if hasattr(self.vae, "kl_weight") else 1.0
        vae_loss = recon_loss + kl_weight * kl_div

        # Adversarial loss: fool the discriminator (flip labels)
        pred = self.critic(z).squeeze()
        adv_loss = (
            F.binary_cross_entropy(pred, confounder_labels.float()) * self.lambda_critic * -1
        )
        total_loss = vae_loss + adv_loss

        self.log("val_vae_loss", vae_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_adv_loss", adv_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss
