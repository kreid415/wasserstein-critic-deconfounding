from dataclasses import dataclass
from typing import List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from wcd_vae.metrics import compute_metrics


# -- Config Dataclass -----------------------------------------------------
@dataclass
class VAEConfig:
    input_dim: int = 784
    latent_dim: int = 20
    encoder_hidden_dims: List[int] = (400,)
    decoder_hidden_dims: List[int] = (400,)
    activation: str = "relu"
    use_batchnorm: bool = False
    dropout: float = 0.0
    lr: float = 1e-3
    batchsize: int = 64
    num_epochs: int = 10
    weight_decay: float = 0.0
    kl_anneal_start: int = 0  # epoch to start annealing
    kl_anneal_end: int = 10  # epoch to reach full KL weight
    kl_anneal_max: float = 1.0  # maximum KL weight


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
        h = self.net(x)
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

        layers.append(nn.Linear(dims[-1], config.input_dim))
        layers.append(nn.Sigmoid())  # for binary input images
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# -- VAE Lightning Module -------------------------------------------------
class VAE(pl.LightningModule):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.val_embeddings = []
        self.val_batches = []
        self.val_cell_types = []

        self.automatic_optimization = True

    def kl_weight(self):
        # Linear annealing from kl_anneal_start to kl_anneal_end
        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        if current_epoch < self.config.kl_anneal_start:
            return 0.0
        elif current_epoch >= self.config.kl_anneal_end:
            return self.config.kl_anneal_max
        else:
            progress = (current_epoch - self.config.kl_anneal_start) / max(
                1, self.config.kl_anneal_end - self.config.kl_anneal_start
            )
            return progress * self.config.kl_anneal_max

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
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        x = x.view(x.size(0), -1)
        x_hat, _, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        kl_weight = self.kl_weight()
        loss = recon_loss + kl_weight * kl_div

        # Debug: check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(
                f"NaN or Inf loss at batch {batch_idx}: recon_loss={recon_loss.item()}, kl_div={kl_div.item()}, kl_weight={kl_weight}"
            )
            raise ValueError("Loss is NaN or Inf")

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0)
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
        return loss

    def validation_step(self, batch, batch_idx):
        x, batch_label, cell_label = batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)
        batch_label = batch_label.to(self.device)
        cell_label = cell_label.to(self.device)

        x_hat, embed, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        loss = recon_loss + kl_div

        # Store data for metrics
        self.val_embeddings.append(embed.detach().cpu())
        self.val_batches.append(batch_label.detach().cpu())
        self.val_cell_types.append(cell_label.detach().cpu())

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

        return loss

    def on_validation_epoch_end(self):
        if not self.val_embeddings:
            return

        embeddings = torch.cat(self.val_embeddings, dim=0)
        batches = torch.cat(self.val_batches, dim=0)
        cell_types = torch.cat(self.val_cell_types, dim=0)

        # Clear for next epoch
        self.val_embeddings.clear()
        self.val_batches.clear()
        self.val_cell_types.clear()

        # Compute metrics (import compute_metrics at top of file)
        metrics = compute_metrics(
            embeddings=embeddings,
            batch_labels=batches,
            cell_type_labels=cell_types,
        )

        # Log metrics
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
