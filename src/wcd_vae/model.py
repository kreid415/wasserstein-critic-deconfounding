from dataclasses import dataclass
from typing import List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


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
        last_dim = dims[-1]
        self.mu = nn.Linear(last_dim, config.latent_dim)
        self.logvar = nn.Linear(last_dim, config.latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


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
        return self.decode(z), mu, logvar

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat, mu, logvar = self.forward(x)
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_div
        self.log_dict(
            {"train_loss": loss, "recon_loss": recon_loss, "kl_div": kl_div}, prog_bar=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
