"""Alternative VAE backbones for the architecture-generalization experiment (E2).

# WHY: Reviewers R2.1/R3.1 ask whether the discriminator-vs-critic trade-off is a
#      property of the ADVERSARIAL OBJECTIVE or an artefact of the one scCRAFT backbone.
#      To answer it we hold the adversarial head byte-for-byte fixed and swap only the
#      z-producing backbone. If the trade-off reproduces across backbones it is
#      objective-driven (Tier-2 generality); if it flips it is architecture-specific.
# HOW: Every backbone implements ONE interface so the training engine (_train_batch) is
#      untouched:
#        forward(x, x_raw, ec, warmup) -> (reconst_loss[N,G], kl[N], z[N,dz], x_tilde[N,G])
#        .encoder(x, warmup) -> (q_m, q_v, z)   (used by obtain_embeddings)
#      The four backbones differ ONLY in the decoder likelihood / feature pathway:
#        - scCRAFT  : upstream NB decoder + batch-effect pathway (imported, the reference)
#        - scVI_NB  : plain NB VAE, NO batch-effect pathway, NO triplet/cosine reliance
#        - Gaussian : MSE (Gaussian) decoder on log-normalised X
#        - ZINB     : zero-inflated NB decoder
#      x_tilde is always the decoder MEAN on the gene scale so the cosine term in
#      _train_batch (1 - cos(log1p(x_tilde), x)) is well-defined for every backbone.
"""
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from wcd_vae.scCRAFT.networks import VAE as scCRAFTVAE  # noqa: N811
from wcd_vae.scCRAFT.networks import log_nb_positive, reparameterize_gaussian


class _Encoder(nn.Module):
    """Shared encoder trunk (matches scCRAFT Encoder capacity)."""

    def __init__(self, p_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(p_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x, warmup):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        q_m = self.fc_mean(x)
        q_v = torch.exp(torch.clamp(self.fc_var(x), max=15)) + 1e-4
        z = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, z


def _kl_standard_normal(q_m, q_v):
    return kl(Normal(q_m, torch.sqrt(q_v)), Normal(torch.zeros_like(q_m), torch.ones_like(q_v))).sum(dim=1)


class ScviNBVAE(nn.Module):
    """scVI-style NB VAE: NB decoder conditioned on batch, no scCRAFT batch-effect
    residual pathway. Represents the mainstream NB-VAE design (R3.1)."""

    def __init__(self, p_dim, v_dim, latent_dim):
        super().__init__()
        self.encoder = _Encoder(p_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + v_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
        )
        self.px_scale = nn.Linear(1024, p_dim)
        self.px_r = nn.Linear(1024, p_dim)

    def forward(self, x, x_raw, ec, warmup):
        q_m, q_v, z = self.encoder(x, warmup)
        h = self.decoder(torch.cat((z, ec), dim=-1))
        px_scale = torch.exp(torch.clamp(self.px_scale(h), max=15, min=-15))
        px_r = torch.exp(torch.clamp(self.px_r(h), max=15, min=-15))
        reconst_loss = -log_nb_positive(x_raw, px_scale, px_r)
        kl_div = _kl_standard_normal(q_m, q_v)
        return reconst_loss, kl_div, z, px_scale


class GaussianVAE(nn.Module):
    """Vanilla Gaussian VAE: MSE reconstruction on log-normalised X. The simplest
    possible decoder likelihood — a strong architecture-generality control."""

    def __init__(self, p_dim, v_dim, latent_dim):
        super().__init__()
        self.encoder = _Encoder(p_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + v_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, p_dim),
        )

    def forward(self, x, x_raw, ec, warmup):
        q_m, q_v, z = self.encoder(x, warmup)
        x_hat = self.decoder(torch.cat((z, ec), dim=-1))  # reconstructs log-norm X
        # per-cell,per-gene squared error; keep [N,G] shape to match the engine's mean().
        reconst_loss = (x_hat - x) ** 2
        kl_div = _kl_standard_normal(q_m, q_v)
        # x_tilde on gene scale for the cosine term: expm1 of the (nonneg) reconstruction.
        x_tilde = torch.clamp(torch.expm1(F.relu(x_hat)), max=1e6)
        return reconst_loss, kl_div, z, x_tilde


class ZinbVAE(nn.Module):
    """Zero-inflated NB VAE: NB decoder plus a per-gene dropout-logit head. Captures
    excess zeros common in droplet scRNA-seq / ATAC gene-activity."""

    def __init__(self, p_dim, v_dim, latent_dim):
        super().__init__()
        self.encoder = _Encoder(p_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + v_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
        )
        self.px_scale = nn.Linear(1024, p_dim)
        self.px_r = nn.Linear(1024, p_dim)
        self.px_dropout = nn.Linear(1024, p_dim)  # zero-inflation logits

    def forward(self, x, x_raw, ec, warmup):
        q_m, q_v, z = self.encoder(x, warmup)
        h = self.decoder(torch.cat((z, ec), dim=-1))
        px_scale = torch.exp(torch.clamp(self.px_scale(h), max=15, min=-15))
        px_r = torch.exp(torch.clamp(self.px_r(h), max=15, min=-15))
        pi_logit = self.px_dropout(h)  # logit of the zero-inflation probability
        reconst_loss = -self._log_zinb(x_raw, px_scale, px_r, pi_logit)
        kl_div = _kl_standard_normal(q_m, q_v)
        return reconst_loss, kl_div, z, px_scale

    @staticmethod
    def _log_zinb(x, mu, theta, pi_logit, eps=1e-8):
        # WHY: ZINB log-likelihood; HOW: mixture of a point mass at 0 (weight sigmoid(pi))
        #      and an NB, computed in a numerically stable log-space via softplus.
        softplus = nn.functional.softplus
        nb_ll = log_nb_positive(x, mu, theta, eps=eps)  # NB log-prob [N,G]
        # log P(x=0) under NB = theta*(log theta - log(theta+mu))
        log_nb_zero = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        # case x==0: log( sigmoid(pi) + (1-sigmoid(pi)) * NB(0) )
        #          = softplus(-pi + log_nb_zero) - softplus(-pi)   (log-sum-exp form)
        case_zero = softplus(-pi_logit + log_nb_zero) - softplus(-pi_logit)
        # case x>0 : log(1-sigmoid(pi)) + NB(x) = -softplus(pi) + nb_ll
        case_nonzero = -softplus(pi_logit) + nb_ll
        return torch.where(x < eps, case_zero, case_nonzero)


BACKBONES = {
    "scCRAFT": scCRAFTVAE,
    "scVI_NB": ScviNBVAE,
    "Gaussian": GaussianVAE,
    "ZINB": ZinbVAE,
}


def build_backbone(name, p_dim, v_dim, latent_dim):
    """Factory: return an initialised backbone by registry name."""
    if name not in BACKBONES:
        raise KeyError(f"Unknown backbone '{name}'. Known: {sorted(BACKBONES)}")
    return BACKBONES[name](p_dim=p_dim, v_dim=v_dim, latent_dim=latent_dim)
