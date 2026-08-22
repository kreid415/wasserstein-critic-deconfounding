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

from wcd_vae.wcd.primitives import gaussian_sample, nb_log_likelihood
from wcd_vae.wcd.scvi_backbone import LinearSCVIBackbone as _LinearSCVIBackbone


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
        z = gaussian_sample(q_m, q_v)
        return q_m, q_v, z


def _kl_standard_normal(q_m, q_v):
    return kl(Normal(q_m, torch.sqrt(q_v)), Normal(torch.zeros_like(q_m), torch.ones_like(q_v))).sum(dim=1)



# =============================================================================
# De-scCRAFT multi-VAE study (rebuilt E2): each backbone is a NATIVE VAE trained
# with only its own reconstruction likelihood + KL. The scCRAFT auxiliary losses
# (cosine, triplet) are NOT applied here -- they are backbone-owned and every VAE
# below declares an empty aux set, so the ONLY globally-fixed component across
# backbones is the adversarial head. This isolates the head as the variable and
# makes each backbone a model people actually use.
#
# Two crossed axes:
#   likelihood     : Gaussian | Poisson | NB | ZINB  (+ LDVAE = NB, linear decoder)
#   conditioning   : whether the decoder receives the batch one-hot (decoder-side
#                    batch correction) or not (adversary is the only mixing route).
# Every backbone exposes:
#   .conditioned                     -> bool
#   .aux_losses  (class attribute)   -> tuple of aux-loss names it natively uses ()
#   .encoder(x, warmup)              -> (q_m, q_v, z)
#   forward(x, x_raw, ec, warmup)    -> (reconst_loss[N,G], kl[N], z[N,dz], x_tilde[N,G])
# =============================================================================


def _dec_input(z, ec, conditioned):
    """Concatenate the batch one-hot only when the decoder is batch-conditioned."""
    return torch.cat((z, ec), dim=-1) if conditioned else z


class _NativeVAE(nn.Module):
    """Base for the native (non-scCRAFT) backbones. Native training objective is
    reconstruction + KL only; no cosine/triplet."""

    aux_losses = ()  # names of scCRAFT-style auxiliary losses this backbone uses

    def __init__(self, p_dim, v_dim, latent_dim, conditioned=True):
        super().__init__()
        self.conditioned = conditioned
        self.p_dim, self.v_dim, self.latent_dim = p_dim, v_dim, latent_dim
        self.encoder = _Encoder(p_dim, latent_dim)
        self._dec_in = latent_dim + (v_dim if conditioned else 0)

    def _trunk(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
        )


class GaussianVAE(_NativeVAE):
    """Gaussian VAE: MSE reconstruction on log-normalised X. Simplest control."""

    def __init__(self, p_dim, v_dim, latent_dim, conditioned=False):
        super().__init__(p_dim, v_dim, latent_dim, conditioned)
        self.decoder = nn.Sequential(
            self._trunk(self._dec_in), nn.Linear(1024, p_dim),
        )

    def forward(self, x, x_raw, ec, warmup):
        q_m, q_v, z = self.encoder(x, warmup)
        x_hat = self.decoder(_dec_input(z, ec, self.conditioned))
        reconst_loss = (x_hat - x) ** 2
        kl_div = _kl_standard_normal(q_m, q_v)
        x_tilde = torch.clamp(torch.expm1(F.relu(x_hat)), max=1e6)
        return reconst_loss, kl_div, z, x_tilde


class PoissonVAE(_NativeVAE):
    """Poisson VAE: simplest count likelihood (single rate parameter per gene)."""

    def __init__(self, p_dim, v_dim, latent_dim, conditioned=True):
        super().__init__(p_dim, v_dim, latent_dim, conditioned)
        self.decoder = self._trunk(self._dec_in)
        self.px_rate = nn.Linear(1024, p_dim)

    def forward(self, x, x_raw, ec, warmup):
        q_m, q_v, z = self.encoder(x, warmup)
        h = self.decoder(_dec_input(z, ec, self.conditioned))
        rate = torch.exp(torch.clamp(self.px_rate(h), max=15, min=-15))
        # Poisson negative log-likelihood (drop the constant log x! term):
        reconst_loss = rate - x_raw * torch.log(rate + 1e-8)
        kl_div = _kl_standard_normal(q_m, q_v)
        return reconst_loss, kl_div, z, rate


class NBVAE(_NativeVAE):
    """Negative binomial VAE (the scVI-style count model). Deep decoder."""

    linear = False

    def __init__(self, p_dim, v_dim, latent_dim, conditioned=True):
        super().__init__(p_dim, v_dim, latent_dim, conditioned)
        if self.linear:
            # LDVAE: a single linear decoder from the (optionally conditioned) latent.
            self.decoder = None
            self.px_scale = nn.Linear(self._dec_in, p_dim)
            self.px_r = nn.Linear(self._dec_in, p_dim)
        else:
            self.decoder = self._trunk(self._dec_in)
            self.px_scale = nn.Linear(1024, p_dim)
            self.px_r = nn.Linear(1024, p_dim)

    def _decode(self, z, ec):
        d = _dec_input(z, ec, self.conditioned)
        h = d if self.linear else self.decoder(d)
        px_scale = torch.exp(torch.clamp(self.px_scale(h), max=15, min=-15))
        px_r = torch.exp(torch.clamp(self.px_r(h), max=15, min=-15))
        return px_scale, px_r

    def forward(self, x, x_raw, ec, warmup):
        q_m, q_v, z = self.encoder(x, warmup)
        px_scale, px_r = self._decode(z, ec)
        reconst_loss = -nb_log_likelihood(x_raw, px_scale, px_r)
        kl_div = _kl_standard_normal(q_m, q_v)
        return reconst_loss, kl_div, z, px_scale


class LDVAE(NBVAE):
    """Linearly-decoded NB VAE (scVI's LDVAE): NB likelihood, linear decoder."""

    linear = True


class ZinbVAE(_NativeVAE):
    """Zero-inflated NB VAE: NB decoder + per-gene dropout-logit head."""

    def __init__(self, p_dim, v_dim, latent_dim, conditioned=True):
        super().__init__(p_dim, v_dim, latent_dim, conditioned)
        self.decoder = self._trunk(self._dec_in)
        self.px_scale = nn.Linear(1024, p_dim)
        self.px_r = nn.Linear(1024, p_dim)
        self.px_dropout = nn.Linear(1024, p_dim)

    def forward(self, x, x_raw, ec, warmup):
        q_m, q_v, z = self.encoder(x, warmup)
        h = self.decoder(_dec_input(z, ec, self.conditioned))
        px_scale = torch.exp(torch.clamp(self.px_scale(h), max=15, min=-15))
        px_r = torch.exp(torch.clamp(self.px_r(h), max=15, min=-15))
        pi_logit = self.px_dropout(h)
        reconst_loss = -self._log_zinb(x_raw, px_scale, px_r, pi_logit)
        kl_div = _kl_standard_normal(q_m, q_v)
        return reconst_loss, kl_div, z, px_scale

    @staticmethod
    def _log_zinb(x, mu, theta, pi_logit, eps=1e-8):
        softplus = nn.functional.softplus
        nb_ll = nb_log_likelihood(x, mu, theta, eps=eps)
        log_nb_zero = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        case_zero = softplus(-pi_logit + log_nb_zero) - softplus(-pi_logit)
        case_nonzero = -softplus(pi_logit) + nb_ll
        return torch.where(x < eps, case_zero, case_nonzero)


# -----------------------------------------------------------------------------
# Registry: config name -> (class, kwargs). Conditioned/unconditioned variants
# are explicit configs so the harness can sweep them by name. The old scCRAFT and
# scVI_NB names are retained for backward-compatible replay of pre-pivot artifacts.
# -----------------------------------------------------------------------------
BACKBONE_CONFIGS = {
    # likelihood x conditioning grid (the rebuilt E2)
    "Gaussian":     (GaussianVAE, {"conditioned": False}),
    "Poisson":      (PoissonVAE,  {"conditioned": False}),
    "NB":           (NBVAE,       {"conditioned": True}),
    "NB_uncond":    (NBVAE,       {"conditioned": False}),
    "ZINB":         (ZinbVAE,     {"conditioned": True}),
    "ZINB_uncond":  (ZinbVAE,     {"conditioned": False}),
    "LDVAE":        (LDVAE,       {"conditioned": True}),
    "LDVAE_uncond": (LDVAE,       {"conditioned": False}),
    # Faithful scvi-tools 1.4.2 LinearSCVI (library latent + softmax composition + per-gene
    # dispersion); numeric equivalence to real scVI proven by scripts/scvi_gate_*.py. Always
    # decoder-conditioned (scVI's setup_anndata(batch_key)); pair with an adversary head to
    # ask whether an adversary adds anything on top of scVI's decoder-side conditioning.
    "LinearSCVI":   (_LinearSCVIBackbone, {"conditioned": True}),
    # DELIBERATE ABLATION (not faithful scVI -- scVI's LinearSCVI is ALWAYS decoder-conditioned):
    # the decoder gets NO batch one-hot, so the adversary is the SOLE batch-correction mechanism.
    # This isolates the adversary's contribution -- on the conditioned backbone the decoder's own
    # batch channel does the integration and confounds the critic-vs-discriminator comparison.
    # Shares the exact scVI generative machinery (library latent, softmax composition, per-gene NB)
    # and the scVI training profile; only the decoder conditioning differs.
    "LinearSCVI_uncond": (_LinearSCVIBackbone, {"conditioned": False}),
}

def build_backbone(name, p_dim, v_dim, latent_dim):
    """Factory: return an initialised backbone by config name."""
    if name not in BACKBONE_CONFIGS:
        raise KeyError(f"Unknown backbone '{name}'. Known: {sorted(BACKBONE_CONFIGS)}")
    cls, kw = BACKBONE_CONFIGS[name]
    return cls(p_dim=p_dim, v_dim=v_dim, latent_dim=latent_dim, **kw)
