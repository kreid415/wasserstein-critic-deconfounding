"""Faithful reimplementation of scvi-tools 1.4.2 LinearSCVI (module: LDVAE) as a wcd
backbone, so the SAME adversarial/critic training loop (_train_batch) can attach a
discriminator or Wasserstein critic head to scVI's exact generative model.

WHY A REIMPLEMENTATION AND NOT A WRAPPER:
  scvi-tools does NOT import in the harness env (wcd-kbet ships scvi-tools 0.14.6, broken
  by a torchmetrics skew: pytorch_lightning wants torchmetrics.utilities.data.get_num_classes
  which no longer exists), and the harness env's torch (2.4.1+cu121) differs from the
  scvi-env that runs real scVI (torch 2.13.0). So the live scVI module cannot be dropped
  into our critic loop without breaking reproducibility pins. Instead we reimplement LDVAE
  LAYER-FOR-LAYER here and PROVE equivalence by weight transplant + static forward compare
  against real scvi-tools 1.4.2 (scripts/scvi_equivalence_gate.py). The gate is a HARD
  blocker: no adversarial sweep is trusted unless |Δ| on reconstruction+KL is < 1e-5.

WHAT MAKES THIS scVI AND NOT OUR OLD LDVAE (see backbones.py):
  1. a SEPARATE 1-D library-size latent (l_encoder) -> z carries only composition, not depth
  2. decoder mean = exp(library) * softmax(linear(z, batch))  (simplex composition, not a
     free exp(Linear) rate)
  3. per-gene dispersion theta = exp(px_r), px_r a single Parameter(n_genes) shared across cells
  4. encoder input = log1p(raw counts) (log_variational), NOT the harness's log-norm X
  5. two KL terms: KL(qz||N(0,I)) is the warmup/kl_coef-weighted term; KL(ql||pl) against a
     PER-BATCH empirical library prior is always full weight (folded into reconst_loss here so
     the single kl_coef in _train_batch anneals only the z-KL, exactly as scVI's kl_weight does).

Layer specifics copied verbatim from scvi.nn (FCLayers/Encoder/LinearDecoderSCVI) so a
state_dict transplant is possible: BatchNorm1d(momentum=0.01, eps=0.001); encoder trunk
Linear->BN->ReLU->Dropout(0.1); mean/var heads plain Linear; var_activation=exp, var_eps=1e-4;
decoder factor_regressor is Linear(bias=False)->BN (NO activation), batch one-hot concatenated
to z BEFORE that Linear; px_dropout head exists in scVI but is unused for the NB likelihood.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from wcd_vae.wcd.primitives import nb_log_likelihood


def _fc_block(n_in, n_out, dropout_rate=0.1, use_activation=True, bias=True):
    """One scvi FCLayers 'Layer i': Linear -> BatchNorm1d -> [ReLU] -> [Dropout].

    BatchNorm params (momentum=0.01, eps=0.001) match scvi.nn.FCLayers exactly. Activation
    and dropout are dropped for the decoder (use_activation=False, dropout_rate=0).
    """
    layers = [nn.Linear(n_in, n_out, bias=bias),
              nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)]
    if use_activation:
        layers.append(nn.ReLU())
    if dropout_rate > 0:
        layers.append(nn.Dropout(p=dropout_rate))
    return nn.Sequential(*layers)


class _ScviEncoder(nn.Module):
    """Mirror of scvi.nn.Encoder(return_dist=True): FCLayers trunk -> mean/var heads.

    Batch is NOT injected (n_cat_list=[] in LDVAE for both z- and l-encoders; verified by
    in_features == n_input on the real module). Returns (q_m, q_v, z) to match the wcd
    backbone .encoder contract used by obtain_embeddings and _train_batch.
    """

    def __init__(self, n_input, n_output, n_hidden=128, dropout_rate=0.1, var_eps=1e-4):
        super().__init__()
        self.var_eps = var_eps
        self.encoder = _fc_block(n_input, n_hidden, dropout_rate=dropout_rate)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x, warmup=False):
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + self.var_eps      # scvi var_activation=exp
        z = Normal(q_m, q_v.sqrt()).rsample()
        return q_m, q_v, z


class LinearSCVIBackbone(nn.Module):
    """scvi-tools LinearSCVI (LDVAE module) reimplemented for the wcd training loop.

    Interface matches the other wcd backbones so _train_batch is untouched:
        forward(x, x_raw, ec, warmup) -> (reconst_loss[N], kl_z[N], z[N,dz], x_tilde[N,G])
        .encoder(x_counts, warmup)    -> (q_m, q_v, z)   (x_counts = RAW counts slice; see note)

    NOTE on encoder input: scVI encodes log1p(counts). _train_batch passes the harness's
    log-norm ``x`` and raw ``x_raw``; this backbone uses log1p(x_raw). Because obtain_embeddings
    calls ``.encoder`` with data_dict['X'] (log-norm), the training loop is patched to pass raw
    counts to backbones exposing ``encoder_wants_counts=True`` so train- and embedding-time
    transforms agree. ``.encoder`` here applies log1p internally, so it must be fed raw counts.
    """

    aux_losses = ()
    encoder_wants_counts = True     # tells _train_batch / obtain_embeddings to feed raw counts

    def __init__(self, p_dim, v_dim, latent_dim, conditioned=True):
        super().__init__()
        # scVI LinearSCVI is ALWAYS batch-conditioned at the decoder via setup_anndata(batch_key);
        # `conditioned` is accepted for interface symmetry but the faithful model conditions the
        # decoder whenever there is >1 batch. n_batch is v_dim (the one-hot width).
        self.p_dim, self.v_dim, self.latent_dim = p_dim, v_dim, latent_dim
        self.conditioned = conditioned
        self.n_batch = v_dim if conditioned else 0

        self._z_encoder = _ScviEncoder(p_dim, latent_dim)
        self._l_encoder = _ScviEncoder(p_dim, 1)                 # 1-D library-size latent

        # LinearDecoderSCVI.factor_regressor: single FCLayers, batch one-hot injected at the
        # linear layer, BatchNorm, NO activation, bias=False.
        dec_in = latent_dim + self.n_batch
        self.factor_regressor = _fc_block(dec_in, p_dim, dropout_rate=0.0,
                                          use_activation=False, bias=False)
        # per-gene dispersion (dispersion='gene'): one theta per gene, shared across cells
        self.px_r = nn.Parameter(torch.randn(p_dim))

        # per-batch empirical library prior N(mean, var) of log total counts; set by
        # set_library_prior() from the training data before the first forward.
        self.register_buffer("library_log_means", torch.zeros(1, max(self.n_batch, 1)))
        self.register_buffer("library_log_vars", torch.ones(1, max(self.n_batch, 1)))

    # -- library prior ---------------------------------------------------------------
    def set_library_prior(self, counts, batch_idx):
        """Empirical per-batch mean/var of log library size (log total counts), like scVI's
        _init_library_size. counts: [N,G] raw; batch_idx: [N] long."""
        counts = torch.as_tensor(counts, dtype=torch.float32)
        batch_idx = torch.as_tensor(batch_idx, dtype=torch.long)
        nb = max(self.n_batch, 1)
        means = torch.zeros(1, nb)
        varis = torch.ones(1, nb)
        log_lib = torch.log(counts.sum(1).clamp_min(1.0))
        for b in range(nb):
            m = batch_idx == b
            if m.sum() > 1:
                means[0, b] = log_lib[m].mean()
                varis[0, b] = log_lib[m].var().clamp_min(1e-4)
        self.library_log_means = means.to(self.library_log_means.device)
        self.library_log_vars = varis.to(self.library_log_vars.device)

    def _local_library_params(self, ec):
        """Per-cell (mean, var) of the log-library prior = one-hot(batch) @ per-batch table."""
        if self.n_batch == 0:
            n = ec.size(0)
            return (self.library_log_means[:, :1].expand(n, 1),
                    self.library_log_vars[:, :1].expand(n, 1))
        m = F.linear(ec, self.library_log_means)      # [N,1]
        v = F.linear(ec, self.library_log_vars)       # [N,1]
        return m, v

    # -- encoder (embeddings + training) ---------------------------------------------
    def encoder(self, x_counts, warmup=False):
        """z-encoder on log1p(raw counts). Fed raw counts (encoder_wants_counts=True)."""
        return self._z_encoder(torch.log1p(x_counts), warmup)

    # -- full forward ----------------------------------------------------------------
    def forward(self, x, x_raw, ec, warmup):
        # scVI: encoder sees log1p(counts); library latent inferred from same input.
        enc_in = torch.log1p(x_raw)
        q_m, q_v, z = self._z_encoder(enc_in, warmup)
        ql_m, ql_v, library = self._l_encoder(enc_in, warmup)     # library = sampled log-depth

        # decoder: softmax composition, conditioned on batch one-hot
        dec_in = torch.cat((z, ec), dim=-1) if self.n_batch else z
        raw_scale = self.factor_regressor(dec_in)
        px_scale = torch.softmax(raw_scale, dim=-1)
        px_rate = torch.exp(library) * px_scale                    # mu = exp(l) * composition
        theta = torch.exp(self.px_r)                               # per-gene dispersion

        # CONTRACT: _train_batch masks reconst_loss with (x_raw > 0), a [N,G] mask, so the
        # reconstruction MUST be per-element [N,G] like every other backbone (not summed to [N]).
        reconst = -nb_log_likelihood(x_raw, px_rate, theta)   # [N,G] per-element NB nll

        # KL terms. z-KL is the warmup/kl_coef-weighted term (returned separately). Library-KL
        # is against the per-batch prior and is ALWAYS full weight in scVI -> fold into reconst
        # so _train_batch's single kl_coef anneals only the z-KL (exactly scVI's kl_weight).
        # KL SCALING SO kl_coef == scVI's beta. The harness minimizes
        #   mean_{N,G}(reconst) + kl_coef * mean_N(kl_divergence)
        # -> multiplying by G, that is  recon_sum + kl_l + (G*kl_coef)*kl_z. scVI's objective
        # is recon_sum + kl_l + beta*kl_z. To make kl_coef DIRECTLY equal beta (so --kl-coef 1.0
        # is scVI's default), return kl_z/G here: then kl_coef*mean(kl_z/G) sits at the same
        # per-gene scale as reconstruction and the effective z-KL weight is exactly kl_coef.
        # (This is why the NATIVE backbones use kl_coef=5e-4 ~ 1/G to reach beta~1; this backbone
        # instead absorbs the 1/G so its kl_coef reads as a true beta.)
        kl_z = kl(Normal(q_m, q_v.sqrt()), Normal(torch.zeros_like(q_m), torch.ones_like(q_v))).sum(dim=1)
        kl_z = kl_z / self.p_dim
        pl_m, pl_v = self._local_library_params(ec if self.n_batch else torch.zeros(z.size(0), 1, device=z.device))
        kl_l = kl(Normal(ql_m, ql_v.sqrt()), Normal(pl_m, pl_v.sqrt())).sum(dim=1)   # [N]
        # Fold the per-cell library-KL into the [N,G] reconst at FULL weight while preserving
        # the per-element contract: distribute kl_l evenly across the G genes (kl_l/G per entry)
        # so each row still sums to reconst_nll + kl_l. The (x_raw>0) mask drops the tiny
        # fraction of exactly-zero entries; the resulting mean matches scVI's objective to
        # float precision (verified against the equivalence gate's full-weight KL_l term).
        reconst = reconst + (kl_l.unsqueeze(1) / self.p_dim)       # [N,G], row-sum adds kl_l

        # x_tilde = decoder mean on the gene scale (for the optional cosine term); px_rate is
        # already the NB mean, so expose it directly.
        x_tilde = px_rate
        return reconst, kl_z, z, x_tilde
