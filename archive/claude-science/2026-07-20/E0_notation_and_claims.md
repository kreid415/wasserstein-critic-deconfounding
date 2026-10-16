# E0 — Notation fixes and claim narrowing (tracked changes)

This document lists the manuscript edits that address the reviewers' notation
comments (R1.minor.3) and the claim-scope concerns (R2.1, R2.3, R2.5, R3.1, R3.3).
Each notation correction is grounded in the actual implementation (function named
in parentheses); the claim edits follow a three-tier scoping scheme so the paper
claims exactly what its experiments establish — no more, no less.

---

## Part A — Notation corrections (R1.minor.3)

The reviewer flagged three equations. In every case the *implemented* loss is
well-defined; the manuscript's typeset form was ambiguous or inconsistent with it.
Ground truth is the code in `scCRAFT/networks.py` (`log_nb_positive`, `CrossEntropy`)
and `wcd/training.py` (`_train_batch`).

### A.1 Reconstruction loss — negative binomial

**Flagged:** `P_{NB}(x | mu_g, r_g)` — subscripts undefined; `r_g` vs dispersion unclear;
reads as a probability rather than a (log-)likelihood loss.

**Implemented (`log_nb_positive`):** the per-gene NB log-likelihood with mean
`mu` (decoder `px_scale`) and **inverse-dispersion** `theta` (decoder `px_r`), summed
over genes and averaged over the minibatch, entering the objective with a minus sign.

**Corrected form.** For cell `n`, gene `g`, with observed raw counts `x_{ng}`, decoded
mean `mu_{ng} > 0`, and inverse-dispersion `theta_g > 0`:

$$
\log p_{\mathrm{NB}}(x_{ng}\mid \mu_{ng},\theta_g)
= \theta_g\big(\log\theta_g - \log(\theta_g+\mu_{ng})\big)
+ x_{ng}\big(\log\mu_{ng} - \log(\theta_g+\mu_{ng})\big)
+ \log\frac{\Gamma(x_{ng}+\theta_g)}{\Gamma(\theta_g)\,\Gamma(x_{ng}+1)} .
$$

$$
\mathcal{L}_{\mathrm{recon}}
= -\frac{1}{N}\sum_{n=1}^{N}\sum_{g=1}^{G}\log p_{\mathrm{NB}}\!\left(x_{ng}\mid \mu_{ng},\theta_g\right).
$$

Notes to fix in text: (i) use `theta` for the inverse-dispersion parameter and state
it explicitly (the code's `px_r`), not `r_g`; (ii) numerical `eps` terms are an
implementation detail, omit from the typeset equation; (iii) `mu` is produced per-cell
by the decoder from `(z, batch one-hot)`, so it carries the `n` index.

### A.2 Cosine-similarity loss

**Flagged:** `⟨x_g , log(1+mu_g)⟩` — bare inner product; missing normalisation; index `g`
used for both a gene and (apparently) a vector.

**Implemented (`_train_batch`):** `1 - mean_n < L2normalize(log1p(x_tilde_n)), L2normalize(x_n) >`,
i.e. one minus the cosine similarity between the L2-normalised **log1p decoded mean**
and the L2-normalised (already log-normalised) **input** vector, averaged over cells.

**Corrected form.** With input vector `x_n \in R^G` (log-normalised expression) and
decoded mean `mu_n \in R^G`:

$$
\mathcal{L}_{\cos}
= \frac{1}{N}\sum_{n=1}^{N}\left(1 -
\frac{\big\langle\, \log(1+\mu_{n}),\; x_{n} \,\big\rangle}
{\lVert \log(1+\mu_{n})\rVert_2\,\lVert x_{n}\rVert_2}\right).
$$

Notes: (i) the similarity is between full cell vectors, index by cell `n` not gene `g`;
(ii) both arguments are L2-normalised — this is the missing piece; (iii) it is `1 - cos`
(a loss, minimised at perfect alignment), not the raw inner product.

### A.3 Discriminator cross-entropy loss

**Flagged:** `min_{h_v} E_{z ~ q(z|x)} [L_{CE}(h_v(z), v)]` — `h_v` suggests a per-batch
head; the implemented discriminator is a single `V`-way classifier, and the expectation
should be over the data (and its batch labels), not indexed by `v`.

**Implemented (`CrossEntropy`, `Discriminator`):** a single MLP `h: R^{d_z} -> R^{V}`
producing logits over the `V` batches; standard multi-class cross-entropy
(`log_softmax` + `nll_loss`) against the true batch label `v_n`.

**Corrected form.** Let `h_\phi: R^{d_z}\to R^{V}` be the discriminator with parameters
`\phi`, `z_n \sim q(z\mid x_n)` the encoder posterior sample, and `v_n \in \{1,\dots,V\}`
the batch label of cell `n`:

$$
\min_{\phi}\;
\mathbb{E}_{x}\,\mathbb{E}_{z\sim q(z\mid x)}
\left[-\log \operatorname{softmax}\big(h_\phi(z)\big)_{v}\right]
\;=\;
\min_{\phi}\;
-\frac{1}{N}\sum_{n=1}^{N}\log
\frac{\exp\!\big(h_\phi(z_n)_{v_n}\big)}{\sum_{k=1}^{V}\exp\!\big(h_\phi(z_n)_{k}\big)} .
$$

Notes: (i) one shared classifier `h_\phi` with a `V`-dimensional output — drop the `h_v`
per-batch notation; (ii) index the sum over cells `n`; (iii) the encoder acts as the
generator in the min-max game, trained to *maximise* this loss (adversarial term
`-d_coef * loss_da` in the generator update, where `d_coef = lambda_adv`).

### A.4 (Consistency) name the adversarial weight

Throughout, the adversarial term weight is the code's `d_coef`. The manuscript should
denote it `lambda_adv` consistently and state that it multiplies the adversarial loss
in the generator objective:
`L_gen = L_recon + kl_coef*KL + triplet_coef*L_triplet + cos_coef*L_cos - lambda_adv*L_adv`.
This matters because R2.2/R3.3 (the Pareto-front request) is a sweep over `lambda_adv`.

---

## Part B — Claim narrowing (R2.1, R2.3, R2.5, R3.1)

The reviewers converged on one point: the conclusions are stated more broadly than a
single-backbone ablation licenses (R2.1, R2.3), the mechanism language (topology,
reference bottleneck) outruns the evidence (R2.3, R2.5), and the relation to mainstream
methods (scVI etc.) is not established (R3.1). Rather than a blanket "this one
implementation only" retreat, we scope each claim to exactly the tier its experiment
supports.

### Three claim tiers

- **Tier 1 — Existence (supported now, keep):** *Within a fixed VAE backbone, swapping a
  JS-divergence discriminator for a reference-based Wasserstein critic shifts the
  operating point toward stronger local batch mixing at the cost of global structure
  preservation.* Supported by the original experiments and strengthened by E1/E5/E6.
- **Tier 2 — Generality across architectures (state at exactly E2's level):** the
  existence claim is asserted to *generalise across VAE backbones* **only** to the extent
  E2 shows it reproducing across the four backbones (scCRAFT / scVI-NB / Gaussian / ZINB).
  If it reproduces on all four, we say "across the VAE backbones tested"; if it flips on
  any, we report that boundary explicitly. No claim beyond the tested set.
- **Tier 3 — Universality across all integration methods (permanently capped):** we do
  **not** claim the trade-off is intrinsic to Wasserstein-vs-JS objectives in general, nor
  that it dictates the behaviour of non-adversarial methods. E3 (Harmony/scVI/scANVI/
  Scanorama) *contextualises* our methods against mainstream tools but does not license a
  universal claim.

### Specific text edits

| Location (reviewer) | Current (too broad) | Revised (scoped) |
|---|---|---|
| Title / abstract (R2.1) | "Wasserstein critics vs discriminators for single-cell integration" as method classes | Add scope: "…within a controlled VAE backbone", and frame as a controlled ablation whose generality is tested across backbones (E2). |
| Mechanism (R2.3, R2.5) | "topological bottleneck" / "reference bottleneck" as established cause | Present as a *hypothesis*, explicitly tested: E4 isolates the reference design (fixed vs rotating vs joint), E6 ties the local/global mismatch to collapse. State mechanism only to the level E4/E6 support. |
| Reference-based failure (R2.5) | attributed to "Wasserstein critics" | attribute to *the reference-based formulation* unless E4's joint/all-pairs critic also shows it; separate the objective from the reference design. |
| Multibatch degradation (R1.minor.1) | claimed from iLISI alone | re-state with the full metric suite (E8); soften/qualify if the full suite does not reproduce a monotone degradation. |
| Relation to SOTA (R3.1) | implicit generality | add explicit paragraph placing the adversarial methods relative to Harmony/scVI/scANVI/Scanorama (E3), with the disjoint-support premise (E6) as the condition under which the Wasserstein objective is motivated. |
| Practical guidance (R3.3) | none | add a "when to use which" paragraph derived from the Pareto fronts (E1) and the disjoint-support analysis (E6): prefer the critic when batches have little support overlap and local mixing is the priority; prefer the discriminator (or a low `lambda_adv` critic) when conservation of global/rare structure matters. |

### One-line framing for the response letter

> We have reframed the paper as a *controlled ablation* whose central trade-off claim is
> (i) established within a fixed backbone, (ii) tested for generality across four VAE
> backbones, and (iii) explicitly **not** extended to a universal statement about all
> integration methods; every mechanistic statement is now tied to the specific experiment
> (E4 reference design, E6 local/global) that tests it.
