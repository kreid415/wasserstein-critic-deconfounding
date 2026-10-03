# Specifications: the four arms the Tier 1+2 manifest still lacks (X6, X7, X8, X3)

Branch `missing-arms` from `main` @ 8cda6de, 2026-10-02. Status: **signed off by the user, 2026-10-02** (section 6).
Nothing in sections 1-4 was coded before its sign-off.

## 0. What was checked

| source | how it was checked |
|---|---|
| Mescheder, Geiger & Nowozin 2018, ICML, arXiv:1801.04406v4 | PDF from arxiv.org (sha256 a6aba4d57143), text extracted; Eq. (1), (9), (10), Theorem 4.1, App. experiments (gamma) read |
| Goodfellow et al. 2014, arXiv:1406.2661v1 | PDF (ff5819e3a7b7); Eq. (1), Prop. 1 Eq. (2), Theorem 1, Sec. 3 (non-saturating objective) |
| Tzeng et al. 2017 (ADDA), CVPR, arXiv:1702.05464v1 | PDF (4b301977f5c5); Eq. (7) inverted-label loss, Eq. (9) |
| Ganin & Lempitsky 2015, ICML, arXiv:1409.7495v2 | PDF (d8a8fd9c202a); Sec. 5 "CNN training procedure" |
| Tachet des Combes et al. 2020, NeurIPS, arXiv:2003.04475v3 | PDF (d6562c2adb37); Def. 3.4 Eq. (4), Lemma 3.1, Sec. 3.5 Eq. (6), Eq. (7), Sec. 4.1 (oracle IWDAN-O), Lemma A.1, App. B.5.3 Eq. (22) |
| scvi-tools 1.4.2 (env scvi-api) | `scvi/dataloaders/_data_splitting.py` (DataSplitter, validate_data_split: n_train = ceil(train_size n)), `_ann_dataloader.py` (BatchSampler(RandomSampler), drop_last=False), `model/base/_training_mixin.py` (`_data_splitter_cls`), `train/_trainingplans.py` (n_epochs_kl_warmup=400, epoch-based) |
| committed code @ 8cda6de | `scripts/scvi_adversarial_plan.py`, `src/wcd_vae/wcd/{critic,adversarial,alignment}.py`, `scripts/fit_paper_config.py` (subsample), `scripts/build_paper_manifest.py` (X3/X6/X7/X8 blocks) |
| data (prepped_scib/*__scib.h5ad, obs only) | X3 composition profiles: `docs/x3_shared_support_profile.csv`, `docs/x3_iw_weight_profile.csv`; X8 per-batch training counts (section 3) |

Notation. Minibatch of n cells, batch labels b_i in {0..K-1}, n_k cells of batch k in the minibatch, adversary
input z_i (posterior mean or sample, optionally standardised; `adv_input`, `zstd`, unchanged). l(z) in R^K are
the logits of the existing 3-layer MLP head (`wcd.adversarial.Discriminator`, n_hidden 128). "Discriminator
optimiser" = the existing one (Adam lr 1e-3, eps 0.01, weight decay = scvi-tools' 1e-6). Every JS arm takes
exactly one adversary step per generator step (CONSTRAINTS.md SI-01).

## 1. X6: discriminator with an R1 gradient penalty (`discriminator_r1`)

**Source.** Mescheder et al. 2018 write the GAN objective as L(theta, psi) = E_p(z)[f(D_psi(G_theta(z)))] +
E_pD(x)[f(-D_psi(x))] (Eq. 1, f(t) = -log(1 + exp(-t)) for the original GAN), with D_psi a real-valued logit,
and define

    R1(psi) = (gamma / 2) E_pD(x)[ ||grad_x D_psi(x)||^2 ]          (Eq. 9, true data)
    R2(theta, psi) = (gamma / 2) E_p_theta(x)[ ||grad_x D_psi(x)||^2 ]  (Eq. 10, generator distribution)

Theorem 4.1: training with either penalty is locally convergent. Gamma: "For the R1 and R2 regularizers from
Section 4.1 we use a regularization parameter of gamma = 10" with "1 discriminator update per generator update"
(App., CIFAR-10 experiments); the high-resolution experiments also use R1 with gamma = 10; the 2-D experiments
tried gamma in {1, 3, 10}.

**Multi-class analogue.** Our discriminator is a K-way batch classifier. Define the one-vs-rest log-odds of
batch k,

    D_k(z) = l_k(z) - log sum_{j != k} exp(l_j(z)) = log p_k(z) - log(1 - p_k(z)).

For K = 2, D_1 = l_1 - l_0 is exactly the binary logit D_psi of the paper. Each cell is a true sample of its own
batch, so the penalty is applied to every cell for its own batch's log-odds:

    R1_K(psi) = (gamma / 2) (1/n) sum_i || grad_z D_{b_i}(z_i) ||^2 .

For K = 2 this is the cell-weighted sum of the paper's R1 (on batch 1) and R2 (on batch 0), both covered by
Theorem 4.1. Both sides are penalised because no batch is a fixed "real" distribution here: the encoder moves
every batch.

Rejected definitions: (i) the gradient of the true-class log-probability, ||grad log p_b||^2 =
(1 - p_b)^2 ||grad D_b||^2, which vanishes where the discriminator is confident, i.e. where JS gradients vanish;
(ii) the raw own-class logit l_b, which is not invariant to adding one function to all logits (softmax
invariance), so it would penalise an unidentified component.

**As coded.** Adversary step: L_D = CE(l(z_d), b) + R1_K at the detached minibatch points z_d (one forward
pass with z_d.requires_grad, create_graph=True for the double backward), one step of the discriminator
optimiser. Generator step unchanged (scvi-tools fool loss times d_coef); the penalty is not in the generator
step (as the critics' gradient penalty). `r1_gamma` is a required plan argument for `*_r1` arms and is refused
for every other arm.

**Hyperparameters.** gamma = 10 (Mescheder et al. 2018, above); n_adv = 1 (SI-01); lambda = the X6 block's
matched_lo / matched / matched_hi.

**Manifest.** X6 block: (discriminator_r1, n_adv 1) x {atac_small, immune, pancreas} x 3 seeds x 3 lambda
= 27 fits, extra {"r1_gamma": 10}.

**Tests.** gradcheck of R1_K (float64) w.r.t. z and the head parameters; known answers: R1_K = 0 for a constant
head; linear head l = Wz + c with K = 2 gives R1_K = (gamma/2) ||w_1 - w_0||^2 exactly; for K = 3 the value equals
a per-cell finite-difference evaluation; one discriminator step per generator step; existing arms bit-identical.

## 2. X7: reference-anchored JS discriminator (`discriminator_ref`, "reference JS")

**Sources.** Goodfellow et al. 2014: value function V(D, G) = E_pdata[log D(x)] + E_pz[log(1 - D(G(z)))]
(Eq. 1); optimal D* = p_data / (p_data + p_g) (Prop. 1, Eq. 2); C(G) = max_D V = -log 4 + 2 JSD(p_data || p_g)
(Theorem 1); Sec. 3: "Rather than training G to minimize log(1 - D(G(z))) we can train G to maximize
log D(G(z))". ADDA (Tzeng et al. 2017) aligns a domain to a fixed one with the inverted-label loss
L_advM = -E_xt[log D(M_t(x_t))] (Eq. 7, Eq. 9). Code counterpart: critic formulation `reference`
(`src/wcd_vae/wcd/critic.py`): head k compares batch k with the reference batch r, head r is skipped, the loss
is the mean over active heads, and reference cells receive gradient from every head.

**As coded.** Same MLP head (K outputs); output k is the logit l_k of a binary discriminator "batch k versus
reference r" (k != r); output r is unused. A = batches present in the minibatch other than r (empty, and the
loss 0, if r is absent, as in the critic).

    Discriminator (class-balanced per head, per-side means):
      L_D = mean_{k in A} [ mean_{i in k} softplus(-l_k(z_i)) + mean_{j in r} softplus(l_k(z_j)) ]
    Generator (non-saturating, labels flipped on both sides):
      L_G = mean_{k in A} [ mean_{i in k} softplus(l_k(z_i)) + mean_{j in r} softplus(-l_k(z_j)) ]

Per-side means give each head the equal-prior objective of Eq. (1), so at D* the head's loss is
log 4 - 2 JSD(P_k || P_r) (Theorem 1) whatever n_k and n_r are; this is also how the critic weights its heads.
L_G is scvi-tools' fool loss (cross-entropy to the other class) applied to each two-class head. Reference cells
are not detached (counterpart of `reference`; the detached variant `reference_fixed` exists for W1 only).
1 adversary step per generator step, discriminator optimiser, no penalty; d_coef multiplies L_G.

**Manifest.** X7 block: discriminator_ref x every reference of pancreas (9), sim1 (6), atac_small (3) x 3 seeds,
lambda 'matched' = 54 fits (n_adv 1).

**Tests.** gradcheck of L_D and L_G (float64); known answers: L_D = log 4 when all logits are 0; at the Bayes
logit of two 1-D Gaussians (l = log N(z; mu, 1) - log N(z; 0, 1)) L_D = log 4 - 2 JSD within Monte Carlo error,
JSD by numerical integration; for K = 2 L_D equals the balanced binary cross-entropy; absent batches give
inactive heads; L_G sends gradient to reference cells; one adversary step per generator step; existing arms
bit-identical.

## 3. X8: per-batch stratified minibatch sampler (`sampler: stratified`)

**Source and default being replaced.** Domain-stratified minibatches are standard in domain-adversarial training:
Ganin & Lempitsky 2015 train on "128-sized batches", "A half of each batch is populated by the samples from the
source domain", the rest from the target domain (Sec. 5). Code review N6 (docs/PAPER_PLAN.md): at batch 128 a
small batch contributes 4-5 cells to its critic head, while critics weight heads equally. scvi-tools 1.4.2 default:
DataSplitter.train_dataloader -> AnnDataLoader(shuffle=True, drop_last=False) -> BatchSampler(RandomSampler,
128): ceil(n_train / 128) steps per epoch, the last one partial, per-batch counts multinomial.

**As coded.** A torch Sampler that yields index lists, used for the training loader only (validation loader
unchanged), installed through a DataSplitter subclass that the fit sets as `model._data_splitter_cls` only when
`sampler='stratified'` (the default path is untouched).
- Per step: with V batches in the training set, every batch gives m = floor(128 / V) cells; the remainder
  128 - V m goes one extra cell each to that many batches drawn uniformly at random per step. When V divides 128
  (V = 2, 4, 8, 16 in X8) every minibatch holds exactly 128 / V cells per batch.
- Within a batch: draws without replacement from a permutation of its training cells; when the permutation is
  used up a new one starts, and cells already in the current minibatch are not repeated in it. The cycles
  continue across epoch boundaries (no reshuffle at an epoch start), so each cell is drawn once per cycle.
- Randomness: one generator per epoch, seeded from torch's global RNG (as RandomSampler does), so a fit is
  reproducible from scvi.settings.seed.
- **Too few cells:** if a batch has fewer training cells than its per-step quota (ceil(128 / V)), the fit raises
  ValueError naming the batch. No sampling with replacement, no smaller minibatch. Never triggered in X8: the
  minimum training cells per batch is 158 (sim2, V = 16, quota 8).
- **Epoch:** ceil(n_train / 128) steps of exactly 128 cells, the same number of optimiser steps per epoch as the
  default sampler, so max_epochs (scvi heuristic) and the epoch-based KL warm-up (n_epochs_kl_warmup = 400) are
  unchanged. Each epoch processes up to 127 more cells than the default (no partial last minibatch); stated in
  the fit record. In X8 batches are equal-sized (n_cells / V each), so one epoch is about one pass over every batch.

**Manifest.** X8 block, rows for the critic arms (reference, pooled) with extra {"sampler": "stratified"}:
3 tasks x V {2,4,8,16} x 2 subsets x 3 seeds x 2 arms = 144 fits.

**Tests.** exact per-batch counts 128 / V in every minibatch for V in {2, 4, 8, 16}; counts within 1 and equal in
expectation otherwise; ceil(n_train / 128) steps, 128 distinct indices per minibatch; each training index once
per cycle; reproducible under a fixed seed; ValueError below quota; fit-level instrumented check that the
training step receives equal counts; default path bit-identical.

## 4. X3: oracle importance-weighted control for composition shift (`iw: depletion_oracle`)

**Sources.** Tachet des Combes et al. 2020: importance weights w_y = D_T(Y = y) / D_S(Y = y), assuming
D_S(Y = y) > 0 (Def. 3.4, Eq. 4); under generalized label shift D_T(Z) = sum_y w_y D_S(Z, Y = y) = D_S^w(Z)
(Lemma 3.1); the weighted adversarial loss -1/s sum_i [w_{y_i} log d(z_i^S) + log(1 - d(z_i^T))] (Eq. 7)
minimises D_JS(D_S^w || D_T) when E_S[w] = 1 (Lemma A.1); the same reweighting applies to any IPM (Sec. 3.5,
Eq. 6; the W1 critic is the IPM over 1-Lipschitz functions) and to MMD (IWJAN, App. B.5.3 Eq. 22); the oracle
versions (IWDAN-O, ...) use the true weights (Sec. 4.1).

**Why a common all-batch target is not used.** A common cell-type composition for all K batches needs types
present in every batch. On the X3 tasks no cell type is present in every batch of immune_hum_mou (23 batches)
or sim2 (16 batches), at any dose (`docs/x3_shared_support_profile.csv`).

**Depletion design (revised 2026-10-03, SI-31..SI-33).** Every X3 row names its reference batch: the repo rule
(select_reference_batch, maximum cell-type entropy) applied to the dose-0 subsample, fixed for every dose (SI-31).
The depleted batch is always a non-reference batch: the largest one, depleting its two most abundant cell types
present in >= 3 batches; on sim2 only Group1 is depleted, because Batch3Sub1 holds only Group1 (1,446) and Group2
(484) cells and depleting both removed the batch (SI-32). Only atac_small changed target (it had depleted its own
reference); its new target was signed off with per-batch counts (SI-33).

| task | reference (dose-0 rule) | depleted batch | depleted types |
|---|---|---|---|
| atac_small | Cusanovich et al. - WholeBrainA_62216 | Fang et al. - CEMBA180305_2B (3,750 cells) | Excitatory Neurons, Inhibitory Neurons (2,868) |
| immune_hum_mou | Oetjen_A | MCA_BM_2 | Neutrophils, Monocyte progenitors |
| sim2 | Batch4Sub2 | Batch3Sub1 | Group1 (1,446) |
| pancreas | inDrop1 | inDrop3 | alpha, acinar |

`scripts/fit_paper_config.py` (resolve_reference) refuses an X3 composition row whose reference is 'auto' or an
index, names a batch absent from the fitted cells, or names the depleted batch. `scripts/x3_design_profile.py`
re-derives every reference from the prepped files, checks that the reference and the depleted batch are present at
every dose, and writes both profiles; it also reports the batch the automatic rule would pick at each dose (atac_small
would switch to Fang at dose 80, pancreas to inDrop3 at doses 50-80): the fixed reference removes that confound.

**Target (signed off, SI-25).** Each batch's own pre-depletion composition, i.e. the X3 base condition. The depletion is a
known selection on true labels: a cell of a depleted type in the depleted batch b survives with probability
kappa_d = (n_hit - drop) / n_hit, drop = round(n_hit d / 100) (`fit_paper_config.subsample`), every other cell
with probability 1. Eq. (4) applied to the joint label (batch, cell type), T = pre-depletion and S =
post-depletion population, gives the oracle weights

    w_i = 1 / kappa_d  for cells of the depleted types in batch b,   w_i = 1  otherwise.

**How the weights enter (identical in all four X3 arms, adversary step and generator step).** Every empirical
average in the arm's own objective is replaced by its weighted, self-normalised average:
- discriminator: CE and fool loss, sum_i w_i l_i / sum_i w_i over the minibatch (batch shares and compositions of
  the pre-depletion population);
- reference and pooled critics: E_{P_k}[C_k] -> sum_{i in k} w_i C_k(z_i) / sum_{i in k} w_i, and the same for
  the target side (reference cells; pooled: all cells of the other batches);
- mmd: per-batch kernel mean embeddings with weights normalised within the batch and within its "other batches"
  pool (Eq. 22 with normalised weights).
Self-normalisation instead of Eq. (7)'s 1/s: within one minibatch the weights of a side do not sum to its cell
count, and an IPM objective E_P[C] - E_Q[C] is invariant to shifting C only if both sides' weights sum to one.
With w = 1 each arm reduces to its unweighted objective. The critics' gradient penalty is unchanged (unweighted
interpolates; it enforces the Lipschitz constraint between the supports, which weighting does not move).

**Doses.** w = 1 for every cell at dose 0 (identical to the unweighted rows) and at dose 100 (no depleted-type
cell remains; Eq. 4 needs D_S(Y = y) > 0), so IW rows exist at doses 50, 80, 95 only. At dose 95 (revised design,
`docs/x3_iw_weight_profile.csv`) the weight of a kept depleted-type cell is about 20, the effective-sample-size
fraction of batch b is 0.235 (atac_small), 0.284 (immune_hum_mou), 0.227 (sim2) and 0.183 (pancreas), and a
128-cell minibatch holds 2.14, 0.83, 0.51 and 0.87 depleted-type cells of batch b on average.

**Implementation.** Cell types reach the minibatch through scvi's labels slot (setup_anndata(labels_key=
'celltype'), conditioned models only; a test checks the labels slot changes no latent). IW with an
unconditioned model raises (its labels slot holds the batch). fit_paper_config.py builds the (batch, cell type)
weight table from the row's X3 subsample spec, using the same code that performs the depletion.

**Manifest.** X3 block: IW rows for discriminator, reference, pooled, mmd x {matched_lo, matched_hi} x doses
{50, 80, 95} x 3 seeds x 4 tasks = 288 fits, extra adds {"iw": "depletion_oracle"}. Since 2026-10-03 every X3 row
(unweighted and IW) carries reference = the batch name above, and the seeds are the follow-up seeds 10-12 (SI-26).

**Tests.** every X3_TARGETS entry re-derived from the prepped files (reference rule at dose 0, largest
non-reference batch, type rule, sim2 Group1 only, reference and depleted batch present at every dose); every X3
row carries its reference; resolve_reference refuses 'auto', an index, a missing name and the depleted batch on
composition rows; weights 1 at doses 0 and 100; sum of kept depleted-type weights = n_hit exactly when there is no final
subsample; for each arm, unit weights give the unweighted loss and integer weights equal duplicated cells;
gradcheck of the weighted losses; zero-weight cells get no adversarial gradient; the labels slot changes no
latent (adversary 'none'); existing arms bit-identical.

**Findings on the first X3 design (2026-10-02) and their resolution (2026-10-03).**
1. The 'auto' reference changed with dose (atac_small WholeBrainA_62216 to CEMBA180305_2B; pancreas inDrop1 to
   inDrop3 to inDrop1), mixing reference identity with dose. Resolved: fixed reference (SI-31).
2. sim2: the depleted batch Batch3Sub1 holds only the two depleted types, so X3 changed its size, not its
   composition, and removed it at dose 100 (K 16 to 15). Resolved: only Group1 is depleted (SI-32); K = 16 at
   every dose.
3. docs/paper_experiment_matrix.csv described X3 as composition-matched subsamples; dose 0 keeps the natural
   composition and matching is impossible on immune_hum_mou and sim2. Resolved: wording corrected.

## 5. Common points

- New arms are not in X1, so their 'matched' lambdas have no X1 curve of their own. The existing X6/X7 rows for
  discriminator_sn, pooled_sn, reference_fixed and mmd_ref have the same gap. The resolution rule belongs to the
  pre-registration (branch prereg-rules); the rows here use the same placeholders as their block.
- Cost: 27 + 54 + 144 + 288 = 513 fits; measured step costs are appended to
  docs/throughput_rtx3080_stock_backbone.csv.
- Runner: every key of a row's `extra` is consumed or the fit is refused (new keys r1_gamma, sampler, iw).

## 6. Sign-off

All four answered by the user on 2026-10-02 (ask_user, one question per item, recommendation first). Recorded
in CONSTRAINTS.md and the lab notebook.

Ledger ids: first recorded as SI-18..SI-21 (commit 6c46d68, notebook entries NB-20261002-01..04); renumbered
SI-22..SI-25 because branch prereg-rules (commit 0c3ca24, one minute earlier) assigned SI-18..SI-21 to its rules
R1-R4. Rule texts unchanged; correction logged in the notebook.

| item | question (abridged) | answer (verbatim) | ledger | notebook |
|---|---|---|---|---|
| X6 | How should the R1 penalty (Mescheder et al. 2018, Eq. 9) be defined for our K-way batch classifier, and with which gamma? | "One-vs-rest R1, gamma 10 (Recommended)" | SI-22 | NB-20261002-01 |
| X7 | discriminator_ref ("reference JS"): which generator loss? | "Non-saturating, labels flipped (Recommended)" | SI-23 | NB-20261002-02 |
| X8 | Stratified sampler: which arms get it? | "Critics only: reference, pooled (Recommended)" | SI-24 | NB-20261002-03 |
| X3 | Oracle importance-weighted control: which target composition? | "Undo the induced depletion (Recommended)" | SI-25 | NB-20261002-04 |

Alternatives offered and not chosen: X6 gamma grid {1, 10} (54 fits); X7 minimax (saturating) generator loss;
X8 sampler on all four X8 adversarial arms (288 fits); X3 pairwise full composition matching (3 arms, 360 fits).

Decisions of 2026-10-03 (ask_user; X3 reference and sim2 asked by the lead, the others in this branch):

| item | question (abridged) | answer (verbatim) | ledger | notebook |
|---|---|---|---|---|
| X3 | X3 reference batch | "Fix reference; deplete a non-reference batch (Recommended)" | SI-31 | NB-20261003-01 |
| X3 | X3 on sim2 | "Deplete only Group1 (Recommended)" | SI-32 | NB-20261003-02 |
| X3 | atac_small: depleted batch and types (per-batch counts shown) | "Fang: Excitatory + Inhibitory (Recommended)" | SI-33 | NB-20261003-03 |
| X13 | Harmony's strength knob and its 6 values | "theta {0, 0.5, 1, 2, 4, 8} (Recommended)" | SI-34 | NB-20261003-04 |
| X13 | Scanorama's strength knob and its 6 values | "knn {5, 10, 20, 40, 80, 160} (Recommended)" | SI-35 | NB-20261003-05 |
| X13 | PCA (no strength knob) | "Once, as uncorrected anchor (Recommended)" | SI-36 | NB-20261003-06 |

Alternatives offered and not chosen: atac_small Fang with Excitatory Neurons only, or 10x Genomics with both
types; Harmony theta {0.5, 1, 2, 3, 4, 6} or ridge lambda {0.05 ... 2}; Scanorama alpha {0 ... 0.6} or sigma
{1 ... 60}; PCA over 6 PC counts.

## 7. Implementation and verification (as built, branch missing-arms)

| item | code | tests |
|---|---|---|
| X6 discriminator_r1 | `wcd/discriminator_losses.py` (`one_vs_rest_logodds`, `r1_penalty`); plan arm `discriminator_r1`, `r1_gamma` required | gradcheck; R1 = 0 for a constant head; (gamma/2) times the squared norm of w1 - w0 for a linear two-batch head; finite differences (3 batches); one adversary step per generator step |
| X7 discriminator_ref | `reference_js_losses`; plan arm `discriminator_ref` (n_adv 1, reference required) | gradcheck; log 4 at zero logits; log 4 - 2 JSD at the Bayes logit of two Gaussians; explicit per-head loop; generator gradient reaches reference cells; one adversary step |
| X8 stratified sampler | `wcd/sampling.py` (`StratifiedBatchSampler`); plan `_make_stratified_splitter` (training loader only), option `sampler='stratified'` | exact 128/V counts (V = 2, 4, 8, 16); remainder within 1 and balanced; ceil(n/128) steps; once per cycle; reproducible; refusal below quota; fit-level counts in the training step (conditioned and unconditioned) |
| X3 oracle IW | `critic.py` (`_forward_weighted`), `alignment.py` (`_batch_masks_weighted`), `weighted_ce` / `fool_loss`; runner `depletion_oracle_weights` from `subsample(..., return_info=True)` | unit weights = unweighted loss; integer weights = duplicated cells; zero-weight cells get no gradient; gradcheck; weights 1 at doses 0 / 100 and depleted count restored; labels slot changes no latent; unit-weight fits equal unweighted fits (pooled, mmd bit for bit) |
| runner | `fit_paper_config.py`: every `extra` key consumed or refused; provenance git calls fail loudly | unknown / inconsistent keys refused; end-to-end rows for r1_gamma, discriminator_ref, sampler, iw; X3/X8 cells unchanged from 8cda6de on real obs |
| manifest | `build_paper_manifest.py`: X3 +288 (IW, doses 50/80/95), X6 +27 (R1), X7 +54 (reference JS), X8 +144 (stratified critics); base rows unchanged | `test_manifest_adds_exactly_the_signed_off_rows`; fl_unique_outputs over 6,502 rows |

Test runs at commit c3e3ad4 (code final; later commits change documentation only): wcd-gpu `159 passed`; scvi-api `50 passed, 1 skipped`; wcd-kbet (X13) `6 passed`.
Bit-identity gate (`scripts/check_arm_bitidentity.py --base 8cda6de`, CPU; run on the code of fe4ec94, which differs from the final code only in SI id tokens in comments): 34 of 34 configurations of the existing arms
identical, max|dz| = 0 (`docs/bitidentity_existing_arms.csv`); a 0.025% MMD bandwidth change and a 1e-4 fool-loss change are detected.

**Step cost (immune, single lane, RTX 3080; `docs/throughput_missing_arms_summary.csv`).** Four runs of
`experiments/bench_missing_arms_step_cost` (3 interleaved repeats of a 3-epoch and a 1-epoch fit per arm, 4 existing
arms as same-session controls): run1 (2026-10-02) failed PF-16 under CPU contention; run2 (2026-10-03 02:38-03:05 UTC,
GPU idle, no other compute process) passed for 11 of 12 arms; run3 was stopped when other workloads took 11-12 of 12
cores (15:09 UTC); run4 waited behind a quiet gate that never opened and was cancelled. Lead decision (NB-20261003-10):
the run2 medians are the rows in `docs/throughput_rtx3080_stock_backbone.csv`, and mmd_iw (spread 0.304 > 0.25) stays
marked provisional in `scripts/wall_time_report.py`. Same-session controls vs the existing CSV rows (ms/step):
discriminator 12.08 vs 11.65; reference 46.61 vs 47.03; pooled 43.01 vs 34.96; mmd 12.71 vs 12.71.

| arm | run2 median ms/step | range | spread (max-min)/median | PF-16 | run1 median (superseded) |
|---|---|---|---|---|---|
| discriminator_r1 | 13.56 | 13.14-14.62 | 0.109 | pass | 14.62 |
| discriminator_ref | 15.47 | 15.04-15.68 | 0.041 | pass | 16.53 |
| reference_stratified | 45.97 | 43.43-47.03 | 0.078 | pass | 46.4 |
| pooled_stratified | 41.74 | 33.26-43.64 | 0.249 | pass | 30.51 |
| discriminator_iw | 13.08 | 12.38-13.79 | 0.108 | pass | 13.08 |
| reference_iw | 46.03 | 44.63-48.13 | 0.076 | pass | 50.23 |
| pooled_iw | 37.62 | 32.94-38.08 | 0.137 | pass | 36.92 |
| mmd_iw | 13.08 | 11.68-15.65 | 0.304 | fail (provisional) | 12.38 |

The pooled control ran 23% slower in run2 than its existing CSV row (43.01 vs 34.96), so comparisons between new
arms and existing rows measured in another session carry session drift of that order.

**X13 CPU baselines** (`scripts/run_cpu_baselines.py`, env wcd-kbet): on atac_small the 6 latents (PCA d10/d50, Harmony d10/d50,
Scanorama d10/d100) were scored by `score_scib_native.py` unchanged. Two findings for the X13 design: Harmony through scib
(harmony-pytorch 0.1.7, multithreaded float32) is not reproducible run to run here (atac_small d50: max|dz| 0.114, NMI -0.024
between two runs), and kBET varies by up to 0.0047 on identical latents.

## 8. X13 CPU baselines: strength knobs, budget parity, Harmony reproducibility (2026-10-03)

**Requirement.** The approved matrix gives each X13 method "its native strength knob x 6 configs (deterministic CPU
methods once)" and PAPER_PLAN section 5 requires the same number of configurations per method family (best-of-k
curves). The adversarial arms have 6 lambda values per family (A1, R1); sysVI 6 cycle weights.

**Sources** (full texts read 2026-10-03: PMC author manuscripts PMC6884693 and PMC6551256, fetched by DOI
10.1038/s41592-019-0619-0 and 10.1038/s41587-019-0113-3; every quote below was matched verbatim in them; installed
package sources read in env wcd-kbet).
- Harmony, Korsunsky et al. 2019 (Nat. Methods 16:1289), Methods Eq. 3-4: theta "decides the degree of penalty for
  dependence between batch membership and cluster assignment"; theta = 0 reverts to soft k-means without the
  diversity penalty (Eq. 2); larger theta favours batch-independent clusters and the solution degenerates as theta
  grows without bound. Defaults (Methods 5.4): theta 2, K 100, sigma 0.1, ridge lambda 1; the paper's analyses
  used theta 2-4. harmony-pytorch 0.1.7: harmonize(theta=2.0, ridge_lambda=1.0, n_jobs=-1, random_state=0);
  scib 1.1.7 scib.integration.harmony calls harmonize with its defaults and does not forward kwargs.
- Scanorama, Hie et al. 2019 (Nat. Biotechnol. 37:685), Methods: mutual nearest-neighbour matching among all
  dataset pairs with 20 nearest neighbours, chosen "to identify a robust set of matches without also being overly
  permissive"; alignment-score cutoff alpha and Gaussian smoothing sigma. scanorama 1.7.4: correct(knn=20,
  alpha=0.10, sigma=15, dimred=100, seed=0); every dataset is translated by the kernel-weighted mean of its matched
  differences (no partial-strength parameter). scib 1.1.7 forwards kwargs to correct_scanpy.
- PCA: no batch-correction parameter; scIB's unintegrated embedding.

**Signed off (SI-34..SI-36).** Harmony theta in {0, 0.5, 1, 2, 4, 8}; Scanorama knn in {5, 10, 20, 40, 80, 160};
each at the backbone's 10 dimensions (SI-04), run once (Harmony random_state 0, Scanorama seed 0), plus the tool's
default dimensions at the default knob value as sensitivity (Harmony on 50 PCs at theta 2, Scanorama dimred 100 at
knn 20). PCA once at 10 PCs plus 50 PCs, shown as the uncorrected anchor and not as a best-of-k curve.

**Manifest.** `build_paper_manifest.CPU_BASELINES`: per task 7 Harmony + 7 Scanorama + 2 PCA rows = 16, 128 rows
over the 8 tasks; experiment X13, seed 0, knob value in 'lam', dimensions in 'n_latent', extra {"knob": ...};
scVI-backbone fields 'na' / 0. `fit_paper_config.py` refuses these arms; `cost_model.cost` gives them 0 GPU
lane-hours (they run on CPU; scoring time is counted as for every row).

**Runner.** `scripts/run_cpu_baselines.py --manifest M --prepped-dir D --out-dir O [--task T | --tag ...]` runs the
selected rows (refuses non-CPU tags and rows whose knob or seed differ from the design). Harmony: scib's two
statements (sc.tl.pca(n_comps), harmonize(theta)) in a fresh Python process with OMP/MKL/OpenBLAS/NumExpr threads
and harmonize(n_jobs) pinned (default 1). Scanorama: scib.integration.scanorama(dimred, knn), output re-ordered to
the input cells. PCA: sc.tl.pca(n_comps). Output format unchanged (latents/<tag>.npz, scored by
score_scib_native.py).

**Harmony reproducibility.** In a shared process two runs on atac_small gave d50 latents differing by up to 0.114
(NB-20261002-10). The fresh-process runs with pinned threads are tested for bit identity on atac_small at 50 PCs.

**Knob probes (exploratory, atac_small, 2026-10-03, NB-20261003-09; machine under external load).** kNN(30) batch
entropy (normalised by log K) at 10 dimensions: PCA 0.282; Harmony theta 0 / 2 / 8: 0.558 / 0.706 / 0.731 (19.5-32.6 s
per single-thread fit); Scanorama knn 5 / 20: 0.401 / 0.434 (19.6 s / 115 s); knn 160 had not finished after 16.6 min
(peak RSS 13.7 GB) and was stopped. So both knobs move batch mixing in the expected direction, and the top of the
Scanorama grid is expensive: its cost on the 85-98k-cell tasks (atac_large, immune_hum_mou) is not yet measured.

**Verification (2026-10-03, branch missing-arms-v2 = main 878d8ea + this work).** Test runs at commit 126812c, CPU only
(CUDA_VISIBLE_DEVICES empty): wcd-gpu `243 passed, 1 skipped` (tests/ without scvi and x13, incl. prereg); scvi-api `60 passed, 1 skipped`
(tests/scvi, incl. tests/scvi/test_x3_x13_design.py); wcd-kbet `17 passed` (tests/x13 and tests/scoring), including the
bit-identity of two fresh-process Harmony fits on atac_small at 50 PCs. The test that the runner's Harmony statements
reproduce scib.integration.harmony (tolerance 1e-4) has two negative controls, a shuffled batch column and the
uncorrected PCA (theta 2.5 instead of 2 changes the toy output by only 1e-05, so it is not used as a control); the
knob test compares theta 0 with theta 2 at 50 PCs. Mutation checks (fl_mutation_check) catch: a
reference resolver that accepts 'auto', an index or the depleted batch on composition rows; a CPU_BASELINES with theta
8 missing or Scanorama's default dimensions wrong; X3_TARGETS depleting both sim2 types or atac_small's reference; a
Harmony comparator fed a theta-0 fit or a one-ulp perturbation; the PF-16 rule fed a 30% spread or a warm-up repeat.
Manifest (stock, pilot, uncond 5, barycenter 10): 6,846 rows = main's 6,718 + 128 X13 CPU rows; 828 X3 rows, all with
their fixed reference; output paths unique (fl_unique_outputs, 0 collisions).
