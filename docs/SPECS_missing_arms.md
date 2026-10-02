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
cell remains; Eq. 4 needs D_S(Y = y) > 0), so IW rows exist at doses 50, 80, 95 only. Effective sample size of
batch b at dose 95: 0.19 (atac_small), 0.28 (immune_hum_mou), 0.18 (pancreas) of its cells; sim2 1.00 (see
findings); about 0.7-1.0 depleted-type cells of batch b per minibatch at dose 95 (`docs/x3_iw_weight_profile.csv`).

**Implementation.** Cell types reach the minibatch through scvi's labels slot (setup_anndata(labels_key=
'celltype'), conditioned models only; a test checks the labels slot changes no latent). IW with an
unconditioned model raises (its labels slot holds the batch). fit_paper_config.py builds the (batch, cell type)
weight table from the row's X3 subsample spec, using the same code that performs the depletion.

**Manifest.** X3 block: IW rows for discriminator, reference, pooled, mmd x {matched_lo, matched_hi} x doses
{50, 80, 95} x 3 seeds x 4 tasks = 288 fits, extra adds {"iw": "depletion_oracle"}.

**Tests.** weights 1 at doses 0 and 100; sum of kept depleted-type weights = n_hit exactly when there is no final
subsample; for each arm, unit weights give the unweighted loss and integer weights equal duplicated cells;
gradcheck of the weighted losses; zero-weight cells get no adversarial gradient; the labels slot changes no
latent (adversary 'none'); existing arms bit-identical.

**Findings on the X3 design (reported, not changed here).**
1. The 'auto' reference (maximum cell-type entropy on the cells used) changes with dose: atac_small
   WholeBrainA_62216 (doses 0-80) to CEMBA180305_2B (95, 100); pancreas inDrop1 (0) to inDrop3 (50, 80) to
   inDrop1 (95, 100). The reference arm's dose-response then mixes reference identity with dose.
2. sim2: the depleted batch Batch3Sub1 contains only the two depleted types (Group1 1446, Group2 484 cells), so
   X3 on sim2 changes that batch's size, not its composition, and at dose 100 the batch disappears (K 16 to 15).
3. docs/paper_experiment_matrix.csv describes X3 as composition-matched subsamples; at dose 0 the subsample
   keeps the natural composition.

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

## 7. Implementation and verification (as built, branch missing-arms)

| item | code | tests |
|---|---|---|
| X6 discriminator_r1 | `wcd/discriminator_losses.py` (`one_vs_rest_logodds`, `r1_penalty`); plan arm `discriminator_r1`, `r1_gamma` required | gradcheck; R1 = 0 for a constant head; (gamma/2) times the squared norm of w1 - w0 for a linear two-batch head; finite differences (3 batches); one adversary step per generator step |
| X7 discriminator_ref | `reference_js_losses`; plan arm `discriminator_ref` (n_adv 1, reference required) | gradcheck; log 4 at zero logits; log 4 - 2 JSD at the Bayes logit of two Gaussians; explicit per-head loop; generator gradient reaches reference cells; one adversary step |
| X8 stratified sampler | `wcd/sampling.py` (`StratifiedBatchSampler`); plan `_make_stratified_splitter` (training loader only), option `sampler='stratified'` | exact 128/V counts (V = 2, 4, 8, 16); remainder within 1 and balanced; ceil(n/128) steps; once per cycle; reproducible; refusal below quota; fit-level counts in the training step (conditioned and unconditioned) |
| X3 oracle IW | `critic.py` (`_forward_weighted`), `alignment.py` (`_batch_masks_weighted`), `weighted_ce` / `fool_loss`; runner `depletion_oracle_weights` from `subsample(..., return_info=True)` | unit weights = unweighted loss; integer weights = duplicated cells; zero-weight cells get no gradient; gradcheck; weights 1 at doses 0 / 100 and depleted count restored; labels slot changes no latent; unit-weight fits equal unweighted fits (pooled, mmd bit for bit) |
| runner | `fit_paper_config.py`: every `extra` key consumed or refused; provenance git calls fail loudly | unknown / inconsistent keys refused; end-to-end rows for r1_gamma, discriminator_ref, sampler, iw; X3/X8 cells unchanged from 8cda6de on real obs |
| manifest | `build_paper_manifest.py`: X3 +288 (IW, doses 50/80/95), X6 +27 (R1), X7 +54 (reference JS), X8 +144 (stratified critics); base rows unchanged | `test_manifest_adds_exactly_the_signed_off_rows`; fl_unique_outputs over 6,502 rows |

Test runs on the final code (src/ and scripts/ unchanged since fe4ec94; X13 suite re-run after its negative-control fix): wcd-gpu `159 passed, 66 warnings in 81.65s (0:01:21)`; scvi-api `50 passed, 1 skipped, 132 warnings in 50.75s`; wcd-kbet (X13) `6 passed, 4 warnings in 121.97s (0:02:01)`.
Bit-identity gate (`scripts/check_arm_bitidentity.py --base 8cda6de`, CPU): 34 of 34 configurations of the existing arms
identical, max|dz| = 0 (`docs/bitidentity_existing_arms.csv`); a 0.025% MMD bandwidth change and a 1e-4 fool-loss change are detected.

**Step cost (immune, single lane, `docs/throughput_missing_arms_summary.csv`), PROVISIONAL.** The run failed PF-16: repeat 2 ran
while other workloads kept 8-13 of 12 CPU cores busy. Same-session controls vs the existing CSV (ms/step): discriminator 12.71 vs 11.65; reference 47.03 vs 47.03; pooled 46.19 vs 34.96; mmd 11.65 vs 12.71.

| arm | median ms/step (3 repeats) | range | PF-16 | control-ratio estimate |
|---|---|---|---|---|
| discriminator_r1 | 14.62 | 13.14-14.83 | pass | 12.04 |
| discriminator_ref | 16.53 | 15.89-19.92 | pass | 16.18 |
| reference_stratified | 46.4 | 42.8-47.46 | pass | 45.46 |
| pooled_stratified | 30.51 | 21.19-36.23 | fail | 25.55 |
| discriminator_iw | 13.08 | 12.38-28.27 | fail | 13.32 |
| reference_iw | 50.23 | 41.82-65.42 | fail | 53.35 |
| pooled_iw | 36.92 | 35.28-60.28 | fail | 30.92 |
| mmd_iw | 12.38 | 11.21-28.5 | fail | 13.51 |

The 8 medians are appended to `docs/throughput_rtx3080_stock_backbone.csv`; re-measure on a quiet machine before relying on them.

**X13 CPU baselines** (`scripts/run_cpu_baselines.py`, env wcd-kbet): on atac_small the 6 latents (PCA d10/d50, Harmony d10/d50,
Scanorama d10/d100) were scored by `score_scib_native.py` unchanged. Two findings for the X13 design: Harmony through scib
(harmony-pytorch 0.1.7, multithreaded float32) is not reproducible run to run here (atac_small d50: max|dz| 0.114, NMI -0.024
between two runs), and kBET varies by up to 0.0047 on identical latents.
