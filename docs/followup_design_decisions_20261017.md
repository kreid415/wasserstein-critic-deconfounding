# Follow-up experiments: design decisions to take before R4-x1

*Prepared 2026-10-17 by the rigor reviewer from branch `final-review-fixes` @ `1ad44cb` (tag `prereg-tier12-v3` = `10a1ca5`), read-only. No X1 or X13 score, latent, status record or runner log was opened (blinding rules SI-78/SI-79). Costs are GPU lane-hours on the RTX 3080 (graphics processing unit hours per training lane). They come from `scripts/cost_model.py` applied to the manifest of record (`manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv`) and reproduce `docs/manifest_v3_cost.csv`.*

**Terms used.** scVI = single-cell variational inference (the backbone). scIB = single-cell integration benchmarking metrics. B = batch-removal score; C = biology-conservation score. lambda (λ) = adversary penalty strength. b* = the target batch score, half-way between scVI alone and the batch removal every arm reaches (PREREG section 4). JS = Jensen-Shannon (the discriminator's divergence). W1 = Wasserstein-1 (the critic's). MMD = maximum mean discrepancy. SN = spectral normalisation. GP = gradient penalty. R1 = gradient penalty on data points. R4-x1 = the registered rule that sets every follow-up's λ from the X1 scores. SI-xx = an entry in the user-decision ledger `CONSTRAINTS.md`. Ixxx = a confirmed issue of the final review (`review/lead/verified_issues.csv`). D-xx = a finding of the design re-check (`review/rereview/design_fable/REVIEW_design_recheck.md`).

## 1. Timing

Order of stages (PREREG.md section 0; HANDOFF-final-review-fixes-4.md): Sinkhorn redo stage 1 (180 A1 refits, preflight GO at 1ad44cb) → Sinkhorn parts of R1, R4-a1 and R3 redone (SI-57) → X1 core benchmark launched and fully scored → R4-x1 (freeze_matched_lambda.py --stage x1: sets every follow-up penalty strength from all X1 scores, and may ask for extra X1-seed fits at grid edges for the cells follow-up rows draw on) → follow-ups X3, X6, X7, X8, X12, X15 on seeds 10-12 (SI-26). Must exist before the follow-ups, whatever is decided: (1) every X1 score, because R4-x1 needs them; (2) these decisions and the rebuilt follow-up manifest committed BEFORE R4-x1 runs, because R4-x1's grid-edge extensions depend on which cells follow-up rows draw on (PREREG.md:97, grid-edge rule) and design recheck D-06 asks for the decision before R4-x1 resolves the manifest; (3) the follow-up analysis specification (P2 dose-response, P3, P4, R3 factor effects, X15) written and locked while X1 runs, before R4-x1 and before anyone examines an X1 result (SI-68). Under SI-78 a script that uses X1 scores counts as examining, so R4-x1 itself is the latest possible lock point; (4) a new code tag after prereg-tier12-v3 for the follow-up code changes (the I036 gradient-penalty fix, plus any new X8 draw, X6 arm, X7 depleted-reference arm or X12 test chosen below), with a check that X1 fit paths give results identical to the tag (as EQUIVALENT_COMMITS.json did for SI-54). This can be done while X1 runs, without touching the X1 launch at v3; (5) an experiment preflight GO for each follow-up wave (X3 and X8 need the single-cell pack's perturbation items, SC-07/SC-08). None of this blocks the X1 launch. Current follow-up cost: 2,136 fits, 594.3 GPU lane-hours (docs/manifest_v3_cost.csv).

## 2. Per-experiment status

Status key: FIXED (commit) / DECIDED-NOT-FIXED / OPEN. Each was verified in the code or notebook at `1ad44cb`, not taken from the review alone. `git log --grep` and `grep` over notebook/, docs/ and CONSTRAINTS.md found no fix or decision for any OPEN item. The only matches were the ANALYSIS_SPEC.md:56-57 'NO-GO as designed' notes and SI-68's 'lock later'.

| Experiment | Question (plain) | Design as written (manifest v3, seeds 10-12, conditioned decoder) | Fits / GPU-h | Open issues and verified status |
|---|---|---|---|---|
| X3 composition dose-response | Does removing cell types from one batch make adversaries over-correct, and does oracle reweighting undo it? | atac_small, immune_hum_mou, sim2, pancreas; doses 0/50/80/95/100 % of two types removed from one non-reference batch, equal cell count per dose (SI-41); λ=0 + discriminator, reference, pooled, MMD at matched_lo/hi; importance-weighted copies at 50/80/95 | 828 / 259.7 | I088 wording + effective sample size: OPEN, default (section 4). D-15 wording: OPEN, default. I048 inheritance: has lo/hi, so default analysis rule. CR-04 count: FIXED 900f0f8. I033: FIXED c4be14b. |
| X6 divergence × regularisation × steps | Is the critic-discriminator difference due to the divergence or to the penalty, normalisation or number of adversary steps? | atac_small, immune, pancreas; discriminator, discriminator_sn, discriminator_r1 (1 step); pooled critic at 1/5/10 steps; pooled_sn at 1/5; λ lo/matched/hi | 216 / 74.3 | I036 GP computed for SN critics: OPEN in code (scvi_adversarial_plan.py:396, :288), default fix. I087 optimiser mismatch / incomplete factorial: OPEN → Q6 + default wording. I034: FIXED a4f4be0. |
| X7 reference choice | Does the reference W1 critic depend on which batch is the reference more than reference JS or reference MMD do? | every batch as reference: pancreas 9, sim1 6, atac_small 3; reference, reference_fixed, mmd_ref, discriminator_ref; one λ ('matched') | 216 / 68.3 | D-06 / I048 single inherited λ: OPEN → Q3. I089 no causal coverage contrast: OPEN → Q4. I092 label use: OPEN, default disclosure. I033 no-pair minibatches: FIXED c4be14b/96f4258. Review verdict NO-GO as designed (ANALYSIS_SPEC.md:57). |
| X8 batch count | Does the critic-discriminator difference change as the number of batches V grows? | immune_hum_mou 16,000, lung 6,000, sim2 3,000 cells; V 2/4/8/16 × 2 subsets; λ=0, discriminator, MMD, reference, pooled; reference and pooled also with stratified sampler (SI-24); one λ | 504 / 101.3 | I015, I077, I078, D-07 (per-head size not fixed, composition varies with V, doc drift): OPEN, code unchanged since bc4303d → Q2. I016 duplicate subsets: OPEN, re-verified here → default. I048 single λ: OPEN → Q3. Stratified discriminator missing for P3: OPEN, default rewording (follows SI-24). NO-GO as designed (ANALYSIS_SPEC.md:56). |
| X12 architecture robustness | Is the sign of the critic-discriminator difference stable across backbone and adversary settings? | atac_small, immune, pancreas; 2^(5-2) fraction, 8 runs: n_latent 10/30, adversary width 32/128, adversary lr 1e-4/1e-3, minibatch 128/512, decoder SCVI/LinearSCVI; discriminator vs reference at matched_lo/hi | 288 / 66.7 | I028 resolution III labelled IV (matrix csv:14): OPEN → Q5. D-06 inherited λ: OPEN, default analysis rule. |
| X15 KL warm-up | Are the X1 comparisons unchanged when the KL (Kullback-Leibler) warm-up completes instead of the stock 400 epochs? | immune, atac_large; stock vs complete warm-up; λ=0, discriminator, pooled, MMD at matched_lo/hi | 84 / 24.1 | D-16 missing from PREREG stage table: OPEN, default. 12 rows equal X6 settings: DECIDED (fitted separately, disclosed). No design question. |
| All follow-ups | | | 2,136 / 594.3 | I029 conditioned-only follow-ups: OPEN → Q1. |

## 3. Decision questions (in dependency order)

Order: Q1 changes the cost of every later option. Q2 changes the X8 part of Q3. Q3 changes the cost of Q4. Each is one decision; the recommended option is listed first.

### Q1 · X3, X6, X7, X8, X12 (mechanism follow-ups) · [Decoder]

**Question.** Should the mechanism follow-ups stay on the batch-conditioned decoder only, or also run with the unconditioned decoder?

**Problem.** All 2,052 mechanism-follow-up fits (X3 composition dose-response, X6 regularisation, X7 reference choice, X8 batch count, X12 architecture) use the batch-conditioned decoder (cond=1). The unconditioned decoder, where the adversary is the only thing removing batch effects, is the original paper's setting (PAPER_PLAN.md:68) and one of the two main P1 results (P1 = biology kept at matched batch removal; SI-70). Example: if X7 shows the reference critic depends on reference coverage with the conditioned decoder, the paper cannot say this explains the unconditioned P1 result.

| Option | What it does | Cost versus current design |
|---|---|---|
| Conditioned only; restrict the mechanism claims (Recommended) | Keep 2,052 fits; Methods state mechanisms explain conditioned-decoder results only. | +0 fits, +0 GPU-h |
| Add unconditioned X7 and X8 | Test the two original-paper claims (reference choice, batch count) in the original setting too. | +720 fits, about +169.6 GPU-h (provisional), plus any extra R4-x1 grid-edge fits |
| Add unconditioned to every mechanism follow-up | Every mechanism follow-up answered for both decoders. | +2,052 fits, about +570.2 GPU-h (provisional), plus any extra R4-x1 grid-edge fits |

**Why the first option.** The conditioned decoder is how scVI is normally used, and P1 is already reported separately for each decoder. Adding the unconditioned decoder doubles the cost of a secondary layer of the paper; option 2 is the cheaper middle ground if the original claims must be tested in their own setting.

**If the user does nothing.** The follow-ups run conditioned-only (as now) and the restriction is written into Methods as a default (review I029).

**Evidence.** I029 (verified_issues.csv); manifest v3 cond counts (X3 828, X6 216, X7 216, X8 504, X12 288, all cond=1; build_paper_manifest.py:213-271 passes cond True); PAPER_PLAN.md:68; CONSTRAINTS.md:477 (SI-70). Costs: scripts/cost_model.py on manifest v3; the cost model has no decoder term, so unconditioned cost = conditioned cost (provisional).

### Q2 · X8 (batch-count scaling) · [X8 design]

**Question.** What should X8 hold fixed as the number of batches V grows, and how should its batches be picked?

**Problem.** With total cells fixed (lung 6,000), cells per batch fall from 3,000 at V=2 to 375 at V=16, and cells per batch per minibatch from 64 to 8, so the registered claim 'per-head sample size and total cells are fixed' cannot hold. Each V also draws its own batches: at V=2 only 3 lung batches are big enough and both 'random' subsets pick the same 2; at V=16 both subsets are all 16 batches. Example: the lung macrophage share falls from 0.439 (V=2) to 0.173 (V=16) (review probe), so a change in the critic-discriminator gap with V is mixed with a change of cell types.

| Option | What it does | Cost versus current design |
|---|---|---|
| Fixed total cells, nested batch sets (Recommended) | V=2 batches inside V=4 inside 8 inside 16; claim reworded; per-head cells and composition as covariates. | +0 fits, +0 GPU-h (504 fits, 101.3 GPU-h); new draw code and a test |
| Fixed cells per batch, nested sets | Each batch keeps N/16 cells at every V, so total cells grow with V. | +0 fits, -52.8 GPU-h (48.4 GPU-h); total cells and training steps then change with V |
| Keep current draws; reword only | Keep random per-V draws; drop the 'fixed' claims; add covariates. | +0 fits, +0 GPU-h |

**Why the first option.** Nesting means a higher V only adds batches to a lower V, so composition changes by addition rather than by unrelated jumps, and fixed total cells keeps data volume and training steps equal across V. Matching composition directly is not possible because no cell type is present in every batch of immune_hum_mou or sim2.

**If the user does nothing.** X8 stays NO-GO (review D-07, I015, I077, I078): it cannot launch as designed, and P3 (the batch-count scaling endpoint) cannot be stated as registered.

**Depends on.** Q1

**Evidence.** scripts/fit_paper_config.py:192-205 (eligible = batches with >= N/V cells; pick with default_rng(20000+100*subset+V); per = N//V), unchanged since bc4303d; build_paper_manifest.py:109-111, 220-230; re-run by this check on prepped_scib obs 'batch': eligible batches at V=2 = 3 for all three tasks; subsets 0 and 1 identical at V=2 (all three tasks) and V=16 (lung, sim2); current draws not nested; sim2 V=16 total 2,992. Macrophage share from review probe out_rerun/I077_I078.csv (not re-run). Shared-support infeasibility: docs/paper_experiment_matrix.csv:7 (X3 row). Costs: cost_model.py with n_cells changed per V.

### Q3 · X7 (reference choice), X8 (batch count) · [X7/X8 λ]

**Question.** Should X7 and X8 get the low and high neighbouring penalty strengths, as X3, X6, X12 and X15 already have, so that biology can be read at the target batch score?

**Problem.** Every X7 and X8 fit uses a single penalty strength (lambda), inherited from another arm or another dataset size. Example: X7's reference-anchored JS discriminator (discriminator_ref) takes the lambda that brought the ordinary all-batch discriminator to the target batch score b*; its per-batch binary heads have a different loss scale, so its batch removal can land well off b*. With one lambda there is no way to read its biology at b* afterwards.

| Option | What it does | Cost versus current design |
|---|---|---|
| Add the λ window to X7 only (Recommended) | X7 gets lo/matched/hi and reads biology at b*; X8 is compared at fixed λ, achieved batch score as covariate. | +432 fits, +136.6 GPU-h |
| Add the λ window to X7 and X8 | Both get lo/matched/hi; X8 needs a b* defined for each V. | +1,296 fits, +331.1 GPU-h (X8 part +194.5; +93.0 if Q2 option 2) |
| No window; fixed-λ comparisons | One λ everywhere; report achieved batch score; drop 'matched batch removal' for X7/X8. | +0 fits, +0 GPU-h |

**Why the first option.** X7 runs on full X1 tasks, so the X1 target b* applies directly and a window lets biology be read at it, as in X3, X6, X12 and X15. On X8 subsets the scVI-alone batch score changes with V, so the full-task b* is not a defined target and a window would not give a clean matched comparison.

**If the user does nothing.** X7 stays NO-GO (review D-06). P4, the reference-choice endpoint ('formulation x coverage at matched batch removal'), would have to be reworded as a fixed-λ comparison.

**Depends on.** Q1, Q2

**Evidence.** I048, D-06; build_paper_manifest.py:216,219 (X7 lam 'matched' only), :225,230 (X8 'matched' only), :240,248,252-258,261,279 (X3/X6/X12/X15 lo/hi); PREREG.md:98 (variant inheritance; subsampled tasks inherit the full task's value); SPECS_missing_arms.md:232; freeze_matched_lambda.py:32 supports matched_lo/matched/matched_hi for follow-ups. Costs: cost_model.py (X7 68.31 GPU-h x2; X8 adversarial rows 97.26 x2).

### Q4 · X7 (reference choice) · [X7 coverage]

**Question.** Should X7 add an arm in which cell types are removed from one reference batch, so that the effect of reference coverage can be claimed as causal?

**Problem.** X7 only compares the references that happen to exist. In pancreas, 8 of 9 references cover 93-100% of the cell types and only 'smarter' covers 29%; each reference is also a different sequencing technology; and coverage moves with cell-type entropy (Spearman rho 0.55-0.97 across the three tasks). Example: if the critic does badly with 'smarter' as reference, X7 cannot say whether missing cell types, the technology or the batch size caused it.

| Option | What it does | Cost versus current design |
|---|---|---|
| Add a depleted-reference arm (Recommended) | Pancreas and sim1: remove reference cell types at 2 levels, refill to the same size; 4 X7 arms. | +48 fits, about +15.3 GPU-h (+144 fits, about +46.0 GPU-h with a Q3 window); new code (provisional estimate) |
| Observational only; associational wording | P4 reported as an association, with coverage, size and entropy as covariates. | +0 fits, +0 GPU-h |

**Why the first option.** Within one batch, removing cell types changes coverage while technology and size stay fixed. That is the only way to test the original paper's 'dense reference' claim, because natural coverage barely varies in pancreas. The design reuses the X3 depletion-and-refill code.

**If the user does nothing.** P4 is worded as an association (review I089 default). The original paper's 'dense reference' claim then has no causal test.

**Depends on.** Q1, Q3

**Evidence.** I089 (probe out_rerun/I089.csv, not re-run: coverage and Spearman values are the review's); PAPER_PLAN.md:101 (P4); build_paper_manifest.py:213-219 (X7 = every batch of pancreas 9, sim1 6, atac_small 3); fit_paper_config.py x3_draw (depletion with refill, SI-41/SI-42). Cost = 2 tasks x 2 levels x 3 seeds x the X7 per-fit cost of each of the 4 arms (pancreas 0.201/0.149/0.608/0.509, sim1 0.149/0.111/0.451/0.378 GPU-h).

### Q5 · X12 (architecture robustness) · [X12 design]

**Question.** Should X12 use the 16-run design, where each factor's effect is clear of two-factor interactions, instead of the current 8-run design, which is resolution III and not resolution IV as the matrix says?

**Problem.** X12 varies 5 settings in 8 runs (latent size, adversary width, adversary learning rate, minibatch size, decoder type). With generators D=AB and E=AC, the minibatch-size effect cannot be told apart from the latent size x adversary width interaction, and the decoder effect from latent size x learning rate. Example: if latent size and width interact, X12 would wrongly report a minibatch-size effect. The 8 runs still show whether the critic-minus-discriminator sign is stable across 8 configurations.

| Option | What it does | Cost versus current design |
|---|---|---|
| 16-run design, resolution V (Recommended) | 2^(5-1) with E=ABCD, already coded (x12_runs=16); main effects and two-factor interactions separable. | +288 fits, +66.6 GPU-h; needs one test |
| Keep 8 runs; relabel resolution III | Report sign stability over 8 configurations and main effects with their aliases stated. | +0 fits, +0 GPU-h |

**Why the first option.** Resolution III designs are meant for screening and assume two-factor interactions are negligible. In adversarial training, minibatch size, learning rate and width plausibly interact, so the 16-run design gives clean attributions for about 11% more follow-up compute.

**If the user does nothing.** X12 runs 8 runs and the matrix label is corrected to resolution III (review I028); factor attributions carry the stated aliases.

**Depends on.** Q1

**Evidence.** I028; build_paper_manifest.py:259-271 (8 runs D=A^B, E=A^C; 16 runs E=A^B^C^D), :310 (--x12-runs {8,16}, default 8); docs/paper_experiment_matrix.csv:14 ('resolution-IV'); no test covers x12_runs=16 (grep tests/). Costs: manifest_v3_cost.csv X12 66.65 GPU-h x1.

### Q6 · X6 (divergence vs regularisation) · [X6 optim]

**Question.** Should X6 add a spectral-norm JS discriminator trained with the critic's optimiser settings, so that the cleanest JS-versus-Wasserstein contrast differs only in the divergence?

**Problem.** X6's cleanest contrast of the two divergences is the spectral-norm JS discriminator against the spectral-norm W1 critic (W1 = Wasserstein-1), both at one adversary step. Their optimisers differ: the discriminator uses Adam lr 1e-3, eps 0.01, weight decay (scvi-tools' classifier settings), the critic Adam lr 1e-4, betas (0, 0.9) (WGAN-GP settings). Example: if the critic keeps more biology, X6 cannot say whether the divergence or the 10-fold smaller learning rate did it.

| Option | What it does | Cost versus current design |
|---|---|---|
| Add JS+SN with critic optimiser (Recommended) | One new X6 arm: discriminator_sn trained with critic Adam settings; inherits the discriminator λ window. | +27 fits, about +5.5 GPU-h; small code change |
| Add both optimiser swaps | Also the spectral-norm critic with the discriminator's optimiser. | +54 fits, about +11.5 GPU-h |
| No new arm; name the confound | X6 states the divergence contrast is confounded with optimiser. | +0 fits, +0 GPU-h |

**Why the first option.** Large GAN comparisons compare losses over the same optimiser settings. Here a single 27-fit arm turns the R3 divergence-versus-regularisation question from confounded into a one-factor contrast.

**If the user does nothing.** X6 runs as designed and its divergence contrast is reported as confounded with the optimiser (review I087).

**Depends on.** Q1

**Evidence.** I087; scripts/scvi_adversarial_plan.py:256-260 (opt_d: critic Adam lr critic_lr=1e-4, betas (0,0.9); discriminator Adam lr 1e-3, eps 0.01, weight_decay), :162 defaults; build_paper_manifest.py:250-258 (X6 arms). Cost: X6 per-fit means discriminator_sn 0.202, pooled_sn(1) 0.224 GPU-h x 27.

**Literature behind Q5 and Q6.** Q5: in resolution-III designs main effects are confounded with two-factor interactions, which are assumed minor (NIST/SEMATECH e-Handbook, section 5.3.3.2.6). Resolution V is 'often the most useful if it is achievable' (ETH Zürich design-of-experiments notes, chapter 9.6). Q6: Kurach et al. 2019 (ICML, arXiv:1807.04720) compare the non-saturating JS loss and the Wasserstein loss with the same normalisation and penalty options over one shared grid of learning rates and discriminator updates. The other recommendations rest on the design logic stated with each question.

## 4. Defaults applied without asking (wording, tooling, provenance, implementation; SI-92)

- I036 (X6 spectral-norm critics still compute the gradient penalty, then discard it, which also advances the spectral-norm power iteration): pass with_gp=False when spectral_norm is set (scripts/scvi_adversarial_plan.py:396 calls _head(with_gp=True) for every critic; :288 only zeroes the value). One-line fix plus one test, on the follow-up tag. Status OPEN, verified in code at 1ad44cb.
- I016 / D-07 (duplicate X8 subsets): whatever Q2 decides, the 'subset' factor is modelled as nested in V. Subsets that select identical batches (V=2 on all tasks, V=16 on lung and sim2; re-verified here) are not counted as independent composition replicates.
- I015 part (b) (sim2 V=16 gives 2,992 cells, not 3,000, because of floor division): keep it and report the realised cell count per cell of the design.
- D-07 stratified sampler: the P3 primary contrast (critic minus discriminator slope over log2 V) uses the default sampler for every arm. The stratified rows (reference and pooled critics only, user decision SI-24) give a secondary, critic-only contrast. PAPER_PLAN.md:99 is reworded to match.
- D-07 / I057 document drift: correct PAPER_PLAN.md:99, :143, :217 and docs/paper_experiment_matrix.csv:10 (lung, not BMMC; 16,000 / 6,000 / 3,000 cells; no 'fixed per-head size' or 'fixed composition' claim), and regenerate the matrix from the manifest.
- X8 covariates (I078): report cell-type composition distance from the V=2 draw (total variation) and cells per batch per minibatch for every X8 cell. Use them as covariates in the P3 model.
- I048 analysis rule for follow-ups with a lambda window (X3, X6, X12, X15, and X7 if Q3 adds it): report achieved batch score minus b* for every variant cell. Interpolate biology at b* only where the cell's own fits bracket b*; otherwise mark the cell 'not evaluable at b*' and report it at fixed lambda. Never call a single-lambda contrast 'matched batch removal'.
- D-06 (X12 inherited lambda): X12 results are reported at the fixed lo and hi strengths with achieved batch score, and are not called matched.
- I087 (X6 is an incomplete factorial): the X6 specification names the identified contrasts. These are JS with {none, spectral norm, R1} at one step; W1 with {gradient penalty, spectral norm} x steps; and JS+SN vs W1+SN at one step as the divergence contrast. It also discloses that R1 (penalty on data points) and the W1 gradient penalty (on interpolates) differ in form, and that per-batch versus per-cell weighting was not probed.
- I088 (X3 importance weighting): word the control as 'oracle reweighting of the alignment objective'. Report effective sample size per dose (18-42% of cells at dose 95, review probe) and state that the unweighted dose effect includes a batch-share shift.
- I092: disclose in Methods that the reference arm's automatic target uses cell-type labels and that X3's importance weights are an oracle. No label-free reference sensitivity run (X7 already uses every batch as reference).
- I089 wording (if Q4 adds no manipulated arm): P4 is reported as an association, with coverage, size and entropy as covariates. If Q4 adds the arm, only its contrast is described causally.
- I029 wording (if Q1 keeps conditioned-only): Methods state that the mechanism follow-ups explain conditioned-decoder results only. PAPER_PLAN.md:223 gets the decoder restriction.
- I028 label: docs/paper_experiment_matrix.csv:14 is corrected to match the Q5 answer (resolution III with aliases listed, or resolution V).
- D-15: correct PREREG.md wording on immune_hum_mou's X3 size (20,000 is a cap; the dose-100 size is 86,988). State the epoch counts of subsampled tasks (400 epochs) next to the inheritance rule (full tasks: immune_hum_mou 82).
- D-16: add X15 to PREREG.md section 0 stage table and to the fresh-seed list (seeds 10-12).
- D-09 follow-up endpoint definitions (written into the follow-up analysis specification and approved with it, as SI-80 did for X1). X3 response = bio score and batch score at each fixed lambda, regressed on dose with the X1 model structure. P3 'ceiling-normalised iLISI' (integration local inverse Simpson's index) = iLISI divided by its maximum attainable value at that V. P4 coverage = share of the task's cell types present in the reference with at least a minimum count. R3 window width defined on the bio score.
- Q6 implementation if chosen: the new arm gets a BASE_ARM mapping to discriminator (prereg_rules.py:59-61), so R4-x1 resolves it. Q4 implementation if chosen: depletion levels about 2/3 and 1/3 of natural coverage, chosen from the reference's most abundant types, with refill as in SI-41/SI-42.
- X15: 12 stock-warm-up rows equal X6 rows and are fitted separately (already disclosed, PREREG section 9). No action.

**Already fixed or decided (relevant to the follow-ups).**

- I033 minibatches with nothing to compare crashed MMD, reference MMD and Sinkhorn (affects X7 reference MMD on pancreas and sim1, X3 atac_small, X8 sim2): FIXED, commits c4be14b and 96f4258 (no_pair_steps counted), both in tag prereg-tier12-v3.
- I034 non-finite inner critic steps went undetected (affects X6 pooled at 5 and 10 steps): FIXED, commit a4f4be0, in the tag.
- I042 / G-03 inapplicable design-file settings accepted silently: FIXED, commit 96f4258 (fit_paper_config.py refuses unconsumed extra keys), in the tag.
- D-01 P1 comparator named a 10-step discriminator: DECIDED and amended (PREREG section 9: 1-step discriminator, SI-01; the update-budget control is X6's pooled critic only).
- CR-04 X3 cell count varied with dose: FIXED, SI-41/SI-42, commit 900f0f8 (equal cell count per dose, nested draw nested_v1), in the tag.
- SI-26 fresh seeds 10-12 for follow-ups: in code (build_paper_manifest.py:70, :209-211 check) and in freeze_matched_lambda.py check_no_x1_reuse.
- SI-81 (c) no_pair_steps reporting script written after the runs: DECIDED by the user (2026-10-15/16).

## 5. Sources checked

- Design documents at `1ad44cb`: docs/PREREG.md sections 0, 4 and 9 (lines 9-22, 90-101, 143-end, grep for X3/X6/X7/X8/X12/X15); docs/ANALYSIS_SPEC.md:1-12, 49-70, 286-300, 476-497; docs/PAPER_PLAN.md:68, 86-102, 118-150, 180-223; docs/paper_experiment_matrix.csv; CONSTRAINTS.md SI-01, SI-22..SI-26, SI-41, SI-42, SI-68, SI-70, SI-78, SI-79, SI-90..SI-92; docs/SPECS_missing_arms.md (grep).
- Code at `1ad44cb`: scripts/build_paper_manifest.py:70-111, 153, 209-282, 310; scripts/fit_paper_config.py:184-205 (X8 subsample; last changed bc4303d); scripts/scvi_adversarial_plan.py:162, 244-260, 278-290, 388-420; src/wcd_vae/wcd/adversarial.py:105-131; src/wcd_vae/wcd/sampling.py; scripts/freeze_matched_lambda.py:1-140 (grep); scripts/prereg_rules.py:59-61; scripts/cost_model.py; tests/ (grep for x12_runs).
- Re-run here (no score data): X8 batch eligibility and subset identity from `prepped_scib/*__scib.h5ad` obs 'batch' with the code's own seeds; follow-up costs by `cost_model.cost` on manifest v3 and on a copy with X8 cell counts changed (Q2 option 2).
- Review reports: review/lead/verified_issues.csv rows I015, I016, I028, I029, I036, I048, I057, I077, I078, I087, I088, I089, I092; docs/review_final/REVIEW_FINAL.md:50-90; review/rereview/design_fable/REVIEW_design_recheck.md D-01..D-19 and sections 4-5. Values quoted only from review probes (macrophage share, coverage and Spearman values, effective sample size, GP timing) were not re-run.
- Notebook and history: notebook/INDEX.md, notebook/2026-10-16.md, HANDOFF-final-review-fixes-4.md (grep); `git log --all --grep` for each issue id; commits c4be14b, 96f4258, a4f4be0, 900f0f8, bc4303d checked as ancestors of the tag.
- Not opened (blinding): D/tier12*, D/jhpce_tier12, D/x13_cpu_baselines_*, D/review/identity_*, D/review/out, D/review/verify, D/review_sealed, any score, latent, failures.csv or attempts/*.jsonl.