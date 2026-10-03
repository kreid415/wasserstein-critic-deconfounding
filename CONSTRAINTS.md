# Standing constraints
<!-- The user's standing rules for this project. Managed by the standing-instructions skill
     (si_add / si_retire / si_link). Entries are never deleted; retire them. Check every deliverable
     against the active entries before presenting it. -->

## SI-01 · active · method
- rule: don't run the discriminator at 10 update steps, that is not how they are trained and will result in failure
- source: "don't run the discriminator at 10 update steps, that is not how they are trained and will result in failure." (user, 2026-10-02)
- added: 2026-10-02
- keywords: discriminator, n_critic, update, steps

## SI-02 · active · data
- rule: The datasets are the standard scIB datasets; no new datasets (the standard in literature and open problems is scIB)
- source: "The datasets are the standard scIB datasets now, why would we deviates? ... why would we add new datasets when the standard in literature and open problems is scIB?" (user, 2026-10-02)
- added: 2026-10-02
- keywords: dataset, datasets, task, tasks
- files: scripts/*manifest*.tsv
- check: {"forbid_literal": ["hlca_subset", "bmmc", "ding_pbmc", "cellbench"]}

## SI-03 · active · data
- rule: accept scIB standard keys
- source: "accept scIB standard keys." (user, 2026-10-02)
- added: 2026-10-02
- keywords: batch key, label key, keys

## SI-04 · active · method
- rule: match baselines and experiments to the same number of epochs, batch size, and latents
- source: "We will match baselines and experiments to the same number of epochs, batch size, and latents." (user, 2026-10-02)
- added: 2026-10-02
- keywords: epochs, batch size, latent, baseline, baselines

## SI-05 · active · process
- rule: we will do tier 1 + 2 but we need to reduce wall time
- source: "we will do tier 1 + 2 but we need to reduce wall time." (user, 2026-10-02)
- added: 2026-10-02
- keywords: tier, wall time, design, manifest

## SI-06 · active · process
- rule: We will discuss GPU once the compute wall time is finalized based on design
- source: "We will discuss GPU once the compute wall time is finalized based on design." (user, 2026-10-02)
- added: 2026-10-02
- keywords: GPU, cluster, compute

## SI-07 · active · process
- rule: For the preprint, leave it for now and replace it with the next version when we have results
- source: "For the preprint, we will just leave it for now and replace it with the next version when we have results." (user, 2026-10-02)
- added: 2026-10-02
- keywords: preprint, bioRxiv

## SI-08 · active · method
- rule: switch to scIB native scoring; the comparison is meaningless on a wrong loss
- source: "switch to scIB native. The comparison is meaningless on a wrong loss, so -1 month of time." (user, 2026-10-01)
- added: 2026-10-02
- keywords: score, scoring, metric, metrics, scIB

## SI-09 · active · method
- rule: We only care about scVI as people use it
- source: "We only care about scVI as people use it" (user, 2026-10-01)
- added: 2026-10-02
- keywords: scVI, backbone, baseline

## SI-10 · active · method
- rule: Shared backbone for every arm and baseline: scvi-tools defaults (n_latent 10, 1 layer, ZINB, 90% train split, batch 128, heuristic epochs)
- source: "scvi-tools defaults (Recommended)" (user, ask_user answer to "Which scVI configuration should every arm and baseline share?", 2026-10-02)
- added: 2026-10-02
- keywords: backbone, n_latent, scVI, configuration, manifest

## SI-11 · active · method
- rule: experiment with batch conditioned arms; review all code for fair comparison between the arms before running
- source: "before running, review all code to ensure fair comparison between the arms. We will also need to experiment with batch conditioned arms" (user, 2026-10-02)
- added: 2026-10-02
- keywords: conditioned, fair, arms

## SI-12 · active · writing
- rule: take the previous paper and make it more rigorous and expand it as needed
- source: "we will take the previous paper and make it more rigorous and expand it as needed" (user, 2026-10-02)
- added: 2026-10-02
- keywords: paper, outline, sections

## SI-13 · active · process
- rule: recreate all sweeps from base; reproducible from the baseline; no cherry-picking the best from past experiments
- source: "we need to recreate all sweeps from base because this needs to be reproducible from the baseline. we can't just cherry pick the best based on past experiments. This is the reproducible final run." (user, 2026-08-26)
- added: 2026-10-02
- keywords: sweep, lambda, grid, reuse, past

## SI-14 · active · method
- rule: Barycenter target: cold start (init = minibatch cells), 10 fixed-point iterations per training step; no warm start
- source: "10 iterations" (user, ask_user answer to "Cold-start fixed-point iterations per training step for the barycenter target?", 2026-10-02)
- added: 2026-10-02
- keywords: barycenter, bary_iter, warm

## SI-15 · active · data
- rule: Fit the scIB files as distributed (the 'counts' layer exactly as scIB fed scVI); no strict-count main run
- source: "scIB files as distributed" (user, ask_user answer to "Which data should our fits use?", 2026-10-02)
- added: 2026-10-02
- keywords: counts, valid, strict, non-integer

## SI-16 · active · method
- rule: Tier 1+2 lambda design: A1 pilot (full 10-point grid on atac_small, immune, sim1; 3 seeds; both decoders) fixes a 6-point grid per arm family for X1; 5 seeds in both X1 blocks
- source: "A1 pilot, 5 uncond seeds (Recommended)" (user, ask_user answer to the lambda design question, 2026-10-02)
- added: 2026-10-02
- keywords: lambda, grid, pilot, A1, seeds

## SI-17 · active · process
- rule: Compute: split the Tier 1+2 run between the local RTX 3080 and JHPCE by task: all arms, seeds and lambda of a task run on one machine, so arm comparisons are never confounded by hardware. (Agent-derived corollary, not a user statement: on JHPCE each task runs on one pinned GPU model.)
- source: "Local + one cluster split" and "JHPCE (Recommended)" (user, ask_user answers, 2026-10-02; split-by-task stated in the question)
- added: 2026-10-02
- keywords: GPU, cluster, JHPCE, split, compute

## SI-18 · active · method
- rule: R1 lambda-grid families: divergence x decoder -- {discriminator}, {reference, pooled, barycenter}, {mmd}, {sinkhorn}, each with a separate 6-point grid for conditioned and unconditioned decoders (8 grids), fixed from A1 by docs/PREREG.md section 2
- source: "Divergence × decoder (Recommended)" (user, ask_user answer to "R1 (A1 → X1 λ grid). Which arms should share one 6-point λ grid?", 2026-10-02)
- added: 2026-10-02
- keywords: lambda, grid, family, families, A1, X1, prereg
- files: docs/PREREG.md

## SI-19 · active · method
- rule: R2 adversary input: posterior mean is the X1 default; switch to posterior sample only if the PREREG.md section 3 criteria hold (mean dbio@b* > 0.01, positive in each task, >= 4 of 6 cells evaluable, no excess failures); masking => mean
- source: "Posterior mean (Recommended)" (user, ask_user answer to "R2 (A2 → X1 adversary input) ... Which input is the default?", 2026-10-02)
- added: 2026-10-02
- keywords: adv_input, adversary input, posterior mean, posterior sample, A2, prereg
- files: docs/PREREG.md

## SI-20 · active · method
- rule: R3 standardisation: per-dimension standardisation of the adversary input is off by default in X1 and the follow-ups; switch on only if the PREREG.md section 3 criteria hold (mean dbio@b* > 0.01, positive in each task, >= 7 of 10 cells evaluable, no excess failures)
- source: "Off (Recommended)" (user, ask_user answer to "R3 (A3 → X1 per-dimension standardisation of the adversary input) ... Which setting is the default?", 2026-10-02)
- added: 2026-10-02
- keywords: zstd, standardisation, standardization, A3, prereg
- files: docs/PREREG.md

## SI-21 · active · method
- rule: R4 matched-lambda target: b* = b0 + 0.5*(b_common - b0) per task x decoder on unscaled batch scores (mean of the 5 raw scIB batch metrics; b0 = lambda=0, b_common = lowest of the 6 arms' best failure-free seed-mean batch score); also proposed for P1
- source: "Midpoint of common range (Recommended)" (user, ask_user answer to "R4 (matched λ for the follow-ups ...) How should b* be defined?", 2026-10-02)
- added: 2026-10-02
- keywords: matched, b*, bstar, matched lambda, follow-up, P1, prereg
- files: docs/PREREG.md

## SI-22 · active · method
- rule: X6 discriminator_r1: one-vs-rest R1 penalty (gamma/2) mean_i ||grad_z D_{b_i}(z_i)||^2 with D_k = l_k - log sum_{j!=k} exp(l_j), at every minibatch cell, gamma = 10, in the discriminator step only, 1 discriminator step per generator step
- source: "One-vs-rest R1, gamma 10 (Recommended)" (user, ask_user answer to "X6 discriminator_r1: how should the R1 penalty (Mescheder et al. 2018, Eq. 9) be defined for our K-way batch classifier, and with which gamma?", 2026-10-02)
- added: 2026-10-02
- keywords: R1, discriminator_r1, gamma, gradient penalty, X6
- files: docs/SPECS_missing_arms.md, scripts/build_paper_manifest.py
- notebook: NB-20261002-01

## SI-23 · active · method
- rule: X7 discriminator_ref (reference JS): per-batch binary heads 'batch k vs reference', class-balanced cross-entropy, non-saturating generator loss with labels flipped on both sides, reference cells not detached, 1 discriminator step per generator step
- source: "Non-saturating, labels flipped (Recommended)" (user, ask_user answer to "X7 discriminator_ref ("reference JS"): which generator loss?", 2026-10-02)
- added: 2026-10-02
- keywords: reference JS, discriminator_ref, X7, reference
- files: docs/SPECS_missing_arms.md, scripts/build_paper_manifest.py
- notebook: NB-20261002-02

## SI-24 · active · method
- rule: X8 stratified sampler for the critic arms only (reference, pooled): 128/V cells per batch per minibatch, without replacement per batch, epoch = ceil(n_train/128) steps, fit fails when a batch has fewer training cells than its quota
- source: "Critics only: reference, pooled (Recommended)" (user, ask_user answer to "X8 stratified sampler: which arms get it?", 2026-10-02)
- added: 2026-10-02
- keywords: stratified, sampler, X8, minibatch
- files: docs/SPECS_missing_arms.md, scripts/build_paper_manifest.py
- notebook: NB-20261002-03

## SI-25 · active · method
- rule: X3 oracle importance weights undo the induced depletion: target = each batch's pre-depletion composition, w = 1/keep-fraction for depleted-type cells of the depleted batch and 1 otherwise, same weights in discriminator, reference, pooled and MMD, self-normalised in adversary and generator steps, doses 50/80/95 only
- source: "Undo the induced depletion (Recommended)" (user, ask_user answer to "X3 oracle importance-weighted control: which target composition?", 2026-10-02)
- added: 2026-10-02
- keywords: importance, IW, iw, X3, composition, weights
- files: docs/SPECS_missing_arms.md, scripts/build_paper_manifest.py
- notebook: NB-20261002-04

## SI-26 · active · method
- rule: Fresh seeds for all lambda-matched follow-ups: X3, X6, X7, X8 and X12 use seeds 10-12, disjoint from X1 (0-4) and A1/A2/A3 (100-102), so no follow-up cell reuses an X1 fit that chose its matched lambda; X13 is not lambda-matched and keeps seeds 0-4
- source: "Fresh seeds for all follow-ups (Recommended)" (user, ask_user answer to "The matched λ for the follow-ups is chosen from X1 seed-means (seeds 0–4), and the follow-ups run on seeds 0–2. ... These reuse the X1 fits that chose that λ, so they are biased toward the target batch score b*. Which seeds should the follow-ups use?", 2026-10-02/03)
- added: 2026-10-03
- keywords: seeds, follow-up, follow-ups, matched, reuse, X3, X6, X7, X8, X12
- notebook: NB-20261002-21

## SI-27 · active · method
- rule: A2/A3 also run with the unconditioned decoder, one pooled decision: the same R2/R3 rule over task x arm x decoder cells (A2 12, A3 20; at least 8 and 14 evaluable), the sign criterion checked per task x decoder; the unconditioned default halves are A1 unconditioned fits
- source: "Add it, one pooled decision (Recommended)" (user, ask_user answer to "A2 (posterior mean vs sample) and A3 (standardisation off vs on) run only with the conditioned decoder, but their decisions set both X1 blocks. Should A2/A3 also run with the unconditioned decoder?", 2026-10-02/03)
- added: 2026-10-03
- keywords: A2, A3, unconditioned, decoder, pooled decision, adv_input, zstd
- notebook: NB-20261002-22

## SI-28 · active · method
- rule: Include trajectory conservation in the bio score (score_scib_native.BIO_METRICS), as scIB does: computed where the prepped file has obs dpt_pseudotime (immune, immune_hum_mou), NaN and skipped elsewhere
- source: "Include trajectory (Recommended)" (user, ask_user answer to "scIB includes trajectory conservation in the bio score (scib-reproducibility plotSingleTaskRNA.R, group_bio; scIB computed it for its two immune tasks). Our scorer computes it for immune and immune_hum_mou but leaves it out of the bio aggregate. Should it be included? No Tier 1+2 results are scored yet.", 2026-10-02/03)
- added: 2026-10-03
- keywords: trajectory, bio score, BIO_METRICS, bio, metric, scIB
- notebook: NB-20261002-23

## SI-29 · active · process
- rule: Every cluster job submission (JHPCE: setup, tests, benchmarks, experiments) needs the user's explicit go; show what will run, where and for how long first. Local runs do not need a go but are reported.
- source: "only cluster jobs" (user, free-text answer to "Should every job submission need your explicit \"go\" from now on, recorded as a standing rule in CONSTRAINTS.md?", 2026-10-03; wording from the offered option "Every submission, incl. tests" restricted to cluster jobs)
- added: 2026-10-03
- keywords: cluster, JHPCE, job, submission, go, approval, launch, sbatch
- notebook: NB-20261002-24

## SI-30 · active · process
- rule: Task placement: atac_small, immune and sim1 (the A1/A2/A3 pilot tasks) run on the local RTX 3080 for every experiment (SI-17); the pilots start locally once the full design is merged and tagged and the preflight passes, with a report to the user before the first fit. The other five tasks are placed after the JHPCE GPU test, from a split recomputed with measured speed and local scoring.
- source: "Local; start pilots once design is tagged (Recommended)" (user, ask_user answer to "The pilots (A1, A2/A3) use only atac_small, immune and sim1. ... Where should they go?", 2026-10-03)
- added: 2026-10-03
- keywords: placement, local, JHPCE, split, pilot, A1, tasks

## SI-31 · active · method
- rule: X3 reference batch is fixed at each task's dose-0 automatic choice (select_reference_batch on the dose-0 subsample) for every dose, and the depleted batch is always a non-reference batch
- source: "Fix reference; deplete a non-reference batch (Recommended)" (user, ask_user relayed by the lead, 2026-10-03)
- added: 2026-10-03
- keywords: X3, reference, depletion, composition
- notebook: NB-20261003-01

## SI-32 · active · method
- rule: X3 on sim2 depletes only Group1 in Batch3Sub1 (which holds only Group1 and Group2), so the batch stays at every dose
- source: "Deplete only Group1 (Recommended)" (user, ask_user relayed by the lead, 2026-10-03)
- added: 2026-10-03
- keywords: X3, sim2, Group1, Batch3Sub1
- notebook: NB-20261003-02

## SI-33 · active · method
- rule: X3 on atac_small depletes Excitatory and Inhibitory Neurons in 'Fang et al. - CEMBA180305_2B' (largest non-reference batch; its two most abundant types present in all 3 batches)
- source: "Fang: Excitatory + Inhibitory (Recommended)" (user, ask_user, 2026-10-03)
- added: 2026-10-03
- keywords: X3, atac_small, Fang, Excitatory, Inhibitory
- notebook: NB-20261003-03

## SI-34 · active · method
- rule: X13 Harmony strength knob is theta in {0, 0.5, 1, 2, 4, 8} at 10 PCs, each run once (harmonize random_state 0, fresh process, pinned threads), plus the 50-PC tool default at theta 2 as sensitivity
- source: "theta {0, 0.5, 1, 2, 4, 8} (Recommended)" (user, ask_user, 2026-10-03)
- added: 2026-10-03
- keywords: X13, Harmony, theta, baseline, budget
- notebook: NB-20261003-04

## SI-35 · active · method
- rule: X13 Scanorama strength knob is knn in {5, 10, 20, 40, 80, 160} at dimred 10, each run once (seed 0), plus the dimred-100 tool default at knn 20 as sensitivity
- source: "knn {5, 10, 20, 40, 80, 160} (Recommended)" (user, ask_user, 2026-10-03)
- added: 2026-10-03
- keywords: X13, Scanorama, knn, baseline, budget
- notebook: NB-20261003-05

## SI-36 · active · method
- rule: X13 PCA runs once at 10 PCs (plus the 50-PC tool default as sensitivity) and is shown as the uncorrected anchor, not as a best-of-k budget curve
- source: "Once, as uncorrected anchor (Recommended)" (user, ask_user, 2026-10-03)
- added: 2026-10-03
- keywords: X13, PCA, baseline, budget
- notebook: NB-20261003-06

## SI-37 · active · process
- rule: Pre-registration tag goes ahead with the signed-off Scanorama grid (SI-35); knn 80 and 160 are timed on immune_hum_mou locally while A1 runs. If either is infeasible, the X13 grid is amended before any X13 fit, with a dated note in PREREG.md (X13's grid is not chosen from results).
- source: "Tag now; time it during A1 (Recommended)" (user, ask_user answer to "Scanorama's signed-off grid (SI-35) goes up to knn 160 ... How should I handle this?", 2026-10-03)
- added: 2026-10-03
- keywords: Scanorama, knn, X13, tag, amendment, timing

## SI-38 · active · method
- rule: Pre-specified sensitivity analysis (PREREG sec 9, amendment 2026-10-03): every X1 comparison on immune and immune_hum_mou is also reported with the bio score C recomputed without trajectory conservation (7 bio metrics). The primary analysis keeps trajectory in C as scIB does (SI-28). No fit changes.
- source: "Add it (Recommended)" (user, ask_user answer to "The primary bio score stays as you chose it ... Should the pre-registration also name a sensitivity analysis that recomputes the immune bio scores without trajectory? ...", 2026-10-03)
- added: 2026-10-03
- keywords: trajectory, sensitivity, bio score, immune, amendment, X1
