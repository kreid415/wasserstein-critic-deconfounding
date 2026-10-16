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
- rule: R2 adversary input: posterior mean is the X1 default; switch to posterior sample only if the PREREG.md section 3 criteria hold (mean dbio@b* > 0.01, positive in each task, >= 4 of 6 cells evaluable, no excess failures); masking => mean [cell counts superseded by SI-27: A2 12 / A3 20 cells, at least 8 / 14 evaluable, sign checked per task x decoder]
- source: "Posterior mean (Recommended)" (user, ask_user answer to "R2 (A2 → X1 adversary input) ... Which input is the default?", 2026-10-02)
- added: 2026-10-02
- keywords: adv_input, adversary input, posterior mean, posterior sample, A2, prereg
- files: docs/PREREG.md

## SI-20 · active · method
- rule: R3 standardisation: per-dimension standardisation of the adversary input is off by default in X1 and the follow-ups; switch on only if the PREREG.md section 3 criteria hold (mean dbio@b* > 0.01, positive in each task, >= 7 of 10 cells evaluable, no excess failures) [cell counts superseded by SI-27: A2 12 / A3 20 cells, at least 8 / 14 evaluable, sign checked per task x decoder]
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

## SI-35 · retired · method
- rule: X13 Scanorama strength knob is knn in {5, 10, 20, 40, 80, 160} at dimred 10, each run once (seed 0), plus the dimred-100 tool default at knn 20 as sensitivity
- source: "knn {5, 10, 20, 40, 80, 160} (Recommended)" (user, ask_user, 2026-10-03)
- added: 2026-10-03
- keywords: X13, Scanorama, knn, baseline, budget
- notebook: NB-20261003-05
- retired: 2026-10-04 · replaced by SI-46: "Use knn 2-80 (Recommended)" (user, 2026-10-04) after knn 160 timed out at 8 h on immune_hum_mou

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

## SI-39 · active · process
- rule: JHPCE placement for the Tier 1+2 benchmark: 2 L40S GPUs; atac_large, immune_hum_mou, lung, pancreas and sim2 run on JHPCE L40S (every fit of these tasks on L40S, SI-17), atac_small, immune and sim1 locally (SI-30). Split recomputed on the tagged 6,846-row design: 13.8 d. Each JHPCE submission still needs the user's go (SI-29).
- source: "2 L40S (Recommended)" (user, ask_user answer to "For the real X1 and follow-up fits of atac_large, immune_hum_mou, lung, pancreas and sim2: which JHPCE setup? ...", 2026-10-03)
- added: 2026-10-03
- keywords: JHPCE, L40S, placement, split, GPU, tasks

## SI-40 · active · process
- rule: Code check before any JHPCE launch: a RIGOR_REVIEWER review of tag prereg-tier12-v1 (fit, arms, CPU baselines, runner, scoring, decision rules, manifest builder, JHPCE scripts); the lead verifies every critical/high finding before reporting. A1 keeps running meanwhile; if the check changes any fit or scoring code, A1's fits up to that point are redone.
- source: "Review sub-agent (Recommended)" and "Keep A1 running (Recommended)" (user, ask_user answers to "How should the check run?" and "What should A1 do meanwhile?", 2026-10-03; preceded by the free-text answer "I want to do a code check before launch")
- added: 2026-10-03
- keywords: code check, review, RIGOR_REVIEWER, launch, JHPCE, A1

## SI-41 · active · method
- rule: X3 composition shift: every dose of a task uses the same cell count, the task's dose-100 size (atac_small 8,402, pancreas 14,409, sim2 17,872, immune_hum_mou 20,000), drawn as nested subsamples from one dose-independent permutation, so the dose effect is not confounded with cell count or training steps. Dated PREREG amendment; X3 rows rebuilt and re-tagged before any X3 fit; A1, A2, A3 and X1 rows unchanged.
- source: "Equal cell count per dose (Recommended)" (user, ask_user answer to "CR-04 ... How should X3 be handled?", 2026-10-03)
- added: 2026-10-03
- keywords: X3, composition, cell count, dose, subsample, amendment, CR-04

## SI-42 · active · method
- rule: X3 draw and oracle weights under SI-41 + SI-25: K0 = first N cells of one dose-independent permutation per task; dose d removes the first round(d x h0_t) cells of each declared type t of K0 in permutation order and refills with the next non-declared cells; oracle weights w[b, y] = n_K0(b, y) / n_Kd(b, y) per (batch, cell type) from the realised draws (doses 50/80/95, same arms, self-normalised); every X3 spec carries draw 'nested_v1' and all 828 X3 tags change. Agent-derived implementation of the user's SI-41 and SI-25 decisions.
- source: NB-20261003-22 DECISION (lead, 2026-10-03)
- added: 2026-10-03
- keywords: X3, draw, nested, importance, IW, weights, SI-25, SI-41
- notebook: NB-20261003-22

## SI-43 · active · process
- rule: A1 restarts from zero at the post-fix tag (prereg-tier12-v2) once the re-review passes; A1 was paused on 2026-10-04 at 09:27 EDT. The 202 A1 fits made at prereg-tier12-v1 (tier12/A1) stay on disk, unused, and are disclosed; no A1 metric value was used for any decision (two A1 rows were rescored only to test scorer identity).
- source: "Restart at the new tag (Recommended)" (user, ask_user answer to "... Your rule (SI-40) said A1's fits are redone if the check changes fit or scoring code ... What should happen to A1?", 2026-10-04)
- added: 2026-10-04
- keywords: A1, restart, tag, prereg-tier12-v2, SI-40
- notebook: NB-20261004-02

## SI-44 · active · method
- rule: KL warm-up sensitivity (PAPER_PLAN N8): immune (local) and atac_large (JHPCE L40S) x {lambda=0, discriminator, pooled critic, MMD at the matched lo/hi lambda} x seeds 10-12 x {stock 400-epoch KL warm-up, warm-up that completes (length = the task's epoch count)}; 84 fits, conditioned decoder, run after X1 with the follow-ups; dated PREREG amendment before the new tag.
- source: "Add it as planned (Recommended)" (user, ask_user answer to "The approved paper plan (PAPER_PLAN N8) promises a warm-up-complete sensitivity on 2 datasets ... Should it be added?", 2026-10-04)
- added: 2026-10-04
- keywords: N8, KL, warm-up, sensitivity, kl_weight, follow-up
- notebook: NB-20261004-03

## SI-45 · active · method
- rule: A2/A3 edge case (code check CR-11): if an R4 stage-a1 window neighbour is edge_unresolved, the default setting is fitted at that lambda on A1's seeds (100-102, same decoder) as A1 extension rows, then R2/R3 are applied as written.
- source: "Fit the missing point (Recommended)" (user, ask_user answer to "An undefined case in the A2/A3 rule (CR-11) ... Which handling should the amendment pre-register?", 2026-10-04)
- added: 2026-10-04
- keywords: CR-11, edge_unresolved, R2, R3, A2, A3, extension
- notebook: NB-20261004-04

## SI-46 · active · method
- rule: X13 Scanorama strength knob is knn in {2, 5, 10, 20, 40, 80} at dimred 10, each run once (seed 0), plus the dimred-100 tool default at knn 20 as sensitivity (replaces SI-35's knn 160: on immune_hum_mou knn 160 did not finish within 8 h and reached 43.9 GB; knn 80 took 2.1 h and 23.2 GB).
- source: "Use knn 2-80 (Recommended)" (user, ask_user answer to "The SI-37 timing finished ... knn 160 did not finish within the 8-hour limit ... What should replace 160?", 2026-10-04)
- added: 2026-10-04
- keywords: X13, Scanorama, knn, baseline, SI-35, SI-37
- notebook: NB-20261004-05

## SI-47 · active · process
- rule: JHPCE filler jobs jobA (atac_large, immune_hum_mou; 100 fits) and jobB (lung, pancreas, sim2; 150 fits) are submitted together at tag prereg-tier12-v2 (c88cce3): 1 L40S, 10 CPUs, 96 GB, 24-h wall each; fit only; latents harvested to jhpce_tier12/fillers_x1_x13 and scored locally from the worktree pinned at the tag.
- source: "Submit both now (Recommended)" (user, ask_user answer to "The two JHPCE filler jobs passed preflight at the tag ... Submit?", 2026-10-04)
- added: 2026-10-04
- keywords: JHPCE, filler, jobA, jobB, submit, go, SI-29
- notebook: NB-20261004-14

## SI-48 · active · process
- rule: JHPCE fit env repair: one shared CPU job at b698a00 deletes the purge-damaged wcd-fit env and rebuilds it with cluster/jhpce/build_envs.sh (fresh conda package cache, conda md5 and pip versions equal to the local env, all files re-dated); when it passes, the two filler jobs of SI-47 are resubmitted unchanged at prereg-tier12-v2 (c88cce3).
- source: "Rebuild, then resubmit both (Recommended)" (user, ask_user answer to "Both filler jobs landed on L40S nodes at once ... Go?", 2026-10-04)
- added: 2026-10-04
- keywords: JHPCE, env, rebuild, purge, fastscratch, filler, resubmit
- notebook: NB-20261004-16

## SI-49 · active · process
- rule: Cluster storage (user rule, all projects): software environments (conda envs, venvs, the R kBET library) live in $HOME; data lives on scratch (job dirs, inputs/outputs/logs/checkpoints, TMPDIR, caches). JHPCE: FIT_ENV, SCORE_ENV, WCD_R_LIBS = $HOME/envs/{wcd-fit, wcd-score, Rlib_kbet} (cluster/jhpce/env.sh); prereg-tier12-v2's fastscratch env paths become symlinks to them (build_envs.sh compat_link), so the tagged jobs run unchanged. The SI-48 rebuild therefore targets $HOME/envs/wcd-fit, and build_envs.sh refuses to build unless HOME keeps 5,000 MiB free under its 100 GB cap (HOME also holds every agent session's job workdirs).
- source: "store environments in home. Store data on scratch." (user, 2026-10-04, message after the SI-48 go; the fastscratch rebuild job 9d66854c was cancelled at 18:17 EDT)
- added: 2026-10-04
- keywords: storage, home, scratch, environments, conda, JHPCE, fastscratch
- notebook: NB-20261004-17

## SI-50 · active · process
- rule: JHPCE HOME space for SI-49: the cluster copy of the superseded sweep's latents, /users/kreid/wcd_repo/durable/embeddings_final (6,305 npz, 51,590,355,673 bytes), is deleted. The local copy (experiment_data/wasserstein-critic-deconfounding/embeddings_final_cluster; every file md5-matched 2026-09-30, names and sizes equal on 2026-10-04, name+size digest 206aec95b6d6dffc7b2d07ef17a75eb4) becomes the only copy. This frees room in HOME for $HOME/envs/wcd-fit.
- source: "Delete the cluster copy (Recommended)" (user, ask_user answer to "The JHPCE fit environment doesn't fit in `$HOME` ... What should I do with the cluster copy?", 2026-10-04)
- added: 2026-10-04
- keywords: JHPCE, HOME, latents, delete, embeddings_final, space, SI-49
- notebook: NB-20261004-18

## SI-51 · active · method
- rule: Sinkhorn arm: fix the loss gradient (no double-counted final transform) and the final-epsilon convergence, then refit all 234 Sinkhorn pilot fits (A1 180, R1 extension 18, A3 36) at the new tag and re-run the Sinkhorn parts of R1, R4-a1 and R3 before X1.
- source: "Fix it and redo all 234 (Recommended)" (user, ask_user answer to the Sinkhorn question after the final review, 2026-10-13)
- added: 2026-10-13
- keywords: Sinkhorn, gradient, refit, final review, new tag
- notebook: NB-20261013-02

## SI-52 · active · process
- rule: Final-review fixes: draft all code guards (I033, I034, I042, I050, I051, I094), launch-file fixes (I008, CR2-01 L40S gate re-run at the new tag) and the document amendments (P1 comparator, underpowered-design claim, b* anchoring, grid-edge rule, R1 trajectory note, frozen P1-P4 analysis code) on a branch with tests; prove non-Sinkhorn fits and scores bit-identical to c88cce3; independent review of each fix; show the user the diff before any new tag.
- source: "Draft all, show me, then tag (Recommended)" (user, ask_user answer, 2026-10-13)
- added: 2026-10-13
- keywords: final review, fixes, new tag, bit-identity, amendments
- notebook: NB-20261013-02

## SI-53 · active · method
- rule: R1 trajectory (I091): keep the registered R1 and the frozen X1 grids; correct the PREREG §9 sentence that R1 is unaffected and report R1 without trajectory as a disclosed sensitivity analysis.
- source: "Keep the registered R1 and disclose (Recommended)" (user, ask_user answer to I091, 2026-10-13)
- added: 2026-10-13
- keywords: R1, trajectory, sensitivity, grid, I091
- notebook: NB-20261013-08

## SI-54 · active · process
- rule: Mixed commits: declare c88cce3 and the new tag result-identical for non-Sinkhorn arms in an explicit committed list backed by the identity checks; rule scripts and the runner accept either commit for those arms only; Sinkhorn rows must carry the new tag; no rescoring.
- source: "Declare the two commits equivalent (Recommended)" (user, ask_user answer to the provenance question, 2026-10-13)
- added: 2026-10-13
- keywords: provenance, commit, scorer_git_sha, allow-mixed-sha, equivalent, tag
- notebook: NB-20261013-08

## SI-55 · active · method
- rule: X1 grid edge (I006): correct PAPER_PLAN to PREREG §4: grid extensions only for matching; P1 and the frontier use the six registered lambda points; edge optima are disclosed, no extension fits.
- source: "Correct PAPER_PLAN to match PREREG (Recommended)" (user, ask_user answer to I006, 2026-10-13)
- added: 2026-10-13
- keywords: grid edge, extension, PAPER_PLAN, frontier, I006
- notebook: NB-20261013-08

## SI-56 · active · method
- rule: P1 at b* (I072, D-04): anchor every arm's bio-versus-batch curve at lambda=0 (the unintegrated model's scores that define b0); bstar_low_signal cells are not evaluable for P1 and R2/R3 and are reported separately.
- source: "Anchor at λ=0; low-signal cells not evaluable (Recommended)" (user, ask_user answer to I072/D-04, 2026-10-13)
- added: 2026-10-13
- keywords: b*, bstar_low_signal, P1, anchor, R2, R3, I072
- notebook: NB-20261013-08

## SI-57 · active · method
- rule: Sinkhorn redo in rule order: refit the 180 A1 Sinkhorn fits, run R1 for the Sinkhorn family and its extension rows, run R4-a1 over all six arms with the refitted Sinkhorn arm, fit the A3 rows it resolves (including any non-Sinkhorn cell whose window moved), then re-run R2 and R3 in full.
- source: "Run R4-a1 as registered, in rule order (Recommended)" (user, ask_user answer to the R4-a1 scope question, 2026-10-13)
- added: 2026-10-13
- keywords: Sinkhorn, R4-a1, b_common, b*, refit, rule order, A3, R3
- notebook: NB-20261013-10

## SI-58 · active · method
- rule: Sinkhorn solve precision: validate a float32 solve on the 598 hardest minibatches and the GPU timing; adopt it only if the worst gradient cosine and value error stay within the float64 results (cosine >= 0.99999, value error <= 1e-4), then re-run the tests and the independent review; otherwise keep float64.
- source: "Validate float32 first; adopt only if it matches" (user, ask_user answer to the precision question, 2026-10-13)
- added: 2026-10-13
- keywords: Sinkhorn, float32, float64, precision, dtype, speed
- notebook: NB-20261013-14

## SI-59 · retired · method
- rule: Sinkhorn iteration cap: raise SINKHORN_MAX_ITER to 500,000 and keep raising an error at the cap (stage stops loudly); a non-finite marginal error ends the loop at once (review F1).
- source: "Raise the cap to 500,000 and keep the error (Recommended)" (user, ask_user answer to the cap question, 2026-10-13)
- added: 2026-10-13
- keywords: Sinkhorn, SINKHORN_MAX_ITER, cap, iterations, RuntimeError, F2
- notebook: NB-20261013-15
- retired: 2026-10-13 · user accepted NaN-only stop: "Accept NaN-only; reword SI-59 (Recommended)" (2026-10-13); replaced by the next rule

## SI-60 · active · process
- rule: New tag name: prereg-tier12-v3 (the tag that SI-54's equivalence list names; c88cce3 = prereg-tier12-v2 is result-identical to it for non-Sinkhorn arms).
- source: "prereg-tier12-v3 (Recommended)" (user, ask_user answer to the tag-name question, 2026-10-14)
- added: 2026-10-13
- keywords: tag, prereg-tier12-v3, equivalent
- notebook: NB-20261013-28

## SI-61 · active · method
- rule: Sinkhorn iteration cap: SINKHORN_MAX_ITER = 500,000 and the cap still raises an error (stage stops loudly). A NaN marginal error ends the solve loop at once; a +infinity error does not (it occurs for finite latents of about 1e18 or more and the solve converges a few iterations later; re-review R2-01).
- source: "Accept NaN-only; reword SI-59 (Recommended)" (user, ask_user answer to decision 2, F1 stop rule, 2026-10-13)
- added: 2026-10-13
- keywords: Sinkhorn, SINKHORN_MAX_ITER, cap, NaN, F1
- notebook: NB-20261013-30

## SI-62 · active · method
- rule: Sinkhorn solve precision: keep float64 (float32 met the SI-58 criterion on the 598 but gives no speed-up at 128-cell minibatches and fails the value threshold on all 3,076).
- source: "Keep float64 (Recommended)" (user, ask_user answer to decision 3, SI-58 precision, 2026-10-13)
- added: 2026-10-13
- keywords: Sinkhorn, float32, float64, precision
- notebook: NB-20261013-31

## SI-63 · active · method
- rule: I042 unread settings: accept the per-arm refusals as implemented (5f30f17 plus G-05), not the review's list; none reads decoder, likelihood and cond, scANVI reads likelihood.
- source: "Accept as implemented (Recommended)" (user, ask_user answer to decision 4, I042 refusals, 2026-10-13)
- added: 2026-10-13
- keywords: I042, allow-list, unread, check_extra
- notebook: NB-20261013-32

## SI-64 · active · process
- rule: I050: keep refusing latents without cell names; move fit_stock_scvi.py and scvi_fit_pancreas.py to legacy/ as retired scripts.
- source: "Keep refusal; move both to legacy/ as retired (Recommended)" (user, ask_user answer to decision 5, I050 legacy, 2026-10-13)
- added: 2026-10-13
- keywords: I050, legacy, fit_stock_scvi, scvi_fit_pancreas
- notebook: NB-20261013-33

## SI-65 · active · method
- rule: scib 1.1.7 NMI-0 edge case (no clustering kept when NMI is 0 at every resolution): keep the loud error; decide case by case if it occurs.
- source: "Keep the loud error (Recommended)" (user, ask_user answer to decision 6, NMI-0 case, 2026-10-13)
- added: 2026-10-13
- keywords: NMI, scib, cluster_optimal_resolution, edge case
- notebook: NB-20261013-34

## SI-66 · active · method
- rule: R2/R3 counting of low-signal cells (SI-56): cells of a task x decoder flagged bstar_low_signal stay in n and count as not evaluable; so while such a block is flagged, the 'at least one evaluable cell per task x decoder' condition fails and R2/R3 keep the default (posterior mean; standardisation off).
- source: "A: count it as 'could not be judged'" (user, ask_user answer to decision 1, R2/R3 counting, 2026-10-13)
- added: 2026-10-13
- keywords: R2, R3, bstar_low_signal, evaluable, coverage, SI-56
- notebook: NB-20261013-35

## SI-67 · active · writing
- rule: Communicate more clearly: plain language a genetics PhD student can follow without earlier messages; explain what each step entails and why; spell out labels and acronyms (profile NO-JARGON rule).
- source: "you need to communicate more clearly. What are the remaining steps before experimentation? Explain what they entail and why." (user, 2026-10-13)
- added: 2026-10-13
- keywords: *
- notebook: NB-20261013-36

## SI-68 · active · process
- rule: Lock all X1 analyses now; the rest during X1. Lock now: P1 (biology kept at equal batch removal) and its trade-off area; P2 (how often runs collapse, i.e. fail or lose 0.10 or more biology score); the local-versus-global test; tables of each method at one penalty strength chosen on the other datasets; the comparison with standard tools (Harmony, Scanorama, sysVI, scANVI) when every method gets the same number of tries; and how wide each method's range of good penalty strengths is. Lock later, while X1 runs and before anyone looks at an X1 result: P3, P4 and the other follow-up analyses, together with the X7/X8 redesign. Exploratory (no confirmed or refuted claim): the premise check (R0) and X10 (downstream biology at matched batch removal).
- source: "Lock all X1 analyses now; the rest during X1 (Recommended)" (user, ask_user answer to "Which analyses should be written as code and locked before X1 starts?", 2026-10-14)
- added: 2026-10-14
- keywords: lock, scope, exploratory, R0, X10, P3, P4, follow-up analyses
- files: docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py, docs/PREREG.md, docs/PAPER_PLAN.md
- notebook: NB-20261014-01

## SI-69 · active · method
- rule: Average the 5 runs first: at each penalty strength, average the batch and biology scores over the 5 runs, then read the target once; uncertainty by repeating the whole reading 5 times, each time leaving one seed out of both methods and of plain scVI (the reading R2 and R3 already use). A curve that crosses the target twice uses the first crossing (the weaker penalty); reading each run separately is reported as a sensitivity analysis.
- source: "Average the 5 runs first (Recommended)" (user, ask_user answer to "Should the 5 runs be averaged before this reading, or read one by one?", 2026-10-14)
- added: 2026-10-14
- keywords: P1, bio@b*, leave-one-seed-out, reading, crossing, seed mean
- files: docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py
- notebook: NB-20261014-02

## SI-70 · active · method
- rule: Two P1 results, one per decoder (conditioned and unconditioned): both are main results; they are never averaged together.
- source: "Two results, one per decoder (Recommended)" (user, ask_user answer to "How should P1 treat the two [decoders]?", 2026-10-14)
- added: 2026-10-14
- keywords: decoder, conditioned, unconditioned, primary, P1
- files: docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py
- notebook: NB-20261014-03

## SI-71 · active · method
- rule: Leave failed runs out of P1 and its trade-off area: each penalty strength's score is the average of its successful runs (PREREG section 1); a strength where all 5 runs failed is skipped; sensitivity analysis: drop every strength with any failed run. Every failure is counted in P2, the collapse rate.
- source: "Leave failed runs out of P1 (Recommended)" (user, ask_user answer to "How should failed runs enter P1 and its trade-off area?", 2026-10-14)
- added: 2026-10-14
- keywords: failed, failure, diverged, nonfinite_latent, impute, P2
- files: docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py
- notebook: NB-20261014-04

## SI-72 · active · method
- rule: Trade-off area corner at the collapse line: plain scVI's batch score and 0.10 below plain scVI's biology score (the R1 collapse threshold). Only settings that remove more batch effect than scVI and keep biology above the collapse line add area; overlapping rectangles count once; to combine datasets each area is divided by the dataset's batch range (plain scVI's batch score to the highest batch score every method reaches).
- source: "Corner at the collapse line (Recommended)" (user, ask_user answer to "Where should the corner be? (P1 trade-off area)", 2026-10-14)
- added: 2026-10-14
- keywords: area, hypervolume, frontier, corner, trade-off area
- files: docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py
- notebook: NB-20261014-05

## SI-73 · active · method
- rule: Combine the 6 real datasets with a family term: each dataset's true difference = the overall difference + a family effect + a dataset effect (families: pancreas; lung; immune = immune and immune_hum_mou; ATAC = atac_small and atac_large). Prior assumptions: the overall difference and the spreads between datasets each at scale 0.1, with scales 0.05 and 0.2 as sensitivity analyses; exact calculation (no random sampling), so results reproduce digit for digit.
- source: "Family term (Recommended)" (user, ask_user answer to "How should the 6 real datasets be combined into one overall answer per decoder?", 2026-10-14)
- added: 2026-10-14
- keywords: family, combining datasets, hierarchical, prior, spread, pooled
- files: docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py
- notebook: NB-20261014-07

## SI-74 · active · method
- rule: Margin 0.01, the registered smallest difference that counts (PREREG section 1), for P1, the trade-off area (in biology-score units) and the local-versus-global test. A direction is declared when the probability that the difference lies beyond the margin in that direction is at least 95%; practically equivalent when the probability that it lies within plus or minus the margin is at least 95%; otherwise inconclusive. Measurement noise can still be reported, descriptively.
- source: "Margin 0.01 (Recommended)" (user, ask_user answer to "Which margin? (when the overall difference counts)", 2026-10-14)
- added: 2026-10-14
- keywords: margin, ROPE, declaration, equivalent, inconclusive, measurement noise
- files: docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py
- notebook: NB-20261014-08

## SI-75 · active · method
- rule: P2: a run collapses when it fails, or when its biology score is 0.10 or more below the same seed's plain-scVI run. Rates per method and decoder, per dataset and pooled over the 6 real datasets, with exact (Clopper-Pearson) 95% intervals; also combine the per-dataset differences in collapse rate (critic minus discriminator) with the same model as P1 (prior scales 0.25), and declare a direction only beyond a margin of 5 percentage points.
- source: "Rates plus a formal comparison (Recommended)" (user, ask_user answer to "Should P2 also formally compare the reference critic with the discriminator?", 2026-10-14)
- added: 2026-10-14
- keywords: P2, collapse rate, collapse, Clopper-Pearson, percentage points
- files: docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py
- notebook: NB-20261014-09

## SI-76 · active · method
- rule: Local versus global as one signed question: at matched batch removal, does the critic get there through more local mixing (local average of iLISI, kBET and graph connectivity above the discriminator's, which is the same as the global average of principal-component regression and batch silhouette width below it)? The local average is read at the same crossing of the target as P1; per decoder an exact two-sided sign test over the real datasets plus the P1 combining model with margin 0.01. Rewords the paper plan's hypothesis (dated note in the plan).
- source: "One signed question (Recommended)" (user, ask_user answer to "How should this analysis be set up? (local versus global batch metrics)", 2026-10-14)
- added: 2026-10-14
- keywords: local, global, local-global, iLISI, sign test, local mixing
- files: docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py, docs/PAPER_PLAN.md
- notebook: NB-20261014-10

## SI-77 · active · method
- rule: Selection score for the single-strength tables, the comparison with standard tools at equal numbers of tries, and the width of the range of good strengths: 0.4 x batch score + 0.6 x biology score (scIB's weights), without scIB's min-max scaling.
- source: "scIB weights, unscaled (Recommended)" (user, ask_user answer to "Which score? (ranking a method's penalty strengths)", 2026-10-14)
- added: 2026-10-14
- keywords: selection score, transfer lambda, best-of-k, window width, min-max, scIB weights
- files: docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py
- notebook: NB-20261014-11

## SI-78 · active · process
- rule: A score is examined when a person reads its value or an analysis script uses it; checks that print only counts or flags do not count. This applies to every X1 and X13 score, of any arm (not only X1 adversarial-arm scores).
- source: "Every X1 and X13 score, any arm (Recommended)" (user, ask_user answer to "What counts as having 'examined' an X1 result?", 2026-10-15)
- added: 2026-10-15
- keywords: examined, blinding, X1 score, X13 score, freeze
- files: docs/PREREG.md, docs/ANALYSIS_SPEC.md
- notebook: NB-20261015-02

## SI-79 · active · process
- rule: Record SHA-256 hashes of every existing X1 and X13 score file (hashing reads bytes and shows no value) and the date GitHub shows for the tag prereg-tier12-v3 as an outside time stamp; commit the hash list and state it in PREREG.
- source: "Record hashes and the tag's GitHub date (Recommended)" (user, ask_user answer to "Should we record a fingerprint (a SHA-256 hash) of every existing X1 and X13 score file?", 2026-10-15)
- added: 2026-10-15
- keywords: hash, SHA-256, score files, time stamp, tag date
- files: docs/PREREG.md
- notebook: NB-20261015-03

## SI-80 · active · method
- rule: The agent's proposals in docs/ANALYSIS_SPEC.md are approved as rules: section 2.3 (not-evaluable rule), section 5 (marginal declarations; Holm over the six primary results for the descriptive Wilcoxon companion only), section 8 (leave-one-family-out transfer rule), section 9 (best-of-k placement against standard tools), section 10 (width of the range of good strengths), the section 11 sensitivity analyses that no decision names, and section 15 (implementation details). Parts that implement the rigour-review designs asked separately at the same review (the zero-spread exclusion in sections 2.3 and 4, question 1; the t-based sign-table limits in section 15 and the other designs of question 2) follow the user's answers to those questions.
- source: "Approve all (Recommended)" (user, ask_user answer to "Do you approve the agent's proposals in the analysis specification (docs/ANALYSIS_SPEC.md)?", 2026-10-15)
- added: 2026-10-15
- keywords: proposal, not evaluable, Holm, transfer rule, best-of-k, window width, sensitivity, implementation details
- files: docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py
- notebook: NB-20261015-04

## SI-81 · active · process
- rule: (a) Each case where scib finds zero normalised mutual information at every clustering resolution (SI-65) is decided by the user and listed with the results. (b) A Sinkhorn iteration-cap hit (500,000 iterations, SI-61) stops the stage until the user decides whether the fit counts as failed; failed fits count as collapses in P2, and each such decision is listed with the results (clause confirmed by the user, 2026-10-16). (c) The no-pair training-step counts (CF-06) and, per arm, the number of cells not evaluable because the arm never reaches b* (AN-R15) are tabulated by a reporting script written after the runs; the tag freezes this promise, not the script (user, 2026-10-16).
- source: "Approve all three (Recommended)" (user, ask_user answer to "Do you approve three process proposals in the pre-registration?", 2026-10-15); "Keep, add to ledger (Recommended)" and "Promise only (Recommended)" (user, ask_user answers on (b) and (c), 2026-10-16)
- added: 2026-10-15
- keywords: NMI-0, NMI, iteration cap, no-pair, unreached, reporting script
- files: docs/PREREG.md, docs/ANALYSIS_SPEC.md
- notebook: NB-20261015-05, NB-20261016-11, NB-20261016-12

## SI-82 · active · method
- rule: PAPER_PLAN's four scaling checks (leave-one-method-out rescaling, z-score aggregation, a composite without the silhouette metrics, rank stability by Kendall's tau) are all reported, labelled not pre-registered, and used for no conclusion (exploratory).
- source: "Report all four as exploratory (Recommended)" (user, ask_user answer to "What should happen to the paper plan's four scaling checks?", 2026-10-15)
- added: 2026-10-15
- keywords: scaling sensitivities, z-score, leave-one-method-out, silhouette, Kendall, exploratory
- files: docs/PAPER_PLAN.md, docs/ANALYSIS_SPEC.md
- notebook: NB-20261015-06

## SI-83 · active · method
- rule: A dataset whose leave-one-seed-out standard error is below 0.000001 is kept in the combined analysis with its standard error raised to 0.000001, instead of being left out as not evaluable (zero_spread); the same applies to the paired-datasets sensitivity analysis; each dataset raised this way is listed in the report.
- source: "Keep it; raise its standard error to the cut-off 0.000001 (Recommended)" (user, ask_user answer to "What should happen to a dataset whose run-to-run uncertainty comes out as exactly zero?", 2026-10-15); earlier in the same round: "If we have a 'certain' dataset, why would we drop it as it is performing well and this would bias the results towards the uncertain datasets" (user, 2026-10-15)
- added: 2026-10-15
- keywords: zero_spread, zero spread, standard error, floor, S_MIN, cut-off
- files: scripts/analysis_x1.py, docs/ANALYSIS_SPEC.md
- notebook: NB-20261015-16

## SI-84 · active · method
- rule: Approved: the log-spaced 201-point grid for the between-dataset spreads (AN-01), the solve-based formula for the overall difference, recomputing b0, b_common, b* and the low-signal flag from the same rows and stopping on a mismatch (AN-03), t-based sign-table limits (AN-06), and the per-run sensitivity limited to P1 and the local-versus-global test (AN-05); the specification states where the grid is accurate and that the sign-table limits are approximate.
- source: "Approve all five (Recommended)" (user, ask_user answer to "Do you approve five design choices the agent made in the X1 analysis code after the rigour review?", 2026-10-15)
- added: 2026-10-15
- keywords: log grid, Gauss-Legendre, precision, recompute, t-based, sign table, per-run
- files: scripts/analysis_x1.py, docs/ANALYSIS_SPEC.md
- notebook: NB-20261015-17

## SI-85 · active · process
- rule: The stage runner refuses a divergence record of any arm (user, 2026-10-16) that carries no git commit or was made from a checkout with uncommitted changes, as it refuses such results (code re-check CF-R03, rule A; runner-made nonfinite_latent records stay accepted); one test covers the refusals for records and results together.
- source: "Refuse them, as for results (Recommended)" (user, ask_user answer to "Should the stage runner refuse a Sinkhorn 'divergence record' that lacks the code version, or that was made from code with uncommitted changes?", 2026-10-15); "Every arm (Recommended)" (user, ask_user answer to "Should the stage runner refuse a divergence record (the file saying a fit failed) with no code version for every arm, or only for Sinkhorn?", 2026-10-16)
- added: 2026-10-15
- keywords: divergence record, git_sha, git_dirty, uncommitted, run_stage
- files: scripts/run_stage.py, tests/stage/test_run_stage.py, tests/stage/fake_env.py
- notebook: NB-20261015-18, NB-20261016-09

## SI-86 · active · method
- rule: prereg_rules.power_report restarts its root search at 0.01 when the search from 1.0 fails; every result that works today stays bit-identical; applied before the tag (documents re-check DOC-R12).
- source: "Apply the fix before the tag (Recommended)" (user, ask_user answer to "Should the error in the power function be fixed before the tag?", 2026-10-15)
- added: 2026-10-15
- keywords: power_report, power, minimal detectable effect, non-central t, NaN
- files: scripts/prereg_rules.py, tests/prereg
- notebook: NB-20261015-19

## SI-87 · active · process
- rule: The Sinkhorn redo overwrites docs/prereg/r1_record.json, r1_record_round1.json, r4_a1_record.json, r4_a1_record_round1.json and r2r3_record.json in place (no *_sinkhorn_redo.json names); the versions before the redo stay at tag prereg-tier12-v3, which is made before the redo, and in the commits that froze them (they do not exist at prereg-tier12-v2, which predates them; correction NB-20261015-21); PREREG says so, and the redo's notebook entry lists every decision for the other five arms that changed.
- source: "Overwrite; old versions kept at the tags (Recommended)" (user, ask_user answer to "Do you approve these file names for the redone Sinkhorn decision records?", 2026-10-15); first answer: "if the previous sinkhorns are not relevant, just overwrite them" (user, 2026-10-15)
- added: 2026-10-15
- keywords: record, r1_record, r4_a1_record, r2r3_record, sinkhorn_redo, overwrite, redo
- files: docs/PREREG.md, docs/ANALYSIS_SPEC.md
- notebook: NB-20261015-20, NB-20261015-21

## SI-88 · active · process
- rule: The two displays of score values found by the 2026-10-15 audit (2026-10-02: all scores of six test fits from one of the two X13 runner-test runs on atac_small; 2026-10-13: PCR_batch and cell-cycle conservation of one X1 discriminator run and one X13 sysVI run in the scorer identity check) are listed in PREREG as shown to the agent only, not as examined under SI-78; the facts, dates and limits are stated in full.
- source: "No, list them as shown to the agent only" (user, ask_user answer to "Your rule SI-78 says a score is 'examined' when a person reads its value. Should these two cases count as examined?", 2026-10-15)
- added: 2026-10-15
- keywords: examined, displayed, shown to the agent, identity check, audit, SI-78
- files: docs/PREREG.md, docs/ANALYSIS_SPEC.md
- notebook: NB-20261015-25

## SI-89 · active · process
- rule: The stage runner refuses a latent that names no git commit (no git_sha in its config), as it refuses a divergence record without one.
- source: "Refuse it (Recommended)" (user, ask_user answer to "Should the stage runner also refuse a latent (a fitted embedding) that names no git commit?", 2026-10-16)
- added: 2026-10-16
- keywords: latent, git_sha, run_stage, provenance
- files: scripts/run_stage.py, tests/stage/test_run_stage.py
- notebook: NB-20261016-10
