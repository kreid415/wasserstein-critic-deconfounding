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
