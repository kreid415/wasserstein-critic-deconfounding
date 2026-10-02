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
