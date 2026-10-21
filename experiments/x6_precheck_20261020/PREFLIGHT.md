# PREFLIGHT: x6_precheck_20261020

**Verdict: GO**  (checked 2026-10-21T16:16:54Z; plan_sha 69ceead66a17e69c)

## Plan

- **question**: Per arm (discriminator_sn_critic_opt; pooled_sn at n_critic 1, SI-107): does the batch-score curve over the base arm's inherited pilot window (plain discriminator; pilot pooled critic) bracket the pilot target b*?
- **primary_outcome**: per-cell status (reached / unreached_low / unreached_high) of batch score B vs pilot b*
- **unit_of_analysis**: cell = arm x task x decoder (per arm: 4 counted, 2 low-signal reported only)
- **split_unit**: none: no train/test split; one pilot seed (100), fits on full pilot tasks as in A1
- **selection_rule**: per arm separately: shift_up if >=3 of 4 counted cells unreached_low, shift_down if >=3 unreached_high, else keep (PLAN.md and addendum 2026-10-21)
- **n_comparisons**: 2
- **multiple_comparison_correction**: none: two separate design decisions (one per arm), each moving only its own arm's strengths by at most one grid step; no claim is made
- **data_filters**: tasks atac_small, immune, sim1 as prepared for the pilot (D/prepped_scib); no extra filtering

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha bfbbd672fab892ea) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=2 (one design decision per arm); no correction: each decision moves only its own arm by at most one grid step and supports no claim |
| PF-03 | blocking | Unit of analysis | n/a | no classifier/probe/resampling evaluated; outcome is the frozen batch score B per fit |
| PF-04 | blocking | ID integrity | n/a | no merging of sources; each pilot task is the prepared file used by the pilot (D/prepped_scib) |
| PF-05 | major | Nesting of factors | n/a | single categorical comparison: status against b* per cell |
| PF-06 | blocking | Selection bias | n/a | no best-of selection precedes a number; strengths are fixed in advance from the frozen pilot record |
| PF-07 | blocking | Matched comparison | pass | each pre-check row equals the pilot base-arm row of the same task/cond/seed/strength except experiment, arm and (pooled_sn) n_critic 1; sim1/1 pooled lo 0.03 has no pilot fit and copies the pilot pooled row of the same task/cond/seed, which differs only in strength (build_manifest.py at fbd407c; ... |
| PF-08 | blocking | Aggregation | pass | no averaging across tasks or arms; each arm's decision counts its own per-cell statuses over the 4 named counted cells (decide.summarise, COUNTED) |
| PF-09 | blocking | No-op baselines | n/a | no perturbation/noise knob in this pre-check; adversary strengths only |
| PF-10 | blocking | Sample size | n/a | no conditions created by filtering or subsampling |
| PF-11 | major | Metric vs model structure | n/a | same latent structure (scVI, n_latent 10) as the pilot discriminator rows |
| PF-12 | blocking | Metric implementation | n/a | no new metric: B = mean of frozen batch metrics via prereg_rules.outcomes on the frozen scorer |
| PF-13 | major | Estimator noise | pass | single seed by design; stated in PLAN.md as a design check, not a result; reading noise acknowledged |
| PF-14 | major | Test vs control | pass | design and rule committed before any pre-check fit: arm 1 at 859e572 (2026-10-20), arm 2 addendum at 6163c4f and frozen by tag prereg-x6-v1 (635be7f, 2026-10-21); builder note fbd407c; no pre-check fit exists (dry run needs_fit 36) |
| PF-15 | major | Attribution | n/a | no mechanism is attributed; result is a design check only |
| PF-16 | blocking | Timing confounds | n/a | no runtime number is reported; the cost estimate (PF-17) uses measured pilot fit times |
| PF-17 | blocking | Cost estimate | pass | cost from measured A1 fit wall times (tier12_v2 attempts, RTX 3080): plain discriminator 19.9 min mean, pooled critic 63.0 min mean at n_critic 5 (an upper bound for pooled_sn at n_critic 1, which runs 1 critic step and no gradient penalty). 18x19.9 + 18x63.0 = 24.9 GPU-h serial; at the 6.8 effec... |
| PF-18 | major | Co-scheduling | pass | run_precheck.sh waits (10-min checks, two in a row) until no run_stage.py is alive and GPU memory < 500 MiB, so it starts only after the main-benchmark local run (tmux x1loc) ends; it never shares the GPU |
| PF-19 | blocking | Launch manifest | pass | manifest built by build_manifest.py at fbd407c from m.r4a1.tsv sha256 549f2216.. and r4_a1_record.json sha256 15b4e8ea..; 36 rows (18 discriminator_sn_critic_opt, 18 pooled_sn n_critic 1); strengths equal the record's lo/matched/hi for every arm x cell; adv_input mean, zstd 0, seed 100; the 18 ea... |
| PF-20 | major | Count reconciliation | n/a | no schedule depends on grid size beyond the 36-row cost estimate in PF-17 |
| PF-21 | major | Replicates vary | n/a | no aggregation over seeds: single pilot seed 100 by design (PLAN.md) |
| PF-22 | blocking | Output persistence | pass | run_stage.py ledger/claims make the run resumable; outputs under tier12_v3/x6_precheck/run (new dir, absent before launch); exit line appended to logs/x6_precheck_exit.txt; failures go to failures.csv with frozen statuses |
| PF-23 | blocking | Provenance and inclusion | pass | inputs are the pilot's prepared files D/prepped_scib (same as stage-2 runs) and the frozen pilot rule record r4_a1_record.json sha256 15b4e8ea.. |
| PF-24 | major | Config vs input | pass | strengths carried over from the pilot record by design (the question is whether they transfer); adv_input mean, zstd 0 copied from the pilot rows that defined b*; pooled_sn n_critic 1 is the X6 setting being checked |
| SC-01 | blocking | splits | n/a | no held-out label metric or classifier |
| SC-02 | blocking | splits | n/a | no cohorts pooled; per task x decoder cells only |
| SC-03 | blocking | splits | n/a | no train/test split beyond scvi-tools' internal train_size 0.9 copied from pilot rows |
| SC-04 | major | splits | n/a | no custom split function |
| SC-05 | blocking | gene panel | n/a | no gene panel built; prepared pilot inputs reused unchanged |
| SC-06 | major | gene panel | n/a | no foundation model |
| SC-07 | blocking | perturbation | n/a | no perturbation axis |
| SC-08 | blocking | perturbation | n/a | no severity-varying filtering |
| SC-09 | blocking | perturbation | n/a | no dropout/thinning before fitting |
| SC-10 | major | gene panel | n/a | no HVG step changed; pilot inputs reused |
| SC-11 | blocking | inputs | pass | same loader and obs keys as pilot fits: rows copied from A1 rows, fitted by fit_paper_config.py at fbd407c, whose fit code is identical to tag prereg-x6-v1 (identity check vs prereg-tier12-v3 57/57, NB-20261021-06) |
| SC-12 | major | claims | n/a | no disentanglement claim |
| SC-13 | major | metrics | n/a | no covariates scored against each other |
| SC-14 | blocking | metrics | n/a | no block-latent models |
| SC-15 | major | metrics | n/a | no block-aware metric |
| SC-16 | blocking | metrics | pass | kBET/LISI computed by the frozen scorer on full tasks, as in the pilot; no subsampling |
| SC-17 | major | metrics | pass | prereg_rules.outcomes refuses non-finite required metrics and records failed fits as failed; decide.py marks a cell not_readable with <2 ok fits |
| SC-18 | major | aggregation | n/a | no averaging across datasets |
| SC-19 | major | inputs | pass | output dir tier12_v3/x6_precheck/run does not exist before launch (folder lists only the manifests); dry run planned needs_fit 36 |
| SC-20 | blocking | runners | pass | runner retries infrastructure errors once (max-attempts 2) and stops; fit failures are recorded with frozen statuses, never silently refit |
| SC-21 | minor | metrics | n/a | no classifier/MI metric on continuous factors |
| SC-22 | minor | claims | n/a | metric suite not justified by convention here; frozen pre-registered suite |
