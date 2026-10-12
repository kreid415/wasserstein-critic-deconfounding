# PREFLIGHT: X1

**Verdict: GO**  (checked 2026-10-12T16:39:31Z; plan_sha d0d66db23582b874)

## Plan

- **question**: X1 main benchmark (PREREG sec 0, after R2/R3): fit and score the 3,000 X1 rows of manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv (sha256 7c2fe8679631...): 8 scIB tasks x 2 decoders (conditioned, unconditioned) x seeds 0-4; six adversarial arms at R1's 6 lambda per arm x decoder, adversary input = posterior mean and no standardisation (R2/R3), plus none and scvi_adv. 75 rule-free rows on the JHPCE tasks were fitted by the filler jobs; 1,125 rows (atac_small, immune, sim1) run locally as...
- **primary_outcome**: P1 (PAPER_PLAN sec 3/5): difference in bio conservation C at the matched batch-removal target b* between the reference W1 critic and the discriminator, on unscaled scIB aggregates; computed in the X1 analysis after all 3,000 rows are scored, not at launch
- **unit_of_analysis**: dataset (task) for the primary contrasts: hierarchical model with seeds nested in datasets, simulations a separate family (PAPER_PLAN sec 5); one fit = task x decoder x arm x lambda x seed
- **split_unit**: seed: X1 uses seeds 0-4, disjoint from the pilots A1-A3 (100-102) and the follow-ups (10-12); no pilot fit is reused as an X1 fit; scIB metrics use all cells (no cell-level split)
- **selection_rule**: nothing is selected at launch: X1's lambda grid (R1), adversary input and standardisation (R2/R3) were frozen from the pilots on seeds 100-102 before any X1 adversarial fit; R4-x1 (matched lambda for the follow-ups) runs on X1's scores afterwards; every X1 row is fitted and reported
- **n_comparisons**: 4
- **multiple_comparison_correction**: P1-P4 form one family: Bayesian hierarchical model with ROPE = max(0.01, 2 x measurement SD) for the declarations; frequentist companion exact Wilcoxon on dataset means with Holm over P1-P4 (PAPER_PLAN sec 5); X1 supplies P1 and P2
- **data_filters**: the 8 scIB task files as distributed (SI-15), prepped by scripts/prep_scib_task.py; fingerprints re-verified 2026-10-12 locally (8/8 OK) and re-checked on the JHPCE node by each job; training features = var.highly_variable; no cell, batch or gene filtering beyond the prep

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha d0d66db23582b874) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=4; correction: P1-P4 form one family: Bayesian hierarchical model with ROPE = max(0.01, 2 x measurement SD) for the declarations; frequentist companion exact Wilcoxon on dataset means with Holm over P1-P4 (PAPER_PLAN sec 5); X1 supplies P1 and P2 |
| PF-03 | blocking | Unit of analysis | n/a | no classifier, probe or held-out evaluation: scIB metrics use all cells of each task (scIB convention); scVI's internal 90/10 split only sets the validation loss |
| PF-04 | blocking | ID integrity | n/a | no unit-level evaluation; batches are the integration targets from each scIB file's own key (SC-11 key check passed on all 8 files); the 3,000 X1 tags are unique (make_x1_tags.py and pf_check_manifest) |
| PF-05 | major | Nesting of factors | pass | scpf_check_nesting(batch, celltype): no deterministic nesting in any of the 8 tasks (2026-10-12) |
| PF-06 | blocking | Selection bias | pass | [auto:pf_check_selection] selection and report sets disjoint (1143 vs 3000 ids) |
| PF-07 | blocking | Matched comparison | pass | [auto:pf_check_matched_arms] arms matched on 9 fields (varied: ['extra', 'n_critic']) |
| PF-08 | blocking | Aggregation | pass | [auto:pf_check_coverage] all 8 items cover all 8 units |
| PF-09 | blocking | No-op baselines | pass | lambda=0 'none' arm = stock scVI: tests/scvi (incl. test_none_is_bit_identical_to_stock_scvi) 110 passed, 1 skipped at c88cce3 in the X1 job rehearsal (immune_hum_mou, 2026-10-12); every JHPCE job reruns tests/scvi on its node before any fit (prod_fit_job.sh step 3) |
| PF-10 | blocking | Sample size | n/a | no conditions are created by filtering, subsampling or perturbing: every fit uses the full prepped task (11,270 to 97,861 cells) |
| PF-11 | major | Metric vs model structure | pass | all arms produce 10-dimensional unstructured posterior means scored by the same scIB functions; no metric assumes a latent structure some arms lack |
| PF-12 | blocking | Metric implementation | pass | no new or ported metric: the tagged scorer of A1-A3 (score_scib_native.py at c88cce3, scib 1.1.7), known-answer tests as in the A1 v2 record; re-scoring two A1 latents on 2026-10-12 reproduced all 18 metric columns of their A1 score files exactly, in the chat sandbox and as a cs-local host job (i... |
| PF-13 | major | Estimator noise | pass | replicate noise = 5 seeds per configuration; ROPE = max(0.01, 2 x measurement SD) (PAPER_PLAN sec 5); scIB metrics use all cells (no subsampling); kBET seeded per call (seed 0), recorded in every score row as in A1-A3 |
| PF-14 | major | Test vs control | pass | X1 contains the tested arms (reference, pooled and barycenter critics, discriminator, MMD, Sinkhorn; 6 lambda each) and the controls (none = lambda 0, scvi_adv) on every task and decoder; b* is set from all arms' own batch ranges (R4), not from the critic-discriminator contrast |
| PF-15 | major | Attribution | n/a | the launch attributes no mechanism; mechanism analyses are the follow-ups (X3, X7, X8) |
| PF-16 | blocking | Timing confounds | pass | timings are used only to size walltimes and estimate finish dates; no speed claim. L40S repeats: 15 configurations (filler X1 none/scvi_adv, 5 seeds each, same nodes and job body) spread 0.4-15.5% of the median (< 25%), no warm-up (pf_check_timing_repeats ok on all 15); local: the cost model with... |
| PF-17 | blocking | Cost estimate | pass | [auto:pf_estimate_cost] estimate 52.27 h (x1.3 safety = 67.96 h); cpu-hours 52.3; recommended walltime 244680s |
| PF-18 | major | Co-scheduling | pass | local: the same 8 fit lanes + 3 scorers as A1 on the same 3 tasks (A1: 1,098 fits, 0 failures; GPU <= 3 GB of 10 GB; ~35 GB RAM) plus ONE big-task filler scorer (25 GB, measured 2026-10-06) = ~60 GB of 94 GB; the second filler scorer is stopped when the local run starts (two would leave ~9 GB). T... |
| PF-19 | blocking | Launch manifest | pass | [auto:pf_check_manifest] 3000 rows carry all 12 intended flags |
| PF-20 | major | Count reconciliation | pass | independent count: 8 tasks x [2 decoders x 5 seeds x (6 arms x 6 lambda + none) + 1 decoder x 5 seeds x scvi_adv] = 8 x 375 = 3,000 = manifest X1 rows = 75 filler rows + 1,125 local (dry run: 1,125 needs_fit) + 1,800 JHPCE (5 tags files x 360; each rehearsal: 360 needs_fit) |
| PF-21 | major | Replicates vary | pass | [auto:pf_check_replicates_differ] 5 of 5 replicates distinct |
| PF-22 | blocking | Output persistence | pass | same runner as A1-A3: per tag latents/<tag>.npz, models/<tag>/, scores/<tag>.csv, attempts, ledger, failures.csv; the gate exits 0 only if every row is final. JHPCE jobs fit only and pack a harvest (latents, models, per-tag records) that is downloaded and scored here. Space: local 2.5 TB free; JH... |
| PF-23 | blocking | Provenance and inclusion | pass | [auto:pf_check_inclusion] all 8 items inside ['OK'] |
| PF-24 | major | Config vs input | pass | training genes <= features (2,000 of 8,135-19,093; ATAC tasks all features); max_epochs = scvi-tools heuristic on all 8 tasks (94-400); per-fit timeout 8 h vs longest predicted fit 2.0 h on an L40S (barycenter, immune_hum_mou); walltime 3 d vs <= 1.9 d predicted per job (2.7 d at 1.5x); the deadl... |
| SC-01 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in X1; scIB integration metrics are computed on all cells |
| SC-02 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in X1; scIB integration metrics are computed on all cells |
| SC-03 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in X1; scIB integration metrics are computed on all cells |
| SC-04 | major | splits | n/a | no custom grouped-split function is used |
| SC-05 | blocking | gene panel | n/a | no gene panel from a reference: each task's features are its own scIB HVGs (2,000) or, for the ATAC tasks, all 3,429 / 3,580 features |
| SC-06 | major | gene panel | n/a | no foundation model with a fixed gene vocabulary is used |
| SC-07 | blocking | perturbation | n/a | no perturbation or filtering axis in X1 (X3 and X8 get their own preflight) |
| SC-08 | blocking | perturbation | n/a | no perturbation or filtering axis in X1 (X3 and X8 get their own preflight) |
| SC-09 | blocking | perturbation | pass | no perturbation generator; scpf_check_empty_cells on layers['counts'] restricted to the training features: 0 empty cells in all 8 tasks (2026-10-12) |
| SC-10 | major | gene panel | pass | the fitter keeps var.highly_variable for every arm (fit_paper_config.py l.299): 2,000 features for immune, sim1, immune_hum_mou, lung, pancreas, sim2; all 3,429 (atac_small) and 3,580 (atac_large); one HVG selection, from scIB's files |
| SC-11 | blocking | inputs | pass | scpf_check_obs_keys: 'batch' and 'celltype' present by name in all 8 prepped files (batches 3/10/6/11/23/16/9/16, cell types 7/16/7/7/19/17/14/4 for atac_small/immune/sim1/atac_large/immune_hum_mou/lung/pancreas/sim2); the fitter reads obs['batch'] and obs['celltype'] by name |
| SC-12 | major | claims | pass | every adversarial arm is fitted with both decoders (conditioned and unconditioned) and each decoder has its own lambda=0 control, so integration claims are reported per decoder, separately from what the decoder's batch conditioning does |
| SC-13 | major | metrics | pass | same check as PF-05: scpf_check_nesting finds no deterministic nesting of batch and cell type in the 8 tasks |
| SC-14 | blocking | metrics | n/a | every arm yields an unstructured 10-dimensional posterior-mean latent; no block-structured models |
| SC-15 | major | metrics | n/a | no block-aware metric is used; latents are unstructured |
| SC-16 | blocking | metrics | pass | kNN graphs are built over all cells of a task: graph metrics k=15, LISI k=90, smallest task 11,270 cells (scpf_check_knn_validity ok on the full graphs of all 8). Per-label kBET follows scib 1.1.7, which sizes k0 from each label's batches or skips the label; skips and forced labels are written to... |
| SC-17 | major | metrics | pass | the tagged scorer writes non-computable metrics as NaN, never placeholders, and reports kBET skips (A1 v2 record); the runner fails any row with a non-finite required metric; all 288 A2/A3 and 48 scored filler X1 rows have every required metric finite |
| SC-18 | major | aggregation | pass | PASS SC-18 scpf_check_pool_coverage (n_models=8, n_datasets=8, rank_flip=False) (3,000 X1 rows, arm x task) |
| SC-19 | major | inputs | pass | PASS SC-19 scpf_check_allowed_values (n_checked=123, n_disallowed=0) (git sha of the 75 filler X1 latents and the 48 scored filler X1 rows, all c88cce3; fresh out dirs tier12_v2/X1_local and JHPCE tier12/x1_jhpce, both absent; the runner refuses dirty checkouts and mixed commits) |
| SC-20 | blocking | runners | pass | scpf_scan_fallbacks: 0 hits in the 7 new X1 scripts (make_x1_tags.py, x1_jhpce_command.py, rehearse_x1_job.sh, run_X1_local.sh, check_cslocal_identity.sh, env_record.py, compare_identity.py); fl_lint 0 findings; fitter, runner and scorer are the tagged files used by A1-A3 |
| SC-21 | minor | metrics | pass | trajectory conservation uses continuous DPT pseudotime (scib trajectory_conservation, Spearman), nothing binned; X1 tasks with dpt_pseudotime: immune, immune_hum_mou |
| SC-22 | minor | claims | pass | unchanged since the A1 record: the metric suite and bio/batch grouping follow scIB (Luecken et al. 2022, Nat. Methods) and scib-reproducibility (PREREG sec 1) |
