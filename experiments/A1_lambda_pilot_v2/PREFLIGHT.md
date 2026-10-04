# PREFLIGHT: A1_lambda_pilot_v2

**Verdict: GO**  (checked 2026-10-04T21:41:20Z; plan_sha da141677a23f99af)

## Plan

- **question**: A1 lambda pilot (PREREG R1): for each arm family x decoder, which 6 lambda values on the half-decade grid 0.1-3000 span no effect to over-correction or collapse; fixes X1's lambda grid and gives sigma/tau for the power report
- **primary_outcome**: seed-paired change in the PREREG batch score B (mean of the 5 raw scIB batch metrics) relative to lambda=0; the bio score C enters R1 only as the collapse bound
- **unit_of_analysis**: one fit = task x decoder x arm x lambda x seed; R1 decides per arm family x decoder
- **split_unit**: seed: A1 uses seeds 100-102, disjoint from X1 (0-4) and follow-ups (10-12); scoring uses all cells as in scIB, no cell-level evaluation split
- **selection_rule**: R1 chooses 6 lambda per family from A1 (atac_small, immune, sim1; seeds 100-102); X1 evaluates them on fresh seeds 0-4 on all 8 tasks; no A1 fit is reused as an X1 fit (SI-13, PREREG sec 0)
- **n_comparisons**: 1
- **multiple_comparison_correction**: none needed: A1 makes no inferential claim; R1 is a deterministic pre-registered rule (PREREG sec 2)
- **data_filters**: scIB task files as distributed (SI-15), prepped by scripts/prep_scib_task.py --counts scib, fingerprints verified (docs/prepped_fingerprints_scib.json); training features = scIB HVGs (var.highly_variable)

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha da141677a23f99af) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=1; correction: none needed, A1 makes no inferential claim; R1 is a deterministic pre-registered rule (PREREG sec 2); plan identical to the v1 record (sha da141677a23f99af) |
| PF-03 | blocking | Unit of analysis | n/a | no classifier, probe or held-out evaluation: scIB metrics are computed on all cells of each task (scIB convention); scVI's internal 90/10 split only sets the validation loss |
| PF-04 | blocking | ID integrity | n/a | no unit-level evaluation; batches are the integration targets from the scIB key as distributed (SC-11 key check passed on all 3 files) |
| PF-05 | major | Nesting of factors | pass | scpf_check_nesting(batch, celltype) re-run 2026-10-04 on the 3 prepped A1 files: no deterministic nesting, no warnings (atac_small 3 batches/7 types, immune 10/16, sim1 6/7) |
| PF-06 | blocking | Selection bias | pass | [auto:pf_check_selection] selection and report sets disjoint (1098 vs 3000 ids) |
| PF-07 | blocking | Matched comparison | pass | [auto:pf_check_matched_arms] arms matched on 9 fields (varied: ['adv_opt', 'adv_updates', 'adv_width', 'lipschitz']) |
| PF-08 | blocking | Aggregation | pass | scpf_check_pool_coverage(arm x task) on the A1 rows of manifest v3: every arm covers atac_small, immune, sim1 x 2 decoders x seeds 100-102; each adversarial arm has 10 lambda; problems [] |
| PF-09 | blocking | No-op baselines | pass | lambda=0 'none' arm = stock scVI: tests/scvi (incl. test_none_is_bit_identical_to_stock_scvi) 110 passed, 1 skipped at 32cbbce; A1 latents at 32cbbce equal the tag's (14/14, max /dz/ 0; experiments/integration_checks_20261004/a1_fit_identity.sh) |
| PF-10 | blocking | Sample size | n/a | A1 creates no conditions by filtering, subsampling or perturbing: every fit uses the full prepped task (11,270 / 33,506 / 12,097 cells) |
| PF-11 | major | Metric vs model structure | pass | all arms produce 10-dimensional unstructured posterior means scored by the same scIB functions; no metric assumes a latent structure some arms lack |
| PF-12 | blocking | Metric implementation | pass | metrics from scib 1.1.7 via score_scib_native.py; tests/prereg (B/C aggregation known answers) and tests/scoring pass at 32cbbce: other 312 passed, x13+scoring 29 passed |
| PF-13 | major | Estimator noise | pass | A1 estimates the paired seed SD; R1's effect threshold max(delta_min 0.01, 3 x noise) (PREREG sec 1); kBET seeded per call (score rows record kbet_seed and kbet_r_calls); CR-07 pairing wording amended in PREREG sec 9 |
| PF-14 | major | Test vs control | pass | A1 contains the lambda=0 control and every test arm at all 10 lambda on both decoders (manifest v3 A1 rows) |
| PF-15 | major | Attribution | n/a | A1 attributes no mechanism; it only sets X1's lambda grid |
| PF-16 | blocking | Timing confounds | pass | step costs from single-lane fits with startup removed (docs/throughput_rtx3080_stock_backbone.csv); calibrated 2026-10-04 on 202 v1 A1 fits (atac_small): cost-weighted observed/predicted 1.128, pooled critic 1.54 (2,997 vs 1,949 s median), other arms 1.00-1.13; to be rechecked on the first immune... |
| PF-17 | blocking | Cost estimate | pass | [auto:pf_estimate_cost] estimate 110.77 h (x1.5 safety = 166.15 h); cpu-hours 110.8; recommended walltime 598140s |
| PF-18 | major | Co-scheduling | pass | 8 fit lanes + 3 scoring workers during fits (8 after), as in the v1 A1 run: 202 fits, 0 failures (tier12/A1/failures.csv empty), GPU 99% at 3.1 GB of 10 GB; no other job runs locally (Scanorama timing and JHPCE-latent scoring ended) |
| PF-19 | blocking | Launch manifest | pass | [auto:pf_check_manifest] 1098 rows carry all 12 intended flags |
| PF-20 | major | Count reconciliation | pass | independent count 3 tasks x 2 decoders x 3 seeds x (1 lambda=0 + 6 arms x 10 lambda) = 1098 = A1 rows of manifest v3; v3 rebuilt by the builder -> sha256 6efbe422c8f0... identical; A1 rows byte-identical to the tagged manifest; run_stage --dry-run at 32cbbce (run_A1 arguments): 1098 rows needs_fit |
| PF-21 | major | Replicates vary | pass | [auto:pf_check_replicates_differ] 3 of 3 replicates distinct |
| PF-22 | blocking | Output persistence | pass | run_stage.py writes per tag OUT/latents/<tag>.npz, OUT/models/<tag>/, OUT/scores/<tag>.csv, ledger, attempts and failures.csv; fit failures classified (fit_outcome.py) and quarantined; real fitter+scorer e2e passed at 32cbbce (tests/stage 62 passed); the scorer (unchanged since 080f417) wrote a r... |
| PF-23 | blocking | Provenance and inclusion | pass | inputs = scIB task files as distributed (SI-15); fingerprint_prepped.py --compare docs/prepped_fingerprints_scib.json exit 0 on all 8 prepped files, 2026-10-04 |
| PF-24 | major | Config vs input | pass | training genes <= features (2,000 of 12,303 / 9,979; atac_small all 3,429); kNN k (15 graph, 90 LISI, <=70 kBET) below the smallest cell type (226/129/204, re-measured 2026-10-04) |
| SC-01 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1; scIB integration metrics are computed on all cells |
| SC-02 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1; scIB integration metrics are computed on all cells |
| SC-03 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1; scIB integration metrics are computed on all cells |
| SC-04 | major | splits | n/a | no custom grouped-split function is used anywhere in A1 |
| SC-05 | blocking | gene panel | n/a | no gene panel built from a reference: each task's features are scIB's own HVGs (2,000) or, for atac_small, all 3,429 gene-activity features |
| SC-06 | major | gene panel | n/a | no foundation model with a fixed gene vocabulary is used |
| SC-07 | blocking | perturbation | n/a | no perturbation axis in A1 (X3's depletion dose gets its own preflight) |
| SC-08 | blocking | perturbation | n/a | no perturbation or filtering axis in A1 (X3/X8 get their own preflight) |
| SC-09 | blocking | perturbation | pass | scpf_check_empty_cells on layers['counts'] restricted to the training features, re-run 2026-10-04: 0 empty cells in atac_small (11,270), immune (33,506), sim1 (12,097) |
| SC-10 | major | gene panel | pass | training genes = var.highly_variable: 3,429 (atac_small, all features), 2,000 (immune), 2,000 (sim1), re-measured 2026-10-04; equal to n_hvg in docs/prepped_fingerprints_scib.json (compare exit 0) |
| SC-11 | blocking | inputs | pass | scpf_check_obs_keys: 'batch' and 'celltype' present by name in all 3 files; batches 3/10/6 and cell types 7/16/7 (re-run 2026-10-04) |
| SC-12 | major | claims | n/a | A1 makes no integration or disentanglement claim; it only fixes X1's lambda grid |
| SC-13 | major | metrics | pass | same check as PF-05: scpf_check_nesting(batch, celltype) finds no deterministic nesting in the 3 A1 tasks (2026-10-04) |
| SC-14 | blocking | metrics | n/a | every arm yields an unstructured 10-dimensional posterior-mean latent; no block-structured models |
| SC-15 | major | metrics | n/a | no block-aware metric is used; latents are unstructured |
| SC-16 | blocking | metrics | pass | scpf_check_knn_validity: smallest cell type 226 / 129 / 204 cells exceeds LISI's 90 neighbours and kBET's k0 <= 70 (ok on all 3 tasks, 2026-10-04) |
| SC-17 | major | metrics | pass | new scorer row (atac_small pooled, 1-epoch latent, 080f417): non-computable metrics are NaN, not placeholders (cell_cycle_conservation, hvg_overlap, trajectory); traj_root_fallback blank when trajectory is off; kBET reports kbet_labels_skipped=1, kbet_labels_forced_one=0, kbet_r_calls=6 of 7 labe... |
| SC-18 | major | aggregation | pass | PASS SC-18 scpf_check_pool_coverage (n_models=7, n_datasets=3, rank_flip=False) (A1 rows of manifest v3, arm x task) |
| SC-19 | major | inputs | pass | fresh output directory tier12_v2/A1 (absent before launch; tier12/A1 holds the discarded v1 fits); run_stage.py refuses a dirty checkout (--allow-dirty is test-only) and latents from more than one commit (latent_git_shas check unless --allow-mixed-sha) |
| SC-20 | blocking | runners | pass | scpf_scan_fallbacks on 11 pipeline files at 32cbbce (incl. harvest_local.py, freeze_matched_lambda.py: 0 hits): 7 hits, all docstrings/comments describing refusals or code that RECORDS scib's own fallbacks (score_scib_native.py l.37/41/42/58/228/232, CR-08; run_cpu_baselines.py l.56 comment); no ... |
| SC-21 | minor | metrics | pass | trajectory conservation uses continuous DPT pseudotime (scib trajectory_conservation, Spearman), nothing binned; among the A1 tasks only immune carries dpt_pseudotime ({'atac_small': False, 'immune': True, 'sim1': False}), so trajectory is computed there only |
| SC-22 | minor | claims | pass | unchanged since the v1 record: the metric suite and bio/batch grouping follow scIB (Luecken et al. 2022, Nat. Methods) and scib-reproducibility plotSingleTaskRNA.R l.45-46 and l.142-143 (PREREG sec 1); trajectory in the bio score per SI-28 |
