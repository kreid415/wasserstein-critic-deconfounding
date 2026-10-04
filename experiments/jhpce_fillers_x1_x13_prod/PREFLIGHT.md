# PREFLIGHT: jhpce_fillers_x1_x13_prod

**Verdict: GO**  (checked 2026-10-04T21:41:43Z; plan_sha 22405057878fe677)

## Plan

- **question**: JHPCE production filler fits at the review commit: jobA (atac_large, immune_hum_mou; 100 rows) and jobB (lung, pancreas, sim2; 150 rows) = the 250 manifest-v3 rows that PREREG sec 0 allows before the lambda grid is frozen (X1 none c0/c1, X1 scvi_adv, X13 scANVI, X13 sysVI), one L40S per job (SI-39), fit only; latents scored locally after harvest
- **primary_outcome**: completion gate of each job: every tagged row final (fitted, diverged or nonfinite_latent) with valid latents harvested by SHA-256; the scientific outcome of these rows (PREREG B and C) is computed later by local scoring
- **unit_of_analysis**: one fit = manifest row (task x arm x decoder x knob x seed)
- **split_unit**: seed (X1/X13 seeds 0-4); no cell-level evaluation split (scIB convention: metrics on all cells)
- **selection_rule**: none in this job: rows fixed by manifest v3 (sha256 6efbe422...) and the committed tags files; no lambda, epoch or checkpoint is selected
- **n_comparisons**: 1
- **multiple_comparison_correction**: none needed: the job makes no inferential comparison
- **data_filters**: scIB task files as distributed (SI-15), prepped by scripts/prep_scib_task.py, fingerprints compared on the node with docs/prepped_fingerprints_scib.json; training features = var.highly_variable

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha 22405057878fe677) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=1 and no inferential comparison: the job fits fixed manifest rows; B/C are computed later by local scoring under PREREG rules |
| PF-03 | blocking | Unit of analysis | n/a | no held-out evaluation: scIB metrics use all cells (scIB convention); scVI's internal 90/10 split only logs validation loss (early_stopping False in fit_paper_config.py) |
| PF-04 | blocking | ID integrity | n/a | no unit-level evaluation in the job; batch/celltype come from the scIB files as distributed; prepped files are fingerprint-checked on the node (rehearsal with tag-carrying bundle: 8/8 OK) |
| PF-05 | major | Nesting of factors | n/a | no overlap/collision statistic is computed by a fit-only job; batch x celltype nesting is a scoring-time property (checked in the A1 preflight for its tasks) |
| PF-06 | blocking | Selection bias | pass | no selection in this job; the 250 rows (seeds 0-4) are disjoint from A1/A2/A3 seeds 100-102 and follow-up seeds 10-12 (build_paper_manifest seed check; tests/prereg in the 'other' suite, 312 passed at 32cbbce) |
| PF-07 | blocking | Matched comparison | pass | [auto:pf_check_matched_arms] arms matched on 8 fields (varied: ['kl_warmup', 'lambda_policy', 'likelihood', 'post_epochs']) |
| PF-08 | blocking | Aggregation | pass | no pooled mean is computed in the job; coverage is balanced: each of the 5 tasks has 10 none (c0+c1), 5 scvi_adv, 5 scanvi, 30 sysvi rows (tags files vs manifest v3: 250 = expected set, jobA 100, jobB 150, disjoint) |
| PF-09 | blocking | No-op baselines | pass | lambda=0 'none' = stock scVI: tests/scvi incl. test_none_is_bit_identical_to_stock_scvi 110 passed, 1 skipped at 32cbbce (WCD_REQUIRE_DATA=1); the same suite passes inside each rehearsal's bundle clone |
| PF-10 | blocking | Sample size | n/a | the job creates no condition by filtering or subsampling: every X1/X13 row uses the full prepped task |
| PF-11 | major | Metric vs model structure | n/a | no metric is computed on JHPCE; scoring happens locally with the scorer whose metric path is unchanged (A1 barycenter latent rescored at 080f417: 15/15 columns identical) |
| PF-12 | blocking | Metric implementation | n/a | no metric is computed in the job; the local scorer reproduces the tagged scorer exactly (15/15 columns on 4 latents: 3 in experiments/fix_score_fit_regression, 1 re-scored in this review) |
| PF-13 | major | Estimator noise | n/a | no claim or effect estimate is made by the job |
| PF-14 | major | Test vs control | n/a | no test-vs-control claim is made by the job; these are the lambda=0 and baseline arms the later comparisons need |
| PF-15 | major | Attribution | n/a | no causal explanation is offered by the job |
| PF-16 | blocking | Timing confounds | n/a | no timing, throughput or memory benchmark is reported; walltime sizing is covered by PF-17 |
| PF-17 | blocking | Cost estimate | pass | estimate 4 h (jobA) / 5.5 h (jobB) = local cost-model lane-h / 4.62 (L40S 8-lane factor, gate G5, 3-epoch immune, compute-171); requested 24 h = 4.4-6x; per-fit timeout 7200 s = 5.7x the longest predicted fit (21 min); stop guard at end-2700 s, resumable. Caveat: factor untested for scANVI/sysVI ... |
| PF-18 | major | Co-scheduling | pass | 8 lanes sized from measured per-fit peak RSS (docs/jhpce/prod_data/fit_peak_rss_cpu_1epoch.csv: ~4.3 GiB/lane largest tasks) and the gate's 8-lane L40S MaxRSS 18.4 GiB; 96 GB gives ~2.6x headroom; 10 CPUs = 8 lanes + 2 |
| PF-19 | blocking | Launch manifest | pass | [auto:pf_check_manifest] 250 rows carry all 7 intended flags |
| PF-20 | major | Count reconciliation | pass | independent count 5 tasks x (5 none c1 + 5 none c0 + 5 scvi_adv + 5 scanvi + 30 sysvi) = 250 = rows in fillers_x1_x13.tags; jobA 100 + jobB 150, disjoint; runner dry runs at 32cbbce: 100 rows (key ...__tags-d5da370a4d0c) and 150 rows (...__tags-91daef08ab82) |
| PF-21 | major | Replicates vary | pass | [auto:pf_check_replicates_differ] 2 of 2 replicates distinct |
| PF-22 | blocking | Output persistence | pass | end-to-end rehearsal of both jobs at 32cbbce44b62 (cluster/jhpce/local_rehearsal.sh after the RR-01 fix: guard test pins 129142b): exit 0 for jobA and jobB, problems [], fingerprints 8/8, on-node tests/scvi pass in the bundle clone, runner dry runs '[stage] X1+X13__atac_large+immune_hum_mou__tags... |
| PF-23 | blocking | Provenance and inclusion | pass | inclusion = the scIB task files as distributed; fingerprint_prepped.py --compare docs/prepped_fingerprints_scib.json: 8/8 OK in both tag-fixed rehearsals (rr/out/rehearsal_tagfix_job*/rehearsal.json) |
| PF-24 | major | Config vs input | pass | sizes valid: n_latent 10 << HVGs; batch 128; GPU smoke at 080f417 on RTX 3080 (pancreas X1 none c0/c1, scvi_adv, X13 scanvi, sysvi; 1 epoch) all exit 0 with finite latents; runner refuses placeholders and CPU rows without lanes (dry runs) |
| SC-01 | blocking | splits | n/a | no donor-level label is evaluated; no train/test split |
| SC-02 | blocking | splits | n/a | no donor/patient grouping key is used |
| SC-03 | blocking | splits | n/a | no train/test assignment exists (scIB metrics on all cells) |
| SC-04 | major | splits | n/a | no custom grouped-split function |
| SC-05 | blocking | gene panel | n/a | scIB task files as distributed (SI-15); no foundation-model gene vocabulary; species per task from the prepped file |
| SC-06 | major | gene panel | n/a | no model with a fixed gene vocabulary |
| SC-07 | blocking | perturbation | n/a | no perturbation axis |
| SC-08 | blocking | perturbation | n/a | no perturbation axis; full tasks |
| SC-09 | blocking | perturbation | n/a | no perturbation generator |
| SC-10 | major | gene panel | pass | every GPU arm is fitted on var.highly_variable of the prepped file (fit_paper_config.py main(): a = a[:, a.var['highly_variable']]) before any model; scANVI/sysVI receive the same features |
| SC-11 | blocking | inputs | pass | batch/celltype keys are read by name from the prepped file (uns batch_key/celltype_key; fitter uses obs['batch'], obs['celltype']); prepped fingerprints 8/8 on the node path |
| SC-12 | major | claims | n/a | no disentanglement or batch-demotion claim is made by a fit-only job |
| SC-13 | major | metrics | n/a | no collision/overlap count is computed |
| SC-14 | blocking | metrics | n/a | all latents are 10-d unstructured posterior means (no block structure) |
| SC-15 | major | metrics | n/a | no block-to-factor pairing |
| SC-16 | blocking | metrics | n/a | kNN metrics run at local scoring time on full tasks (smallest task 11,270 cells; scib k=15/50/kBET k0<=70) |
| SC-17 | major | metrics | n/a | no metric computed in the job; at scoring, scib fallbacks are flagged per row (traj_root_fallback, kbet_labels_*) and non-finite metrics are refused by the runner |
| SC-18 | major | aggregation | n/a | no pooled model means in the job |
| SC-19 | major | inputs | pass | the runner rejects latents by value: recorded manifest row (REQUIRED columns) must equal the row, device must contain L40S, dirty checkouts refused, X3 draw other than nested_v1 refused (run_stage.py Outputs.latent; fit_paper_config.x3_draw) |
| SC-20 | blocking | runners | pass | no fallback to another method: a divergence is recorded as 'diverged' (NonFiniteLossError, guarded for scANVI/sysVI since CR-06), any other non-zero exit is an infrastructure error retried at most twice; GPU smoke logs show no refit |
| SC-21 | minor | metrics | n/a | no continuous factor is binned |
| SC-22 | minor | claims | n/a | no 'standard metric' statement in the job |
