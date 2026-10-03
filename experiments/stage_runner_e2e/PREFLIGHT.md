# PREFLIGHT: stage_runner_e2e

**Verdict: GO**  (checked 2026-10-03T16:18:44Z; plan_sha 1289be6e0638665f)

## Plan

- **question**: Does scripts/run_stage.py, on the local RTX 3080 (4 fit lanes x 1 thread, 2 scorers while and after fitting, the lead's limits of 2026-10-03; KBET_SEED 0, clean committed checkout, --expect-device 'RTX 3080'), fit, score and classify a 6-row atac_small stage that includes one forced divergence, and pass its completion gate? Functional check only: the CPU is shared with other agents' scoring, so no timing from this run is a throughput measurement.
- **primary_outcome**: runner exit status 0 with ledger states 5 scored + 1 diverged (the lambda=1e39 row)
- **unit_of_analysis**: manifest row (one fit)
- **split_unit**: none: no model is evaluated on held-out data (engineering test of the runner)
- **selection_rule**: none: nothing is selected; no output of this run enters the paper, a rule or a decision record
- **n_comparisons**: 1
- **multiple_comparison_correction**: none needed: one primary outcome, no contrast between settings
- **data_filters**: atac_small as prepped (prepped_scib/atac_small__scib.h5ad: 11,270 cells x 3,429 features, all highly_variable); no filtering

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha 1289be6e0638665f) |
| PF-02 | major | Multiple comparisons | n/a | one primary outcome and no contrast between models, metrics or settings |
| PF-03 | blocking | Unit of analysis | n/a | no classifier, probe or resampling scheme is evaluated on held-out data |
| PF-04 | blocking | ID integrity | n/a | one prepped scIB file (atac_small); no merge of sources and no re-keyed units |
| PF-05 | major | Nesting of factors | n/a | no factors or labels are compared with each other |
| PF-06 | blocking | Selection bias | n/a | no best-of selection precedes any reported number |
| PF-07 | blocking | Matched comparison | n/a | arms are not compared: the rows only exercise the training-step branches of the plan |
| PF-08 | blocking | Aggregation | n/a | nothing is averaged or ranked across datasets |
| PF-09 | blocking | No-op baselines | n/a | no perturbation or identity knob is part of this test |
| PF-10 | blocking | Sample size | n/a | no conditions are created by filtering, subsampling or perturbing |
| PF-11 | major | Metric vs model structure | n/a | no metric is interpreted; scores are only checked for presence, finiteness and range by the runner |
| PF-12 | blocking | Metric implementation | n/a | no new or rewritten metric: scripts/score_scib_native.py is unchanged from main 878d8ea |
| PF-13 | major | Estimator noise | n/a | no effect is claimed from any metric |
| PF-14 | major | Test vs control | n/a | no new experimental design or discovery filter; engineering test |
| PF-15 | major | Attribution | n/a | no mechanism or advantage is attributed to a method |
| PF-16 | blocking | Timing confounds | n/a | no runtime, speed-up or memory number from this run is reported as a benchmark (the guard cost is a separate record, experiments/stage_runner_guard_cost) |
| PF-17 | blocking | Cost estimate | n/a | under one GPU-hour: 6 fits x 2 epochs on 11,270 cells; the CPU pilot of the same pipeline on a 1,200-cell subsample took 126 s (tests/stage/test_run_stage_e2e.py at 757f62f) |
| PF-18 | major | Co-scheduling | pass | 4 concurrent fits (lead's limit) on the RTX 3080 (10 GiB) on atac_small (11,270 cells); the 8-lane calibration on immune (33,506 cells) ran 64 fits with no failure (docs/concurrency_calibration_stock.json: fits 64, effective factor 3.77); 2 scorers on CPU with CUDA hidden |
| PF-19 | blocking | Launch manifest | pass | [auto:pf_check_manifest] 6 rows carry all 15 intended flags |
| PF-20 | major | Count reconciliation | pass | 6 rows = 5 design rows (A1 atac_small seed 100: none l0 c1, discriminator l1 c1, reference l1 c0, barycenter l1 c1, mmd l1 c1) + 1 forced divergence (discriminator l1e39 c1); experiments/stage_runner_e2e/make_manifest.py PICK + 1 |
| PF-21 | major | Replicates vary | n/a | no aggregation over seeds, folds or resamples |
| PF-22 | blocking | Output persistence | pass | the runner keeps latents/, models/, scores/, status/, attempts/, logs/, failures.csv and ledger/ in OUT_DIR; ledger csv/json, failures.csv, scores and runner log are saved as artifacts after the run |
| PF-23 | blocking | Provenance and inclusion | n/a | inputs are the single prepped scIB file used by every stage (scripts/prep_scib_task.py output) |
| PF-24 | major | Config vs input | pass | n_latent 10 < 3,429 features; scanpy neighbours k=15 < smallest batch 3,618 cells and kBET k=15 < 11,270 cells (scpf_check_knn_validity PASS, both kinds) |
| SC-01 | blocking | splits | n/a | no donor-level label and no split |
| SC-02 | blocking | splits | n/a | no pooled cohorts |
| SC-03 | blocking | splits | n/a | no train/test assignment is evaluated |
| SC-04 | major | splits | n/a | no custom split function |
| SC-05 | blocking | gene panel | n/a | no gene panel is built (prepped features as distributed) |
| SC-06 | major | gene panel | n/a | no foundation model |
| SC-07 | blocking | perturbation | n/a | no perturbation axis |
| SC-08 | blocking | perturbation | n/a | no perturbation or filtering axis |
| SC-09 | blocking | perturbation | n/a | no dropout or thinning |
| SC-10 | major | gene panel | n/a | no HVG selection (all 3,429 prepped features are highly_variable) |
| SC-11 | blocking | inputs | pass | PASS SC-11 scpf_check_obs_keys (); obs batch and celltype read by name in fit_paper_config.py, uns batch_key/celltype_key = batch/celltype in the prepped file |
| SC-12 | major | claims | n/a | no disentanglement or integration claim |
| SC-13 | major | metrics | n/a | no covariates are scored against each other |
| SC-14 | blocking | metrics | n/a | no block-structured latent |
| SC-15 | major | metrics | n/a | no block-aware metric |
| SC-16 | blocking | metrics | pass | PASS SC-16 scpf_check_knn_validity (kind=knn, k=15, n_min=3618, max_valid_k_at_n_min=3617, n_invalid_conditions=0); PASS SC-16 scpf_check_knn_validity (kind=kbet, k=15, n_min=11270, max_valid_k_at_n_min=11270, n_invalid_conditions=0) |
| SC-17 | major | metrics | pass | the scorer writes NaN for metrics it does not compute (atac_small: cell_cycle_conservation and trajectory NaN in the CPU e2e score rows) and the runner requires finiteness only for the task's metrics (prereg_rules.bio_metrics_for); a NaN required metric stops the stage (tests/stage test_nan_metri... |
| SC-18 | major | aggregation | n/a | no pooling across datasets |
| SC-19 | major | inputs | n/a | a fresh OUT_DIR is used; the runner refuses stale latents and scores by value (row, device, kBET seed, latent SHA-256) |
| SC-20 | blocking | runners | pass | fl_lint clean on run_stage.py, fit_outcome.py, fit_paper_config.py, scvi_adversarial_plan.py (9 files, 0 findings); the runner never substitutes a method or file: contract breaks are infrastructure errors; the run logs are scanned with scpf_scan_fallbacks after the run |
| SC-21 | minor | metrics | n/a | no continuous factor is binned |
| SC-22 | minor | claims | n/a | no field-standard claim is made |
