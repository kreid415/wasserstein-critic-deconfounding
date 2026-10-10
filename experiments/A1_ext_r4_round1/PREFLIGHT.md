# PREFLIGHT: A1_ext_r4_round1

**Verdict: GO**  (checked 2026-10-10T03:32:22Z; plan_sha 095283995cd9b8a3)

## Plan

- **question**: A1 extension, R4-a1 round 1 (PREREG sec. 4): fit and score the 9 A1 rows R4-a1 requested (immune, conditioned decoder c1, discriminator / pooled / reference at lambda 0.03, seeds 100-102; their matched point lies on the low edge 0.1) with A1's code, device and settings, then rerun R4-a1
- **primary_outcome**: unscaled batch score B (mean of the 5 raw scIB batch metrics) per fit, which R4 uses to place the matched point near b* = b0 + 0.5 (b_common - b0)
- **unit_of_analysis**: one fit = task x decoder x arm x lambda x seed; R4 decides per task x arm x decoder
- **split_unit**: seed: 100-102 as in A1, disjoint from X1 (0-4) and follow-ups (10-12); scIB metrics use all cells, no cell-level split
- **selection_rule**: R4-a1 is rerun with the pre-registered rule (1 point per round, <= 2 rounds per edge); R4 extension points are used for matching only, frontier and P1 analyses use the six pre-registered points per family (PREREG sec. 4); no A1 or extension fit is reused as an A2/A3/X1 fit
- **n_comparisons**: 1
- **multiple_comparison_correction**: none needed: no inferential claim; R4 is a deterministic pre-registered rule (PREREG sec 4)
- **data_filters**: same as A1: scIB task files as distributed (SI-15), prepped by scripts/prep_scib_task.py, fingerprints verified (fingerprint_prepped.py --compare exit 0, 2026-10-09); training features = var.highly_variable

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha 095283995cd9b8a3) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=1; correction none needed: no inferential claim, R1 is a deterministic pre-registered rule (PREREG sec 2) |
| PF-03 | blocking | Unit of analysis | n/a | no classifier, probe or held-out evaluation: scIB metrics are computed on all cells of each task (scIB convention); scVI's internal 90/10 split only sets the validation loss |
| PF-04 | blocking | ID integrity | n/a | no unit-level evaluation; batches are the integration targets from the scIB key as distributed (SC-11 key check passed on all 3 files) |
| PF-05 | major | Nesting of factors | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): scpf_check_nesting(batch, celltype) re-run 2026-10-04 on the 3 prepped A1 files: no deterministic nesting, no warni... |
| PF-06 | blocking | Selection bias | pass | [auto:pf_check_selection] selection and report sets disjoint (9 vs 3000 ids) |
| PF-07 | blocking | Matched comparison | pass | the 9 rows equal A1's immune c1 rows of the same arm in every manifest column except tag and lam (constant columns identical; n_critic discriminator 1, pooled 5, reference 5; extra identical), so the A1 v2 matched-arms check (9 fields) carries over |
| PF-08 | blocking | Aggregation | pass | each of the 3 arms has seeds 100-102 at lambda 0.03 (3 rows, 3 cells); the lambda=0 pairs and every other grid point are A1's own scored immune c1 rows |
| PF-09 | blocking | No-op baselines | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): lambda=0 'none' arm = stock scVI: tests/scvi (incl. test_none_is_bit_identical_to_stock_scvi) 110 passed, 1 skipped... |
| PF-10 | blocking | Sample size | n/a | A1 creates no conditions by filtering, subsampling or perturbing: every fit uses the full prepped task (11,270 / 33,506 / 12,097 cells) |
| PF-11 | major | Metric vs model structure | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): all arms produce 10-dimensional unstructured posterior means scored by the same scIB functions; no metric assumes a... |
| PF-12 | blocking | Metric implementation | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): metrics from scib 1.1.7 via score_scib_native.py; tests/prereg (B/C aggregation known answers) and tests/scoring pa... |
| PF-13 | major | Estimator noise | pass | R4 matches on the unscaled batch score B with the pre-registered target b* = b0 + 0.5 (b_common - b0) (constants in docs/prereg/r4_a1_record.json); kBET seeded per call as in A1 |
| PF-14 | major | Test vs control | pass | controls are A1's lambda=0 'none' rows for immune c1 (seed-paired); the extension adds one test point per arm only |
| PF-15 | major | Attribution | n/a | A1 attributes no mechanism; it only sets X1's lambda grid |
| PF-16 | blocking | Timing confounds | pass | timings used only to plan: 90 measured A1 fits of immune c1 for the same 3 arms at the production setting (8 lanes, RTX 3080); fit time barely depends on lambda (CV across lambda 0.008-0.047); no speed claim is made |
| PF-17 | blocking | Cost estimate | pass | [auto:pf_estimate_cost] estimate 11.29 h (x1.5 safety = 16.93 h); cpu-hours 11.3; recommended walltime 7620s |
| PF-18 | major | Co-scheduling | pass | same 8 fit lanes + 3 scorers as A1 and the R1 extension (36 fits, 0 failures, 2026-10-09; GPU 99% at 3.0 GB of 10 GB; >= 46 GB RAM available); filler pass A stays paused (critical path; NB-20261006-03..06) |
| PF-19 | blocking | Launch manifest | pass | [auto:pf_check_manifest] 9 rows carry all 12 intended flags |
| PF-20 | major | Count reconciliation | pass | independent counts agree: R4-a1 record n_extension_rows 9 = manifest rows 9 = dry run 9 needs_fit = 3 arms x 3 seeds x 1 lambda |
| PF-21 | major | Replicates vary | pass | [auto:pf_check_replicates_differ] 3 of 3 replicates distinct |
| PF-22 | blocking | Output persistence | pass | same runner as A1 (per-tag latents, scores, attempts, ledger, failures.csv; gate exits 0 only if every row is final); fresh out dir tier12_v2/A1_ext_r4_round1; failed fits are listed in failures.csv, which R4 reads; the launcher pins the manifest SHA-256 and the clean tag checkout |
| PF-23 | blocking | Provenance and inclusion | pass | inputs = the scIB task files as distributed (SI-15); fingerprint_prepped.py --compare docs/prepped_fingerprints_scib.json exit 0 on all 8 prepped files (2026-10-09) |
| PF-24 | major | Config vs input | pass | lambda 0.03 lies below the A1 grid's 0.1, where these arms trained and scored with every required metric finite ({'discriminator': 3, 'reference': 3, 'pooled': 3} of 3 seeds each); kNN parameters as in A1 (k <= 90 < smallest immune cell type 129) |
| SC-01 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1 or its extension; scIB integration metrics are computed on all cells |
| SC-02 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1 or its extension; scIB integration metrics are computed on all cells |
| SC-03 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1 or its extension; scIB integration metrics are computed on all cells |
| SC-04 | major | splits | n/a | no custom grouped-split function is used anywhere in A1 or its extension |
| SC-05 | blocking | gene panel | n/a | no gene panel built from a reference: each task's features are scIB's own HVGs (2,000) or, for atac_small, all 3,429 gene-activity features |
| SC-06 | major | gene panel | n/a | no foundation model with a fixed gene vocabulary is used |
| SC-07 | blocking | perturbation | n/a | no perturbation axis in A1 or its extension (X3's depletion dose gets its own preflight) |
| SC-08 | blocking | perturbation | n/a | no perturbation or filtering axis in A1 or its extension (X3/X8 get their own preflight) |
| SC-09 | blocking | perturbation | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): scpf_check_empty_cells on layers['counts'] restricted to the training features, re-run 2026-10-04: 0 empty cells in... |
| SC-10 | major | gene panel | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): training genes = var.highly_variable: 3,429 (atac_small, all features), 2,000 (immune), 2,000 (sim1), re-measured 2... |
| SC-11 | blocking | inputs | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): scpf_check_obs_keys: 'batch' and 'celltype' present by name in all 3 files; batches 3/10/6 and cell types 7/16/7 (r... |
| SC-12 | major | claims | n/a | A1 makes no integration or disentanglement claim; it only fixes X1's lambda grid |
| SC-13 | major | metrics | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): same check as PF-05: scpf_check_nesting(batch, celltype) finds no deterministic nesting in the 3 A1 tasks (2026-10-04) |
| SC-14 | blocking | metrics | n/a | every arm yields an unstructured 10-dimensional posterior-mean latent; no block-structured models |
| SC-15 | major | metrics | n/a | no block-aware metric is used; latents are unstructured |
| SC-16 | blocking | metrics | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): scpf_check_knn_validity: smallest cell type 226 / 129 / 204 cells exceeds LISI's 90 neighbours and kBET's k0 <= 70 ... |
| SC-17 | major | metrics | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): new scorer row (atac_small pooled, 1-epoch latent, 080f417): non-computable metrics are NaN, not placeholders (cell... |
| SC-18 | major | aggregation | pass | PASS SC-18 scpf_check_pool_coverage (n_models=3, n_datasets=1, rank_flip=False) (the 9 rows, arm x task) |
| SC-19 | major | inputs | pass | fresh out dir tier12_v2/A1_ext_r4_round1 (absent; the dry run created nothing); the runner refuses a dirty checkout and mixed commits; --expect-device RTX 3080 = the GPU model of every A1 fit (SI-17) |
| SC-20 | blocking | runners | pass | PASS SC-20 scpf_scan_fallbacks (n_lines_scanned=25, n_hits=0) (new launcher; pipeline files are the tagged ones) |
| SC-21 | minor | metrics | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): trajectory conservation uses continuous DPT pseudotime (scib trajectory_conservation, Spearman), nothing binned; am... |
| SC-22 | minor | claims | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): unchanged since the v1 record: the metric suite and bio/batch grouping follow scIB (Luecken et al. 2022, Nat. Metho... |
