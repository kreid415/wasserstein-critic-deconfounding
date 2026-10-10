# PREFLIGHT: A1_ext_r1_round1

**Verdict: GO**  (checked 2026-10-10T01:38:14Z; plan_sha 7198165209b67cc3)

## Plan

- **question**: A1 extension, R1 round 1 (PREREG sec. 2): fit and score the 36 A1 rows R1 requested on the complete A1 pilot (js/0 discriminator c0 low edge at lambda 0.01/0.03; sinkhorn/0 c0 high edge at lambda 10000/30000; atac_small, immune, sim1; seeds 100-102) with A1's code, device and settings, then rerun R1
- **primary_outcome**: seed-paired change in the PREREG batch score B relative to lambda=0 (as in A1); the bio score C enters R1 only as the collapse bound
- **unit_of_analysis**: one fit = task x decoder x arm x lambda x seed; R1 decides per arm family x decoder
- **split_unit**: seed: 100-102 as in A1, disjoint from X1 (0-4) and follow-ups (10-12); scIB metrics use all cells, no cell-level split
- **selection_rule**: R1 is rerun on A1 + this extension with the pre-registered constants (2 points per round, <= 2 rounds per edge); X1 evaluates the frozen grid on fresh seeds 0-4; no A1 or extension fit is reused as an X1 fit (SI-13)
- **n_comparisons**: 1
- **multiple_comparison_correction**: none needed: no inferential claim; R1 is a deterministic pre-registered rule (PREREG sec 2)
- **data_filters**: same as A1: scIB task files as distributed (SI-15), prepped by scripts/prep_scib_task.py, fingerprints verified (fingerprint_prepped.py --compare exit 0, 2026-10-09); training features = var.highly_variable

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha 7198165209b67cc3) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=1; correction none needed: no inferential claim, R1 is a deterministic pre-registered rule (PREREG sec 2) |
| PF-03 | blocking | Unit of analysis | n/a | no classifier, probe or held-out evaluation: scIB metrics are computed on all cells of each task (scIB convention); scVI's internal 90/10 split only sets the validation loss |
| PF-04 | blocking | ID integrity | n/a | no unit-level evaluation; batches are the integration targets from the scIB key as distributed (SC-11 key check passed on all 3 files) |
| PF-05 | major | Nesting of factors | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): scpf_check_nesting(batch, celltype) re-run 2026-10-04 on the 3 prepped A1 files: no deterministic nesting, no warni... |
| PF-06 | blocking | Selection bias | pass | [auto:pf_check_selection] selection and report sets disjoint (36 vs 3000 ids) |
| PF-07 | blocking | Matched comparison | pass | the 36 rows equal A1's rows of the same arm and decoder in every manifest column except tag and lam (compared with the A1 rows of manifest v3: constant columns identical; n_critic discriminator 1 / sinkhorn 0; extra {}; max_epochs 400 / 239 / 400), so the A1 v2 matched-arms check (9 fields) carri... |
| PF-08 | blocking | Aggregation | pass | each (arm, lambda) covers atac_small, immune and sim1 x seeds 100-102 (9 rows, 4 cells); the lambda=0 pairs are A1's own scored 'none' rows (same tasks, seeds, decoder) |
| PF-09 | blocking | No-op baselines | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): lambda=0 'none' arm = stock scVI: tests/scvi (incl. test_none_is_bit_identical_to_stock_scvi) 110 passed, 1 skipped... |
| PF-10 | blocking | Sample size | n/a | A1 creates no conditions by filtering, subsampling or perturbing: every fit uses the full prepped task (11,270 / 33,506 / 12,097 cells) |
| PF-11 | major | Metric vs model structure | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): all arms produce 10-dimensional unstructured posterior means scored by the same scIB functions; no metric assumes a... |
| PF-12 | blocking | Metric implementation | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): metrics from scib 1.1.7 via score_scib_native.py; tests/prereg (B/C aggregation known answers) and tests/scoring pa... |
| PF-13 | major | Estimator noise | pass | R1's noise threshold uses A1's paired seed SD (constants in docs/prereg/r1_record.json, unchanged); kBET seeded per call as in A1 |
| PF-14 | major | Test vs control | pass | the control is A1's lambda=0 'none' rows, seed-paired with every extension row; the extension adds test points only |
| PF-15 | major | Attribution | n/a | A1 attributes no mechanism; it only sets X1's lambda grid |
| PF-16 | blocking | Timing confounds | pass | timings used only to plan: 180 measured A1 fits of the same arm x decoder (discriminator and sinkhorn, c0) at the production setting (8 lanes, RTX 3080); fit time barely depends on lambda (CV across lambda within task x arm 0.012-0.127); no speed claim is made |
| PF-17 | blocking | Cost estimate | pass | [auto:pf_estimate_cost] estimate 14.57 h (x1.5 safety = 21.85 h); cpu-hours 14.6; recommended walltime 9840s |
| PF-18 | major | Co-scheduling | pass | same 8 fit lanes + 3 scorers as A1 (1,098 fits, 0 failures; GPU 99% at <= 3 GB of 10 GB; RAM in use <= 64 GB of 94 including outside load, 2026-10-09); filler pass A (25 GB scorers) is NOT co-scheduled: beside A1 it slowed fits 11.8% (NB-20261006-03..06); it starts after the extension's fits end |
| PF-19 | blocking | Launch manifest | pass | [auto:pf_check_manifest] 36 rows carry all 12 intended flags |
| PF-20 | major | Count reconciliation | pass | independent counts agree: R1 record n_extension_rows 36 = manifest rows 36 = dry run 36 needs_fit = 2 families x 2 lambda x 3 tasks x 3 seeds |
| PF-21 | major | Replicates vary | pass | [auto:pf_check_replicates_differ] 3 of 3 replicates distinct |
| PF-22 | blocking | Output persistence | pass | same runner as A1 (per-tag latents, scores, attempts, ledger, failures.csv; gate exits 0 only if every row is final); fresh out dir tier12_v2/A1_ext_r1_round1; failed fits are classified and listed in failures.csv, which R1 reads; launcher pins the manifest SHA-256 and the clean tag checkout |
| PF-23 | blocking | Provenance and inclusion | pass | inputs = the scIB task files as distributed (SI-15); fingerprint_prepped.py --compare docs/prepped_fingerprints_scib.json exit 0 on all 8 prepped files (2026-10-09) |
| PF-24 | major | Config vs input | pass | lambda 0.01-30000 only scales the adversarial term; A1's nearest edge rows trained and scored with every required metric finite (discriminator c0 lambda 0.1: 9/9; sinkhorn c0 lambda 3000: 9/9); a non-finite latent is classified and recorded (nonfinite_latent) and enters R1 as a failure; kNN k as ... |
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
| SC-18 | major | aggregation | pass | PASS SC-18 scpf_check_pool_coverage (n_models=2, n_datasets=3, rank_flip=False) (the 36 extension rows, arm x task) |
| SC-19 | major | inputs | pass | fresh out dir tier12_v2/A1_ext_r1_round1 (absent; the dry run created nothing); the runner refuses a dirty checkout and latents from more than one commit; --expect-device RTX 3080 = the GPU model of every A1 fit (SI-17) |
| SC-20 | blocking | runners | pass | PASS SC-20 scpf_scan_fallbacks (n_lines_scanned=25, n_hits=0) (new launcher run_A1_ext.sh; every pipeline file is the tagged one A1 used, scanned in the A1 v2 record) |
| SC-21 | minor | metrics | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): trajectory conservation uses continuous DPT pseudotime (scib trajectory_conservation, Spearman), nothing binned; am... |
| SC-22 | minor | claims | pass | as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): unchanged since the v1 record: the metric suite and bio/batch grouping follow scIB (Luecken et al. 2022, Nat. Metho... |
