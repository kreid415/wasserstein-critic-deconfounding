# PREFLIGHT: A2A3_pilot

**Verdict: GO**  (checked 2026-10-10T05:18:39Z; plan_sha 4c59135928fb43a7)

## Plan

- **question**: A2/A3 pilots (PREREG sec. 3): fit and score the 288 new halves (A2 posterior-sample adversary input, 108 rows; A3 per-dimension standardisation on, 180 rows; atac_small + immune, both decoders, seeds 100-102, lambda = R4-a1 lo/matched/hi) so that R2/R3 (decide_a2_a3.py) set X1's adv_input and zstd
- **primary_outcome**: per cell (task, decoder, arm): D = bio@b*(alternative) - bio@b*(default), the seed-mean bio score C interpolated at the stage-a1 batch target b* (PREREG sec 3)
- **unit_of_analysis**: cell = task x decoder x arm (A2 12, A3 20); one fit = task x decoder x arm x lambda x seed; one pooled decision per pilot
- **split_unit**: seed: 100-102 (the A1 seeds, so the default halves are the A1 fits), disjoint from X1 (0-4) and follow-ups (10-12); scIB metrics use all cells
- **selection_rule**: R2 and R3 are pre-registered deterministic rules (evaluable >= 2n/3 with >= 1 per task x decoder, mean D > ROPE, mean D > 0 within every task x decoder, no more failed fits; A2 masking => mean); X1 then runs on fresh seeds 0-4; no A2/A3 fit is reused as an X1 fit
- **n_comparisons**: 2
- **multiple_comparison_correction**: none needed: two separate pre-registered decision rules (R2, R3) with a ROPE threshold, no inferential claim (PREREG sec 3); a combined switch is disclosed
- **data_filters**: same as A1 for its tasks atac_small and immune: scIB task files as distributed (SI-15), prepped by scripts/prep_scib_task.py, fingerprints verified (fingerprint_prepped.py --compare exit 0, 2026-10-09); training features = var.highly_variable

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha 4c59135928fb43a7) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=2 (R2, R3); correction none needed: two pre-registered deterministic decision rules with a ROPE, no inferential claim (PREREG sec 3) |
| PF-03 | blocking | Unit of analysis | n/a | no classifier, probe or held-out evaluation: scIB metrics are computed on all cells of each task (scIB convention); scVI's internal 90/10 split only sets the validation loss |
| PF-04 | blocking | ID integrity | n/a | no unit-level evaluation; batches are the integration targets from the scIB key as distributed (SC-11 key check passed on all 3 files) |
| PF-05 | major | Nesting of factors | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): scpf_check_nesting(batch, celltype) re-run 2026-10-0... |
| PF-06 | blocking | Selection bias | pass | [auto:pf_check_selection] selection and report sets disjoint (288 vs 3000 ids) |
| PF-07 | blocking | Matched comparison | pass | every one of the 288 new-half rows has a default-half row (A1 or its extension rounds) equal in every manifest column except tag, experiment and the one switched setting (adv_input for A2, zstd for A3): 288/288 matched (180 distinct A1 fits, 24 from the extension rounds), 288/288 scored; asserted... |
| PF-08 | blocking | Aggregation | pass | A2: 3 arms x 2 tasks x 2 decoders x 3 lambda x 3 seeds = 108; A3: 5 arms x 2 x 2 x 3 x 3 = 180; every cell has its lo/matched/hi window on all 3 seeds (counts from manifests/paper_manifest_stock_pilot_u5_b10_v3.r4a1.tsv) |
| PF-09 | blocking | No-op baselines | pass | the default half of every cell is the A1 fit itself (not refitted). The new half differs only in the switched setting, and the switch is live: 1-epoch CPU fits in tier12_v2/checks/A2A3_switch_check_20261010T011644/switch_check.csv (experiments/A2A3_pilot/check_switches.sh): default half bit-ident... |
| PF-10 | blocking | Sample size | n/a | A1 creates no conditions by filtering, subsampling or perturbing: every fit uses the full prepped task (11,270 / 33,506 / 12,097 cells) |
| PF-11 | major | Metric vs model structure | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): all arms produce 10-dimensional unstructured posteri... |
| PF-12 | blocking | Metric implementation | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): metrics from scib 1.1.7 via score_scib_native.py; te... |
| PF-13 | major | Estimator noise | pass | R2/R3 use the seed-mean C and B (3 seeds) with the pre-registered ROPE and b* from the R4-a1 record; kBET seeded per call (seed 0) as in A1 |
| PF-14 | major | Test vs control | pass | the control of every cell is its default half (A1 fits, same seeds/lambda/decoder); D is alternative minus default at the same b* |
| PF-15 | major | Attribution | n/a | A1 attributes no mechanism; it only sets X1's lambda grid |
| PF-16 | blocking | Timing confounds | pass | timings used only to plan: each new half's cost is taken from its own default half's measured fit time (the same configuration except the switched setting, 8 lanes on this RTX 3080); the switch adds at most one O(n x d) standardisation or uses the already-drawn sample; no speed claim is made |
| PF-17 | blocking | Cost estimate | pass | [auto:pf_estimate_cost] estimate 235.43 h (x1.3 safety = 306.05 h); cpu-hours 235.4; recommended walltime 137760s |
| PF-18 | major | Co-scheduling | pass | same 8 fit lanes + 3 scorers as A1 on the same 2 tasks (A1: 1,098 fits, 0 failures; GPU 99% at <= 3 GB of 10 GB; RAM in use <= 64 GB of 94 including outside load); filler pass A (25 GB scorers) is not co-scheduled (NB-20261006-03..06) |
| PF-19 | blocking | Launch manifest | pass | [auto:pf_check_manifest] 288 rows carry all 9 intended flags |
| PF-20 | major | Count reconciliation | pass | independent counts agree: R4-a1 record resolved_rows 288 = manifest A2+A3 rows 288 (108 + 180) = dry run 288 needs_fit = PREREG sec 3 (108 + 180 new fits) |
| PF-21 | major | Replicates vary | pass | seeds differ in the new code path: A2 reference seed 100 vs 101 at 1 epoch, max /dz/ 31.5 (tier12_v2/checks/A2A3_switch_check_20261010T011644/switch_check.csv (experiments/A2A3_pilot/check_switches.sh)) |
| PF-22 | blocking | Output persistence | pass | same runner as A1 (per-tag latents, scores, attempts, ledger, failures.csv; gate exits 0 only if every row is final); fresh out dir tier12_v2/A2A3; failed fits are listed in failures.csv, which decide_a2_a3.py reads (rule 4 counts them); the launcher pins the manifest SHA-256 and the clean tag ch... |
| PF-23 | blocking | Provenance and inclusion | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): inputs = the scIB task files as distributed (SI-15); fingerprint_prepped.py --compare docs/prepped_fingerprints_scib.json exit 0 on all 8 prepped files (2026-10-09) |
| PF-24 | major | Config vs input | pass | every window lambda of A2/A3 already trained and scored with all required metrics finite in its default half (288/288 scored); the switched settings ran finite at 1 epoch (tier12_v2/checks/A2A3_switch_check_20261010T011644/switch_check.csv (experiments/A2A3_pilot/check_switches.sh)); a non-finite... |
| SC-01 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1, its extensions or A2/A3; scIB integration metrics are computed on all cells |
| SC-02 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1, its extensions or A2/A3; scIB integration metrics are computed on all cells |
| SC-03 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1, its extensions or A2/A3; scIB integration metrics are computed on all cells |
| SC-04 | major | splits | n/a | no custom grouped-split function is used anywhere in A1, its extensions or A2/A3 |
| SC-05 | blocking | gene panel | n/a | no gene panel built from a reference: each task's features are scIB's own HVGs (2,000) or, for atac_small, all 3,429 gene-activity features |
| SC-06 | major | gene panel | n/a | no foundation model with a fixed gene vocabulary is used |
| SC-07 | blocking | perturbation | n/a | no perturbation axis in A1, its extensions or A2/A3 (X3's depletion dose gets its own preflight) |
| SC-08 | blocking | perturbation | n/a | no perturbation or filtering axis in A1, its extensions or A2/A3 (X3/X8 get their own preflight) |
| SC-09 | blocking | perturbation | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): scpf_check_empty_cells on layers['counts'] restricte... |
| SC-10 | major | gene panel | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): training genes = var.highly_variable: 3,429 (atac_sm... |
| SC-11 | blocking | inputs | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): scpf_check_obs_keys: 'batch' and 'celltype' present ... |
| SC-12 | major | claims | n/a | A1 makes no integration or disentanglement claim; it only fixes X1's lambda grid |
| SC-13 | major | metrics | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): same check as PF-05: scpf_check_nesting(batch, cellt... |
| SC-14 | blocking | metrics | n/a | every arm yields an unstructured 10-dimensional posterior-mean latent; no block-structured models |
| SC-15 | major | metrics | n/a | no block-aware metric is used; latents are unstructured |
| SC-16 | blocking | metrics | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): scpf_check_knn_validity: smallest cell type 226 / 12... |
| SC-17 | major | metrics | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): new scorer row (atac_small pooled, 1-epoch latent, 0... |
| SC-18 | major | aggregation | pass | PASS SC-18 scpf_check_pool_coverage (n_models=5, n_datasets=2, rank_flip=False) (288 rows, arm x task) |
| SC-19 | major | inputs | pass | fresh out dir tier12_v2/A2A3 (absent; the dry run created nothing); runner refuses a dirty checkout and mixed commits; --expect-device RTX 3080 = the GPU of every A1 fit (SI-17) |
| SC-20 | blocking | runners | pass | PASS SC-20 scpf_scan_fallbacks (n_lines_scanned=25, n_hits=0) (new launcher; pipeline files are the tagged ones) |
| SC-21 | minor | metrics | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): trajectory conservation uses continuous DPT pseudoti... |
| SC-22 | minor | claims | pass | same tasks' files, code and settings as A1 (2 of its 3 tasks): as in the A1 v2 record (experiments/A1_lambda_pilot_v2, GO; same code c88cce3, same 3 prepped files unchanged per fingerprint compare exit 0 on 2026-10-09, same seeds and settings): unchanged since the v1 record: the metric suite and ... |
