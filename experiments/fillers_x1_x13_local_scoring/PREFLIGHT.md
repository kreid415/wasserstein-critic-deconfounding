# PREFLIGHT: fillers_x1_x13_local_scoring

**Verdict: GO**  (checked 2026-10-06T14:04:30Z; plan_sha 7ee4d33faa8f59d8)

## Plan

- **question**: Score the 250 JHPCE filler latents (X1 none c0/c1 + stock scVI-adversarial, X13 scANVI + sysVI; atac_large, immune_hum_mou, lung, pancreas, sim2; seeds 0-4; fitted on L40S at c88cce3; harvested and verified 2026-10-06) with the tagged scorer, locally beside A1, as planned at the SI-48 go (local scoring within A1's spare scoring capacity)
- **primary_outcome**: per-latent PREREG batch score B (mean of the 5 raw scIB batch metrics) and bio score C, written by scripts/score_scib_native.py at c88cce3 to OUT/scores/<tag>.csv; the run is complete when the runner's gate finds all 250 rows final
- **unit_of_analysis**: one latent = task x arm x knob value x decoder condition x seed
- **split_unit**: seed (0-4, disjoint from A1's 100-102 and follow-ups' 10-12); scIB metrics use all cells of each task, no cell-level split
- **selection_rule**: nothing is selected: every harvested filler latent is scored; the X1/X13 analysis rules of PREREG apply later and are not run here
- **n_comparisons**: 1
- **multiple_comparison_correction**: none needed: this step only computes per-latent scores; inference happens later in the pre-registered X1/X13 analyses
- **data_filters**: scIB task files as distributed (SI-15), prepped by scripts/prep_scib_task.py, fingerprints verified (docs/prepped_fingerprints_scib.json); the runner accepts a latent only if it was fitted on an L40S at c88cce3 and records its own manifest row (SI-17, SI-19)

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha 7ee4d33faa8f59d8) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=1; this step computes per-latent B/C and tests no contrast |
| PF-03 | blocking | Unit of analysis | n/a | no held-out evaluation: scIB metrics are computed on all cells of each task (scIB convention) |
| PF-04 | blocking | ID integrity | n/a | no unit-level evaluation; batch and cell type are the scIB keys of the files as distributed |
| PF-05 | major | Nesting of factors | pass | scpf_check_nesting(batch, celltype) on the 5 prepped tasks (atac_large, immune_hum_mou, lung, pancreas, sim2): no deterministic nesting; immune_hum_mou has 1 of 19 cell types in a single batch (kBET skips it as NaN, scib kbet.py) |
| PF-06 | blocking | Selection bias | pass | nothing is selected; the 250 rows (seeds 0-4) are disjoint from A1's seeds 100-102 and the follow-up seeds 10-12 |
| PF-07 | blocking | Matched comparison | pass | arms as fitted (filler preflight experiments/jhpce_fillers_x1_x13_prod, pf_check_matched_arms on 8 fields); scoring applies one scorer with identical settings to every arm |
| PF-08 | blocking | Aggregation | pass | no pooled mean is computed here; coverage counted from the 250 harvested latents: every task has 10 none (c0+c1), 5 scvi_adv, 5 scanvi, 30 sysvi |
| PF-09 | blocking | No-op baselines | pass | lambda=0 'none' = stock scVI: tests/scvi incl. test_none_is_bit_identical_to_stock_scvi passed on both L40S nodes before fitting (110 passed, 1 skipped; Slurm 36155707/36155708 logs) |
| PF-10 | blocking | Sample size | n/a | no condition is created by filtering or subsampling: every latent is scored on its full prepped task |
| PF-11 | major | Metric vs model structure | pass | all 250 latents are 10-dimensional unstructured posterior means scored by the same scIB functions; no metric assumes a latent structure some arms lack |
| PF-12 | blocking | Metric implementation | pass | scorer = scripts/score_scib_native.py at c88cce3, the commit A1 is scored with; metrics from scib 1.1.7; B/C aggregation and seeded kBET are tested (A1 preflight PF-12: tests/prereg known answers, tests/scoring same latent twice identical) |
| PF-13 | major | Estimator noise | n/a | no claim or effect estimate is made by this scoring step |
| PF-14 | major | Test vs control | n/a | no test-vs-control claim is made by this scoring step |
| PF-15 | major | Attribution | n/a | no causal explanation is offered by this scoring step |
| PF-16 | blocking | Timing confounds | pass | timings and peak memory are used only to plan, and were measured at the production setting on purpose: two scorers at once, nice 19, beside the running A1, each a fresh process as in production (scratch/score_probe/probe.py at c88cce3): immune_hum_mou 3,016 s / 19.36 GB, atac_large 2,489 s / 24.7... |
| PF-17 | blocking | Cost estimate | pass | [auto:pf_estimate_cost] estimate 107.38 h (x1.5 safety = 161.08 h); cpu-hours 214.8; recommended walltime 289980s |
| PF-18 | major | Co-scheduling | pass | one scorer per pass, sized from peak RSS measured at the production setting (probe/probe.json): atac_large 24.79 GB, immune_hum_mou 19.36 GB, the two largest tasks (pass A). Pass B's tasks have at most 38% of atac_large's cells (32,472 vs 84,813) and dense X of at most 2.0 GB (lung), so its peak ... |
| PF-19 | blocking | Launch manifest | pass | the runner reads the 250 rows from manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv via the two tagged tags files (sha256 d5da370a..., 91daef08...); these rows passed pf_check_manifest (7 intended flags) in experiments/jhpce_fillers_x1_x13_prod |
| PF-20 | major | Count reconciliation | pass | independent counts agree: dry runs plan 100 + 150 needs_score rows and 0 fits; 250 latents on disk; harvest n_latents 100 + 150; filler record: 5 tasks x 50 rows |
| PF-21 | major | Replicates vary | pass | pf_check_replicates_differ on harvested X1_immune_hum_mou_none_l0_c1 seeds 0/1/2: 3 of 3 replicates distinct |
| PF-22 | blocking | Output persistence | pass | the runner writes OUT/scores/<tag>.csv, status/<tag>.json and attempts/<tag>.jsonl per tag and rebuilds ledger/ and failures.csv; its gate exits 0 only if every row is final; a killed run resumes from the per-tag files; the downloaded parts stay in jhpce_tier12/_parts until the scores are checked |
| PF-23 | blocking | Provenance and inclusion | pass | inclusion = all harvested filler latents: harvest_local.py verified every part's SHA-256, the reassembled tars (d527c45d..., 267ba98a...), the manifests and every extracted file's size and SHA-256; merged with 0 conflicts |
| PF-24 | major | Config vs input | pass | parameters valid for these inputs: features 3,580 (atac_large, all) and 2,000 HVGs of 8,135-19,093 (others); LISI k=90 and graph k=15 are far below n (16,382-97,861 cells); kBET sets k0 = min(70, max(10, quarter of the mean batch size)) per cell type and skips cell types under 10 cells or in one ... |
| SC-01 | blocking | splits | n/a | no donor-level labels, probes or classifiers; scIB metrics use all cells |
| SC-02 | blocking | splits | n/a | no donor or patient grouping key is used by the scorer |
| SC-03 | blocking | splits | n/a | no train/test assignment exists (scIB metrics on all cells) |
| SC-04 | major | splits | n/a | no custom grouped-split function is used |
| SC-05 | blocking | gene panel | n/a | no gene panel built from a reference; features are the scIB files' own HVGs (atac_large: all 3,580) |
| SC-06 | major | gene panel | n/a | no foundation model with a fixed gene vocabulary is used |
| SC-07 | blocking | perturbation | n/a | no perturbation axis in the filler rows |
| SC-08 | blocking | perturbation | n/a | no perturbation or filtering axis in the filler rows |
| SC-09 | blocking | perturbation | pass | no perturbation generator; scpf_check_empty_cells on layers['counts'] restricted to the training features: 0 empty cells in atac_large, immune_hum_mou, lung, pancreas, sim2 |
| SC-10 | major | gene panel | pass | features as fitted: var.highly_variable of the prepped file = 3,580 of 3,580 (atac_large), 2,000 of 8,135 / 15,148 / 19,093 / 10,000 (immune_hum_mou, lung, pancreas, sim2); one HVG selection (scIB prep), same mask for every arm |
| SC-11 | blocking | inputs | pass | scpf_check_obs_keys: 'batch' and 'celltype' present by name in all 5 files; batches 11/23/16/9/16 and cell types 7/19/17/14/4 |
| SC-12 | major | claims | n/a | no integration or disentanglement claim is made by this scoring step |
| SC-13 | major | metrics | pass | same check as PF-05: scpf_check_nesting(batch, celltype) finds no deterministic nesting in the 5 tasks |
| SC-14 | blocking | metrics | n/a | every latent is an unstructured 10-dimensional posterior mean; no block-structured models |
| SC-15 | major | metrics | n/a | no block-aware metric is used; latents are unstructured |
| SC-16 | blocking | metrics | n/a | applies to metrics on subsampled, filtered or perturbed data of varying size; here every metric runs on the full fixed-size task, identically for all arms. Facts: LISI (k=90) and graph metrics (k=15) build kNN on all cells (16,382-97,861); kBET builds kNN within a cell type with k0 = min(70, max(... |
| SC-17 | major | metrics | pass | metrics outside a task's set are NaN, never a placeholder (cell_cycle only on human RNA tasks, trajectory only with dpt_pseudotime); kBET-skipped cell types are NaN and left out of the mean (scib kbet.py); prereg_rules requires finite values for the task's metrics by name and stops otherwise |
| SC-18 | major | aggregation | n/a | no pooled model means are computed in this scoring step |
| SC-19 | major | inputs | pass | the runner rejects latents by value: the --expect-device dry run refused all 250 L40S latents when 'RTX 3080' was expected (SI-17), and the git SHA, manifest row and latent SHA-256 are checked per tag; with L40S every row is needs_score |
| SC-20 | blocking | runners | pass | scorer and runner are the tagged files A1 uses (scpf_scan_fallbacks: no refit/fallback patterns, A1 preflight SC-20); a refit cannot run here because the local fit env sees an RTX 3080 and the runner requires L40S before any fit |
| SC-21 | minor | metrics | pass | trajectory conservation, where defined, uses continuous DPT pseudotime (scib trajectory_conservation, Spearman); nothing is binned |
| SC-22 | minor | claims | pass | the metric suite and bio/batch grouping follow scIB (Luecken et al. 2022, Nat. Methods) and scib-reproducibility plotSingleTaskRNA.R (PREREG sec 1) |
