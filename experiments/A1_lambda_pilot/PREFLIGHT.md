# PREFLIGHT: A1_lambda_pilot

**Verdict: GO**  (checked 2026-10-03T19:07:01Z; plan_sha da141677a23f99af)

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
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=1; correction: none needed: A1 makes no inferential claim; R1 is a deterministic pre-registered rule (PREREG sec 2) |
| PF-03 | blocking | Unit of analysis | n/a | no classifier, probe or held-out evaluation: scIB metrics are computed on all cells of each task (scIB convention); scVI's internal 90/10 split only logs validation loss, early stopping is off (fit_paper_config.py lines 258-271; scvi-tools 1.4.2 default False) |
| PF-04 | blocking | ID integrity | n/a | no unit-level evaluation; batches are the integration targets, taken from the scIB key as distributed (SC-11 key check; prepped fingerprints identical to files regenerated from committed code) |
| PF-05 | major | Nesting of factors | pass | scpf_check_nesting(batch, celltype) on the 3 prepped A1 files: no deterministic nesting; atac_small has 1 cell type present in only one batch (Cerebellar Granule Cells), which scib 1.1.7 kBET skips (NaN, kbet.py l.118) before nanmean (l.200) |
| PF-06 | blocking | Selection bias | pass | [auto:pf_check_selection] selection and report sets disjoint (1098 vs 3000 ids) |
| PF-07 | blocking | Matched comparison | pass | [auto:pf_check_matched_arms] arms matched on 9 fields (varied: ['adv_opt', 'adv_updates', 'adv_width', 'lipschitz']) |
| PF-08 | blocking | Aggregation | pass | scpf_check_pool_coverage on the A1 manifest (arm x task): every arm covers atac_small, immune, sim1 x 2 decoders x seeds 100-102 x 10 lambda; prereg_rules raises PreregError on any missing row, so no pooled mean over unequal coverage is possible |
| PF-09 | blocking | No-op baselines | pass | lambda=0 'none' arm = stock scVI: tests/scvi/test_adversarial_plan_v2.py::test_none_is_bit_identical_to_stock_scvi and test_unconditioned_decoder_never_sees_the_batch pass on the merged tree (scvi-api tests/scvi 72 passed, 1 skipped, 2026-10-03) |
| PF-10 | blocking | Sample size | n/a | A1 creates no conditions by filtering, subsampling or perturbing: every fit uses the full prepped task (11,270 / 33,506 / 12,097 cells) |
| PF-11 | major | Metric vs model structure | pass | all arms produce 10-dimensional unstructured posterior means scored by the same scIB functions; no metric assumes a latent structure some arms lack |
| PF-12 | blocking | Metric implementation | pass | metrics come from scib 1.1.7 via score_scib_native.py; our code on top is tested: B/C aggregation known answers (tests/prereg: C 0.45/0.40/0.35 for 8/7/6 metrics) and seeded kBET (tests/scoring: same latent twice identical, other seed differs); wcd-gpu 269 passed, wcd-kbet 17 passed on the merged... |
| PF-13 | major | Estimator noise | pass | A1 itself estimates the paired seed SD; R1's effect threshold is max(delta_min 0.01, 3 x noise) (PREREG sec 1); kBET is seeded; the measurement-SD step later varies the kBET seed and machine |
| PF-14 | major | Test vs control | pass | A1 contains the lambda=0 control and every test arm at all 10 lambda on both decoders |
| PF-15 | major | Attribution | n/a | A1 attributes no mechanism; it only sets X1's lambda grid |
| PF-16 | blocking | Timing confounds | pass | A1 step costs: single-lane fits at production settings with startup removed via 1-epoch runs (docs/throughput_rtx3080_stock_backbone.csv); 8-lane factor 3.77x from a 64-fit mixed-arm queue (docs/concurrency_calibration_stock.json; identical-arm minimum 3.17x kept as conservative). Run2 same-sessi... |
| PF-17 | blocking | Cost estimate | pass | [auto:pf_estimate_cost] estimate 98.18 h (x1.19 safety = 116.83 h); cpu-hours 98.2; recommended walltime 420600s |
| PF-18 | major | Co-scheduling | pass | 8 fit lanes as in the 64-fit calibration on immune, the largest A1 task (0 failures); launch waits until the JHPCE gate's local scoring ends; 3 scoring workers during fits (8 + 3 = 11 of 12 cores); Scanorama timing at nice 19, 2 threads, capped at 24 GB; 94 GB RAM; per-process memory read in the ... |
| PF-19 | blocking | Launch manifest | pass | [auto:pf_check_manifest] 1098 rows carry all 12 intended flags |
| PF-20 | major | Count reconciliation | pass | independent count 3 tasks x 2 decoders x 3 seeds x (1 lambda=0 + 6 arms x 10 lambda) = 1098 = rows in build_paper_manifest.py --backbone stock --design pilot --uncond-seeds 5 --bary-iter 10 (main 3288641) |
| PF-21 | major | Replicates vary | pass | [auto:pf_check_replicates_differ] 3 of 3 replicates distinct |
| PF-22 | blocking | Output persistence | pass | run_stage.py writes per tag OUT/latents/<tag>.npz (posterior mean, obs names, batch, celltype, config, history), OUT/models/<tag>/, OUT/scores/<tag>.csv, OUT/status/<tag>.json; failures.csv in prereg_rules format; completion gate exits 0 only when every row is scored or failed; OUT on /home/kenda... |
| PF-23 | blocking | Provenance and inclusion | pass | inputs are the scIB task files as distributed (SI-15): raw md5s recorded (scib_raw_sources.tsv), prepped files regenerate bit-identically from committed prep_scib_task.py locally and on JHPCE (fingerprint compare 8/8 OK); A1 tasks atac_small (mouse ATAC gene activity), immune (human), sim1 (simul... |
| PF-24 | major | Config vs input | pass | training genes <= features (2,000 of 12,303 / 9,979; atac_small all 3,429); kNN k (15 graph, 90 LISI, <=70 kBET) below the smallest cell type (129); epochs recomputed from the scvi-tools heuristic min(400, round(20000/n*400)) = 400/239/400 = manifest; reference 'auto' resolved per fit by select_r... |
| SC-01 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1; scIB integration metrics are computed on all cells |
| SC-02 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1; scIB integration metrics are computed on all cells |
| SC-03 | blocking | splits | n/a | no donor-level labels, probes, classifiers or train/test evaluation in A1; scIB integration metrics are computed on all cells |
| SC-04 | major | splits | n/a | no custom grouped-split function is used anywhere in A1 |
| SC-05 | blocking | gene panel | n/a | no gene panel built from a reference: each task's features are scIB's own HVGs (2,000) or, for atac_small, all 3,429 gene-activity features |
| SC-06 | major | gene panel | n/a | no foundation model with a fixed gene vocabulary is used |
| SC-07 | blocking | perturbation | n/a | no perturbation axis in A1 (X3's depletion dose gets its own preflight) |
| SC-08 | blocking | perturbation | n/a | no perturbation or filtering axis in A1 (X3/X8 get their own preflight) |
| SC-09 | blocking | perturbation | pass | scpf_check_empty_cells on layers['counts'] restricted to the training features: 0 cells with zero counts in atac_small (11,270), immune (33,506), sim1 (12,097) |
| SC-10 | major | gene panel | pass | training genes = var.highly_variable (fit_paper_config.py l.205): 3,429 (atac_small, all features), 2,000 (immune of 12,303), 2,000 (sim1 of 9,979), equal to docs/prepped_fingerprints_scib.json; one HVG selection (scIB prep), same mask for every arm |
| SC-11 | blocking | inputs | pass | scpf_check_obs_keys: 'batch' and 'celltype' present by name in all 3 files; batches 3/10/6 and cell types 7/16/7 as in the scIB task definitions |
| SC-12 | major | claims | n/a | A1 makes no integration or disentanglement claim; it only fixes X1's lambda grid |
| SC-13 | major | metrics | pass | same check as PF-05: scpf_check_nesting(batch, celltype) finds no deterministic nesting in the 3 A1 tasks |
| SC-14 | blocking | metrics | n/a | every arm yields an unstructured 10-dimensional posterior-mean latent; no block-structured models |
| SC-15 | major | metrics | n/a | no block-aware metric is used; latents are unstructured |
| SC-16 | blocking | metrics | pass | smallest cell type 226 / 129 / 204 cells (atac_small / immune / sim1) exceeds LISI's 90 neighbours and kBET's k0 <= 70 (scpf_check_knn_validity ok); scib kBET skips single-batch clusters as NaN |
| SC-17 | major | metrics | pass | metrics outside a task's set are NaN, never a placeholder (cell_cycle only on human RNA tasks, trajectory only with dpt_pseudotime; kBET-skipped clusters NaN, scib kbet.py l.118); prereg_rules requires finite values for the task's metrics by name and stops otherwise |
| SC-18 | major | aggregation | pass | scpf_check_pool_coverage(arm x task) on the A1 manifest: all arms cover the same 3 tasks |
| SC-19 | major | inputs | pass | fresh output directory for A1; run_stage.py refuses a dirty checkout (--allow-dirty is test-only) and latents fitted at different commits (--allow-mixed-sha); each npz records its tag and git SHA |
| SC-20 | blocking | runners | pass | scpf_scan_fallbacks on scripts/run_stage.py, fit_paper_config.py, score_scib_native.py: no refit/fallback patterns; fl_lint reported 0 findings on the changed files (stage-runner and missing-arms reports) |
| SC-21 | minor | metrics | pass | trajectory conservation uses continuous DPT pseudotime (scib trajectory_conservation, Spearman); nothing is binned |
| SC-22 | minor | claims | pass | the metric suite and bio/batch grouping follow scIB (Luecken et al. 2022, Nat. Methods) and scib-reproducibility plotSingleTaskRNA.R l.45-46 and l.142-143 (PREREG sec 1) |
