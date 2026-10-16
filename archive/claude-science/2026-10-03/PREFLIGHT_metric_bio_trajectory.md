# PREFLIGHT: metric_bio_trajectory

**Verdict: GO**  (checked 2026-10-03T02:37:01Z; plan_sha bf55037984adfd20)

## Plan

- **question**: Do the two scorer changes (trajectory conservation added to score_scib_native.BIO_METRICS, SI-28; R's RNG set before every R kBET call) reproduce scIB's definitions, change the bio score only on the trajectory tasks, and make kBET identical on identical latents?
- **primary_outcome**: bio score per task (prereg C and scib_overall bio_score) under the new BIO_METRICS
- **unit_of_analysis**: one scored latent (fit)
- **split_unit**: none: a metric-definition change, no train/test split
- **selection_rule**: none: adopted before any Tier 1+2 fit is scored (user answer 'Include trajectory (Recommended)'; lead addendum 2026-10-03 for kBET)
- **n_comparisons**: 1
- **multiple_comparison_correction**: none: single definition check, no inferential test
- **data_filters**: trajectory only where the prepped obs holds dpt_pseudotime (immune, immune_hum_mou); kBET labels as scib 1.1.7 (>= 10 cells and > 1 batch)

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha bf55037984adfd20) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=1; correction: none: single definition check, no inferential test |
| PF-03 | blocking | Unit of analysis | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-03) |
| PF-04 | blocking | ID integrity | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-04) |
| PF-05 | major | Nesting of factors | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-05) |
| PF-06 | blocking | Selection bias | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-06) |
| PF-07 | blocking | Matched comparison | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-07) |
| PF-08 | blocking | Aggregation | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-08) |
| PF-09 | blocking | No-op baselines | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-09) |
| PF-10 | blocking | Sample size | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-10) |
| PF-11 | major | Metric vs model structure | pass | trajectory_conservation (scib 1.1.7 metrics/trajectory.py) = (Spearman(DPT before, DPT after on the integrated kNN graph)+1)/2 on cells with obs dpt_pseudotime; every arm yields one posterior-mean latent of the same size, so the metric applies identically to all arms |
| PF-12 | blocking | Metric implementation | pass | [auto:pf_check_metric_known_answer] 6 known-answer cases reproduced |
| PF-13 | major | Estimator noise | pass | kBET sampling variance is now pinned by set.seed(KBET_SEED) before every R kBET call (tests/scoring: identical kBET on the same latent, different kBET for seeds 1 vs 2); PAPER_PLAN sec 5 measurement-SD re-scoring varies KBET_SEED explicitly (PREREG.md sec 1). Trajectory has no sampling step |
| PF-14 | major | Test vs control | pass | known-answer cases cover the test condition (immune, immune_hum_mou: 8 bio metrics, C = 0.45) and controls (pancreas/lung 7, ATAC/sims 6); experiments/metric_bio_trajectory/check_bio_trajectory.json: scib_overall bio_score change pancreas 1.1e-16, immune 0.143 |
| PF-15 | major | Attribution | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-15) |
| PF-16 | blocking | Timing confounds | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-16) |
| PF-17 | blocking | Cost estimate | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-17) |
| PF-18 | major | Co-scheduling | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-18) |
| PF-19 | blocking | Launch manifest | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-19) |
| PF-20 | major | Count reconciliation | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-20) |
| PF-21 | major | Replicates vary | pass | replicates of the measurement-SD step must vary KBET_SEED: tests/scoring/test_kbet_determinism.py::test_the_seed_is_live shows seeds 1 and 2 give different kBET on the same latent |
| PF-22 | blocking | Output persistence | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-22) |
| PF-23 | blocking | Provenance and inclusion | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (PF-23) |
| PF-24 | major | Config vs input | pass | KBET_SEED is validated as an integer (score_scib_native.kbet_seed raises ValueError otherwise; test_a_non_integer_seed_is_refused); kBET k0 and label filters are scib 1.1.7's own |
| SC-01 | blocking | splits | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-01) |
| SC-02 | blocking | splits | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-02) |
| SC-03 | blocking | splits | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-03) |
| SC-04 | major | splits | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-04) |
| SC-05 | blocking | gene panel | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-05) |
| SC-06 | major | gene panel | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-06) |
| SC-07 | blocking | perturbation | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-07) |
| SC-08 | blocking | perturbation | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-08) |
| SC-09 | blocking | perturbation | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-09) |
| SC-10 | major | gene panel | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-10) |
| SC-11 | blocking | inputs | pass | score_scib_native.py resolves batch/label keys from the prepped uns and trajectory from obs['dpt_pseudotime'] by name; h5py read of prepped_scib on 2026-10-02: dpt_pseudotime only in immune and immune_hum_mou |
| SC-12 | major | claims | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-12) |
| SC-13 | major | metrics | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-13) |
| SC-14 | blocking | metrics | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-14) |
| SC-15 | major | metrics | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-15) |
| SC-16 | blocking | metrics | n/a | kBET's kNN parameters (k0 = min(70, max(10, mean batch size/4)), labels with >= 10 cells and > 1 batch) are scib 1.1.7's and unchanged; this record changes only the RNG seed of kBET's test-cell sampling and adds trajectory; per-condition validity belongs to each launch preflight |
| SC-17 | major | metrics | pass | trajectory is NaN where scib does not compute it (trajectory_=False) and skipped in every mean (scib_overall skipna; prereg bio_metrics_for adds it only for TRAJ_TASKS); a NaN in a required metric stops the rule scripts (test_trajectory_changes_bio_only_on_the_trajectory_tasks) |
| SC-18 | major | aggregation | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-18) |
| SC-19 | major | inputs | pass | prereg_rules.read_scores rejects score files without kbet_seed (pre-2026-10-03, unseeded) and outcomes() rejects mixed kBET seeds (test_scores_with_mixed_kbet_seeds_raise) |
| SC-20 | blocking | runners | pass | score_scib_native.py has no try/except (fl_lint clean on the edited file); seed_kbet wraps scib.metrics.kbet.kBET_single without catching, and main() raises if kBET is finite but no seeded R call ran |
| SC-21 | minor | metrics | n/a | metric-definition change only: no split, selection, arm comparison, perturbation, launch or timing in this record (SC-21) |
| SC-22 | minor | claims | pass | theislab/scib-reproducibility main@3afbffd367 visualization/plotSingleTaskRNA.R: group_bio incl. 'trajectory conservation' (lines 45-46), bio score = rowMeans of scaled bio metrics na.rm (142-143); theislab/scib-pipeline scripts/metrics/metrics.py lines 201-205: trajectory iff dpt_pseudotime in obs |
