# PREFLIGHT: fix_score_fit_regression

**Verdict: GO**  (checked 2026-10-03T23:57:51Z; plan_sha e633edc79ed88eff)

## Plan

- **question**: Do the patched scorer (fix-score-fit) and the tagged scorer (prereg-tier12-v1) give identical metric values on full-data latents, and is the patched subset PCR reference the all-feature PCA of the subset?
- **primary_outcome**: exact equality (bitwise float64) of every BATCH_METRICS + BIO_METRICS column, kbet_seed and kbet_r_calls between paired score rows
- **unit_of_analysis**: score row (one latent)
- **split_unit**: none: regression comparison, no train/test split or selection
- **selection_rule**: none: the latents are fixed in advance (2 A1 atac_small latents named by the lead, 1 immune 3-epoch fit, 1 X3 pancreas dose-50 subset)
- **n_comparisons**: 1
- **multiple_comparison_correction**: none needed: exact-equality check, any difference is a failure
- **data_filters**: prepped_scib files as distributed (read-only); A1 files copied read-only from tier12/A1

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha e633edc79ed88eff) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=1; correction: none needed: exact-equality check, any difference is a failure |
| PF-03 | blocking | Unit of analysis | n/a | no train/test split: a regression comparison of score rows of fixed latents |
| PF-04 | blocking | ID integrity | n/a | units are latents identified by manifest tag and latent sha256; A1 copies verified (sha256 prefixes c8e73fb7 / 9b82c899 equal the score rows' latent_sha256) |
| PF-05 | major | Nesting of factors | n/a | no factor overlap analysis |
| PF-06 | blocking | Selection bias | n/a | no selection: the four latents are fixed before the run (named by the lead / derived deterministically) |
| PF-07 | blocking | Matched comparison | n/a | no arms compared: the same latent is scored by two scorer versions in the same env (wcd-kbet), same env vars |
| PF-08 | blocking | Aggregation | n/a | no pooled mean: each comparison is one latent, all columns compared |
| PF-09 | blocking | No-op baselines | n/a | no identity baseline in this design; the regression itself checks that the full-data path is unchanged |
| PF-10 | blocking | Sample size | n/a | no per-condition n: pass/fail is exact equality per score row |
| PF-11 | major | Metric vs model structure | n/a | no new metric assumption: PCR uses scib 1.1.7's own pcr_comparison; the subset reference follows scib-pipeline's full-data convention |
| PF-12 | blocking | Metric implementation | pass | tests/scoring/test_scorer_provenance_and_subset.py 7 passed (wcd-kbet, nice 19, 1 thread): PCR_batch equals an independent scib.metrics.pcr_comparison with the all-feature reference on full and subset toys; HVG-reference mutant fails; kBET skipped=2/forced=0 and RootCellError flag known answers; ... |
| PF-13 | major | Estimator noise | n/a | exact-equality check; sampling variance does not apply (kBET seeded, all other metrics deterministic on one CPU) |
| PF-14 | major | Test vs control | pass | test condition present: patched scorer (fix-score-fit 1c16443) vs tagged scorer (prereg-tier12-v1 129142b worktree) on identical latents; A1 reference rows from the running tier12 A1 stage |
| PF-15 | major | Attribution | n/a | no causal claim |
| PF-16 | blocking | Timing confounds | n/a | no timing claim: score_seconds is excluded from the comparison and not reported |
| PF-17 | blocking | Cost estimate | n/a | no cost claim; total run < 1 CPU-hour, single process at nice 19 (below the skill's threshold) |
| PF-18 | major | Co-scheduling | pass | one process at a time (sequential driver), nice -n 19, OMP/MKL/OPENBLAS/NUMBA threads 1, CUDA hidden; fit on CPU (3 epochs) so no VRAM use next to A1's GPU lanes |
| PF-19 | blocking | Launch manifest | pass | [auto:pf_check_manifest] 1 rows carry all 21 intended flags |
| PF-20 | major | Count reconciliation | pass | 4 latents = 2 A1 atac_small (lead-named) + 1 immune 3-epoch fit + 1 X3 pancreas dose-50 subset (X3_pancreas_none_l0_c1_s10_31994c65), counted from the brief |
| PF-21 | major | Replicates vary | n/a | no seed replicates: each latent is scored once per scorer version (kBET seed 0 in both) |
| PF-22 | blocking | Output persistence | pass | every score CSV and comparison table is written under regress/ and harvested into docs/regression_fix_score_fit_*.csv (committed) and project artifacts; the immune latent and model are kept in regress/ii |
| PF-23 | blocking | Provenance and inclusion | pass | inputs are the prepped_scib files the A1 stage reads (PREPPED_DIR=/home/kendall/experiment_data/wasserstein-critic-deconfounding/prepped_scib), read-only, no filter applied by the regression |
| PF-24 | major | Config vs input | pass | max_epochs=3 >= 1 (the fitter requires an int); scib PCR uses n_comps = min(50, min(X.shape)); kBET / neighbours / clustering at the production defaults of score_scib_native.py |
