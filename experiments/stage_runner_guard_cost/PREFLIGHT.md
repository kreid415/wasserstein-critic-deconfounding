# PREFLIGHT: stage_runner_guard_cost

**Verdict: GO**  (checked 2026-10-03T16:15:49Z; plan_sha 3701c8757304a840)

## Plan

- **question**: What does the non-finite-loss guard's per-step host sync cost on the RTX 3080, and are guarded GPU fits bit-identical to fits of the base commit 878d8ea?
- **primary_outcome**: ratio of median fit_seconds, guarded over base, per arm (none; discriminator lambda 1), atac_small seed 100, 20 epochs, one fit at a time
- **unit_of_analysis**: one fit (one fitter process)
- **split_unit**: none: no held-out evaluation
- **selection_rule**: none: every repeat is reported; medians over 3 repeats per arm and version
- **n_comparisons**: 2
- **multiple_comparison_correction**: exploratory, uncorrected: descriptive ratios, no test
- **data_filters**: atac_small as prepped (11,270 cells x 3,429 features); no filtering

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha 3701c8757304a840) |
| PF-02 | major | Multiple comparisons | pass | 2 descriptive ratios (one per arm), declared exploratory and uncorrected |
| PF-03 | blocking | Unit of analysis | n/a | no held-out evaluation in a timing check |
| PF-04 | blocking | ID integrity | n/a | one prepped file, no source merge |
| PF-05 | major | Nesting of factors | n/a | no factors or labels are compared |
| PF-06 | blocking | Selection bias | n/a | no best-of selection; every repeat is reported |
| PF-07 | blocking | Matched comparison | n/a | the two versions differ only by the guard (that is the contrast measured) |
| PF-08 | blocking | Aggregation | n/a | no pooling across datasets or tasks |
| PF-09 | blocking | No-op baselines | n/a | no perturbation or identity knob |
| PF-10 | blocking | Sample size | n/a | no conditions are created by filtering or subsampling |
| PF-11 | major | Metric vs model structure | n/a | no metric is computed (timing only) |
| PF-12 | blocking | Metric implementation | n/a | no metric is computed (timing only) |
| PF-13 | major | Estimator noise | n/a | no metric effect; timing spread is reported per version |
| PF-14 | major | Test vs control | n/a | measurement of a code change, not an experimental design |
| PF-15 | major | Attribution | n/a | no mechanism attributed beyond the guard's host sync |
| PF-16 | blocking | Timing confounds | pass | design: one fit at a time, base and guard alternated (repeat 1 base first, 2 guard first, 3 base first), 3 repeats per arm and version, fit_seconds = the fitter's own timer (no process start); launch only after the lead's 'GPU free' message, nvidia-smi checked idle before and after; repeats are c... |
| PF-17 | blocking | Cost estimate | n/a | 12 fits x 20 epochs on 11,270 cells, single lane: well under one GPU-hour |
| PF-18 | major | Co-scheduling | n/a | one fit at a time: no co-scheduling |
| PF-19 | blocking | Launch manifest | pass | time_guard.py copies the two rows from experiments/stage_runner_e2e/manifest.tsv (checked by pf_check_manifest: all 15 intended flags) and changes only max_epochs to 20; it refuses a fit whose latent was not produced at the expected commit from a clean checkout |
| PF-20 | major | Count reconciliation | n/a | 12 fits = 2 arms x 2 versions x 3 repeats, fixed in the script |
| PF-21 | major | Replicates vary | pass | repeats are not seeds: the same seed is refitted on purpose; bit identity of all 6 fits of an arm (max/dz/ over all pairs) is part of the output |
| PF-22 | blocking | Output persistence | pass | fits.csv (every fit) and summary.json are written and saved as artifacts |
| PF-23 | blocking | Provenance and inclusion | n/a | single prepped scIB file, no multi-source build |
| PF-24 | major | Config vs input | n/a | the design rows' settings, unchanged except epochs |
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
| SC-11 | blocking | inputs | n/a | obs keys as in the e2e record (same file, same fitter code path) |
| SC-12 | major | claims | n/a | no disentanglement or integration claim |
| SC-13 | major | metrics | n/a | no covariates are scored against each other |
| SC-14 | blocking | metrics | n/a | no block-structured latent |
| SC-15 | major | metrics | n/a | no block-aware metric |
| SC-16 | blocking | metrics | n/a | no kNN-based metric is computed in a timing check |
| SC-17 | major | metrics | n/a | no metric is computed in a timing check |
| SC-18 | major | aggregation | n/a | no pooling across datasets |
| SC-19 | major | inputs | n/a | a fresh OUT_DIR is used; the runner refuses stale latents and scores by value (row, device, kBET seed, latent SHA-256) |
| SC-20 | blocking | runners | n/a | the fitter's exit status is checked; any failure stops the script with the log path |
| SC-21 | minor | metrics | n/a | no continuous factor is binned |
| SC-22 | minor | claims | n/a | no field-standard claim is made |
