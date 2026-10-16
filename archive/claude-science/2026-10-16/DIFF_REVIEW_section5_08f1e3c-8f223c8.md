# Section 5 of the diff review: what changed after your 2026-10-15 review

Range 08f1e3c (2026-10-14 15:50, end of the range in sections 1-4 of the guide) to 8f223c8 (2026-10-16 16:12): 139 commits; 57 lab-notebook entries, 10 ledger (CONSTRAINTS.md) entries, 72 work commits. 97 files, +16,633 / -418 lines; about 15,000 of the added lines are record files (hash lists, audit outputs, number scripts and their outputs) under experiments/.

## What the tag freezes, and what changed in it

| File | Lines +/- | What changed | Your decisions behind it |
|---|---|---|---|
| docs/PREREG.md | 286 changed | Section 9: disclosure of every X1/X13 score that existed before the tag (five groups, 158 filler score files, local X1 ledger 19 scored + 1,097 need fit + 9 pending, failures.csv 0 rows); three displays of values to an AI agent (2026-10-02, 2026-10-12, 2026-10-13), disclosed, not examined; the display audit, its cut-off and the repeat before the tag; amendment-table rows 5, 8, 9; the 19 questions with answers | SI-78 to SI-91, NB-20261016-09 to -25 |
| docs/ANALYSIS_SPEC.md | 299 changed | Agent proposals approved as rules (SI-80); zero-spread dataset kept with standard error raised to 1e-6 and counted (SI-83); scaling checks exploratory (SI-82); cap/NMI-0 case by case as a limitation (SI-81); 'examined' wording (SI-78) | SI-78 to SI-84 |
| scripts/run_stage.py (+13) | code | Refuses a divergence record of any arm that names no commit or comes from an uncommitted checkout (SI-85), and a latent that names no commit (SI-89) | SI-85, SI-89 |
| scripts/analysis_x1.py (+49/-17) | code | Floor of 1e-6 on standard errors with n_raised_datasets / n_raised_pairs output; R4-target recomputation refuses a mismatch | SI-83, AN-R fixes |
| scripts/prereg_rules.py (+22/-5) | code | Usage line names the R4 extension manifests; refusal prints recorded and recomputed targets | AN-R03, AN-R07 |
| tests/ (6 files, +296) | tests | Tests for each code change above; one test for scripts/check_tests.py (SI-91) | SI-91, fewest-tests rule |
| docs/PAPER_PLAN.md, docs/EQUIVALENT_COMMITS.json | 33, 2 | Scaling checks labelled exploratory; pointer to the PREREG disclosure; which check sees which Sinkhorn output | SI-82 |
| CONSTRAINTS.md | +117 | Ledger entries SI-78 to SI-92 | (the decisions) |

No completed fit, score or rule decision of R1 to R4 changes. The Sinkhorn redo (SI-57) has not started; X1 is paused.

Full patch of these files: final_diff_08f1e3c-8f223c8_frozen_files.patch (1805 lines).

## Record files under experiments/ (not rules; listed for completeness)

```
experiments/analysis_numerics_20261014/README.md                                               |  329 ++-
 experiments/analysis_numerics_20261014/ceilings.json                                           | 1046 ++++++++
 experiments/analysis_numerics_20261014/ceilings.py                                             |   50 +
 experiments/analysis_numerics_20261014/check_pool_numerics.py                                  |   15 +-
 experiments/analysis_numerics_20261014/compare_with_reference.json                             |  538 +++++
 experiments/analysis_numerics_20261014/compare_with_reference.py                               |  125 +
 experiments/analysis_numerics_20261014/first_review/p2_clustering_sim.json                     |   13 +
 experiments/analysis_numerics_20261014/first_review/p2_clustering_sim.py                       |   52 +
 experiments/analysis_numerics_20261014/first_review/pilot_grid_reading.csv                     |    7 +
 experiments/analysis_numerics_20261014/first_review/pilot_grid_reading.py                      |   80 +
 experiments/analysis_numerics_20261014/first_review/run_simulations.sh                         |   17 +
 experiments/analysis_numerics_20261014/first_review/sim_pipeline.py                            |  180 ++
 experiments/analysis_numerics_20261014/first_review/sim_pipeline_S0_null.txt                   |    9 +
 experiments/analysis_numerics_20261014/first_review/sim_pipeline_S1_het.txt                    |    9 +
 experiments/analysis_numerics_20261014/first_review/sim_pipeline_S2_effect.txt                 |    9 +
 experiments/analysis_numerics_20261014/first_review/sim_pipeline_S3_noisy_null.txt             |    9 +
 experiments/analysis_numerics_20261014/first_review/sim_pipeline_S4_curved.txt                 |    9 +
 experiments/analysis_numerics_20261014/first_review/sim_pipeline_W1_wiggle.txt                 |    9 +
 experiments/analysis_numerics_20261014/known_answers_effects.py                                |   43 +
 experiments/analysis_numerics_20261014/known_answers_effects.txt                               |    5 +
 experiments/analysis_numerics_20261014/known_answers_small_s.csv                               |    6 +
 experiments/analysis_numerics_20261014/known_answers_small_s.py                                |   63 +
 experiments/analysis_numerics_20261014/mutation/mutants_lead.py                                |   34 +
 experiments/analysis_numerics_20261014/mutation/mutants_r3.py                                  |   47 +
 .../mutation/{mutation_results_20261014.json => mutation_results_20261015.json}                |  373 ++-
 experiments/analysis_numerics_20261014/mutation/run_mut.py                                     |  137 +-
 experiments/analysis_numerics_20261014/ref_mp.py                                               |   90 +
 experiments/analysis_numerics_20261014/ref_pool.py                                             |   99 +
 experiments/analysis_numerics_20261014/validate_algebra.json                                   |    8 +
 experiments/analysis_numerics_20261014/validate_algebra.py                                     |   32 +
 experiments/doc_numbers_20261015/README.md                                                     |   85 +
 experiments/doc_numbers_20261015/common.py                                                     |   67 +
 experiments/doc_numbers_20261015/filler_fit_counts.py                                          |   28 +
 experiments/doc_numbers_20261015/no_pair_assumptions.py                                        |   68 +
 experiments/doc_numbers_20261015/no_pair_expectations.py                                       |  138 ++
 experiments/doc_numbers_20261015/p2_pilot_recount.py                                           |   79 +
 experiments/doc_numbers_20261015/pilot_ranges.py                                               |   48 +
 experiments/doc_numbers_20261015/results/blinding_guard.txt                                    |    1 +
 experiments/doc_numbers_20261015/results/filler_fit_counts.txt                                 |    6 +
 experiments/doc_numbers_20261015/results/filler_fit_counts_inputs.csv                          |    2 +
 experiments/doc_numbers_20261015/results/no_pair_assumptions.txt                               |   12 +
 experiments/doc_numbers_20261015/results/no_pair_assumptions_inputs.csv                        |    5 +
 experiments/doc_numbers_20261015/results/no_pair_expectations.txt                              |   50 +
 experiments/doc_numbers_20261015/results/no_pair_expectations_inputs.csv                       |   10 +
 experiments/doc_numbers_20261015/results/no_pair_groups.csv                                    |   39 +
 experiments/doc_numbers_20261015/results/p2_pilot_recount.json                                 |   46 +
 experiments/doc_numbers_20261015/results/p2_pilot_recount.txt                                  |   14 +
 experiments/doc_numbers_20261015/results/p2_pilot_recount_inputs.csv                           | 1138 +++++++++
 experiments/doc_numbers_20261015/results/pilot_ranges.txt                                      |    7 +
 experiments/doc_numbers_20261015/results/pilot_ranges_inputs.csv                               | 1100 +++++++++
 experiments/doc_numbers_20261015/results/validation_minibatch_counts.txt                       |   11 +
 experiments/doc_numbers_20261015/results/validation_minibatch_counts_inputs.csv                |    2 +
 experiments/doc_numbers_20261015/run_all.sh                                                    |   16 +
 experiments/doc_numbers_20261015/validation_minibatch_counts.py                                |   45 +
 experiments/final_review_amendments_20261013/05_power_report_nan_probe.py                      |   42 +
 experiments/final_review_amendments_20261013/power_without_sim1.py                             |    9 +-
 experiments/final_review_amendments_20261013/results/05_power_report_nan_probe.txt             |   11 +
 experiments/lead_gates_20261016/README.txt                                                     |   53 +
 experiments/lead_gates_20261016/check_tests_tier_a.exit                                        |    1 +
 experiments/lead_gates_20261016/check_tests_tier_a.log                                         |   41 +
 experiments/lead_gates_20261016/check_tests_tier_b.exit                                        |    1 +
 experiments/lead_gates_20261016/check_tests_tier_b.log                                         |   43 +
 experiments/lead_gates_20261016/check_tests_tier_c.exit                                        |    1 +
 experiments/lead_gates_20261016/check_tests_tier_c.log                                         |   97 +
 experiments/lead_gates_20261016/closures_by_entry.py                                           |   43 +
 experiments/lead_gates_20261016/closures_by_entry.txt                                          |   44 +
 experiments/lead_gates_20261016/ct_union_eval.py                                               |   54 +
 experiments/lead_gates_20261016/import_closure_reused_unchanged.py                             |   86 +
 experiments/lead_gates_20261016/import_closure_reused_unchanged.txt                            |   31 +
 experiments/lead_gates_20261016/union_of_tiers.txt                                             |    3 +
 experiments/x1_x13_score_hashes_20261015/README.md                                             |   64 +
 experiments/x1_x13_score_hashes_20261015/audit_displayed_values.json                           | 5958 ++++++++++++++++++++++++++++++++++++++++++++++
 experiments/x1_x13_score_hashes_20261015/audit_displayed_values.py                             |  258 ++
 experiments/x1_x13_score_hashes_20261015/hash_score_files.py                                   |  152 ++
 experiments/x1_x13_score_hashes_20261015/score_file_hashes.csv                                 | 1130 +++++++++
 experiments/x1_x13_score_hashes_20261015/summary.json                                          |  486 ++++
 experiments/x1_x13_score_hashes_20261015/x1_local_failures_count.py                            |   56 +
 experiments/x1_x13_score_hashes_20261015/x1_local_failures_count.txt                           |    3 +
 experiments/x1_x13_score_hashes_20261015/x1_local_status_tally.py                              |   38 +
 experiments/x1_x13_score_hashes_20261015/x1_local_status_tally.txt                             |    5 +
 80 files changed, 14982 insertions(+), 188 deletions(-)
```

## Checks on 8f223c8

- Confirming re-review (Opus) of 3a5b728..fb7edac: 15 of 16 findings fixed, 1 partly; 6 new items (1 major: a wrong commit id in PREREG). All fixed in 3b9cf67..8f223c8 and checked by the lead: hash list 1,129 rows with all 851 earlier rows unchanged, 278 added rows exactly the attempt logs; failures.csv header only, SHA-256 matches; drafting session of 6d0ca1f is 9a6f1964, not baa34fcd (found in the stored messages); no wrong commit id outside hash-chained notebook lines.
- Tests (lead, fresh clone of 8f223c8, NB-20261016-31): 445 passed, 18 skipped; check_tests OK, no WARN.
- Display audit before the tag (NB-20261016-29, folder experiments/pretag_audit_20261016): 74 of 75 sessions since 2026-10-02 (the audit's own excluded), 42,592 messages, 26,561 files, 565 names of the 1,129-file list, cut-off 2026-10-16 16:35:57 New York. Found Displays 1-3 again and one candidate line, which the lead read digit-masked and classified as latent-file fit settings, no score value (NB-20261016-30). Messages after the cut-off are checked again just before the tag.
- The tag goes on the commit that adds these audit records (the frozen files are as in 8f223c8; the later commits add only the audit folder and notebook entries).
