# Part 0 — measured line coverage (scope track, claude-opus-5-5)

Review copy `wcd_review_copy` HEAD aa7a27dc43834b002d37c26b1c2338689597e5ed, measured in a fresh clone (`out/scope_claude/clone`). coverage.py 7.6.1 (branch mode), installed as a private target dir and started in every child process through `sitecustomize` + `COVERAGE_PROCESS_START` (`out/scope_claude/covhook`, `bin/py-*`). CPU only (`CUDA_VISIBLE_DEVICES=`), nice 19, ≤2 threads per process.

## Pools

| pool | what ran | statements | missed | branches | partial | cover |
|---|---|---|---|---|---|---|
| probe | (1) run_stage.py dry run, a run interrupted by sandbox teardown, a resume with --clear-foreign-claims refused then cleared, the real 1-epoch stage of 47 probe rows (43 GPU-lane rows fitted on CPU by fit_paper_config.py + 4 X13 CPU rows by run_cpu_baselines.py), scoring of all 47 latents by score_scib_native.py (wcd-kbet + R kBET), gate PASS 47/47, then --report-only; (2) freeze_a1_grid.py (round 1 -> exit 3, final), freeze_matched_lambda.py --stage a1 (round 1 -> exit 3, final; and from the committed R1 inputs), decide_a2_a3.py (from regenerated and from committed inputs) on the stored A1/A1-ext/A2A3 scores; (3) build_paper_manifest.py --backbone stock --design pilot --uncond-seeds 5 --bary-iter 10; (4) make_x1_tags.py --check and a regeneration | 8235 | 5370 | 3119 | 307 | 36% |
| rehearsal | experiments/X1_jhpce/rehearse_x1_job.sh --task pancreas, twice: run 1 at the copy's HEAD stopped at job step 3 (tests/scvi needs git history the single-commit copy lacks); run 2 at a harness commit that only replaces that pytest line, exit 0, dry-run plan 360 fits | 8347 | 6715 | 3279 | 213 | 18% |
| tests | full pytest suite: tests/scvi, stage, prereg, jhpce, test_gate_task_split, test_legacy_retired in scvi-api; tests/wcd_vae in wcd-gpu (PYTHONPATH=src); tests/scoring, tests/x13 in wcd-kbet; reruns of test_trajectory_metric (wcd-kbet) and the opt-in test_run_stage_e2e | 11449 | 1691 | 4263 | 342 | 83% |
| union | all of the above | 12139 | 1935 | 4626 | 390 | 82% |

Shell and R files cannot be measured by coverage.py. Executed by this track: `experiments/X1_jhpce/rehearse_x1_job.sh` (2 runs), `cluster/jhpce/prod_fit_job.sh` (run 2: steps 1-7 and the dry-run summary; run 1: steps 1-3), `cluster/jhpce/env.sh` (sourced, main version). Not executed: every other `.sh`/`.R` file, including the launchers (they hard-code the tag checkout and paused output folders) and `cluster/jhpce/kbet_fingerprint.R`.

Measurement gaps: (a) `tests/jhpce/test_prod_job.py` runs prod_helpers.py / harvest_local.py inside throw-away clones outside the measured source tree, so their test execution is not counted; (b) `scripts/prep_scib_task.py` and `cluster/jhpce/harvest_local.py` were executed by nothing in any pool; (c) processes killed by sandbox teardown (the first stage run and rehearsal run 1) wrote no coverage data.

## Active pipeline files: lines executed by nothing (union of all pools)

| file | lines never executed |
|---|---|
| `scripts/run_stage.py` | 141,167,191-194,217,233,274,277,288-289,309,314,392,395-396,398,413,425-426,429,439,450,454,457-458,465,481,484,492,495-497,554,578,587,603,614,616,630,640,663,693,701,723-724,726-727,739-747,749-750,755-766,768-771,781,818,847-849,869,878,884,897,923,925-926,929,932,991,1000-1002,1008-1009,1022-1024,1053,1063-1064,1086,1151-1152,1209,1211,1213,1215,1222 |
| `scripts/fit_paper_config.py` | 107,142,151,154,158,176,182,189,216,236,261-262,279,286-287,289,353,455 |
| `scripts/scvi_adversarial_plan.py` | 78,127,142,174,197,276,454,480,483,486,516,551 |
| `scripts/score_scib_native.py` | 84,103-105,223,227,231 |
| `scripts/run_cpu_baselines.py` | 67-68,89,98,134,157,183,185,215,226,228,270-271 |
| `scripts/prereg_rules.py` | 37,125,153-154,226,481-482,488-489,676-677,763,765 |
| `scripts/freeze_matched_lambda.py` |  |
| `scripts/decide_a2_a3.py` | 59 |
| `scripts/build_paper_manifest.py` | 139,164,175,212,234,267-268,297 |
| `scripts/fit_outcome.py` | 57,71,75,77,80,84,100,102,104,106,108,110 |
| `src/wcd_vae/wcd/critic.py` | 34,51,81,85,88,99,101,113,125,128,173,190,196 |
| `src/wcd_vae/wcd/adversarial.py` | 64,88,103,118-119 |
| `src/wcd_vae/wcd/alignment.py` | 64,66,87,105,175,180-181,183 |
| `src/wcd_vae/wcd/barycenter.py` | 51,61,65 |
| `src/wcd_vae/wcd/discriminator_losses.py` | 24,62,71-72,86 |
| `src/wcd_vae/wcd/sampling.py` | 34 |
| `cluster/jhpce/prod_helpers.py` | 67-71,73-83,92,94,98,101,105,107,110,112,126-128,140-151,153-159,162,166-167,174,185-200,202,215,217,219,226-230,234-239,241-249,270,273-275,279-281,283,285,290,292,294,296-298,309,321-330,336,345-347,351-359,364-371,374-375,377-383,395-409,414-417,421-433,435-437,439,441-443,450-475,477-479,482-484,536-538 |
| `cluster/jhpce/harvest_local.py` | never executed |
| `scripts/fingerprint_prepped.py` | 62-70,77,80,85,99,102-103,109 |
| `scripts/prep_scib_task.py` | never executed |

## Executed lines per file (all measured files)

Columns: executed lines and branches taken per pool; `label` from inventory_part0.csv; `not executed in probes/rehearsal` = lines of the file that only tests, or nothing, executed.

| file | label | stmts | probe | rehearsal | tests | union | branches taken (union) | not executed in probes or rehearsal |
|---|---|---|---|---|---|---|---|---|
| `cluster/jhpce/env_intact.py` | active | 55 | 0 | 0 | 55 | 55 | 33/34 | all |
| `cluster/jhpce/env_versions.py` | active | 50 | 0 | 33 | 0 | 33 | 7/14 | 31-32,50-52,54,56-60,63-68 |
| `cluster/jhpce/make_tags.py` | active | 79 | 0 | 0 | 65 | 65 | 29/41 | all |
| `cluster/jhpce/prod_command.py` | active | 54 | 0 | 0 | 15 | 15 | 1/23 | all |
| `cluster/jhpce/prod_helpers.py` | active | 421 | 0 | 201 | 0 | 201 | 85/250 | 67-71,73-83,92,94,98,101,105,107,110,112,126-128,140-151,153-159,162,166-167,174,185-200,202,215,217,219,226-230,234-239,241-249,270,273-275,279-281,283,285,... |
| `experiments/X1_jhpce/x1_jhpce_command.py` | active | 64 | 0 | 56 | 0 | 56 | 15/24 | 41,43,63,65,67,69,71,74 |
| `legacy/analyze_final.py` | historical | 1 | 0 | 0 | 1 | 1 | 0/0 | all |
| `legacy/build_final_manifest.py` | historical | 1 | 0 | 0 | 1 | 1 | 0/0 | all |
| `legacy/build_final_report.py` | historical | 1 | 0 | 0 | 1 | 1 | 0/0 | all |
| `legacy/build_unified_manifest.py` | historical | 1 | 0 | 0 | 1 | 1 | 0/0 | all |
| `legacy/run_final_baselines.py` | historical | 1 | 0 | 0 | 1 | 1 | 0/0 | all |
| `legacy/score_final_config.py` | historical | 1 | 0 | 0 | 1 | 1 | 0/0 | all |
| `legacy/scvi_adv_fit.py` | historical | 1 | 0 | 0 | 1 | 1 | 0/0 | all |
| `scripts/build_manifest.py` | historical | 122 | 0 | 0 | 122 | 122 | 76/76 | all |
| `scripts/build_paper_manifest.py` | active | 165 | 153 | 35 | 157 | 157 | 101/109 | 139,164,175,202,204,206,208,212,234,267-268,297 |
| `scripts/check_arm_bitidentity.py` | active | 78 | 0 | 0 | 26 | 26 | 4/28 | all |
| `scripts/collect_results.py` | historical | 55 | 0 | 0 | 49 | 49 | 13/18 | all |
| `scripts/cost_model.py` | active | 139 | 0 | 0 | 52 | 52 | 15/56 | all |
| `scripts/decide_a2_a3.py` | active | 79 | 73 | 0 | 77 | 78 | 26/27 | 52-56,59 |
| `scripts/fingerprint_prepped.py` | active | 77 | 0 | 61 | 0 | 61 | 20/40 | 62-70,77,80,85,99,102-103,109 |
| `scripts/fit_outcome.py` | active | 71 | 16 | 49 | 59 | 59 | 35/50 | 57,71,75,77,80,84,96-111 |
| `scripts/fit_paper_config.py` | active | 311 | 256 | 184 | 283 | 293 | 131/146 | 66,71,74,76,84,86,88,90,93,95,105,107,131,136,142,151,154,158,176,182,188-189,207,209,214,216,221,228,236,259,261-262,274,279,286-287,289,353,407,416-420,438... |
| `scripts/freeze_a1_grid.py` | active | 50 | 50 | 0 | 49 | 50 | 12/12 | nan |
| `scripts/freeze_matched_lambda.py` | active | 92 | 80 | 0 | 86 | 92 | 24/24 | 76-80,101-102,123,126-127,130,140 |
| `scripts/gate_task_split.py` | active | 206 | 0 | 0 | 57 | 57 | 18/110 | all |
| `scripts/host_info.py` | active | 15 | 14 | 0 | 14 | 14 | 11/14 | 19 |
| `scripts/manifest_v3_report.py` | active | 107 | 0 | 0 | 67 | 67 | 41/63 | all |
| `scripts/prereg_rules.py` | active | 552 | 520 | 90 | 538 | 539 | 323/335 | 37,83,124-126,153-154,226,247-248,296-297,429,435,481-482,485-486,488-489,515,565,584,587,628,676-677,728-729,735,763,765 |
| `scripts/run_baselines.py` | historical | 100 | 0 | 0 | 28 | 28 | 1/22 | all |
| `scripts/run_biology.py` | historical | 107 | 0 | 0 | 30 | 30 | 1/26 | all |
| `scripts/run_cpu_baselines.py` | active | 187 | 143 | 23 | 174 | 174 | 79/89 | 67-68,78,89,98,117,120,126,130,134,152-162,177,180,183,185,189,192,215,226,228,238-241,243-246,252,264-266,270-271 |
| `scripts/run_experiment.py` | historical | 124 | 0 | 0 | 68 | 68 | 38/68 | all |
| `scripts/run_formulations.py` | historical | 78 | 0 | 0 | 34 | 34 | 1/28 | all |
| `scripts/run_reference.py` | historical | 76 | 0 | 0 | 25 | 25 | 1/26 | all |
| `scripts/run_stage.py` | active | 951 | 672 | 340 | 841 | 841 | 426/513 | 141,167,177-178,191-194,217,233,235,238,240,243,249-250,260,274,277,288-289,294,297,300,309,314,319,322,340-341,345,347,392,395-396,398,402,405,410,413,423-4... |
| `scripts/score_scib_native.py` | active | 137 | 107 | 0 | 130 | 130 | 32/36 | 66-67,84,103-105,122-124,141,166-178,190,195,198,220,223,227,231 |
| `scripts/scvi_adversarial_plan.py` | active | 383 | 344 | 301 | 365 | 371 | 149/162 | 70,75,78,127,140-146,167,169,172,174,197,201,205,209,215,276,310,322-326,342,454,480,483,486,492,516,518,520,551 |
| `scripts/support_overlap.py` | historical | 77 | 0 | 0 | 27 | 27 | 1/16 | all |
| `scripts/wave_status.py` | historical | 77 | 0 | 0 | 32 | 32 | 13/48 | all |
| `scripts/x3_design_profile.py` | active | 98 | 0 | 0 | 93 | 93 | 41/46 | all |
| `src/wcd_vae/__init__.py` | historical | 7 | 0 | 0 | 7 | 7 | 0/0 | all |
| `src/wcd_vae/wcd/__init__.py` | historical | 7 | 0 | 0 | 7 | 7 | 0/0 | all |
| `src/wcd_vae/wcd/adversarial.py` | active | 56 | 45 | 46 | 51 | 51 | 20/26 | 64,88,103,118-119 |
| `src/wcd_vae/wcd/alignment.py` | active | 115 | 94 | 56 | 107 | 107 | 22/30 | 64,66,87,105,175,180-181,183 |
| `src/wcd_vae/wcd/backbones.py` | historical | 116 | 0 | 0 | 115 | 115 | 5/6 | all |
| `src/wcd_vae/wcd/barycenter.py` | active | 38 | 23 | 6 | 35 | 35 | 13/16 | 51,61,65,68,79-89 |
| `src/wcd_vae/wcd/critic.py` | active | 136 | 93 | 112 | 121 | 123 | 39/52 | 34,51,81,85,88,99,101,113,123,125,128,173,175,190,196,202 |
| `src/wcd_vae/wcd/data.py` | active | 79 | 16 | 16 | 16 | 16 | 6/32 | 10-11,13-17,19-26,61,66,99,102-103,106-108,110-111,115-116,119,121-122,125,127,136-138,142-149,151-154,157-159,163-164,166-171,176,179-181,183 |
| `src/wcd_vae/wcd/discriminator_losses.py` | active | 63 | 52 | 23 | 58 | 58 | 14/18 | 24,41,62,65-66,71-72,86,88,90,109 |
| `src/wcd_vae/wcd/evaluation.py` | historical | 220 | 0 | 0 | 117 | 117 | 27/76 | all |
| `src/wcd_vae/wcd/experiment.py` | historical | 205 | 0 | 0 | 120 | 120 | 29/86 | all |
| `src/wcd_vae/wcd/hyperparameter.py` | historical | 266 | 0 | 0 | 82 | 82 | 18/96 | all |
| `src/wcd_vae/wcd/primitives.py` | historical | 36 | 0 | 0 | 34 | 34 | 3/4 | all |
| `src/wcd_vae/wcd/sampling.py` | active | 49 | 46 | 0 | 48 | 48 | 19/20 | 34,40,46 |
| `src/wcd_vae/wcd/scvi_backbone.py` | historical | 83 | 0 | 0 | 68 | 68 | 6/10 | all |
| `src/wcd_vae/wcd/training.py` | historical | 260 | 0 | 0 | 244 | 244 | 62/78 | all |
| `tests/__init__.py` | active | 0 | 0 | 0 | 0 | 0 | 0/0 | all |
| `tests/jhpce/test_env_intact.py` | active | 44 | 0 | 0 | 44 | 44 | 4/4 | all |
| `tests/jhpce/test_prod_job.py` | active | 409 | 0 | 0 | 409 | 409 | 89/90 | all |
| `tests/prereg/__init__.py` | active | 0 | 0 | 0 | 0 | 0 | 0/0 | all |
| `tests/prereg/checks.py` | active | 136 | 0 | 0 | 136 | 136 | 54/54 | all |
| `tests/prereg/conftest.py` | active | 6 | 0 | 0 | 6 | 6 | 3/4 | all |
| `tests/prereg/synth.py` | active | 73 | 0 | 0 | 71 | 71 | 23/24 | all |
| `tests/prereg/test_cli_e2e.py` | active | 146 | 0 | 0 | 146 | 146 | 20/20 | all |
| `tests/prereg/test_manifest_design.py` | active | 69 | 0 | 0 | 69 | 69 | 32/32 | all |
| `tests/prereg/test_manifest_v3.py` | active | 94 | 0 | 0 | 91 | 91 | 21/24 | all |
| `tests/prereg/test_mutation.py` | active | 22 | 0 | 0 | 22 | 22 | 6/6 | all |
| `tests/prereg/test_provenance_rule.py` | active | 49 | 0 | 0 | 48 | 48 | 7/8 | all |
| `tests/prereg/test_rules.py` | active | 282 | 0 | 0 | 282 | 282 | 69/72 | all |
| `tests/prereg/test_rules_unchanged.py` | active | 63 | 0 | 0 | 29 | 29 | 6/22 | all |
| `tests/prereg/test_si45_edge_extension.py` | active | 107 | 0 | 0 | 107 | 107 | 30/32 | all |
| `tests/prereg/test_trajectory_metric.py` | active | 31 | 0 | 0 | 31 | 31 | 8/8 | all |
| `tests/scoring/__init__.py` | active | 0 | 0 | 0 | 0 | 0 | 0/0 | all |
| `tests/scoring/test_kbet_determinism.py` | active | 79 | 0 | 0 | 77 | 77 | 20/20 | all |
| `tests/scoring/test_scorer_provenance_and_subset.py` | active | 172 | 0 | 0 | 170 | 170 | 37/39 | all |
| `tests/scvi/__init__.py` | active | 0 | 0 | 0 | 0 | 0 | 0/0 | all |
| `tests/scvi/_require.py` | active | 19 | 0 | 0 | 13 | 13 | 3/8 | all |
| `tests/scvi/test_adversarial_plan_v2.py` | active | 92 | 0 | 0 | 92 | 92 | 21/22 | all |
| `tests/scvi/test_baseline_guard.py` | active | 190 | 0 | 0 | 181 | 181 | 51/54 | all |
| `tests/scvi/test_kl_warmup.py` | active | 93 | 0 | 0 | 76 | 76 | 24/28 | all |
| `tests/scvi/test_missing_arms_plan.py` | active | 113 | 0 | 0 | 113 | 113 | 42/44 | all |
| `tests/scvi/test_nonfinite_guard.py` | active | 135 | 0 | 0 | 135 | 135 | 41/42 | all |
| `tests/scvi/test_runner_and_manifest_missing_arms.py` | active | 130 | 0 | 0 | 69 | 69 | 16/42 | all |
| `tests/scvi/test_x3_si41.py` | active | 235 | 0 | 0 | 225 | 225 | 140/149 | all |
| `tests/scvi/test_x3_x13_design.py` | active | 100 | 0 | 0 | 100 | 100 | 39/40 | all |
| `tests/stage/__init__.py` | active | 0 | 0 | 0 | 0 | 0 | 0/0 | all |
| `tests/stage/fake_env.py` | active | 160 | 0 | 0 | 153 | 153 | 86/94 | all |
| `tests/stage/test_run_stage.py` | active | 465 | 0 | 0 | 463 | 463 | 93/95 | all |
| `tests/stage/test_run_stage_e2e.py` | active | 51 | 0 | 0 | 48 | 48 | 5/6 | all |
| `tests/test_gate_task_split.py` | active | 32 | 0 | 0 | 32 | 32 | 4/4 | all |
| `tests/test_legacy_retired.py` | active | 38 | 0 | 0 | 37 | 37 | 17/18 | all |
| `tests/wcd_vae/__init__.py` | active | 0 | 0 | 0 | 0 | 0 | 0/0 | all |
| `tests/wcd_vae/test_adversarial_engages.py` | active | 75 | 0 | 0 | 75 | 75 | 8/8 | all |
| `tests/wcd_vae/test_alignment.py` | active | 57 | 0 | 0 | 57 | 57 | 20/22 | all |
| `tests/wcd_vae/test_backbones.py` | active | 37 | 0 | 0 | 37 | 37 | 12/12 | all |
| `tests/wcd_vae/test_critic_vectorised.py` | active | 125 | 0 | 0 | 110 | 110 | 44/48 | all |
| `tests/wcd_vae/test_divergences_v2.py` | active | 103 | 0 | 0 | 103 | 103 | 14/14 | all |
| `tests/wcd_vae/test_formulations.py` | active | 39 | 0 | 0 | 39 | 39 | 10/10 | all |
| `tests/wcd_vae/test_metrics.py` | active | 559 | 0 | 0 | 559 | 559 | 123/125 | all |
| `tests/wcd_vae/test_missing_arms.py` | active | 245 | 0 | 0 | 245 | 245 | 75/76 | all |
| `tests/x13/__init__.py` | active | 0 | 0 | 0 | 0 | 0 | 0/0 | all |
| `tests/x13/test_cpu_baselines.py` | active | 206 | 0 | 0 | 203 | 203 | 44/47 | all |

Full per-file data: `coverage_per_file.csv`; raw reports `cov/report/coverage_{probe,rehearsal,tests,union}.{txt,json}`; HTML for the probe pool `cov/report/html_probe/`.
