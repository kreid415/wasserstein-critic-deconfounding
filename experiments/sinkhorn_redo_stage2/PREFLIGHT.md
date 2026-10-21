# PREFLIGHT: sinkhorn_redo_stage2

**Verdict: GO**  (checked 2026-10-19T16:29:25Z; plan_sha a9fe8a0bde249a32)

## Plan

- **question**: Can stage 2 of the Sinkhorn redo (R1 for the Sinkhorn family and its extension fits, R4-a1 over all six arms, the A2/A3 fits it resolves, R2/R3) run from the pinned checkout of tag prereg-tier12-v3 without a tag scored twice, an overwritten frozen file or a pre-fix Sinkhorn output entering a rule, and what will it cost?
- **primary_outcome**: Stage 2 launch gate (no scientific outcome computed here): every rule step exits 0 (frozen) or 3 (extension needed) with one score file per manifest tag, and every fit step passes the run_stage completion gate (exit 0). The registered outcomes of R1-R4 are unchanged and not touched.
- **unit_of_analysis**: one pilot fit = one manifest row (experiment, task, decoder condition, arm, penalty strength, seed); rule cells are task x decoder x arm
- **split_unit**: not applicable: no held-out split; all cells are used for fitting (scVI validation split inside the fitter is registered)
- **selection_rule**: none at this stage: penalty-strength windows are chosen by the registered rules R1 and R4-a1 (SI-57 order) from A1 pilot seeds 100-102, disjoint from X1 seeds 0-4 (prereg_rules.check_seed_separation)
- **n_comparisons**: 1
- **multiple_comparison_correction**: not applicable: one launch decision per step; the rules' own statistics are registered and unchanged
- **data_filters**: A1/A2/A3 rows of manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv and its extensions; pre-fix Sinkhorn outputs (c88cce3) excluded by arm, non-Sinkhorn v2 outputs reused by tag name (SI-54)

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha a9fe8a0bde249a32) |
| PF-02 | major | Multiple comparisons | n/a | one launch decision per step; no hypothesis test or ranking is produced by stage 2 (the rules' statistics are registered and unchanged at the tag) |
| PF-03 | blocking | Unit of analysis | n/a | no probe, classifier or held-out split: unsupervised scVI fits of all cells per task |
| PF-04 | blocking | ID integrity | pass | Re-checked 2026-10-19 at final-review-fixes f33ef95 (review finding F3): tests/prereg/test_stage2_rules.py + tests/stage/test_link_v2_outputs.py + tests/jhpce/test_x1_command.py 16 passed; independent acceptance (artifact accept.py 6f9d4015) fails on 874e482, passes on the fix; check_tests.py vs ... |
| PF-05 | major | Nesting of factors | n/a | no factors are compared in stage 2; factor nesting belongs to the later analyses |
| PF-06 | blocking | Selection bias | pass | penalty-strength selection is the registered R1/R4-a1 rule on A1 pilot seeds 100-102; prereg_rules.check_seed_separation (called by all three rule scripts) refuses overlap with X1 seeds 0-4 and follow-up seeds 10-12; tests/prereg/test_rules.py covers it |
| PF-07 | blocking | Matched comparison | n/a | no arm comparison is made by the wrapper or launchers; the rules compare arms by registered code that is byte-identical at the tag (git diff c3fd92d 10a1ca5 empty over fit/score code) |
| PF-08 | blocking | Aggregation | n/a | no averaging or ranking is done here; rule code unchanged at the tag |
| PF-09 | blocking | No-op baselines | n/a | no perturbation or preprocessing knob is added: lambda=0 rows (arm none) are not refitted and are bit-identical to c88cce3 (59 of 59, NB-20261013-39) |
| PF-10 | blocking | Sample size | n/a | no filtering or subsampling is added; the rules themselves refuse incomplete lambda x seed designs (prereg_rules._cell_points) |
| PF-11 | major | Metric vs model structure | n/a | every arm has one 10-dimensional latent block; no cross-structure metric is applied |
| PF-12 | blocking | Metric implementation | n/a | no metric is written, changed or ported; scorer unchanged (7 of 7 stored latents rescored identically, NB-20261013-40); stage2_rules.py only selects files |
| PF-13 | major | Estimator noise | n/a | stage 2 produces fits and rule decisions, not a new noise estimate; R1's noise estimate is registered code at the tag |
| PF-14 | major | Test vs control | n/a | no new design, probe or filter is proposed; rows are those the registered rules resolve |
| PF-15 | major | Attribution | n/a | no mechanism is attributed to any method in stage 2 |
| PF-16 | blocking | Timing confounds | pass | cost figures are labelled provisional with their basis and confound: pre-fix production fit_seconds (tier12_v2 ledgers, 8 concurrent lanes) x a corrected-code factor of 1.45 (stage-1 preflight per-task predictions) to 2.8 (NB-20261013-11 hardest regime, run alone); no end-to-end corrected-code ti... |
| PF-17 | blocking | Cost estimate | pass | pf_estimate_cost on stage-1 measured wall: 180 Sinkhorn A1 fits+scores in 94,651 s (D/tier12_v3/A1_sinkhorn/ledger/...a19e4683640a.csv, fit_started min to score_ended max) = 525.8 s per row at the same launcher settings (RTX 3080, 8 lanes, 1 thread, 3 scorers, 10a1ca5). Step 2 (r1ext1) = 18 rows:... |
| PF-18 | major | Co-scheduling | pass | same 8 GPU lanes + 3 scorers (during fits) as stage 1 and the v2 A2A3 launcher; run_stage2_fits.sh refuses when any run_stage.py is alive (pattern run_stage[.]py, shown not to self-match) or GPU memory >= 500 MiB; steps run one at a time; refusal paths executed read-only on the real D without sid... |
| PF-19 | blocking | Launch manifest | pass | pf_check_manifest per task on D/tier12_v3/rules/r1_pass1/a1_extension.tsv restricted to fit_tags.txt (18 rows, sha256 aee75716...): 6 rows each atac_small/immune/sim1 carry all 14 intended flags (n_latent 10, n_layers 1, n_hidden 128, zinb, batch_size 128, max_epochs 400/239 immune/400, train_siz... |
| PF-20 | major | Count reconciliation | pass | Independent count: r1_record.json (r1_pass1) families: only sinkhorn/1 and js/0 have status extend; js/0 = 18 discriminator rows reused from tier12_v2/A1_ext_r1_round1 (plan-fits: 18 reused, 0 moved); sinkhorn/1 = 1 edge x 3 tasks x 2 lambdas x 3 seeds = 18 rows to fit = fit_tags.txt count 18 (k=... |
| PF-21 | major | Replicates vary | pass | Seeds 100/101/102 are manifest columns read by run_stage.py per row; stage-1 completion check (LAUNCH_SINKHORN_STAGE1.md script, run 2026-10-19) cells_with_identical_seed_latents = 0 over 60 cells of the same launcher and code; the 18 step-2 rows are 6 cells x 3 seeds. |
| PF-22 | blocking | Output persistence | pass | every fit step writes per-tag latents/scores/attempts/ledger/failures.csv to its own new folder under D/tier12_v3 (2.4 TB free vs about 1.4 GB per 180 fits); run_stage completion gate gives exit 0 only when every row is final; launcher appends an exit-code trailer to D/tier12_v3/logs/stage2_exit.... |
| PF-23 | blocking | Provenance and inclusion | pass | R1 pass 1 (D/tier12_v3/rules/r1_pass1/step.json) ran prereg_rules.outcomes -> check_provenance (prereg_rules.py:320 at 10a1ca5) on the explicit list of 1,098 score files (918 v2 non-Sinkhorn + 180 v3 Sinkhorn) and exited 3 (extension) with no provenance error; n_old_sinkhorn_skipped 180 = expecte... |
| PF-24 | major | Config vs input | pass | fit flags in run_stage2_fits.sh equal those of the v2 A2A3 launcher and the stage-1 launcher (fit-lanes 8, fit-threads 1, score-workers 3/8, expect-device RTX 3080, scvi-api/wcd-kbet pythons, Rlib_kbet); only manifest, tags file, experiments, tasks and out-dir differ per step |
| SC-01 | blocking | splits | n/a | no labels are used for a probe or held-out metric in stage 2 |
| SC-02 | blocking | splits | n/a | one assay per task, no donor-level labels; no cohort pooling is created |
| SC-03 | blocking | splits | n/a | no split exists |
| SC-04 | major | splits | n/a | no split function written |
| SC-05 | blocking | gene panel | n/a | no gene panel is built; prepped files and HVG settings are fixed by the tagged manifest |
| SC-06 | major | gene panel | n/a | no foundation-model vocabulary |
| SC-07 | blocking | perturbation | n/a | no perturbation axis: lambda grid points are the registered knob and lambda 0 is the untouched none arm |
| SC-08 | blocking | perturbation | n/a | no filtering step varies in severity |
| SC-09 | blocking | perturbation | n/a | no dropout or subsampling |
| SC-10 | major | gene panel | n/a | no HVG setting is changed; manifest rows carry the tagged settings |
| SC-11 | blocking | inputs | n/a | no obs column is selected by new code; stage 2 reads manifest and score tables by column name via the tagged prereg_rules |
| SC-12 | major | claims | n/a | no disentanglement claim is made in stage 2 |
| SC-13 | major | metrics | n/a | no covariate comparison |
| SC-14 | blocking | metrics | n/a | one latent block per model |
| SC-15 | major | metrics | n/a | no block-aware metric |
| SC-16 | blocking | metrics | n/a | scorer and datasets unchanged; kNN validity is a scorer property already checked for the pilots |
| SC-17 | major | metrics | pass | a metric that cannot be computed (scib NMI zero at every resolution, SI-65) is not imputed: the runner records it as an infrastructure error, stops the stage and the user decides (SI-81 a); rule scripts raise on non-finite required metrics (prereg_rules.outcomes) |
| SC-18 | major | aggregation | n/a | no per-model pooling across datasets is computed here |
| SC-19 | major | inputs | pass | tier12_v2 folders hold pre-fix Sinkhorn results under the same tags: stage2_rules.py excludes arm sinkhorn from every --old-dir (test_stage2_rules.py mutant 1 killed) and run_stage refuses c88cce3 Sinkhorn output (SI-54); non-Sinkhorn v2 files are reused by exact tag name only |
| SC-20 | blocking | runners | pass | Re-run 2026-10-19 on installed D/code/stage2 (stage2_rules.py sha256 c90bfad6..., identical to merged final-review-fixes f33ef95): scpf_scan_fallbacks ok, 0 problems; F1/F2 refusals are SystemExit; run_stage2_fits.sh sha ca2a2d3d... unchanged from the record. |
| SC-21 | minor | metrics | n/a | no classifier-based disentanglement metric |
| SC-22 | minor | claims | n/a | no field-standard claim is made |

## Update 2026-10-19 (review finding F3)

Answered after stage 1 ended (exit 0, 180/180 scored, completion check PASS) for the first fit step r1ext1 (18 Sinkhorn rows; tags sha256 aee75716...). The record now reflects the merged code (final-review-fixes f33ef95). The per-step checks in the table below (PF-17, PF-19, PF-20, PF-21) are repeated at the launch of every later fit step (r4ext*, a1half, a2a3) and logged in the notebook; the original 2026-10-17 note follows unchanged.

## Verdict note and checks that can only be made when stage 1 has ended ("check at launch")

Verdict as of 2026-10-17: **NO-GO until the five items below are answered with real inputs**. Every other blocking item is pass or n/a with evidence. The five items are left unanswered on purpose: their inputs (stage-1 outputs, the rule-step manifests) do not exist yet. Stage 1 (180 A1 Sinkhorn fits, tmux a1sk) was still running when this record was written; no stage-1 output was opened.

| item | what to check, with what | pass condition |
|---|---|---|
| PF-17 cost | After stage 1: re-estimate stage-2 fit time from the stage-1 ledger's measured `fit_seconds` (names and seconds only) per task: `pf_estimate_cost(pilot_seconds, pilot_units, total_units, safety=2.0, pilot_settings, production_settings)`; the stage-1 tripwire (PF-17 of the stage-1 record) already compares median fit seconds with 2x the prediction | estimate x 2 below the time the user accepts; settings of the pilot equal the production ones (zstd 0 pilot vs A3 zstd 1 rows: state the gap) |
| PF-19 launch manifest | For each fit step, `pf_check_manifest(rows of the step's manifest restricted to the tags file, intended)` with the intended flags of the A1/A2/A3 settings (n_latent 10, n_layers 1, n_hidden 128, zinb, batch_size 128, max_epochs 400 / 239 immune, train_size 0.9, seeds 100-102, adv_input mean, zstd 0 for A1 and A2-default / 1 for A3, lambda numeric, no placeholder) | every row carries every intended flag; no `a1_matched*` or `g1..g6` placeholder left in the rows to fit |
| PF-20 count reconciliation | Each step's plan-fits count against its expected count in PLAN.md (extension rows are multiples of 18 for Sinkhorn; R4 extension rows multiples of 3; A2/A3 rows to fit = 36 plus the `moved` rows: non-Sinkhorn rows whose v2 score records another lambda than the resolved manifest, 3 seeds per moved window point, with `--changes` from step 7); step.json `n_score_files + n_failure_rows` against `n_rows`, and r2r3's `n_old_stale_skipped` against step 8's `moved` | counts equal, or every difference is explained in the notebook |
| PF-21 replicates vary | After each fit step: latents of the 3 seeds of one cell differ (hash bytes only, the completion check of LAUNCH_SINKHORN_STAGE1.md does this for stage 1) | `cells_with_identical_seed_latents == 0` |
| PF-23 provenance and inclusion | The rule steps run `prereg_rules.check_provenance` on the explicit score list: scorer commit (c88cce3 read as v3 by SI-54), CPU vector width, numba target, package versions and kBET seed must agree between the 918 reused v2 files and the new v3 files; also confirm `n_old_sinkhorn_skipped` equals the number of pre-fix Sinkhorn files in the listed old folders | rule step does not raise; skipped count = 180 (A1) / 180 + 36 (R2/R3 step) |

Other open items (not preflight items): G-15 (harness time limits of chained JHPCE jobs) is untouched by this track; the L40S cross-host gate re-run (CR2-01) is a separate cluster go.
