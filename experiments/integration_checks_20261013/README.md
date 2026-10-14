# Integration checks 2026-10-13: tag c88cce3 vs branch final-review-fixes (SI-52)

SI-52 requires the final-review fixes to leave every non-Sinkhorn fit and score bit-identical to the tag
prereg-tier12-v2 (c88cce3). Old = tag checkout `code/wcd_prereg-tier12-v2`; new = worktree `code/wt_identity` at
2b52542 (all six guard commits). Paths below are relative to the data root
`/home/kendall/experiment_data/wasserstein-critic-deconfounding` (D).

## Re-run commands (local machine)
```
C=/home/kendall/.claude-science/conda/envs; R=D/code/wcd_final-review-fixes; I=$R/experiments/integration_checks_20261013
# 1. rows: one per code path (arm, extra keys, subsample kind, adversary input, standardisation, decoder, conditioning)
cd $I && $C/scvi-api/bin/python select_identity_rows.py results/identity_rows.tsv
# 2. fits, CPU, 1 thread, max_epochs 1, 4 lanes (about 35 min)
PREPPED_DIR=D/prepped_scib FIT_PY=$C/scvi-api/bin/python bash fit_identity.sh D/code/wcd_prereg-tier12-v2 D/code/wt_identity OUT results/identity_rows.tsv 4
# 3. all test suites at the branch commit
REPO=D/code/wt_identity OUT=OUT WORKSPACE=unused bash ../integration_checks_20261004/run_all_suites.sh
# 4. scores of 7 stored latents with the runner's thread settings and META, 3 lanes (1 to 2 h)
SCORE_PY=$C/wcd-kbet/bin/python R_HOME=$C/wcd-kbet/lib/R R_LIBS=D/Rlib_kbet PREPPED_DIR=D/prepped_scib \
  bash score_identity.sh D/code/wcd_prereg-tier12-v2 D/code/wt_identity OUT results/score_identity_latents.txt 3
```
(`score_identity_latents.txt` holds paths relative to D; prefix them with D before running.)

## Results
| check | file | outcome |
|---|---|---|
| fit identity, 63 rows | `results/fit_identity.csv` | 59 of 59 non-Sinkhorn rows give equal latents (numpy.array_equal); 4 of 4 Sinkhorn rows differ, as intended (SI-51); no non-zero exit |
| test suites at 2b52542 | `results/suites_2b52542_summary.txt` | stage_e2e 63 passed; scvi 111 passed, 1 skipped; x13+scoring 31; other 322; jhpce 42 |
| score identity, 7 latents | `results/score_identity.csv` | old = new in all 32 columns compared (scorer_git_sha, scorer_dirty, score_seconds excluded) for 7 of 7; new = the stage's own production score row for 7 of 7, apart from score_host for the local X1 latent (scored on host `gimli`) |
| fit identity at c3fd92d (re-run after the guard follow-ups G-01..G-14), 63 rows | `results/fit_identity_c3fd92d.csv` | 59 of 59 non-Sinkhorn rows equal (numpy.array_equal); 4 of 4 Sinkhorn rows differ (SI-51); no non-zero exit; new = worktree `code/wt_identity` moved to c3fd92d; command `results/run_c3fd92d.sh` (fits and scores at once, 4 + 3 lanes; NB-20261013-39, -40) |
| score identity at c3fd92d, 7 latents | `results/score_identity_c3fd92d.csv` | old = new in all 32 compared columns for 7 of 7; new = production row for 7 of 7 apart from score_host for the local X1 latent, as above |

The c3fd92d rows stand for a later candidate commit only if the diff from c3fd92d to it changes no fit or score code
(src/, scripts/fit_paper_config.py, scripts/scvi_adversarial_plan.py, scripts/score_scib_native.py and their imports);
otherwise re-run `results/run_c3fd92d.sh` with the worktree at the candidate.

The 7 latents cover A1 lambda = 0 (atac_small), a pooled critic on immune (trajectory and cell-cycle metrics), a
simulation (sim1), a standardised A3 fit, the barycenter critic at lambda 3000, a local X1 fit and an X13 sysVI filler.

## Notes
- The first fit run selected 4 retired pre-SI-41 X3 rows (composition subsample without `draw: nested_v1`, from
  `manifests/paper_manifest_stock_pilot_u5_b10.tsv`); both commits refuse them with the same error. The selection now
  skips them, and the 4 current-format replacement rows (X3 sim2 mmd x2, none, reference) are identical too; the table
  above is the final selection.
- The first score run did not set OPENBLAS_NUM_THREADS=1 and MKL_THREADING_LAYER=SEQUENTIAL (run_stage.thread_env(1));
  old and new still agreed in every column, but both differed from the production rows by 1.7e-9 to 2.3e-6 in
  PCR_batch and cell_cycle_conservation (PCA based). Kept as `results/score_identity_first_run_without_runner_threads.csv`.
- The independent guard review (RIGOR_REVIEWER, report artifact 9c6d8e2d-67b2-4b63-bf39-9f21059f1680) found 39 of 42
  CPU configurations identical to the tag, the other 3 being Sinkhorn.
