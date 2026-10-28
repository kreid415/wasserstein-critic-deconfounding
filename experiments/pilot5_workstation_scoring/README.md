# Workstation scoring of the penalty-strength pilot (case iii of the scoring-site rule)

Why here: the 20-latent score-match check failed on JHPCE (run2 differs from run1: graph iLISI and cLISI do not repeat), so by the rule of
docs/PREREG.md section 9 item 6 case (iii) the five new pilot datasets are scored on the workstation, at the scorer commit 10a1ca5
(tag prereg-tier12-v3; item 7). User decision 2026-10-28 (NB-20261028-09). The first three datasets keep their workstation scores
(scorer c88cce3, read as 10a1ca5 under SI-54).

Same scorer as the first three datasets: `scripts/run_stage.py` from the clean checkout
`$D/code/wcd_prereg-tier12-v3` (HEAD 10a1ca5, no uncommitted change), score env `wcd-kbet` (R 4.5.2, kBET 0.99.6), `thread_env(1)`,
kBET seed 0, no GPU. D = /home/kendall/experiment_data/wasserstein-critic-deconfounding. Scores land in `$D/jhpce_tier12/pilot5_jhpce/scores/`
(the folder to give the strength-grid rule as `--scores`); fits are never started (`--expect-device L40S` cannot be met here).

## Inputs and provenance
- Manifest `paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv` sha256 5dd65550fdb18e5c80e73dcfa731d72bdeb6128bdeac7ffb36df6c54450396ff and the five
  run lists `pilot5_jhpce_<task>.tags` (hashes in PILOT5_BUILD.md on branch pilot5-amend) come from `git show origin/pilot5-amend:...` into `$D/jhpce_tier12/pilot5_inputs/`.
- Latents, models, attempts of immune_hum_mou, lung, sim2, pancreas: copied from /fastscratch/myscratch/kreid/wcd/tier12/pilot5_jhpce with
  `mirror_from_jhpce.sh`; the JHPCE-side hash list is made by `jhpce_hash_manifest.sh` (1 CPU, 1 GB). Manifest sha256
  87bb6d74d62c3e131a34bec33c16356c4d7fee9bf962a8caa7a739816252b9c0; 4,551 of 4,551 files matched on 2026-10-28.
- atac_large: copied and verified by `cluster/jhpce/chain_ws.py` (below).

## Commands (exact re-run)
1. Quick test (3 scoring runs, about 5 min): `bash experiments/pilot5_workstation_scoring/quick_test.sh $D/jhpce_tier12/pilot5_quicktest_20261028`
   (exit 0 = both comparisons PASS; output saved as `$D/jhpce_tier12/pilot5_quicktest_20261028.out`).
2. Full scoring of the four finished datasets, smaller first, supervised and resumable:
   `bash experiments/pilot5_workstation_scoring/supervise.sh $D/jhpce_tier12/jw_pilot5_ws_scoring`
   (job-watch restarts a crashed or stalled run; every restart scores only what is missing). Check:
   `bash experiments/pilot5_workstation_scoring/validate_pilot_scores.sh` (exit 0 only when 378 of 378 are scored per dataset).
3. atac_large (copy as the JHPCE fit shards finish, then score): `bash experiments/pilot5_workstation_scoring/supervise_chain_atac_large.sh $D/jhpce_tier12/chain_ws_atac_large $D/jhpce_tier12/jw_pilot5_ws_scoring`
   It scores only after the run of step 2 is done (shared memory).
4. Board of the workstation's share: `python cluster/jhpce/status_board.py --config <config with the jhpce site removed> --out-dir DIR --sites local`
   (the full config reads both sites: export the workstation inventory with `--sites local --export-inventory F` and run the board on JHPCE with `--import local=F`).
5. Tests: `python scripts/check_tests.py --base origin/chaining-board --src cluster/jhpce -- tests/jhpce/test_chain_ws.py tests/jhpce/test_status_board.py tests/jhpce/test_chain_score.py`
   (OK on 2026-10-28). `tests/jhpce/test_x1_command.py::test_chained_job_clock_covers_every_job_up_its_chain` fails on the base branch as well (not touched).

`compare_scores.py` is a verbatim copy of cluster/jhpce/compare_scores.py at origin/pilot5-amend (a4989ff), where its tests are.
Environment: score env /home/kendall/.claude-science/conda/envs/wcd-kbet; run_stage env scvi-api (same directory). Nothing here changes a score rule or a score.
