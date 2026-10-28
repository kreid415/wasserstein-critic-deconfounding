# JHPCE scoring of the pilot's two large datasets with the LISI seed fixed (SI-115)

User decision 2026-10-28: pilot scoring may be mixed across the workstation and JHPCE. This folder holds the commands that move the scoring of
immune_hum_mou (378 latents) and atac_large (378; 250 fitted when this started, 128 still to be fitted) to JHPCE shared-partition CPUs.
The workstation keeps pancreas, sim2 and lung (`experiments/pilot5_workstation_scoring/`, which now skips a dataset that has a marker file).
Notebook: NB-20261028-14 (decision), -15 (identity-check design), -16 (result), -17 (issues). Ledger: CONSTRAINTS.md SI-115.
D = /home/kendall/experiment_data/wasserstein-critic-deconfounding (workstation); S = /fastscratch/myscratch/kreid/wcd (JHPCE).

## Why scores can be mixed
scib 1.1.7 scores graph iLISI and cLISI on a random half of the cells, seeded from the CPU's RDRAND. JHPCE's RDRAND is random, so JHPCE LISI
differs run to run; the workstation's RDRAND always returns 0xffffffff, so its LISI always uses one fixed half
(branch `lisi-jhpce-investigation`@ed280a2, `lisi_jhpce_investigation_20261028.md`). `fixed_random_device.so` (SHA-256 fda4d45779c0971626018a0e68abb405b1a1b8f5810841a6bef4da6f5d9feea3,
source and build command in that branch) makes JHPCE draw the same half. Nothing in the scorer changes; no score value changes.

## What runs where
- Scorer commit 10a1ca5 (tag prereg-tier12-v3): clean checkout `S/scorer_10a1ca5` (made with `git clone score_check/repo` + `git checkout 10a1ca5`; the one job file
  `cluster/jhpce/score_array.sh` of branch pilot5-amend@fc12029, blob 58c982c, sits in it untracked, as amendment item 7 says). Score environment `$HOME/envs/wcd-score`
  (same package versions as the workstation's wcd-kbet: compared through the `scorer_versions` column), `thread_env(1)`, kBET seed 0, no GPU, Slurm feature znver2 (AMD EPYC 7702).
- `cluster/jhpce/score_array_fixed_seed.sh` (one array task = one latent, 1 CPU): checks the preload library's SHA-256, loads it (LD_PRELOAD, LISI_RANDOM_DEVICE_VALUE=0xFFFFFFFF),
  runs the repository's `score_array.sh` unchanged, appends `score_site`, `lisi_random_device`, `lisi_preload_sha256` to the score file. Runs a task again (at most twice) only when its log shows the numba cache index collision.
- `cluster/jhpce/chain_score.py --score-array ... --score-env ...` submits the throttled arrays (<= 200 CPUs, <= 2,000 GB; memory and time from `cluster/jhpce/score_sizes.json`) as latents pass the runner's validity check.
- Workstation: `cluster/jhpce/chain_ws.py --no-score --pull-scores` copies finished latents and the JHPCE score files (SHA-256 verified, columns checked) into `D/jhpce_tier12/pilot5_jhpce/{latents,scores}`; a tag the workstation already scored keeps its workstation row.
- Rule: `scripts/prereg_rules.py` `refuse_unknown_scorer_sites` + `docs/SCORER_EQUIVALENCE.json` + `docs/scorer_identity_check_20261028.json` (SI-115).

## Commands (exact re-run)
1. Preload library to JHPCE: copy `fixed_random_device.so` from branch lisi-jhpce-investigation to `S/lisi_fix/`; check its SHA-256.
2. Deploy: `git archive HEAD cluster scripts docs experiments/pilot5_jhpce_scoring experiments/pilot5_workstation_scoring experiments/chaining_board_20261028/expected` -> `S/pilot5_jhpce_scoring/repo_<sha7>/`.
3. Workstation reference of one immune_hum_mou latent (own stage folder, then copied into the mirror): `bash experiments/pilot5_jhpce_scoring/ws_identity_reference.sh A1_immune_hum_mou_none_l0_c1_s100_847db688` (55 min).
4. Identity batch on JHPCE (login node writes files; then `sbatch` each): `WCD_REPO=<deployed repo> bash experiments/pilot5_jhpce_scoring/identity_batch.sh <workdir> <hummou line file>`; `CONSTRAINT=znver3` for the cross-check.
5. Compare: `python experiments/pilot5_jhpce_scoring/identity_check.py --ref-list D/jhpce_tier12/pilot5_jhpce_scoring_identity/ref_list.tsv --cand <dir of the JHPCE score files> --out docs/scorer_identity_check_20261028.json` (exit 0 = all columns identical on every latent); put its SHA-256 in docs/SCORER_EQUIVALENCE.json.
6. Hand the datasets over: marker files `D/jhpce_tier12/pilot5_inputs/scored_on_jhpce.{immune_hum_mou,atac_large}`; stop the workstation's atac_large scoring chain (`jw.py stop --state D/jhpce_tier12/chain_ws_atac_large/jw`).
7. Production chain (login node submits two 1-CPU/1-GB controllers): `bash experiments/pilot5_jhpce_scoring/start_production_chain.sh <deployed repo>` (state `S/pilot5_jhpce_scoring/chain/state`).
8. Pull scores to the workstation, per dataset: `bash experiments/pilot5_jhpce_scoring/supervise_pull_jhpce_scores.sh <task> D/jhpce_tier12/pull_jhpce_scores_<task>`.
9. Board of both sites: `python cluster/jhpce/status_board.py --sites local --export-inventory F` (workstation), then `sbatch --export=ALL,WCD_REPO=<deployed repo> cluster/jhpce/status_board_job.sh OUT F` (JHPCE).
10. Tests: `python scripts/check_tests.py --base pilot5-workstation-scoring --src cluster/jhpce scripts/prereg_rules.py -- tests/jhpce tests/prereg --deselect tests/jhpce/test_x1_command.py::test_chained_job_clock_covers_every_job_up_its_chain`
    (that one test fails on the base branch as well).

## Changes made after the launch (2026-10-28 about 18:00, NB-20261028-20, -22)
The first 18 immune_hum_mou tasks used 19.8-33.7 GiB (request 32 GB; Slurm lets a task run above its request, and a user cannot raise a pending job's memory), so `score_sizes.json` now says 41 GB for later batches
and the chain's `measured.json` says 56 GB for atac_large (largest of three probe peaks, 36.5 GiB, x 1.5; the three peaks were 17.8, 15.7 and 36.5 GiB). The two submitted arrays' throttles were set with `scontrol update JobId=<id> ArrayTaskThrottle=<n>` (immune 24, atac_large 18)
so both datasets run at once (the chain's own budget would have kept atac_large at 3 tasks until immune_hum_mou was done).

## Limits
Identity was shown on 5 latents (atac_small x2, immune, sim1, immune_hum_mou), all 19 metric columns, znver2 only for the rule record; znver3 (EPYC 7713) also gave identical values on all 5 latents (docs/scorer_identity_check_20261028_znver3.json; not used for production: 1 h 54 min per immune_hum_mou latent against 44 min).
The workstation's RDRAND is a hardware fault: a firmware change or another machine would move every workstation LISI value by about 0.005 and make the record untrue for the workstation rows.
