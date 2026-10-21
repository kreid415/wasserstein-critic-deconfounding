# Launch sheet: main benchmark (X1) after the Sinkhorn redo, 2026-10-20

Pre-launch check: `PREFLIGHT.md` in this folder, verdict **GO** (46 items: 32 pass, 14 not applicable, 0 fail; one row-count
item was first recorded as a fail by a checking-script bug and re-recorded, see its evidence). Design file (resolved X1
manifest): `manifests/paper_manifest_stock_pilot_u5_b10_v3.x1_sinkhorn_redo.tsv` = `tier12_v3/manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv`,
sha256 `034403eb206f77296bc857e5776a84e22e6a172f787913875a9ddc159e1fa6c8`. Code: tag `prereg-tier12-v3` (10a1ca5). Nothing below has been started.

## Cluster GPU check (L40S cross-host check, review item CR2-01): already run, now checked

- The job ran on 2026-10-18 (Slurm 36450131, JHPCE compute-171, one NVIDIA L40S, code 10a1ca5, clean tree) after the user's go of 2026-10-17;
  its harvest was in `D/jhpce_gate_v3/` but the comparison step (GATE_V3_PLAN.md section 5, step 3) had not been run or recorded.
- Run 2026-10-20 on the workstation (no cluster use): `python cluster/jhpce/gate_v3_check.py` (from branch jhpce-gate-v3, bf67653) with
  `--new D/jhpce_gate_v3/latents --old <2026-10-03 L40S gate latents, artifact 19d6ee4d> --manifest bench_queue_manifest.tsv (md5 9cca9c8c..., artifact fa640ada)
  --expected-sha 10a1ca5874fbcdc033b148525b5e71d866306bdb`; output `D/jhpce_gate_v3/check_v3/gate_v3_check.csv`.
- Result: harvest tar md5 OK; exit 0; 66 rows, 0 invalid; 56 of 56 non-Sinkhorn latents bit-identical to the 2026-10-03 L40S latents (pass criterion P2);
  10 Sinkhorn queue fits finished with finite latents, exit 0 (P4); on-node tests/scvi 113 passed, 1 skipped; queue wall 488.6 s for 64 fits vs 353.7 s on 2026-10-03 (P5, informational).
- So no further GPU check job is needed before the X1 jobs.

## A. Local part (no cluster; the project rule asks only that it be reported)

| | |
|---|---|
| What | 1,125 X1 rows of atac_small, immune, sim1: 19 already scored and 1 needing a score (linked non-Sinkhorn v2 outputs), 1,105 fits + up to 1,106 scores |
| Where | this workstation, RTX 3080, 8 fit lanes x 1 thread, 3 scorers during fits and 8 after (same as A1-A3) |
| Command | from the repo root (branch final-review-fixes): `tmux new-session -d -s x1loc "bash experiments/X1_local/run_X1_local.sh; echo exit \$? >> /home/kendall/experiment_data/wasserstein-critic-deconfounding/tier12_v3/logs/x1_local_exit.txt"` |
| Output | `D/tier12_v3/X1_local` (new folder; `tier12_v2/X1_local` is never written) |
| Time | 938 fit lane-hours from pilot fit times; about 138 h (5.7 days) at the measured 6.8 effective lanes; up to 275 h (11.5 days) at 2x safety; no walltime or per-fit limit |
| Stops by itself | Sinkhorn iteration-cap hit or scIB zero-NMI case (user's decision, SI-81) |
| Pass | runner completion check exits 0 with 1,125/1,125 final; X1 scores are not opened (blinding) |

## B. Cluster part: 5 JHPCE jobs (each needs the user's explicit go)

Each job: partition `gpu`, account `jhpce`, 1 L40S, 10 CPUs, 96 GB, 3-day limit (`--time=3-00:00:00`), 8 fit lanes, per-fit timeout 8 h,
fit only; harvest (about 2.5 GB) downloaded, checksum-checked and scored on the workstation. At most 2 jobs run at once. Build each with
`python experiments/X1_jhpce/x1_jhpce_command.py --task <dataset> --bundle <git bundle of 10a1ca5> [--chain-position N --after-slurm-job ID]`
and submit the printed plan to `ssh:jhpce`.

| Job | Dataset | Fits | Starts | GPU-days, plan x 1.116 (pre-fix plan) | If 1.5x slower | Longest Sinkhorn fit (est.) |
|---|---|---|---|---|---|---|
| 1 | immune_hum_mou | 360 | at submission | 2.46 (2.2) | 3.01 | 5.1 h |
| 2 | atac_large | 360 | at submission | 1.56 (1.4) | 2.68 | 4.1 h |
| 3 | lung | 360 | after job 1 (chain position 1) | 2.12 (1.9) | 2.79 | 4.6 h |
| 4 | sim2 | 360 | after job 2 (chain position 1) | 1.67 (1.5) | 2.68 | 4.4 h |
| 5 | pancreas | 360 | after job 4 (chain position 2) | 1.67 (1.5) | 2.12 | 3.3 h |

Total about 9.5 GPU-days (pre-fix plan 8.5). The factor 1.116 is the local increase from the corrected Sinkhorn solver
(corrected/pre-fix fit time median 1.87-2.07 on the pilot tasks, times Sinkhorn's share of the fit time). Chains: 1 -> 3 about 4.6 days,
2 -> 4 -> 5 about 4.9 days, plus queue waits (earlier gate jobs waited 7.8-9.4 h). Condition: job 1 can reach the 3-day limit if fits run
1.5x slower than planned; the runner stops 45 min before the limit and keeps finished models, and one resume job fits the rest
(that resume job would need its own go).

Before submission, each job's full dry-run path is rehearsed locally (`experiments/X1_jhpce/rehearse_x1_job.sh`, CPU only, nothing submitted):
results in `D/tier12_v3/logs/x1_rehearsal_20261020/`.



## Rehearsal results (2026-10-20 20:37-20:57)

All 5 cluster jobs rehearsed locally: exit 0 each; 360 fits planned each; tests/scvi 113 passed, 1 skipped; prepped-data
fingerprints 8 of 8 OK; no problems. The rehearsal launcher needed a 2-line fix first (fit environment under HOME/envs as in the
tag's env.sh, SI-49).

