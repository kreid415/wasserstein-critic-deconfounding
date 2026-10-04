# JHPCE first production jobs: summary for the user's go (DRAFT, nothing submitted)

**What runs.** The 250 fits that docs/PREREG.md sec 0 allows before the lambda grid is frozen, for the five JHPCE tasks
of SI-39: X1 `none` (10 per task) and `scvi_adv` (5), X13 `scanvi` (5) and `sysvi` (30), 50 per task. Fit only; every
latent is scored locally afterwards (G4). Before any fit, the node is checked (exactly one L40S, fit-env versions equal the
gate's) and `tests/scvi` runs at the exact commit. A failure aborts the job.

| job | tasks | fits | GPU / CPU / RAM | wall requested | expected run |
|---|---|---|---|---|---|
| A | atac_large, immune_hum_mou | 100 | 1 L40S / 10 / 96 GB | 24 h | ~4 h |
| B | lung, pancreas, sim2 | 150 | 1 L40S / 10 / 96 GB | 24 h | ~5.5 h |

**Where.** JHPCE `gpu` partition, account `jhpce`, `--gres=gpu:l40s:1` (SI-39: these tasks always on L40S). Code, envs,
TMPDIR, inputs, outputs and logs on `/fastscratch/myscratch/kreid/wcd`, nothing in HOME. At most 2 jobs at once (a slot
guard enforces it); the jobs never cancel anything.

**Walltime and stop guard.** 24 h each, against a ~4-5.5 h estimate (cost model x the L40S speed from the gate; no
full-length fit of these tasks has run on an L40S yet). 45 min before the end, the job stops the runner cleanly, writes its
summary and packs what is done. Resubmitting the same job resumes. Per-fit timeout 2 h (longest predicted fit: 21 min).

**Expected queue wait.** Measured once: 7.8 h for a 2-h L40S job (Fri evening). A 24-h job is unmeasured and likely waits
longer. At 19:20 EDT on Oct 3, all 12 L40S GPUs were allocated.

**Outputs.** On fastscratch: `tier12/fillers_x1_x13/` (latents, models, status, attempt records, ledger, logs) and per job
a harvest of at most 250 MiB parts with SHA-256 sums, about 2.2 GB in total (~9 parts; each download of a fastscratch file
shows one approval card). Locally, after checksum checks and an all-or-nothing merge:
`/home/kendall/experiment_data/wasserstein-critic-deconfounding/jhpce_tier12/fillers_x1_x13/`.
Local scoring afterwards: about 110 process-hours, roughly 9 h on 12 free cores or ~28 h alongside the A1 fits.

**Before the go.**
1. fix-runner (`--tags-file`, fda7578), fix-score-fit and this branch are merged into main, and the reviewer has
   re-checked the merged code (SI-40).
2. The commit to fit is fixed; the plan is printed with `cluster/jhpce/prod_command.py --expected-sha <it>`, and
   `cluster/jhpce/local_rehearsal.sh` passes for both jobs at that commit (it passes at 0a87280: jobA 100 rows,
   jobB 150 rows, each under its own stage key `...__tags-<hash>`, tests/scvi 72 passed, fingerprints 8/8).
3. If the manifest is rebuilt for SI-41/SI-42 (X3 re-tag), the tags files are regenerated (`make_tags.py --check`).
4. fastscratch purges files after 30 days: the env (built Oct 2) must be used before ~Nov 1 or rebuilt.

**Not in these jobs.** The 80 X13 CPU baselines of these tasks (harmony, scanorama, pca). The proposal is to run them on
JHPCE (SI-17), preferably as a CPU phase in the same L40S jobs. That needs runner support for CPU rows, harmony-pytorch
added to the JHPCE scoring env (a short env job, its own go), and the pending Scanorama knn 80/160 timing.
