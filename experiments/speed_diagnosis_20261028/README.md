# Speed diagnosis 2026-10-28 (atac_large start-up wait, scorer profile, lanes and scorers per node)

Pilot data only (pilot5 JHPCE fits, local A1 pilot scores). No main-benchmark score was read.
Report: `speed_diagnosis_20261028.md` (saved as an artifact; copy in this folder). Tables and figures: `results/`.

## Re-run
1. Fetch the 5 pilot run logs (JHPCE, read-only): `/fastscratch/myscratch/kreid/wcd/prod/job_<id>/logs/run_stage.log` for ids 36551148-36551152 into `inputs/logs/run_stage_<id>.log`; save `sacct -j 36551148,...,36551152 -P --format=JobID,JobName%24,State,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS,AveRSS,NodeList` plus the two test jobs (36746301, 36746374) as `inputs/sacct_pilot_and_test_jobs.psv`.
2. Workstation scorer runs (cs-local, env `wcd-kbet`, repo at 10a1ca5 cloned into a scratch folder): `python scripts/run_cfg.py <label> <n_procs> <t1|numba2|blas2|all2> <latent.npz,...>` with `SD=<scratch folder> D=<experiment data root>`; it launches `scripts/profile_score.py` (the unchanged scorer plus per-metric timers). Labels used: `quiet_n1/3/6/8` (pancreas latent A1_pancreas_mmd_l10_c0_s100_45b9014b), `large_single_t1`, `large_n3_t1` (immune_hum_mou latents), `exact_t1`, `exact_t1_rerun`, `exact_numba2`, `exact_blas2`, `exact_all2` (A1 latents A1_atac_small_none_l0_c1_s100_8d38e2a3, A1_sim1_none_l0_c1_s102_2ec5aa2e, A1_immune_none_l0_c0_s101_78c7bb6a). Put the outputs under `inputs/ws_out/x/<label>/`.
3. JHPCE: `pre_fit_job.sh` + `pre_fit_timing.py` (Slurm 36746301), `jhpce_scorer_job.sh` + `profile_score.py` (Slurm 36746374); outputs of the second under `inputs/jhpce_scorer/out/`. The earlier 20-latent score-match job outputs (Slurm 36569613, `check_results.tgz`) go to `inputs/check_results/`, the 20 matching local rows of `tier12_v2/A1/scores/` to `inputs/prod_scores/`.
4. `python scripts/analysis.py inputs results` rebuilds every CSV/JSON in `results/` (the committed `fig_*.png` were drawn with the project figure style; analysis.py draws the same wait figure unstyled).

## Environment and seeds
Scoring env: `env/wcd-kbet_pip_freeze.txt`, `env/wcd-kbet_conda_list.txt` (scib 1.1.7, scanpy 1.11.5, numba 0.66.0, pynndescent 0.5.13, numpy 1.26.4, kBET 0.99.6); R_HOME = env lib/R, R_LIBS = the Rlib_kbet folder; kBET seed 0; thread settings: run_stage.thread_env(1) (OMP/MKL/OPENBLAS/NUMBA threads 1, MKL_THREADING_LAYER=SEQUENTIAL) unless the label names another setting. Analysis env: `env/analysis_env.txt`.
Fit-side inputs: pilot job commit 10a1ca5, L40S node compute-172 (job 36551149).

## Not committed
Raw logs, latents and prepared data (fetch recipe above); node-level sacct list (names other cluster users).
