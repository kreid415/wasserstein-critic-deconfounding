# Cluster scripts

`cluster/jhpce/` — JHPCE (Slurm; partitions `shared` / `gpu` / `transfer`, account `jhpce`) setup for the Tier 1+2 rerun:

| script | runs where | what |
|---|---|---|
| `env.sh` | sourced by every job | fastscratch layout and cache redirects (nothing in `$HOME`) |
| `build_envs.sh` | `shared` job | `wcd-fit` (= local scvi-api, torch 2.13.0+cu126) and `wcd-score` (= local wcd-kbet + R kBET afc5f431), verified against `cluster/envspec/` |
| `stage_scib.sh` | `shared` job | serial figshare download + md5, `scripts/prep_scib_task.py --counts scib`, fingerprint comparison |
| `setup_job.sh` | `shared` job | the three steps above + JHPCE side of the scoring-equivalence check (G4) |
| `gate_gpu.sh`, `gate_job.sh` | `gpu` job, one pinned GPU model | GPU witness, `tests/scvi`, the 64-fit 8-lane calibration queue replayed |
| `env_versions.py`, `kbet_fingerprint.R`, `compare_freeze.py` | both hosts | version records and checks |
| `prod_fit_job.sh` (+ `prod_helpers.py`) | `gpu` job, 1 L40S | production fits (`run_stage.py --no-score --tags-file`): node witness, `tests/scvi`, stop guard, <= 2 concurrent jobs, harvest parts + SHA-256 (docs/jhpce/PRODUCTION_JOB.md) |
| `prod_command.py` | local | prints the submission plan of a production job (submits nothing; each submission needs the user's go) |
| `harvest_local.py` | local | verifies a harvest (every part, the tar, every file) and merges it all-or-nothing into the durable OUT_DIR |
| `make_tags.py`, `tags/` | local | tags files of a production job, derived read-only from the tagged manifest |

`cluster/envspec/`: explicit conda specs and pip pins exported from the local envs on 2026-10-02, plus the local
freezes they are verified against. `scripts/score_list.sh` scores a list of latents on either host.
