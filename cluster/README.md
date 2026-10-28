# Cluster scripts

`cluster/jhpce/` — JHPCE (Slurm; partitions `shared` / `gpu` / `transfer`, account `jhpce`) setup for the Tier 1+2 rerun:

| script | runs where | what |
|---|---|---|
| `env.sh` | sourced by every job | fastscratch layout and cache redirects (nothing in `$HOME`) |
| `build_envs.sh` | `shared` job | `wcd-fit` (= local scvi-api, torch 2.13.0+cu126) and `wcd-score` (= local wcd-kbet + R kBET afc5f431), verified against `cluster/envspec/`; envs live in `$HOME/envs` (SI-49); reuses an env only if `env_intact.py` finds no file missing beyond its build-time baseline, builds from a fresh conda package cache on fastscratch, refuses to start without room in `$HOME`, and links the tag's fastscratch env paths to the `$HOME` envs (2026-10-03 the fastscratch purge removed most of the then-fastscratch envs' standard library) |
| `stage_scib.sh` | `shared` job | serial figshare download + md5, `scripts/prep_scib_task.py --counts scib`, fingerprint comparison |
| `setup_job.sh` | `shared` job | the three steps above + JHPCE side of the scoring-equivalence check (G4) |
| `gate_gpu.sh`, `gate_job.sh` | `gpu` job, one pinned GPU model | GPU witness, `tests/scvi`, the 64-fit 8-lane calibration queue replayed |
| `env_versions.py`, `kbet_fingerprint.R`, `compare_freeze.py` | both hosts | version records and checks |
| `prod_fit_job.sh` (+ `prod_helpers.py`) | `gpu` job, 1 L40S | production fits (`run_stage.py --no-score --tags-file`): node witness, `tests/scvi`, stop guard, <= 2 concurrent jobs, harvest parts + SHA-256 (docs/jhpce/PRODUCTION_JOB.md) |
| `prod_command.py` | local | prints the submission plan of a production job (submits nothing; each submission needs the user's go) |
| `local_rehearsal.sh` | local | the job's full dry-run path at one commit (bundle clone, every check, the runner's dry run); stubs only the GPU node and Slurm |
| `harvest_local.py` | local | verifies a harvest (every part, the tar, every file) and merges it all-or-nothing into the durable OUT_DIR |
| `make_tags.py`, `tags/` | local | tags files of a production job, derived read-only from the tagged manifest |
| `status_board.py` (+ `board_config.json`, `status_board_job.sh`, `chain_lib.py`) | `shared` 1-CPU job, or any host for the presence-only groups | one run-status-board for pilot fits, pilot scoring, main-benchmark (X1) fits and scoring: expected units from the manifests, validity by `run_stage.py`'s own output contract (main groups: files only counted, never opened), Slurm accounting for explicit job ids, throughput ETA with an interval; `--export-inventory` / `--import` join two hosts |
| `chain_score.py` (+ `chain_job.sh`, `score_sizes.json`) | one 1-CPU job under `cluster/vendor/jw.py` | fit-to-scoring chain: when latents pass the validity check, submits one throttled scoring array per dataset (`score_array.sh` of the score checkout; at most 200 CPUs and 2,000 GB held), never resubmits a failed score; state in `--state-dir` |

`cluster/envspec/`: explicit conda specs and pip pins exported from the local envs on 2026-10-02, plus the local
freezes they are verified against. `scripts/score_list.sh` scores a list of latents on either host.
