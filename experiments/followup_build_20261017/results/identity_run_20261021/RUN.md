# X1 fit identity for tag prereg-x6-v1, 2026-10-21

Precondition of the follow-up code tag (review item F5; user decision SI-106, NB-20261021-04). Procedure: `../../README.md`,
Steps 0 and 1, run by `run_x6_identity_20261021.sh` (copied here) in tmux on the workstation, CPU only, 1 thread per fit, 4 lanes.

- OLD = tag `prereg-tier12-v3` (10a1ca5874fbcdc033b148525b5e71d866306bdb), NEW = 6163c4f54cfd768c0811fa41a365488b43815444; both clean
  (`commits.txt`). The tag is made on the commit that adds this folder; `git diff 6163c4f <tag> -- scripts src tests` is empty.
- Step 0 (no-fit plan comparison, zero optimiser steps): 57 of 57 rows pass every check; determinism control 5 of 5
  (`plan_identity.csv`, `plan_identity_control.csv`, `plan_exit_codes.txt`). 10:22:55 to 11:09:37 EDT.
- Step 1 (one-epoch fits, sim2, lambda 10): 57 of 57 rows pass every check (exit codes 0, latent arrays identical with
  max_abs_diff 0.0, config identical apart from provenance, loss history identical, model weights identical); determinism
  control 5 of 5 (`fit_identity.csv`, `fit_identity_control.csv`, `fit_exit_codes.txt`: 119 fit runs, all exit 0). 11:09:37 to 11:56:59 EDT.
- Environment: `scvi-api_env.yml` (micromamba env export), SHA-256 in `scvi-api_env.sha256`.
- An earlier attempt on add8f7b (started 09:54) was stopped after about 30 min when the tag candidate moved to 6163c4f
  (pre-check extended to a second arm, SI-107); its partial output is not used.
- Limits (README): one task (sim2), lambda 10, 1 epoch; latents are not committed (they stay in the run folder on the workstation,
  `identity_run_20261021b`).
