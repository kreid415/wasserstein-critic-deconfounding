# No-fit plan identity, 2026-10-17: tag prereg-tier12-v3 vs follow-up build

What ran: `experiments/followup_build_20261017/plan_identity.py` (re-run commands in `../../README.md`, "Step 0"), the 57 rows of
`../x1_identity_rows.tsv` on the real prepared sim2 data, CPU, 1 thread, **zero optimiser steps** (the model, training plan
and optimisers are built exactly as in a fit; no minibatch is run). Nothing here reads an X1 or X13 score.

| item | value |
|---|---|
| OLD | clean checkout of tag `prereg-tier12-v3`, commit 10a1ca5 (`git_dirty` False in all 57 saved configs) |
| NEW | clean checkout of commit 9de6426 of branch `followup-build` (`git_dirty` False in all 57 saved configs; the 5 control configs of OLD also show 10a1ca5, `git_dirty` False). The later commits of the branch change only files under `experiments/` and `tests/`: nothing under `scripts/` or `src/` differs between 9de6426 and the branch head |
| libraries | scvi-tools 1.4.2, torch 2.13.0+cu130, CPU (environment `scvi-api`), identical in every config |
| rows | 57 of 57 pass every check: exit code 0 at both commits; every saved array equal (seeded initial latent, obs names, batch, celltype); saved config equal apart from git_sha, git_dirty, fit_seconds (so the row, reference batch, cell counts, resolved plan: every optimiser parameter group, gradient_penalty, lambda_gp, spectral_norm, adversary settings); loss history equal; model weights equal; 0 optimiser steps at both commits (`plan_identity.csv`) |
| determinism control | OLD run twice on the first 5 rows: 5 of 5 pass the same checks (`plan_identity_control.csv`) |
| sensitivity check | with `critic_betas` of `fit_adversarial_scvi` changed from (0, 0.9) to (0, 0.95) in the NEW code (synthetic input, 3 rows), the comparison fails the pooled-critic row and names `/plan/fit/critic_betas[1]` and `/plan/fit/optimizers[1]/param_groups[0]/betas[1]`; the unchanged rows pass (`mutation_check_critic_betas_plan_identity.csv`). The mutation was reverted |

Limits: this checks the training setup (what a fit would run) and the seeded initial network, not 1-epoch training or the
training-step arithmetic; `gradient_penalty` in the saved plan is computed from `is_critic and not spectral_norm`, as in
`fit_paper_config.resolved_plan`, not observed from the training step. The 57 one-epoch fits of `fit_identity.sh` are still
to be run on the exact commit to be tagged (README "Step 1"); an earlier real-data attempt with all rows in one process was
killed after 20 rows (no result used).
