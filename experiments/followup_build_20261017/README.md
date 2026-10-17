# Follow-up build, 2026-10-17 (SI-93..SI-98): re-run commands

Branch `followup-build` (cut from c241af6). Environment `scvi-api` (`/home/kendall/.claude-science/conda/envs/scvi-api`),
run from the repo root, `OMP_NUM_THREADS=1`. Nothing here reads an X1 or X13 score.

```
# 1. v4 manifest (v3 is rebuilt byte for byte by the same command without the two v4 flags)
python scripts/build_paper_manifest.py --backbone stock --design pilot --uncond-seeds 5 --bary-iter 10 \
    --x12-runs 16 --followup-design v4 --out manifests/paper_manifest_stock_pilot_u5_b10_v4.tsv
# 2. row diff v3 -> v4 and cost (scripts/cost_model.py); fails if any row outside X3/X6/X7/X8/X12 differs
python experiments/followup_build_20261017/diff_report.py            # writes results/manifest_diff_v3_v4.csv
# 3. static check: which X1/X13 rows and code paths the changed code can reach (no fit)
python experiments/followup_build_20261017/static_x1_paths.py        # writes results/x1_static_paths.csv
# 4. tests (touched files only)
WCD_SRC=src PYTHONPATH=. PREPPED_DIR=<prepped_scib> python scripts/check_tests.py --base c241af6 --src scripts -- \
    tests/scvi/test_followup_v4.py tests/scvi/test_adversarial_plan_v2.py tests/scvi/test_runner_and_manifest_missing_arms.py
```

## X1 fit identity against the tag: what it proves, how to run it (precondition of the new code tag, review item F5)

What it must show (rigor review 2026-10-17, "Identity run"): for the 57 rows (53 X1/X13 neural rows: every arm x decoder x
adversary input (mean/sample) x standardisation (0/1) that X1 can take, scANVI and sysVI included, plus 4 conditioned
oracle-weighted X3 rows), fitted for 1 epoch on sim2 at lambda 10 on CPU with 1 thread, OLD = a clean checkout of tag
`prereg-tier12-v3` (10a1ca5) and NEW = the exact commit that will be tagged:
1. every row exits 0 at both commits;
2. every array of the saved latent file is equal (`np.array_equal`: z, obs_names, batch, celltype; `max_abs_diff` 0.0), the
   saved loss history is equal, and the saved model weights are equal;
3. the saved config is equal apart from `git_sha`, `git_dirty`, `fit_seconds`: the row, reference batch, cell counts,
   library versions and the resolved training plan (every optimiser parameter group, `gradient_penalty`, `lambda_gp`,
   `spectral_norm`, adversary settings);
4. determinism control: OLD fitted twice on the first 5 rows is equal by the same checks (otherwise inequality elsewhere
   cannot be interpreted);
5. the record is committed and named in the amendment and the notebook: `fit_identity.csv`, `fit_identity_control.csv`,
   `exit_codes.txt`, both commit shas, the sha256 of `conda env export` of `scvi-api`.
The script `fit_identity.sh` exits non-zero unless 1-4 hold (it has `set -euo pipefail`, records each fit's exit code, and
stops on a failed summary). Limits to state with the result: one task (sim2), lambda 10, 1 epoch; sufficient because the
changed lines do not depend on task or lambda and X1 rows never reach the spectral-norm, weight or subsample branches
(`static_x1_paths.py`, `results/x1_static_paths.csv`). Scoring code is untouched (no changed function is in
`score_scib_native.py` or `src/`), so `score_identity.sh` is a copy kept for completeness and is not needed unless a scoring
file changes.

### Step 0: the no-fit plan comparison (minutes; run it first, on the commit to be tagged)
`plan_identity.py` runs the same 57 rows through each checkout's own `scripts/fit_paper_config.py` with zero optimiser steps
(`nofit_worker.py` builds the trainer with `limit_train_batches=0`): the model, training plan and optimisers are built as in a
real fit, so the saved config holds the resolved plan, and the seeded initial latent and initial weights are compared too.
It makes a setting change visible that a 1-epoch latent could hide. Same pass criteria as 1-4 above, plus zero optimiser
steps at both commits; OLD is run twice on the first 5 rows as the determinism control. About 20-30 s per row per
checkout on the real data (loading dominates), so roughly an hour for 119 row-runs on a busy single core; run it under
`nohup` or `tmux`. Rows run 8 per worker process, because one process for all rows was killed after 20 rows (twice, no
traceback, probably out of memory).
```
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding; PY=/home/kendall/.claude-science/conda/envs/scvi-api/bin/python
M=<commit to be tagged, after all follow-up branches are merged>
git worktree add --detach $D/code/wcd_prereg-tier12-v3_identity prereg-tier12-v3   # OLD: a SECOND clean checkout, not the stage-1 one
git worktree add --detach $D/code/wcd_identity_new $M                              # NEW: clean, so git_dirty is False
O=$D/identity_run_$(date +%Y%m%d)                                                  # a new folder: the script refuses an existing one
nice -n 19 env KMP_AFFINITY=disabled OMP_NUM_THREADS=1 PYTHONPATH=src $PY experiments/followup_build_20261017/plan_identity.py \
    $D/code/wcd_prereg-tier12-v3_identity $D/code/wcd_identity_new $O/plan experiments/followup_build_20261017/results/x1_identity_rows.tsv \
    --prepped $D/prepped_scib --control 5          # exit 0 = pass; writes $O/plan/plan_identity.csv and plan_identity_control.csv
```

### Step 1: the 57 one-epoch fits (needs the CPU free of the Sinkhorn redo)
```
PREPPED_DIR=$D/prepped_scib FIT_PY=$PY CONTROL=5 bash experiments/followup_build_20261017/fit_identity.sh \
    $D/code/wcd_prereg-tier12-v3_identity $D/code/wcd_identity_new $O/fit experiments/followup_build_20261017/results/x1_identity_rows.tsv 4
```
Each fit runs under `nice` with `CUDA_VISIBLE_DEVICES=` (no GPU) and 1 thread; 4 lanes run at a time (114 fits plus 5 control
fits; time per fit not measured: take it from the first logs and run under `nohup` or `tmux`, because a command inside a
session dies with it). The two steps use the same rows file; the rows are regenerated by `python
experiments/followup_build_20261017/select_x1_rows.py results/x1_identity_rows.tsv` (reads the v3 manifest, no score).
Then copy `plan_identity.csv`, `fit_identity.csv`, the control files and `exit_codes.txt` into
`experiments/followup_build_20261017/results/identity_run_<date>/` (small files; never the latents), add the two shas
(`git -C <checkout> rev-parse HEAD`) and `sha256sum` of `conda env export -n scvi-api`, commit, and cite the folder in the
notebook entry and the amendment's "Order" item 2. Do not run on a dirty checkout (the config records `git_dirty`; it is
excluded from the comparison, so a dirty NEW would still pass, but the record would not name the tagged commit).
