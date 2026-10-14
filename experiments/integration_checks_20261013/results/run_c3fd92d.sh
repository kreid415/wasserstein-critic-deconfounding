#!/usr/bin/env bash
# Identity proof at c3fd92d (SI-52/SI-54): old = tag checkout prereg-tier12-v2 (c88cce3), new = worktree wt_identity at c3fd92d.
set -uo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding; C=/home/kendall/.claude-science/conda/envs; I=/home/kendall/experiment_data/wasserstein-critic-deconfounding/code/wcd_final-review-fixes/experiments/integration_checks_20261013; OUT=/home/kendall/experiment_data/wasserstein-critic-deconfounding/review/identity_c3fd92d_20261014
( PREPPED_DIR=$D/prepped_scib FIT_PY=$C/scvi-api/bin/python bash $I/fit_identity.sh $D/code/wcd_prereg-tier12-v2 $D/code/wt_identity $OUT/fits $I/results/identity_rows.tsv 4 > $OUT/fits.log 2>&1 ) &
( SCORE_PY=$C/wcd-kbet/bin/python R_HOME=$C/wcd-kbet/lib/R R_LIBS=$D/Rlib_kbet PREPPED_DIR=$D/prepped_scib bash $I/score_identity.sh $D/code/wcd_prereg-tier12-v2 $D/code/wt_identity $OUT/scores $OUT/score_list.txt 3 > $OUT/scores.log 2>&1 ) &
wait
tail -2 $OUT/fits.log; tail -2 $OUT/scores.log
