#!/usr/bin/env bash
# X1 fit-identity check for tag prereg-x6-v1 (SI-106): OLD = prereg-tier12-v3 (10a1ca5), NEW = 6163c4f. README Steps 0 and 1.
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding; PY=/home/kendall/.claude-science/conda/envs/scvi-api/bin/python
OLD=$D/code/wcd_prereg-tier12-v3_identity; NEW=$D/code/wcd_identity_new; O=$D/identity_run_20261021b; L=$D/identity_logs_20261021b
mkdir -p $L; cd $NEW
ROWS=experiments/followup_build_20261017/results/x1_identity_rows.tsv
echo "start step0 $(date -Is) old=$(git -C $OLD rev-parse HEAD) new=$(git -C $NEW rev-parse HEAD)" >> $L/exit.txt
nice -n 19 env KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  $PY experiments/followup_build_20261017/plan_identity.py $OLD $NEW $O/plan $ROWS --prepped $D/prepped_scib --control 5 > $L/step0.log 2>&1
rc=$?; echo "step0 exit $rc $(date -Is)" >> $L/exit.txt
[ $rc -eq 0 ] || exit $rc
PREPPED_DIR=$D/prepped_scib FIT_PY=$PY CONTROL=5 bash experiments/followup_build_20261017/fit_identity.sh $OLD $NEW $O/fit $ROWS 4 > $L/step1.log 2>&1
rc=$?; echo "step1 exit $rc $(date -Is)" >> $L/exit.txt
exit $rc
