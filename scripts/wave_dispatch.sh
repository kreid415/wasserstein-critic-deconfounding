#!/bin/bash
# SLURM array dispatcher for the NB-primary re-run wave.
# One array task = one (experiment, dataset, arg) tuple, read from tasks.txt by line number.
# Repo is synced centrally before submission; this script does NOT touch git.
set -eo pipefail
source ~/.bashrc 2>/dev/null || true
source activate $HOME/.conda/envs/wcdenv 2>/dev/null || conda activate $HOME/.conda/envs/wcdenv
export KMP_AFFINITY=disabled OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore
cd $HOME/wcd_repo
TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" scripts/wave_tasks.txt)
echo "[task $SLURM_ARRAY_TASK_ID] $TASK"
eval "$TASK"
