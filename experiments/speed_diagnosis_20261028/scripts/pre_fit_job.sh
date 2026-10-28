#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=jhpce
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -eu
J=/fastscratch/myscratch/kreid/wcd/prod/job_36551149
export PREPPED_DIR=/fastscratch/myscratch/kreid/wcd/prepped_scib WCD_SRC=$J/repo/src PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled MKL_THREADING_LAYER=SEQUENTIAL
export TMPDIR=/fastscratch/myscratch/kreid/wcd/tmp NUMBA_CACHE_DIR=/fastscratch/myscratch/kreid/wcd/numba XDG_CACHE_HOME=/fastscratch/myscratch/kreid/wcd/xdg
PY=/users/kreid/envs/wcd-fit/bin/python
echo "node $(hostname)"; 
for task in atac_large lung; do
  echo "== $task single"; $PY pre_fit_timing.py $task
  echo "== $task 8 concurrent"
  for i in 1 2 3 4 5 6 7 8; do $PY pre_fit_timing.py $task & done; wait
done
