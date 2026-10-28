#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=jhpce
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
set -u
SC=/fastscratch/myscratch/kreid/wcd; SE=/users/kreid/envs/wcd-score
export REPO=$SC/score_check/repo PROFILE_PY=$PWD/profile_score.py PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore PYTHONSAFEPATH=1
export R_HOME=$SE/lib/R R_LIBS=/users/kreid/envs/Rlib_kbet CUDA_VISIBLE_DEVICES= KBET_SEED=0 TMPDIR=$SC/tmp NUMBA_CACHE_DIR=$SC/numba XDG_CACHE_HOME=$SC/xdg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled MKL_THREADING_LAYER=SEQUENTIAL PYTHONUNBUFFERED=1
TAG=A1_pancreas_mmd_l10_c0_s100_45b9014b
export PREPPED=$SC/prepped_scib/pancreas__scib.h5ad NPZ=$SC/tier12/pilot5_jhpce/latents/$TAG.npz TAG META='{}'
echo "node $(hostname) nproc $(nproc) allowed $(grep Cpus_allowed_list /proc/self/status)"; lscpu | grep -i 'model name\|thread\|core(s)\|socket'
mkdir -p out
for N in 1 4 8; do
  t0=$(date +%s.%N)
  for i in $(seq 1 $N); do
    OUT_CSV=out/n${N}_p$i.csv PROFILE_JSON=out/n${N}_p$i.json $SE/bin/python $PROFILE_PY > out/n${N}_p$i.log 2>&1 &
  done
  wait
  echo "N=$N wall_s=$(echo "$(date +%s.%N) - $t0" | bc)"
done
