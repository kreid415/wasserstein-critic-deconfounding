#!/bin/bash
#SBATCH --job-name=wcd-board
#SBATCH --partition=shared
#SBATCH --account=jhpce
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# One status-board pass as a 1-CPU job (the first pass hashes every pilot latent; the login node is for light work only).
# sbatch --export=ALL,WCD_REPO=$PWD cluster/jhpce/status_board_job.sh OUT_DIR [IMPORT_FILE_OF_THE_LOCAL_SITE]   (from the repo root)
# Run from a checkout of the repo on fastscratch; the python is the scoring env (numpy, pandas, scipy).
set -euo pipefail
OUT=${1:?out dir}; IMP=${2:-}
REPO=$(cd "${WCD_REPO:?set WCD_REPO to the repo checkout (Slurm runs a copy of this script)}" && pwd -P)
source "$REPO/cluster/jhpce/env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$SCORE_PY" "$REPO/cluster/jhpce/status_board.py" --out-dir "$OUT" --sites jhpce ${IMP:+--import "local=$IMP"}
