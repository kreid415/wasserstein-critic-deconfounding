#!/bin/bash
#SBATCH --job-name=wcdscore-ext-immune_hum_mou
#SBATCH --partition=shared
#SBATCH --account=jhpce
#SBATCH --array=1-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --output=/fastscratch/myscratch/kreid/wcd/replication_20261102/scoring/logs/slurm-%x_%A_%a.out
#SBATCH --constraint=znver2
#SBATCH --exclude=compute-119
set -euo pipefail
export SCORER_REPO=/fastscratch/myscratch/kreid/wcd/scorer_10a1ca5
export FIXED_SEED_LIB=/fastscratch/myscratch/kreid/wcd/lisi_fix/fixed_random_device.so
exec bash /fastscratch/myscratch/kreid/wcd/pilot5_jhpce_scoring/repo_9d23c20/cluster/jhpce/score_array_fixed_seed.sh 10a1ca5874fbcdc033b148525b5e71d866306bdb /fastscratch/myscratch/kreid/wcd/replication_20261102/scoring/ext_immune_hum_mou.tsv /fastscratch/myscratch/kreid/wcd/sweep_replication_scores
