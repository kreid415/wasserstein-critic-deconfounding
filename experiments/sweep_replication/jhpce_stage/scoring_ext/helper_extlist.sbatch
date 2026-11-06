#!/bin/bash
#SBATCH --job-name=wcd-helper-extlist
#SBATCH --partition=shared
#SBATCH --account=jhpce
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/fastscratch/myscratch/kreid/wcd/replication_20261102/scoring/logs/slurm-%x_%j.out
set -euo pipefail
cd /fastscratch/myscratch/kreid/wcd/replication_20261102/scoring
python3 make_list_ext.py
