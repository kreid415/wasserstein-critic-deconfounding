#!/bin/bash
#SBATCH --job-name=wcd-helper-w3list
#SBATCH --partition=shared
#SBATCH --account=jhpce
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/fastscratch/myscratch/kreid/wcd/replication_20261102/scoring/logs/slurm-%x_%j.out
set -euo pipefail
cd /fastscratch/myscratch/kreid/wcd/replication_20261102/scoring
PY=$(command -v python3)
$PY make_list3.py
