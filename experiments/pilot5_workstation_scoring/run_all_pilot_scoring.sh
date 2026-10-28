#!/usr/bin/env bash
# Scores the four finished pilot datasets one after another, smaller first (pancreas, sim2, lung, immune_hum_mou).
# Resumable: run_stage.py keeps every valid score file and scores only what is missing. Run it under job-watch (supervise.sh).
# Scorers at once: 6 for the small datasets, 3 for immune_hum_mou (measured peak 18.4 GB each, 2026-10-28; this machine has 94 GB).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
LOG=${LOG:-/home/kendall/experiment_data/wasserstein-critic-deconfounding/jhpce_tier12/logs/pilot5_ws_scoring.log}
mkdir -p "$(dirname "$LOG")"
free_gb=$(free -g | awk '/^Mem:/{print $7}'); echo "[$(date -Is)] start; available memory ${free_gb} GB, load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$LOG"
for spec in "pancreas 6" "sim2 6" "lung 6" "immune_hum_mou 3"; do
  set -- $spec
  echo "[$(date -Is)] dataset $1, $2 scorers" | tee -a "$LOG"
  bash "$HERE/run_pilot_scoring.sh" "$1" "$2" --clear-stale-claims >> "$LOG" 2>&1
done
echo "[$(date -Is)] all four datasets final" | tee -a "$LOG"
