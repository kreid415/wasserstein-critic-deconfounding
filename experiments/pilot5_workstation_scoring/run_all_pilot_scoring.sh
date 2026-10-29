#!/usr/bin/env bash
# Scores the four finished pilot datasets one after another, smaller first (pancreas, sim2, lung, immune_hum_mou).
# Resumable: run_stage.py keeps every valid score file and scores only what is missing. Run it under job-watch (supervise.sh).
# Memory cap (user, 2026-10-28): all scorers together stay within SCORE_MEM_BUDGET_GB (default 47 GB = half of this 94 GB machine).
# Scorers at once = floor(budget / (1.1 x peak GB of one scorer)), between 1 and 6. Peak per scorer: the measured values in PEAK_GB;
# a dataset without one is measured first by scoring one unscored latent alone under /usr/bin/time (largest process's peak).
# A dataset whose single scorer would exceed the budget stops the run (exit 3) instead of overrunning it.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
LOG=${LOG:-$D/jhpce_tier12/logs/pilot5_ws_scoring.log}
BUDGET=${SCORE_MEM_BUDGET_GB:-47}
SCORES=$D/jhpce_tier12/pilot5_jhpce/scores
mkdir -p "$(dirname "$LOG")"
# Measured 2026-10-28 on this machine: pancreas 29 GB resident for 6 scorers; sim2 VmHWM 10.0-10.5 GB per scorer.
declare -A PEAK_GB=([pancreas]=5.0 [sim2]=10.5)
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
first_unscored() {  # prints the first tag of TASK's run list without a score file, or nothing
  grep -v '^[[:space:]]*\(#\|$\)' "$D/jhpce_tier12/pilot5_inputs/pilot5_jhpce_$1.tags" | while read -r t; do
    [ -s "$SCORES/$t.csv" ] || { echo "$t"; break; }; done
}
free_gb=$(free -g | awk '/^Mem:/{print $7}'); log "start; budget ${BUDGET} GB, available memory ${free_gb} GB, load $(cut -d' ' -f1-3 /proc/loadavg)"
for task in pancreas sim2 lung immune_hum_mou; do
  if [ -e "$D/jhpce_tier12/pilot5_inputs/scored_on_jhpce.$task" ]; then
    bash "$HERE/run_pilot_scoring.sh" "$task" 1 >> "$LOG" 2>&1; continue   # prints 'skipped' and exits 0
  fi
  one=$(first_unscored "$task")
  if [ -z "$one" ]; then log "dataset $task: every latent already scored"; continue; fi
  peak=${PEAK_GB[$task]:-}
  if [ -z "$peak" ]; then
    tmp=$(mktemp -d); echo "$one" > "$tmp/one.tags"
    log "dataset $task: measuring one scorer's peak memory on $one"
    TAGS_FILE="$tmp/one.tags" /usr/bin/time -f %M -o "$tmp/maxrss_kb" bash "$HERE/run_pilot_scoring.sh" "$task" 1 --clear-stale-claims >> "$LOG" 2>&1
    peak=$(awk '{printf "%.1f", $1/1048576}' "$tmp/maxrss_kb"); rm -rf "$tmp"
    [ -s "$SCORES/$one.csv" ] || { log "dataset $task: probe left no score file for $one"; exit 4; }
  fi
  n=$(awk -v b="$BUDGET" -v p="$peak" 'BEGIN{n=int(b/(1.1*p)); if(n>6)n=6; print n}')
  if [ "$n" -lt 1 ]; then log "dataset $task: one scorer (${peak} GB x 1.1) exceeds the ${BUDGET} GB budget; stopping"; exit 3; fi
  log "dataset $task, $n scorers (peak ${peak} GB each, budget ${BUDGET} GB)"
  bash "$HERE/run_pilot_scoring.sh" "$task" "$n" --clear-stale-claims >> "$LOG" 2>&1
done
log "all four datasets final"
