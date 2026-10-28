#!/usr/bin/env bash
# Runs run_all_pilot_scoring.sh under job-watch (restarts a crashed or stalled run; every restart resumes, because run_stage.py
# keeps valid score files). Ends with exit 0 only when validate_pilot_scores.sh passes (all 4 datasets x 378 scored).
# Usage: supervise.sh STATE_DIR   (STATE_DIR = job-watch state; rerun with the same dir to attach to the same job)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ST=${1:?state dir}
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
LOG=$D/jhpce_tier12/logs/pilot5_ws_scoring.log
export LOG
if [ ! -f "$ST/state.json" ]; then
  python3 /home/kendall/job-watch/jw.py start --state "$ST" --kind local --workdir "$HERE" --cmd "bash $HERE/run_all_pilot_scoring.sh" \
    --validate "bash $HERE/validate_pilot_scores.sh" --progress-file "$LOG" --stall-min 240 --retries 5 --max-attempts 8 --tag pilot5-ws-scoring
fi
exec python3 /home/kendall/job-watch/jw.py watch --state "$ST" --interval 120 --deadline-min 1000000
