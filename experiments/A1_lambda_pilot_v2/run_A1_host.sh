#!/usr/bin/env bash
# Resume the A1 lambda pilot OUTSIDE Claude Science (in tmux on the workstation), so a platform daemon restart or an
# idle session cannot kill it (it was killed twice on 2026-10-07: notebook NB-20261007-04 and NB-20261008-01).
# Runs the tagged launcher experiments/A1_lambda_pilot/run_A1.sh from the clean checkout of prereg-tier12-v2
# (c88cce3), unchanged, with --clear-foreign-claims (claims of a runner that ran in another PID namespace).
# Refuses (exit 3) if another runner may be alive: A1's ledger written in the last 3 min, or GPU memory in use.
# Usage:  tmux new -d -s a1 'bash /home/kendall/experiment_data/wasserstein-critic-deconfounding/code/run_A1_host.sh'
#         tmux attach -t a1     (detach: Ctrl-b d)
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
C=$D/code/wcd_prereg-tier12-v2
cd "$C"
[ "$(git rev-parse HEAD)" = c88cce3c569e9de47f2d65c6794ca3890637786a ] || { echo "checkout is not at c88cce3" >&2; exit 2; }
[ -z "$(git status --porcelain)" ] || { echo "checkout is dirty" >&2; exit 2; }
L=$(ls "$D"/tier12_v2/A1/ledger/*.csv | head -1)
age=$(( $(date +%s) - $(stat -c %Y "$L") ))
[ "$age" -gt 180 ] || { echo "A1 ledger written ${age} s ago: a runner may be alive; stop it first" >&2; exit 3; }
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$mem" -lt 500 ] || { echo "GPU memory in use (${mem} MiB): a runner may be alive; stop it first" >&2; exit 3; }
LOG=$D/tier12_v2/logs/A1_runner_$(date +%Y%m%dT%H%M%S)_host.log
echo "A1 host resume $(date -Is) | $(git describe --tags --exact-match) | ledger age ${age} s | log $LOG"
exec bash experiments/A1_lambda_pilot/run_A1.sh --clear-foreign-claims > "$LOG" 2>&1
