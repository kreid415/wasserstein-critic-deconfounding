#!/usr/bin/env bash
# Exit 0 only if run_stage.py's own file check (dry run) finds every latent of the given datasets scored with a valid score row.
# A dataset with a marker scored_on_jhpce.TASK (SI-115) is JHPCE's to score: it is skipped here (its pulled rows are counted by the board).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
rc=0
for t in "${@:-pancreas sim2 lung immune_hum_mou}"; do for task in $t; do
  if [ -e "/home/kendall/experiment_data/wasserstein-critic-deconfounding/jhpce_tier12/pilot5_inputs/scored_on_jhpce.$task" ]; then echo "$task: scored on JHPCE, skipped"; continue; fi
  out=$(bash "$HERE/run_pilot_scoring.sh" "$task" 1 --dry-run 2>&1 | grep '\[stage\]' | tail -1)
  echo "$task: $out"
  echo "$out" | grep -q "378 rows {'scored': 378}" || rc=1
done; done
exit $rc
