#!/usr/bin/env bash
# Exit 0 only if run_stage.py's own file check (dry run) finds every latent of the given datasets scored with a valid score row.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
rc=0
for t in "${@:-pancreas sim2 lung immune_hum_mou}"; do for task in $t; do
  out=$(bash "$HERE/run_pilot_scoring.sh" "$task" 1 --dry-run 2>&1 | grep '\[stage\]' | tail -1)
  echo "$task: $out"
  echo "$out" | grep -q "378 rows {'scored': 378}" || rc=1
done; done
exit $rc
