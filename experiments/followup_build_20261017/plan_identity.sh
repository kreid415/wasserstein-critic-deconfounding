#!/usr/bin/env bash
# Resolved-plan identity WITHOUT fitting, tag prereg-tier12-v3 (OLD) vs the commit to be tagged (NEW) (review F5, item 3).
# Runs the 57 X1/X13/X3-weighted identity rows through each checkout's own scripts/fit_paper_config.py with zero optimiser
# steps (nofit_worker.py): the model, training plan and optimisers are built as in a real fit and the saved config holds
# the resolved plan (every optimiser param group, gradient_penalty, lambda_gp, spectral_norm, adversary settings),
# the reference batch, cell counts and library versions. identity_compare.py then requires, per row: exit code 0 at both
# commits, equal arrays (seeded initial latent), equal config apart from git_sha / git_dirty / fit_seconds, equal model
# weights, zero optimiser steps. Control (determinism): OLD is run a second time on the first CONTROL rows (default 5) and
# compared with the first OLD run by the same checks; if the control fails, inequality elsewhere is not interpretable.
# Runs in minutes on one thread. Not a substitute for fit_identity.sh (which trains for 1 epoch); it makes a setting
# change visible that a 1-epoch latent could hide.
# Usage: plan_identity.sh OLD_CHECKOUT NEW_CHECKOUT OUT_DIR ROWS_TSV   (env: FIT_PY; PREPPED_DIR = real prepped_scib, or
#        SYNTHETIC=1 for the stand-in sim2 input made by nofit_worker.py --make-synthetic; exactly one of the two)
set -euo pipefail
OLD=$1; NEW=$2; OUT=$3; ROWS=$4
E=${FIT_PY:?}; CONTROL=${CONTROL:-5}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "${SYNTHETIC:-0}" = 1 ]; then
  [ -z "${PREPPED_DIR:-}" ] || { echo "set PREPPED_DIR or SYNTHETIC=1, not both" >&2; exit 2; }
  mkdir -p "$OUT/synthetic_prepped"; "$E" "$HERE/nofit_worker.py" --make-synthetic "$OUT/synthetic_prepped/sim2__scib.h5ad"
  export PREPPED_DIR="$OUT/synthetic_prepped"
else
  : "${PREPPED_DIR:?set PREPPED_DIR (real prepped_scib) or SYNTHETIC=1}"
fi
mkdir -p "$OUT"; cp "$ROWS" "$OUT/manifest.tsv"
"$E" -c "import pandas as pd, sys; m = pd.read_csv(sys.argv[1], sep='\t', comment='#', dtype=str); print('\n'.join(m.tag))" \
    "$OUT/manifest.tsv" > "$OUT/tags.txt"
head -n "$CONTROL" "$OUT/tags.txt" > "$OUT/control_tags.txt"
# the control rows as a manifest of their own (header comment line + column line + first CONTROL rows)
{ head -n 2 "$OUT/manifest.tsv"; sed -n "3,$((CONTROL + 2))p" "$OUT/manifest.tsv"; } > "$OUT/control_manifest.tsv"
: > "$OUT/exit_codes.txt"
nice -n 19 "$E" "$HERE/nofit_worker.py" "$OLD" old "$OUT" "$OUT/manifest.tsv" "$OUT/exit_codes.txt"
nice -n 19 "$E" "$HERE/nofit_worker.py" "$NEW" new "$OUT" "$OUT/manifest.tsv" "$OUT/exit_codes.txt"
nice -n 19 "$E" "$HERE/nofit_worker.py" "$OLD" old2 "$OUT" "$OUT/control_manifest.tsv" "$OUT/exit_codes.txt"
rc=0
"$E" "$HERE/identity_compare.py" "$OUT" plan_identity_control.csv --labels old old2 --tags "$OUT/control_tags.txt" --steps-zero || rc=1
"$E" "$HERE/identity_compare.py" "$OUT" plan_identity.csv --steps-zero || rc=1
exit $rc
