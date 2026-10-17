#!/usr/bin/env bash
# Fit identity, tag prereg-tier12-v3 vs the follow-up build (SI-93..SI-98): scripts/fit_paper_config.py of OLD (tag
# checkout) and NEW (checkout of the commit to be tagged) on the same X1 and X13 rows (ROWS = select_x1_rows.py output,
# max_epochs 1), CPU only, 1 thread per fit, LANES fits at a time. identity_compare.py then writes OUT/fit_identity.csv and
# requires, per row: exit code 0 at both commits; every array of the saved latent file equal (z, obs_names, batch,
# celltype: np.array_equal); the saved config equal apart from git_sha / git_dirty / fit_seconds (so the resolved plan:
# every optimiser param group, gradient_penalty, lambda_gp, spectral_norm, adversary settings; reference batch; cell counts;
# library versions); equal loss history; equal saved model weights. Determinism control: OLD is fitted a second time on the
# first CONTROL rows (default 5) and compared with the first OLD fit by the same checks (OUT/fit_identity_control.csv).
# The tag already holds the Sinkhorn fix, so EVERY row must be identical (no Sinkhorn exception).
# Exit code: 0 only if every check passes for every row and the control; a failed fit, a missing file or a failed
# summary stops the script with a non-zero code (set -e) instead of leaving a partial record.
# NOT RUN by the follow-up build (the CPU was in use by the Sinkhorn redo); run it before the new code tag:
# README.md in this folder gives the exact sequence. Cheap no-fit companion: plan_identity.sh.
# Usage: fit_identity.sh OLD_CHECKOUT NEW_CHECKOUT OUT_DIR ROWS_TSV [LANES]   (env: PREPPED_DIR, FIT_PY, CONTROL)
set -euo pipefail
OLD=$1; NEW=$2; OUT=$3; ROWS=$4; LANES=${5:-4}
P=${PREPPED_DIR:?}; E=${FIT_PY:?}; CONTROL=${CONTROL:-5}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/out_old" "$OUT/out_new" "$OUT/out_old2"
cp "$ROWS" "$OUT/manifest.tsv"
"$E" -c "import pandas as pd, sys; m = pd.read_csv(sys.argv[1], sep='\t', comment='#', dtype=str); print('\n'.join(m.tag))" \
    "$OUT/manifest.tsv" > "$OUT/tags.txt"
head -n "$CONTROL" "$OUT/tags.txt" > "$OUT/control_tags.txt"
fit() {   # fit <old|new|old2> <tag>; records the exit code of the fit, never aborts the other fits
  local v=$1 t=$2 R rc=0; R=$([ "$v" != new ] && echo "$OLD" || echo "$NEW")
  ( cd "$R" && MANIFEST="$OUT/manifest.tsv" PREPPED_DIR="$P" OUT_DIR="$OUT/out_$v" WCD_SRC="$R/src" TAG="$t" \
      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled \
      PYTHONWARNINGS=ignore nice -n 19 "$E" scripts/fit_paper_config.py > "$OUT/out_$v/$t.log" 2>&1 ) || rc=$?
  echo "$v $t $rc" >> "$OUT/exit_codes.txt"
}
export -f fit; export OLD NEW OUT P E
: > "$OUT/exit_codes.txt"
{ for t in $(cat "$OUT/tags.txt"); do echo "old $t"; echo "new $t"; done
  for t in $(cat "$OUT/control_tags.txt"); do echo "old2 $t"; done; } | xargs -P "$LANES" -L 1 bash -c 'fit "$0" "$1"'
rc=0
"$E" "$HERE/identity_compare.py" "$OUT" fit_identity_control.csv --labels old old2 --tags "$OUT/control_tags.txt" || rc=1
"$E" "$HERE/identity_compare.py" "$OUT" fit_identity.csv || rc=1
exit $rc
