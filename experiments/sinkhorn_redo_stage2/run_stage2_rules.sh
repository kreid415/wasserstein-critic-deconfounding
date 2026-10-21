#!/usr/bin/env bash
# Sinkhorn redo, stage 2 rule steps and fit planning (CPU only): runs stage2_rules.py with the pinned checkout and commit
# filled in. Install outside the pinned checkout, next to stage2_rules.py (D/code/stage2/).
# Usage: run_stage2_rules.sh r1|r4a1|r2r3 [stage2_rules.py arguments without --checkout/--expect-commit]
#        run_stage2_rules.sh plan-fits [arguments]
# Exit status = stage2_rules.py's (r1 and r4a1: 0 frozen, 3 extension needed, anything else error); a line is appended to
# D/tier12_v3/logs/stage2_exit.txt either way.
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
REPO=$D/code/wcd_prereg-tier12-v3
TAG_COMMIT=10a1ca5874fbcdc033b148525b5e71d866306bdb
PY=/home/kendall/.claude-science/conda/envs/scvi-api/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
die() { echo "$1" >&2; exit "${2:-2}"; }
[ $# -ge 1 ] || die "usage: run_stage2_rules.sh r1|r4a1|r2r3|plan-fits ARGS"
STEP=$1; shift
case "$STEP" in
  r1|r4a1|r2r3) INJECT=(--checkout "$REPO" --expect-commit "$TAG_COMMIT") ;;
  plan-fits) INJECT=() ;;
  *) die "unknown step $STEP" ;;
esac
[ "$(git -C "$REPO" rev-parse HEAD)" = "$TAG_COMMIT" ] || die "checkout is not at $TAG_COMMIT"
[ "$(git -C "$REPO" describe --tags --exact-match)" = prereg-tier12-v3 ] || die "HEAD is not tag prereg-tier12-v3"
[ -z "$(git -C "$REPO" status --porcelain)" ] || die "checkout is dirty"
mkdir -p "$D/tier12_v3/logs"
rc=0
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled nice -n 19 "$PY" "$HERE/stage2_rules.py" "$STEP" ${INJECT[@]+"${INJECT[@]}"} "$@" || rc=$?
echo "rules $STEP exit $rc $(date -Is)" >> "$D/tier12_v3/logs/stage2_exit.txt"
exit "$rc"
