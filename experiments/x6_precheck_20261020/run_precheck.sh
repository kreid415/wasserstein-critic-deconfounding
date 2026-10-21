#!/usr/bin/env bash
# Pre-check fits (PLAN.md). Install OUTSIDE the pinned checkout (D/code/x6_precheck/) and run in tmux.
# Usage: run_precheck.sh COMMIT MAN_SHA TAGS_SHA [--dry-run]
# Without --dry-run it first waits (checking every 10 min) until no run_stage.py is alive and GPU memory is below
# 500 MiB on two checks in a row, so it never shares the GPU with another run (e.g. the main benchmark's local run). Exit status = run_stage.py's; an exit line goes to logs/x6_precheck_exit.txt.
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
REPO=$D/code/wcd_x6-precheck
ENVS=/home/kendall/.claude-science/conda/envs
OUT=$D/tier12_v3/x6_precheck
MAN=$OUT/manifest.tsv; TAGS=$OUT/manifest_tags.txt
EXITLOG=$D/tier12_v3/logs/x6_precheck_exit.txt
die() { echo "$1" >&2; exit "${2:-2}"; }
[ $# -ge 3 ] || die "usage: run_precheck.sh COMMIT MAN_SHA TAGS_SHA [--dry-run]"
COMMIT=$1; MAN_SHA=$2; TAGS_SHA=$3; shift 3
PASS=(); for x in "$@"; do case "$x" in --dry-run) PASS+=("$x") ;; *) die "argument $x not allowed" ;; esac; done
[ "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT" ] || die "checkout is not at $COMMIT"
[ -z "$(git -C "$REPO" status --porcelain)" ] || die "checkout is dirty"
[ "$(sha256sum "$MAN" | cut -d' ' -f1)" = "$MAN_SHA" ] || die "manifest differs from the SHA-256 given"
[ "$(sha256sum "$TAGS" | cut -d' ' -f1)" = "$TAGS_SHA" ] || die "tags file differs from the SHA-256 given"
if [ ${#PASS[@]} -eq 0 ]; then
  free() { ! pgrep -f 'run_stage[.]py' >/dev/null && \
           [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 500 ]; }
  echo "waiting for a free GPU from $(date -Is)"
  until free && sleep 600 && free; do sleep 600; done
fi
mkdir -p "$D/tier12_v3/logs"
LOG=$D/tier12_v3/logs/x6_precheck_runner_$(date +%Y%m%dT%H%M%S).log
echo "x6 pre-check $(date -Is) | $COMMIT | log $LOG"
export KMP_AFFINITY=disabled
cd "$REPO"
rc=0
"$ENVS/scvi-api/bin/python" scripts/run_stage.py \
  --manifest "$MAN" --experiments PC6 --tasks atac_small immune sim1 --tags-file "$TAGS" \
  --out-dir "$OUT/run" --prepped-dir "$D/prepped_scib" \
  --fit-python "$ENVS/scvi-api/bin/python" --score-python "$ENVS/wcd-kbet/bin/python" \
  --r-home "$ENVS/wcd-kbet/lib/R" --r-libs "$D/Rlib_kbet" --expect-device "RTX 3080" \
  --fit-lanes 8 --fit-threads 1 --score-workers-during 3 --score-workers-after 8 \
  ${PASS[@]+"${PASS[@]}"} > "$LOG" 2>&1 || rc=$?
echo "precheck${PASS[@]+ dry-run} exit $rc $(date -Is) log $LOG" >> "$EXITLOG"
exit "$rc"
