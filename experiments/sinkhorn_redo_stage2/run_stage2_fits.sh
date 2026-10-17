#!/usr/bin/env bash
# Sinkhorn redo, stage 2 fits (SI-57; plan: experiments/sinkhorn_redo_stage2/PLAN.md). One script for every stage-2 fit step;
# modelled on experiments/preflight_sinkhorn_redo_20261016/run_A1_sinkhorn_redo.sh (same pinning, clean-checkout, GPU guards).
# INSTALL IT OUTSIDE the pinned checkout (D/code/stage2/) so the checkout stays clean; run it in tmux from an ordinary terminal.
# Usage: run_stage2_fits.sh STEP --manifest PATH --manifest-sha SHA256 --tags PATH --tags-sha SHA256 [--dry-run | --report-only | --clear-stale-claims | --clear-foreign-claims]
#   STEP: r1ext1 r1ext2 (R1 extension rounds, A1 rows) | r4ext1 r4ext2 (R4-a1 extension rounds, A1 rows) |
#         a1half (SI-45 default-half A1 rows) | a2a3 (A2/A3 rows of the R4-a1 resolved manifest)
# The manifest and the tags file are produced by the rule step (stage2_rules.py) and their SHA-256 is given on the command
# line, so a file changed after the rule step is refused. The launcher refuses until stage 1 has passed its gate.
# Exit status = run_stage.py's; a line "STEP exit N <time>" is appended to D/tier12_v3/logs/stage2_exit.txt either way.
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
REPO=$D/code/wcd_prereg-tier12-v3
TAG_COMMIT=10a1ca5874fbcdc033b148525b5e71d866306bdb
ENVS=/home/kendall/.claude-science/conda/envs
STAGE1_DONE=$D/tier12_v3/A1_sinkhorn/ledger/A1__atac_small+immune+sim1__tags-a19e4683640a.done
EXITLOG=$D/tier12_v3/logs/stage2_exit.txt
die() { echo "$1" >&2; exit "${2:-2}"; }
[ $# -ge 1 ] || die "usage: run_stage2_fits.sh STEP --manifest P --manifest-sha S --tags P --tags-sha S [--dry-run|--report-only|--clear-stale-claims|--clear-foreign-claims]"
STEP=$1; shift
case "$STEP" in
  r1ext1) EXPS="A1"; TASKS="atac_small immune sim1"; OUT=$D/tier12_v3/A1_ext_r1_sinkhorn_round1 ;;
  r1ext2) EXPS="A1"; TASKS="atac_small immune sim1"; OUT=$D/tier12_v3/A1_ext_r1_sinkhorn_round2 ;;
  r4ext1) EXPS="A1"; TASKS="atac_small immune sim1"; OUT=$D/tier12_v3/A1_ext_r4_round1 ;;
  r4ext2) EXPS="A1"; TASKS="atac_small immune sim1"; OUT=$D/tier12_v3/A1_ext_r4_round2 ;;
  a1half) EXPS="A1"; TASKS="atac_small immune sim1"; OUT=$D/tier12_v3/A1_ext_r4_default_half ;;
  a2a3)   EXPS="A2 A3"; TASKS="atac_small immune"; OUT=$D/tier12_v3/A2A3 ;;
  *) die "unknown step $STEP" ;;
esac
MAN=""; MAN_SHA=""; TAGS=""; TAGS_SHA=""; PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --manifest) MAN=${2:?}; shift 2 ;;
    --manifest-sha) MAN_SHA=${2:?}; shift 2 ;;
    --tags) TAGS=${2:?}; shift 2 ;;
    --tags-sha) TAGS_SHA=${2:?}; shift 2 ;;
    --dry-run|--report-only|--clear-stale-claims|--clear-foreign-claims) PASS+=("$1"); shift ;;
    *) die "argument $1 is not allowed (only --dry-run, --report-only, --clear-stale-claims, --clear-foreign-claims; never --allow-dirty or --allow-mixed-sha: SI-85, SI-89)" ;;
  esac
done
[ -n "$MAN" ] && [ -n "$MAN_SHA" ] && [ -n "$TAGS" ] && [ -n "$TAGS_SHA" ] || die "--manifest, --manifest-sha, --tags and --tags-sha are all required"
[[ "$MAN_SHA" =~ ^[0-9a-f]{64}$ ]] && [[ "$TAGS_SHA" =~ ^[0-9a-f]{64}$ ]] || die "the SHA-256 arguments must be 64 hex digits"
case "$MAN" in "$D"/tier12_v3/*) ;; *) die "manifest must be a rule-step output under $D/tier12_v3/" ;; esac
case "$TAGS" in "$D"/tier12_v3/*) ;; *) die "tags file must be under $D/tier12_v3/" ;; esac
[ "$(git -C "$REPO" rev-parse HEAD)" = "$TAG_COMMIT" ] || die "checkout is not at $TAG_COMMIT"
[ "$(git -C "$REPO" describe --tags --exact-match)" = prereg-tier12-v3 ] || die "HEAD is not tag prereg-tier12-v3"
[ -z "$(git -C "$REPO" status --porcelain)" ] || die "checkout is dirty"
[ "$(sha256sum "$MAN" | cut -d' ' -f1)" = "$MAN_SHA" ] || die "manifest differs from the SHA-256 given"
[ "$(sha256sum "$TAGS" | cut -d' ' -f1)" = "$TAGS_SHA" ] || die "tags file differs from the SHA-256 given"
[ -e "$STAGE1_DONE" ] || die "stage 1 has not passed its gate (no $STAGE1_DONE): SI-57 rule order"
if pgrep -f 'run_stage[.]py' >/dev/null; then die "another run_stage.py is alive" 3; fi
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$mem" -lt 500 ] || die "GPU memory in use (${mem} MiB): stop the other user first" 3
mkdir -p "$D/tier12_v3/logs"
LOG=$D/tier12_v3/logs/stage2_${STEP}_runner_$(date +%Y%m%dT%H%M%S).log
echo "stage 2 $STEP $(date -Is) | $(git -C "$REPO" describe --tags --exact-match) | log $LOG"
export KMP_AFFINITY=disabled
cd "$REPO"
rc=0
"$ENVS/scvi-api/bin/python" scripts/run_stage.py \
  --manifest "$MAN" --experiments $EXPS --tasks $TASKS --tags-file "$TAGS" \
  --out-dir "$OUT" --prepped-dir "$D/prepped_scib" \
  --fit-python "$ENVS/scvi-api/bin/python" --score-python "$ENVS/wcd-kbet/bin/python" \
  --r-home "$ENVS/wcd-kbet/lib/R" --r-libs "$D/Rlib_kbet" --expect-device "RTX 3080" \
  --fit-lanes 8 --fit-threads 1 --score-workers-during 3 --score-workers-after 8 \
  ${PASS[@]+"${PASS[@]}"} > "$LOG" 2>&1 || rc=$?
echo "$STEP exit $rc $(date -Is) log $LOG" >> "$EXITLOG"
exit "$rc"
