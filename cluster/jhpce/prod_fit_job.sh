#!/bin/bash
# JHPCE production fit job: one pinned L40S, fit only (scripts/run_stage.py --no-score); every latent is scored on the
# local host after harvest (cross-host gate G4). Requirements: docs/jhpce code check CR-03 ("JHPCE production job script:
# required contents"), CONSTRAINTS.md SI-17 / SI-29 / SI-39. Submission needs the user's explicit go (SI-29); the
# submit command is printed by cluster/jhpce/prod_command.py, which also stages the repo bundle and clones the commit.
#
# Steps (any failure before the runner aborts the job; nothing is fitted):
#   1. environment: inside Slurm, scratch-first (OUT, logs, TMPDIR, caches on fastscratch, nothing in $HOME), verified env
#   2. repository: HEAD == --expected-sha, clean tree
#   3. node witness + tests: nvidia-smi = exactly one L40S; env_versions.py --kind fit == expected_fit_versions.json;
#      WCD_SRC=src pytest -q tests/scvi at this commit
#   4. concurrency: one of 2 slots (<= 2 production GPU jobs at once), tasks disjoint from the other live job
#   5. inputs: manifest + tags-file SHA-256, tags in the manifest, tags' experiments/tasks == the job's; prepped files
#      against docs/prepped_fingerprints_scib.json
#   6. claims of earlier jobs on these tags: cleared only if their Slurm job is terminal (sacct); otherwise abort
#   7. runner --dry-run (logged), then the run in the background with a stop guard (SIGTERM at Slurm end - margin, so
#      the runner stops lanes, releases claims and writes its views while the job is still alive). The stage key is
#      the runner's: a --tags-file run gets its own key <experiments>__<tasks>__tags-<sha256[:12]> with its own
#      ledger/<key>.{csv,json,done}; the job reads it from the runner's '[stage] <key>: N rows' line (dry run and
#      run must agree) and never rebuilds it. After an interrupted or killed run the views are rebuilt from the
#      per-tag files with the runner's own --report-only (starts nothing, writes no .done).
#   8. job summary from the key's ledger (checked against the job's facts); harvest of this job's tags plus the key's
#      ledger files packed on fastscratch (parts <= 250 MiB + SHA-256)
# --dry-run: steps 1-6 and the runner's dry run, then the plan in job_summary.json; nothing is fitted or packed.
# Resume-safe: rerun the same command at the same commit; the key is the same (same tags file), fitted rows are kept
# (runner preflight), interrupted rows rerun. (Latents fitted at another commit fail the runner's mixed-SHA gate.)
# Exit: 0 gate passed | 2 node witness/tests | 3 repository | 7 concurrency/claims | 8 inputs | 9 environment |
#       10 harvest pack failed | 11 runner output breaks the stage-key/ledger contract |
#       runner codes 4 preflight refused, 5 infrastructure stop, 6 incomplete, 128+n signal.
set -euo pipefail

usage() { echo "usage: $0 --expected-sha SHA --stage NAME --tags-file REL --tags-sha256 HEX --manifest REL --manifest-sha256 HEX --experiments 'X1 X13' --tasks 'a b' [--fit-lanes 8] [--fit-timeout-s 7200] [--max-attempts 2] [--guard-margin-s 2700] [--time-left-s N] [--min-run-s 1800] [--slots 2] [--poll-s 10] [--stop-wait-s 240] [--dry-run]" >&2; exit 9; }
EXPECTED_SHA= STAGE= TAGS_REL= TAGS_SHA= MANIFEST_REL= MANIFEST_SHA= EXPERIMENTS= TASKS=
FIT_LANES=8 FIT_TIMEOUT_S=7200 MAX_ATTEMPTS=2 GUARD_MARGIN_S=2700 TIME_LEFT_S= MIN_RUN_S=1800 SLOTS=2 STOP_WAIT_S=240
POLL_S=10 DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift; continue;;
    --expected-sha) EXPECTED_SHA=$2;; --stage) STAGE=$2;; --tags-file) TAGS_REL=$2;; --tags-sha256) TAGS_SHA=$2;;
    --manifest) MANIFEST_REL=$2;; --manifest-sha256) MANIFEST_SHA=$2;; --experiments) EXPERIMENTS=$2;; --tasks) TASKS=$2;;
    --fit-lanes) FIT_LANES=$2;; --fit-timeout-s) FIT_TIMEOUT_S=$2;; --max-attempts) MAX_ATTEMPTS=$2;;
    --guard-margin-s) GUARD_MARGIN_S=$2;; --time-left-s) TIME_LEFT_S=$2;; --min-run-s) MIN_RUN_S=$2;; --slots) SLOTS=$2;;
    --poll-s) POLL_S=$2;; --stop-wait-s) STOP_WAIT_S=$2;;
    *) echo "unknown argument $1" >&2; usage;;
  esac
  [ $# -ge 2 ] || { echo "FATAL: $1 needs a value" >&2; usage; }
  shift 2
done
for v in EXPECTED_SHA STAGE TAGS_REL TAGS_SHA MANIFEST_REL MANIFEST_SHA EXPERIMENTS TASKS; do
  [ -n "${!v}" ] || { echo "FATAL: --${v,,} missing" >&2; usage; }
done
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "FATAL: --expected-sha must be a full 40-hex commit id" >&2; exit 9; }
[[ "$STAGE" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "FATAL: bad --stage $STAGE" >&2; exit 9; }
for v in POLL_S STOP_WAIT_S FIT_LANES MAX_ATTEMPTS GUARD_MARGIN_S MIN_RUN_S SLOTS; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "FATAL: --${v,,} must be a non-negative integer, got '${!v}'" >&2; exit 9; }
done

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
H="$REPO/cluster/jhpce/prod_helpers.py"
log() { echo "[$(date -Is)] $*"; }

# ---- 1. environment --------------------------------------------------------------------------------------------
[ -n "${SLURM_JOB_ID:-}" ] || { echo "FATAL: not inside a Slurm job (SLURM_JOB_ID unset)" >&2; exit 9; }
# shellcheck source=/dev/null
source "$REPO/cluster/jhpce/env.sh"                  # SCRATCH, TMPDIR, caches, FIT_ENV, FIT_PY, PREPPED_DIR on fastscratch
[ -f "$FIT_ENV/.verified" ] || { echo "FATAL: $FIT_ENV/.verified missing (cluster/jhpce/build_envs.sh)" >&2; exit 9; }
JOBDIR="$WCD_SCRATCH/prod/job_${SLURM_JOB_ID}"
OUT="$WCD_SCRATCH/tier12/$STAGE"
HARVEST="$WCD_SCRATCH/harvest/$STAGE/job_${SLURM_JOB_ID}"
SLOTDIR="$WCD_SCRATCH/prod/slots"
HOME_REAL=$(cd "$HOME" && pwd -P)       # fl: allow FL180 read only: the scratch-first guard below compares paths with $HOME
for p in "$WCD_SCRATCH" "$TMPDIR" "$REPO"; do
  # fl: allow FL180 this guard refuses any job path under $HOME (scratch-first rule)
  case "$(cd "$p" && pwd -P)/" in "$HOME_REAL"/*) echo "FATAL: $p is under \$HOME ($HOME_REAL): scratch-first rule" >&2; exit 9;; esac
done
mkdir -p "$JOBDIR/logs" "$OUT" "$HARVEST"
LOGS="$JOBDIR/logs"
cd "$JOBDIR"                                         # cwd of the runner and its fits (Lightning, tmp files): fastscratch
export KMP_AFFINITY=disabled PYTHONNOUSERSITE=1
log "job $SLURM_JOB_ID on $(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} stage=$STAGE tasks='$TASKS' out=$OUT"

# ---- 2. repository ---------------------------------------------------------------------------------------------
HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
[ "$HEAD_SHA" = "$EXPECTED_SHA" ] || { echo "FATAL: repo HEAD $HEAD_SHA != expected $EXPECTED_SHA" >&2; exit 3; }
DIRTY=$(git -C "$REPO" status --porcelain --untracked-files=no)
[ -z "$DIRTY" ] || { echo "FATAL: repo $REPO has uncommitted changes:" >&2; echo "$DIRTY" >&2; exit 3; }
log "repo $REPO at $HEAD_SHA (clean)"

# ---- 3. node witness + tests: first step on the node once the commit is confirmed --------------------------------
nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv,noheader > "$LOGS/nvidia_smi.txt" 2>&1 \
  || { cat "$LOGS/nvidia_smi.txt"; echo "FATAL: nvidia-smi failed" >&2; exit 2; }
NGPU=$(awk 'NF' "$LOGS/nvidia_smi.txt" | wc -l)
[ "$NGPU" = 1 ] && grep -q "L40S" "$LOGS/nvidia_smi.txt" \
  || { cat "$LOGS/nvidia_smi.txt"; echo "FATAL: need exactly one visible L40S, got $NGPU GPU(s)" >&2; exit 2; }
"$FIT_PY" "$REPO/cluster/jhpce/env_versions.py" --kind fit --out "$LOGS/versions_fit.json" > /dev/null 2>"$LOGS/versions_fit.err" \
  || { cat "$LOGS/versions_fit.err"; echo "FATAL: env_versions.py failed" >&2; exit 2; }
"$FIT_PY" "$H" versions --got "$LOGS/versions_fit.json" --expected "$REPO/cluster/jhpce/expected_fit_versions.json" \
  > "$LOGS/versions_check.json" || { cat "$LOGS/versions_check.json"; echo "FATAL: fit env differs from cluster/jhpce/expected_fit_versions.json" >&2; exit 2; }
( cd "$REPO" && WCD_SRC=src "$FIT_PY" -m pytest -q -p no:cacheprovider tests/scvi ) > "$LOGS/pytest_tests_scvi.txt" 2>&1 \
  || { tail -30 "$LOGS/pytest_tests_scvi.txt"; echo "FATAL: tests/scvi failed on this node at $HEAD_SHA" >&2; exit 2; }
log "node: $(cat "$LOGS/nvidia_smi.txt"); tests: $(grep -E 'passed|failed' "$LOGS/pytest_tests_scvi.txt" | tail -1)"

# ---- 4. concurrency ---------------------------------------------------------------------------------------------
SLOT_HELD=0
release_slot() {
  if [ "$SLOT_HELD" = 1 ]; then
    "$FIT_PY" "$H" slot release --slots-dir "$SLOTDIR" --job "$SLURM_JOB_ID" > "$LOGS/slot_release.json" 2>&1 \
      || echo "WARNING: slot release failed; the next job reclaims the slot once Slurm shows this job ended" >&2
  fi
}
trap release_slot EXIT
# shellcheck disable=SC2086
"$FIT_PY" "$H" slot acquire --slots-dir "$SLOTDIR" --n "$SLOTS" --job "$SLURM_JOB_ID" --tasks $TASKS > "$LOGS/slot.json" \
  || { cat "$LOGS/slot.json"; echo "FATAL: concurrency guard refused (<= $SLOTS production GPU jobs, disjoint tasks)" >&2; exit 7; }
SLOT_HELD=1
log "slot: $(tr -d '\n ' < "$LOGS/slot.json" | cut -c1-200)"

# ---- 5. inputs --------------------------------------------------------------------------------------------------
MANIFEST="$REPO/$MANIFEST_REL"; TAGS="$REPO/$TAGS_REL"
# shellcheck disable=SC2086
"$FIT_PY" "$H" inputs --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" --tags "$TAGS" --tags-sha256 "$TAGS_SHA" \
  --experiments $EXPERIMENTS --tasks $TASKS > "$LOGS/inputs.json" || { cat "$LOGS/inputs.json"; echo "FATAL: inputs refused" >&2; exit 8; }
"$FIT_PY" "$REPO/scripts/fingerprint_prepped.py" --dir "$PREPPED_DIR" --compare "$REPO/docs/prepped_fingerprints_scib.json" \
  > "$LOGS/prepped_fingerprints.txt" 2>&1 || { tail -20 "$LOGS/prepped_fingerprints.txt"; echo "FATAL: prepped files differ from the reference fingerprints" >&2; exit 8; }
log "inputs OK: $(grep -o '"n_tags": [0-9]*' "$LOGS/inputs.json"); prepped fingerprints OK"

# ---- 6. claims of earlier jobs ----------------------------------------------------------------------------------
"$FIT_PY" "$H" claims --out-dir "$OUT" --tags "$TAGS" --job "$SLURM_JOB_ID" > "$LOGS/claims.json" \
  || { cat "$LOGS/claims.json"; echo "FATAL: tags claimed by a live (or unidentifiable) job" >&2; exit 7; }
CLEAR_FLAGS=$("$FIT_PY" -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["flags"]))' "$LOGS/claims.json")

# ---- 7. runner: dry run, then the run with a stop guard ----------------------------------------------------------
# shellcheck disable=SC2206
RUN=("$FIT_PY" "$REPO/scripts/run_stage.py" --no-score --manifest "$MANIFEST" --experiments $EXPERIMENTS --tasks $TASKS
     --tags-file "$TAGS" --out-dir "$OUT" --prepped-dir "$PREPPED_DIR" --fit-python "$FIT_PY" --wcd-src "$REPO/src"
     --expect-device L40S --fit-lanes "$FIT_LANES" --fit-threads 1 --fit-timeout-s "$FIT_TIMEOUT_S" --max-attempts "$MAX_ATTEMPTS")
KEYARGS=(--out-dir "$OUT" --tags "$TAGS" --tags-sha256 "$TAGS_SHA")
if [ -z "$CLEAR_FLAGS" ]; then
  "${RUN[@]}" --dry-run > "$LOGS/dryrun.log" 2>&1 || { cat "$LOGS/dryrun.log"; echo "FATAL: runner dry run refused" >&2; exit 4; }
  "$FIT_PY" "$H" stagekey --log "$LOGS/dryrun.log" "${KEYARGS[@]}" > "$LOGS/stage_key.dryrun.json" \
    || { cat "$LOGS/stage_key.dryrun.json"; echo "FATAL: the runner's dry-run output breaks the stage-key contract" >&2; exit 11; }
  log "dry run: $(grep '\[stage\]' "$LOGS/dryrun.log" | tail -1)"
elif [ "$DRY_RUN" = 1 ]; then
  echo "FATAL: claims of ended jobs on these tags ($CLEAR_FLAGS, see $LOGS/claims.json): the runner's dry run refuses them and only a real run clears them" >&2
  exit 7
else
  log "dry run skipped: claims of ended jobs present, the run clears them ($CLEAR_FLAGS); see $LOGS/claims.json"
fi
FACTS=(--fact slurm_job_id="$SLURM_JOB_ID" --fact host="$(hostname)" --fact git_sha="$HEAD_SHA" --fact stage="$STAGE"
       --fact tasks="$TASKS" --fact tags_sha256="$TAGS_SHA" --fact manifest_sha256="$MANIFEST_SHA"
       --fact gpu="$(head -1 "$LOGS/nvidia_smi.txt")")
if [ "$DRY_RUN" = 1 ]; then
  "$FIT_PY" "$H" summary --mode dry-run --stage-key-json "$LOGS/stage_key.dryrun.json" --rc 0 "${FACTS[@]}" \
    --out "$LOGS/job_summary.json" > /dev/null || { cat "$LOGS/job_summary.json"; exit 11; }
  log "dry run only: nothing fitted or packed; plan in $LOGS/job_summary.json"
  exit 0
fi
DEADLINE_ARGS=(--margin-s "$GUARD_MARGIN_S" --min-run-s "$MIN_RUN_S"); [ -n "$TIME_LEFT_S" ] && DEADLINE_ARGS+=(--time-left-s "$TIME_LEFT_S")
"$FIT_PY" "$H" deadline "${DEADLINE_ARGS[@]}" > "$LOGS/deadline.json" || { cat "$LOGS/deadline.json"; echo "FATAL: stop guard cannot be set" >&2; exit 9; }
DEADLINE=$("$FIT_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["deadline"])' "$LOGS/deadline.json")
log "stop guard: SIGTERM to the runner at $(date -d @"$DEADLINE" -Is) (Slurm end - ${GUARD_MARGIN_S}s)"

STOPPED_BY= RUNNER=
on_signal() {
  STOPPED_BY="${STOPPED_BY:-signal_$1}"
  if [ -n "$RUNNER" ] && kill -0 "$RUNNER" 2>/dev/null; then kill -TERM "$RUNNER" 2>/dev/null || log "runner already exited"; fi
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
# shellcheck disable=SC2086
"${RUN[@]}" $CLEAR_FLAGS > "$LOGS/run_stage.log" 2>&1 &
RUNNER=$!
TERM_AT=
while kill -0 "$RUNNER" 2>/dev/null; do
  NOW=$(date +%s)
  if [ -z "$TERM_AT" ] && { [ "$NOW" -ge "$DEADLINE" ] || [ -n "$STOPPED_BY" ]; }; then
    STOPPED_BY="${STOPPED_BY:-deadline_guard}"; TERM_AT=$NOW
    log "stopping the runner ($STOPPED_BY)"; kill -TERM "$RUNNER" 2>/dev/null || log "runner already exited"
  fi
  if [ -n "$TERM_AT" ] && [ $((NOW - TERM_AT)) -gt "$STOP_WAIT_S" ]; then
    log "runner still alive ${STOP_WAIT_S}s after SIGTERM: SIGKILL"; kill -KILL "$RUNNER" 2>/dev/null || log "runner already exited"
  fi
  sleep "$POLL_S" & wait "$!" || WAKE_RC=$?        # >128: a trapped signal woke the loop (handled above)
done
RC=0; wait "$RUNNER" || RC=$?
log "runner exit $RC${STOPPED_BY:+ (stopped by $STOPPED_BY)}: $(grep -E '\[gate\]|STOPPED|interrupted' "$LOGS/run_stage.log" | tail -2 | tr '\n' ' ')"

# ---- 8. summary + harvest -------------------------------------------------------------------------------------
# The key whose ledger is harvested comes from a runner invocation's own '[stage]' line: the run's, or after an
# interrupted / killed / crashed run the --report-only pass's, which rebuilds the views from the per-tag files (it
# must name the dry run's key). A run refused in its preflight (exit 4) prints none and wrote no ledger, so no ledger
# is attributed to this job; a run stopped before it planned (128+n, 1) may print none either.
EXPECT=(); [ -s "$LOGS/stage_key.dryrun.json" ] && EXPECT=(--expect-key-json "$LOGS/stage_key.dryrun.json")
MISSING=(); case "$RC" in 0|5|6) ;; *) MISSING=(--allow-missing);; esac
KEY_RC=0
"$FIT_PY" "$H" stagekey --log "$LOGS/run_stage.log" "${KEYARGS[@]}" "${EXPECT[@]}" "${MISSING[@]}" > "$LOGS/stage_key.run.json" || KEY_RC=$?
[ "$KEY_RC" = 0 ] || log "WARNING: the run's output breaks the stage-key contract: $(tr -d '\n' < "$LOGS/stage_key.run.json" | cut -c1-400)"
cp "$LOGS/stage_key.run.json" "$LOGS/stage_key.json"
REPORT_ARGS=()
case "$RC" in
  0|4|5|6) ;;                                       # the runner wrote final views itself (or ran nothing)
  *) REPORT_RC=0
     "${RUN[@]}" --report-only > "$LOGS/report_only.log" 2>&1 || REPORT_RC=$?
     REPORT_ARGS=(--report-only-rc "$REPORT_RC" --also-stage-json "$LOGS/stage_key.run.json")
     REXP=("${EXPECT[@]}")
     if [ ${#REXP[@]} -eq 0 ] && [ "$KEY_RC" = 0 ] && grep -q '"stage_key": "' "$LOGS/stage_key.run.json"; then
       REXP=(--expect-key-json "$LOGS/stage_key.run.json")
     fi
     if "$FIT_PY" "$H" stagekey --log "$LOGS/report_only.log" "${KEYARGS[@]}" "${REXP[@]}" > "$LOGS/stage_key.report_only.json"; then
       cp "$LOGS/stage_key.report_only.json" "$LOGS/stage_key.json"
     else
       KEY_RC=11; log "WARNING: the --report-only output breaks the stage-key contract"
     fi
     log "views rebuilt by run_stage.py --report-only (exit $REPORT_RC): $(grep -E '\[gate\]' "$LOGS/report_only.log" | tail -1)";;
esac
SUM_RC=0
"$FIT_PY" "$H" summary --mode run --stage-key-json "$LOGS/stage_key.json" --tags "$TAGS" --tags-sha256 "$TAGS_SHA" \
  --manifest-sha256 "$MANIFEST_SHA" --head-sha "$HEAD_SHA" --rc "$RC" --stopped-by "$STOPPED_BY" "${REPORT_ARGS[@]}" \
  "${FACTS[@]}" --out "$LOGS/job_summary.json" > /dev/null || SUM_RC=$?
[ "$SUM_RC" = 0 ] || log "WARNING: summary found problems (exit $SUM_RC): $(grep -A3 '"problems"' "$LOGS/job_summary.json" | tr -d '\n' | cut -c1-400)"
if ! "$FIT_PY" "$H" pack --out-dir "$OUT" --tags "$TAGS" --stage-key-json "$LOGS/stage_key.json" \
     --job-dir "$JOBDIR" --dest "$HARVEST" --stage "$STAGE" --job "$SLURM_JOB_ID" > "$JOBDIR/harvest.json"; then
  echo "FATAL: harvest pack failed (runner exit $RC)" >&2; exit 10
fi
log "harvest: $HARVEST ($(grep -c '"name"' "$JOBDIR/harvest.json") parts; $(grep -o '"n_latents": [0-9]*' "$JOBDIR/harvest.json"))"
ls -l "$HARVEST"
if [ "$RC" = 0 ] && { [ "$KEY_RC" != 0 ] || [ "$SUM_RC" != 0 ]; }; then
  echo "FATAL: the runner's gate passed but its output breaks the stage-key/ledger contract (see $LOGS/stage_key.json, $LOGS/job_summary.json)" >&2
  exit 11
fi
exit "$RC"
