#!/bin/bash
# JHPCE production fit job: one pinned L40S, fit only (scripts/run_stage.py --no-score); every latent is scored on the
# local host after harvest (cross-host gate G4). Requirements: docs/jhpce code check CR-03 ("JHPCE production job script:
# required contents"), CONSTRAINTS.md SI-17 / SI-29 / SI-39. Submission needs the user's explicit go (SI-29); the
# submit command is printed by cluster/jhpce/prod_command.py, which also stages the repo bundle and clones the commit.
#
# Steps (any failure before the runner aborts the job; nothing is fitted):
#   1. environment: inside Slurm, scratch-first (OUT, logs, TMPDIR, caches on fastscratch, nothing in $HOME), verified env
#   2. repository: HEAD == --expected-sha, clean tree
#   3. concurrency: one of 2 slots (<= 2 production GPU jobs at once), tasks disjoint from the other live job
#   4. inputs: manifest + tags-file SHA-256, tags in the manifest, tags' experiments/tasks == the job's; prepped files
#      against docs/prepped_fingerprints_scib.json
#   5. node witness + tests: nvidia-smi = exactly one L40S; env_versions.py --kind fit == expected_fit_versions.json;
#      WCD_SRC=src pytest -q tests/scvi at this commit
#   6. claims of earlier jobs on these tags: cleared only if their Slurm job is terminal (sacct); otherwise abort
#   7. runner --dry-run (logged), then the run in the background with a stop guard (SIGTERM at Slurm end - margin, so
#      the runner stops lanes, releases claims and writes its views while the job is still alive)
#   8. job summary from the ledger; harvest of this job's tags packed on fastscratch (parts <= 250 MiB + SHA-256)
# Resume-safe: rerun the same command; fitted rows are kept (runner preflight), interrupted rows rerun.
# Exit: 0 gate passed | 2 node witness/tests | 3 repository | 7 concurrency/claims | 8 inputs | 9 environment |
#       10 harvest pack failed | runner codes 4 preflight refused, 5 infrastructure stop, 6 incomplete, 128+n signal.
set -euo pipefail

usage() { echo "usage: $0 --expected-sha SHA --stage NAME --tags-file REL --tags-sha256 HEX --manifest REL --manifest-sha256 HEX --experiments 'X1 X13' --tasks 'a b' [--fit-lanes 8] [--fit-timeout-s 7200] [--max-attempts 2] [--guard-margin-s 2700] [--time-left-s N] [--min-run-s 1800] [--slots 2]" >&2; exit 9; }
EXPECTED_SHA= STAGE= TAGS_REL= TAGS_SHA= MANIFEST_REL= MANIFEST_SHA= EXPERIMENTS= TASKS=
FIT_LANES=8 FIT_TIMEOUT_S=7200 MAX_ATTEMPTS=2 GUARD_MARGIN_S=2700 TIME_LEFT_S= MIN_RUN_S=1800 SLOTS=2 STOP_WAIT_S=240
while [ $# -gt 0 ]; do
  case "$1" in
    --expected-sha) EXPECTED_SHA=$2;; --stage) STAGE=$2;; --tags-file) TAGS_REL=$2;; --tags-sha256) TAGS_SHA=$2;;
    --manifest) MANIFEST_REL=$2;; --manifest-sha256) MANIFEST_SHA=$2;; --experiments) EXPERIMENTS=$2;; --tasks) TASKS=$2;;
    --fit-lanes) FIT_LANES=$2;; --fit-timeout-s) FIT_TIMEOUT_S=$2;; --max-attempts) MAX_ATTEMPTS=$2;;
    --guard-margin-s) GUARD_MARGIN_S=$2;; --time-left-s) TIME_LEFT_S=$2;; --min-run-s) MIN_RUN_S=$2;; --slots) SLOTS=$2;;
    *) echo "unknown argument $1" >&2; usage;;
  esac
  shift 2
done
for v in EXPECTED_SHA STAGE TAGS_REL TAGS_SHA MANIFEST_REL MANIFEST_SHA EXPERIMENTS TASKS; do
  [ -n "${!v}" ] || { echo "FATAL: --${v,,} missing" >&2; usage; }
done
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "FATAL: --expected-sha must be a full 40-hex commit id" >&2; exit 9; }
[[ "$STAGE" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "FATAL: bad --stage $STAGE" >&2; exit 9; }

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

# ---- 3. concurrency ---------------------------------------------------------------------------------------------
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

# ---- 4. inputs --------------------------------------------------------------------------------------------------
MANIFEST="$REPO/$MANIFEST_REL"; TAGS="$REPO/$TAGS_REL"
# shellcheck disable=SC2086
"$FIT_PY" "$H" inputs --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" --tags "$TAGS" --tags-sha256 "$TAGS_SHA" \
  --experiments $EXPERIMENTS --tasks $TASKS > "$LOGS/inputs.json" || { cat "$LOGS/inputs.json"; echo "FATAL: inputs refused" >&2; exit 8; }
"$FIT_PY" "$REPO/scripts/fingerprint_prepped.py" --dir "$PREPPED_DIR" --compare "$REPO/docs/prepped_fingerprints_scib.json" \
  > "$LOGS/prepped_fingerprints.txt" 2>&1 || { tail -20 "$LOGS/prepped_fingerprints.txt"; echo "FATAL: prepped files differ from the reference fingerprints" >&2; exit 8; }
log "inputs OK: $(grep -o '"n_tags": [0-9]*' "$LOGS/inputs.json"); prepped fingerprints OK"

# ---- 5. node witness + tests -----------------------------------------------------------------------------------
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

# ---- 6. claims of earlier jobs ----------------------------------------------------------------------------------
"$FIT_PY" "$H" claims --out-dir "$OUT" --tags "$TAGS" --job "$SLURM_JOB_ID" > "$LOGS/claims.json" \
  || { cat "$LOGS/claims.json"; echo "FATAL: tags claimed by a live (or unidentifiable) job" >&2; exit 7; }
CLEAR_FLAGS=$("$FIT_PY" -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["flags"]))' "$LOGS/claims.json")

# ---- 7. runner: dry run, then the run with a stop guard ----------------------------------------------------------
# shellcheck disable=SC2206
RUN=("$FIT_PY" "$REPO/scripts/run_stage.py" --no-score --manifest "$MANIFEST" --experiments $EXPERIMENTS --tasks $TASKS
     --tags-file "$TAGS" --out-dir "$OUT" --prepped-dir "$PREPPED_DIR" --fit-python "$FIT_PY" --wcd-src "$REPO/src"
     --expect-device L40S --fit-lanes "$FIT_LANES" --fit-threads 1 --fit-timeout-s "$FIT_TIMEOUT_S" --max-attempts "$MAX_ATTEMPTS")
DEFAULT_KEY="$(tr ' ' '\n' <<< "$EXPERIMENTS" | sort | paste -sd+)__$(tr ' ' '\n' <<< "$TASKS" | sort | paste -sd+)"
if [ -z "$CLEAR_FLAGS" ]; then
  "${RUN[@]}" --dry-run > "$LOGS/dryrun.log" 2>&1 || { cat "$LOGS/dryrun.log"; echo "FATAL: runner dry run refused" >&2; exit 4; }
  log "dry run: $(grep '\[stage\]' "$LOGS/dryrun.log" | tail -1)"
else
  log "dry run skipped: claims of ended jobs present, the run clears them ($CLEAR_FLAGS); see $LOGS/claims.json"
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
  sleep 10 & wait "$!" || WAKE_RC=$?        # >128: a trapped signal woke the loop (handled above)
done
RC=0; wait "$RUNNER" || RC=$?
log "runner exit $RC${STOPPED_BY:+ (stopped by $STOPPED_BY)}: $(grep -E '\[gate\]|STOPPED|interrupted' "$LOGS/run_stage.log" | tail -2 | tr '\n' ' ')"

# ---- 8. summary + harvest -------------------------------------------------------------------------------------
"$FIT_PY" "$H" summary --out-dir "$OUT" --runner-log "$LOGS/run_stage.log" --default-key "$DEFAULT_KEY" --rc "$RC" \
  --stopped-by "$STOPPED_BY" --out "$LOGS/job_summary.json" --fact slurm_job_id="$SLURM_JOB_ID" --fact host="$(hostname)" \
  --fact git_sha="$HEAD_SHA" --fact stage="$STAGE" --fact tasks="$TASKS" --fact tags_sha256="$TAGS_SHA" \
  --fact manifest_sha256="$MANIFEST_SHA" --fact gpu="$(head -1 "$LOGS/nvidia_smi.txt")" > /dev/null || log "WARNING: summary failed"
if ! "$FIT_PY" "$H" pack --out-dir "$OUT" --tags "$TAGS" --runner-log "$LOGS/run_stage.log" --default-key "$DEFAULT_KEY" \
     --job-dir "$JOBDIR" --dest "$HARVEST" --stage "$STAGE" --job "$SLURM_JOB_ID" > "$JOBDIR/harvest.json"; then
  echo "FATAL: harvest pack failed (runner exit $RC)" >&2; exit 10
fi
log "harvest: $HARVEST ($(grep -c '"name"' "$JOBDIR/harvest.json") parts; $(grep -o '"n_latents": [0-9]*' "$JOBDIR/harvest.json"))"
ls -l "$HARVEST"
exit "$RC"
