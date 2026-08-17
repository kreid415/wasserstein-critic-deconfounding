#!/bin/bash
# Sharded launcher for the full experiment matrix.
#
# WHY SHARDED, NOT ONE-DATASET-PER-WORKER: at one dataset per worker the machine idles a
# third of its capacity -- the wall-clock is set by the slowest dataset. Sharding at
# (dataset, experiment, head) and packing longest-first balanced six workers to ~98% and
# took the light-6 wave from ~23 h to ~15.4 h for the same worker-hours.
#
# WHY THESE SHARD BOUNDARIES: each shard is ONE process looping its own config grid, so
# the ~81 s process startup is paid once per shard, not once per config (924 processes
# would be ~21 h of pure startup). Only splits the harnesses support natively are used.
#
# EPOCHS: 500 is a CEILING, not a target -- early stopping decides. Do NOT set
# --epochs 150: that silently truncated pancreas, lung and sim2 on the critic head,
# where early stopping never fired.
set -uo pipefail
cd "$(dirname "$0")/.."

export KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
       MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore
export PYTHONPATH=$(pwd)/src
# DEFAULT ENVIRONMENT IS wcd-kbet, NOT wcd-gpu. kBET is a scIB BATCH-CORRECTION metric
# that wraps the R kBET package through rpy2; wcd-gpu has neither, so a wave run there
# silently records kbet=NaN for every row -- which is exactly how the kbet column ended up
# empty across all 918 rows of the previous wave. wcd-kbet is a fork of wcd-gpu with
# r-base + rpy2 + anndata2ri==1.3.2 + the R kBET package, and it was verified to reproduce
# wcd-gpu BIT-IDENTICALLY on all 11 other metrics for the same config, so results from the
# two environments are directly comparable.
#
# rpy2 needs R_HOME or it dies at import with "openrlib.R_HOME cannot be None", and the
# R kBET package lives in a workspace library because conda's R library is read-only to
# this sandbox. Both are derived from the interpreter path so overriding WCD_PYTHON alone
# keeps them consistent; override WCD_R_HOME / WCD_R_LIBS only for a non-standard layout.
WCD_ENV_DEFAULT=/home/kendall/.claude-science/conda/envs/wcd-kbet
PY=${WCD_PYTHON:-$WCD_ENV_DEFAULT/bin/python}
PY_PREFIX=$(cd "$(dirname "$PY")/.." && pwd)
export R_HOME=${WCD_R_HOME:-$PY_PREFIX/lib/R}
# R_LIBS AND THE DATA ROOT ARE FORCED ABSOLUTE. `$(pwd)/../x` is textually absolute but
# still contains a `..` that R and the shards re-resolve against THEIR cwd; a caller who
# exports a relative WCD_R_LIBS (e.g. ./Rlib_kbet) gets a libPath that only works when cwd
# happens to be the workspace root. Observed live: .libPaths() reported "./Rlib_kbet",
# which resolves for the driver and silently would not for a shard run elsewhere.
# `cd -- "$dir" && pwd` collapses the `..` to a real path, and -P avoids symlink surprises.
#   A bare `cd -P -- "$dir"` is not enough: a relative input is resolved against the
#   SUBSHELL's cwd (the repo root, since the launcher runs from there), so ./Rlib_kbet --
#   which lives one level up in the workspace -- silently stays relative and R never finds
#   it. Resolve relative inputs against the repo's PARENT, which is where the sibling
#   Rlib_kbet/data/embeddings directories actually live, and leave absolute inputs alone.
_abspath() {
  case "$1" in
    /*) ( cd -P -- "$1" 2>/dev/null && pwd ) || echo "$1" ;;
    *)  ( cd -P -- "$(pwd)/../${1#./}" 2>/dev/null && pwd ) || echo "$1" ;;
  esac
}
export R_LIBS=$(_abspath "${WCD_R_LIBS:-$(pwd)/../Rlib_kbet}")
R=$(_abspath "${WCD_DATA:-$(pwd)/../data}")
# EMBEDDINGS ARE ON BY DEFAULT. Metrics-only CSVs cannot be re-analysed: adding any new
# embedding-derived metric (kBET, PAGA, probes) to a finished wave otherwise costs a full
# retraining run. This project paid that bill twice. Override only for smoke runs.
EMB=${WCD_EMBED_OUT:-$(pwd)/../embeddings}
export PY R EMB
mkdir -p results/wave logs/wave "$EMB"

# WHY THIS WARNING EXISTS -- THE THIRD INSTANCE OF THE SAME LOSS.
#   1. A wave ran with no --embed-out at all: PAGA could not be added post-hoc.
#   2. A wave ran with the flag omitted from the manifest: kBET could not be backfilled.
#   3. A wave ran WITH the flag, pointed at the session workspace. The files were written
#      correctly and then the workspace was swept on an idle timeout. All ~990 latents
#      (~18 GB) were lost; only the metrics CSVs survived, because those had been promoted
#      to durable artifact storage and the latents never had been.
# Passing the flag is NOT the same as persisting the data. A path under a scratch or
# session-workspace root satisfies the flag and still evaporates, so warn loudly and name
# the harvest step rather than letting a 26-hour run end with nothing durable.
# tmpfs is checked as well as the path patterns: a RAM-backed mount looks like ordinary
# disk to `df -h` and to any name-based rule, so a path that passes the pattern test can
# still be volatile. This was caught live -- /home/kendall is tmpfs on this box, so a
# "durable" directory created there would have satisfied the pattern check and still been
# lost. Name-based heuristics are not sufficient; ask the filesystem.
EMB_FSTYPE=$(df -PT "$EMB" 2>/dev/null | awk 'NR==2{print $2}')
case "$EMB:$EMB_FSTYPE" in
  */workspaces/*:*|/tmp/*:*|*/scratch/*:*|*:tmpfs|*:ramfs)
    cat >&2 <<WARN

  ================================ EPHEMERAL LATENTS ================================
  --embed-out resolves to:  $EMB
  That path is under a session-workspace / scratch root, which is swept on idle. Latents
  written there do NOT survive the run. This has already cost this project one full wave.

  Either point WCD_EMBED_OUT at durable storage:
      WCD_EMBED_OUT=/durable/path bash scripts/run_wave.sh <manifest>
  or harvest them the moment the wave finishes:
      tar -czf latents.tar.gz -C "$EMB" . && <promote latents.tar.gz to artifact storage>

  Metrics CSVs are not enough: adding any new embedding-derived metric without the
  latents costs a full retraining wave (~186 worker-hours for the light scope).
  ===================================================================================

WARN
    ;;
esac

# PREFLIGHT: fail BEFORE burning days of GPU time if the metric stack is not importable.
# WHY a hard gate rather than a warning: the failure this guards against is silent by
# construction -- full_metric_suite catches kBET's ImportError and records NaN, so the
# wave completes, every CSV looks well-formed, and the gap only surfaces when someone
# asks for the batch-correction axis. Skip only for a deliberate no-kBET run.
if [ -z "${WCD_SKIP_PREFLIGHT:-}" ]; then
  if ! "$PY" - <<'PYEOF'
import sys
try:
    import anndata2ri, rpy2  # noqa: F401
    from rpy2.robjects.packages import importr
    importr("kBET")
except Exception as exc:
    print(f"PREFLIGHT FAIL: kBET stack unavailable -- {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
print("preflight: kBET stack OK")
PYEOF
  then
    echo "" >&2
    echo "The wave would record kbet=NaN for every config. Either fix the environment or" >&2
    echo "re-run with WCD_SKIP_PREFLIGHT=1 to proceed deliberately without kBET." >&2
    exit 1
  fi
fi

MANIFEST=${1:-scripts/wave_manifest.tsv}
WORKERS=${WCD_WORKERS:-6}

# EXPORT IT: _vram_safe_chunk in evaluation.py divides its GPU chunk budget by
# WCD_WORKERS, because the metric suite's kNN claims 33% of FREE VRAM and six lanes each
# claiming a third oversubscribes an 8 GiB card (measured: 1870 MiB per config once the
# suite runs, and the wave lost 31 rows to this). Without the export the shards do not
# inherit it and the divisor silently falls back to 1 -- i.e. the unsafe behaviour, with no
# error. Verified by reading /proc/<shard>/environ: unexported, it is absent there.
export WCD_WORKERS="$WORKERS"

# TAB-delimited, not CSV: the command field is free text and a flag with a
# comma-separated value (--override a,b) would split mid-command under IFS=','.
#
# Manifest columns: phase  worker  tag  est_hours  cmd
#   phase=parallel -> run across $WORKERS lanes
#   phase=serial   -> run ONE AT A TIME after all parallel shards finish
#
# WHY A SERIAL PHASE: E10 sweeps batch_size in {1024, 4096} and the 4096 arm OOMs
# whenever ~6 workers share the 8 GiB card. Observed three times in one wave
# (immune/critic 9 of 12 configs lost, sim1/critic 9 of 12, lung/DISC 4 of 12) -- it hits
# BOTH heads, so scoping the mitigation to the critic is not enough. Running those shards
# alone is what makes them fit.

# shard_ok <tag> -- a shard counts as DONE only if its CSV exists and contains NO failed
# rows. WHY: the harnesses catch per-config exceptions, record them as rows with a
# populated `failed` column, and still EXIT 0. A naive exit-code check therefore writes a
# .done marker for a shard that lost most of its configs to OOM, and --resume then skips
# it permanently -- a silent, unrecoverable hole in the results.
# WHY A SEPARATE INTERPRETER FOR CHECKING: $PY is overridable via WCD_PYTHON so the
# launcher can be dry-run against a stub. The stub must NOT also be what validates
# results -- during a stub run every check would trivially pass and the launcher's
# failure handling would be untested. WCD_CHECK_PYTHON stays a real interpreter.
CHECK_PY=${WCD_CHECK_PYTHON:-$WCD_ENV_DEFAULT/bin/python}
export CHECK_PY

shard_ok() {
  local tag="$1" csv="results/wave/${tag}.csv"
  # E5 (run_biology.py) writes a DIRECTORY of per-dataset CSVs via --outdir rather than a
  # single --out file, so its completion is judged on its summary file instead. Without
  # this the launcher never marks E5 done and re-runs it on every pass forever.
  if [ ! -f "$csv" ] && [ -d "results/wave/E5_${tag%_E5}" ]; then
    csv="results/wave/E5_${tag%_E5}/E5_summary_${tag%_E5}.csv"
  fi
  [ -f "$csv" ] || return 1
  "$CHECK_PY" - "$csv" <<'PYEOF'
import sys
import pandas as pd
try:
    d = pd.read_csv(sys.argv[1])
except Exception:
    sys.exit(1)
if len(d) == 0:
    sys.exit(1)
if "failed" in d.columns and d["failed"].notna().any():
    sys.exit(1)
sys.exit(0)
PYEOF
}

run_shard() {
  local lane="$1" tag="$2" est="$3" cmd="$4"
  local log="logs/wave/${tag}.log"
  if [ -f "results/wave/${tag}.done" ]; then
    echo "[$lane] skip $tag (done)"; return 0
  fi
  echo "[$lane] start $tag (est ${est} h)"
  # --resume makes a re-run of an interrupted shard skip already-completed configs, so a
  # failed shard costs only its missing configs on the next pass.
  eval "$cmd" > "$log" 2>&1
  local rc=$?
  if shard_ok "$tag"; then
    touch "results/wave/${tag}.done"
    echo "[$lane] ok    $tag"
  else
    # Deliberately NO .done marker: leaving it unmarked is what lets a later pass retry.
    echo "[$lane] FAIL  $tag (rc=$rc, or failed rows present) -- see $log" >&2
  fi
}
export -f shard_ok run_shard

# ---- phase 1: parallel lanes ----
for w in $(seq 1 "$WORKERS"); do
  (
    while IFS=$'\t' read -r phase worker tag est cmd; do
      [ "$phase" = "phase" ] && continue
      [ "$phase" != "parallel" ] && continue
      [ "$worker" != "$w" ] && continue
      run_shard "w$w" "$tag" "$est" "$cmd"
    done < "$MANIFEST"
  ) &
done
wait
echo "--- parallel phase complete ---"

# ---- phase 2: serial (VRAM-heavy shards get the card to themselves) ----
while IFS=$'\t' read -r phase worker tag est cmd; do
  [ "$phase" = "phase" ] && continue
  [ "$phase" != "serial" ] && continue
  run_shard "serial" "$tag" "$est" "$cmd"
done < "$MANIFEST"

# COUNT ONLY THIS MANIFEST'S SHARDS. `ls results/wave/*.done | wc -l` counts every marker
# in the directory, including those from previous manifests, so a small follow-up manifest
# printed nonsense like "257/36 shards done ... INCOMPLETE" and exited 1 while all 36 of
# its shards had in fact succeeded. A completion signal that is wrong in the SAFE-looking
# direction ("incomplete" when done) wastes a re-run; wrong the other way hides real gaps.
n_tot=$(( $(wc -l < "$MANIFEST") - 1 ))
n_done=$(awk -F'\t' 'NR>1 && $3!="" {print $3}' "$MANIFEST" \
         | while read -r t; do [ -f "results/wave/${t}.done" ] && echo x; done | wc -l)
echo "wave complete: ${n_done}/${n_tot} shards done, $(ls results/wave/*.csv 2>/dev/null | wc -l) CSVs"
[ "$n_done" -eq "$n_tot" ] || { echo "INCOMPLETE -- re-run the same command to resume the rest" >&2; exit 1; }
