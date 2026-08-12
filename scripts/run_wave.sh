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
PY=${WCD_PYTHON:-/home/kendall/.claude-science/conda/envs/wcd-gpu/bin/python}
R=${WCD_DATA:-$(pwd)/../data}
# EMBEDDINGS ARE ON BY DEFAULT. Metrics-only CSVs cannot be re-analysed: adding any new
# embedding-derived metric (kBET, PAGA, probes) to a finished wave otherwise costs a full
# retraining run. This project paid that bill twice. Override only for smoke runs.
EMB=${WCD_EMBED_OUT:-$(pwd)/../embeddings}
export PY R EMB
mkdir -p results/wave logs/wave "$EMB"

MANIFEST=${1:-scripts/wave_manifest.tsv}
WORKERS=${WCD_WORKERS:-6}

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
CHECK_PY=${WCD_CHECK_PYTHON:-/home/kendall/.claude-science/conda/envs/wcd-gpu/bin/python}
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

n_done=$(ls results/wave/*.done 2>/dev/null | wc -l)
n_tot=$(( $(wc -l < "$MANIFEST") - 1 ))
echo "wave complete: ${n_done}/${n_tot} shards done, $(ls results/wave/*.csv 2>/dev/null | wc -l) CSVs"
[ "$n_done" -eq "$n_tot" ] || { echo "INCOMPLETE -- re-run the same command to resume the rest" >&2; exit 1; }
