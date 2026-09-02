#!/bin/bash
# PARALLEL CPU SCORER for BASELINE latents (tag <dataset>_XZB_<method>_s<seed>).
#
# The adversarial scorer (score_parallel.sh) iterates the manifest, which contains only the
# adversarial/none tags. Baselines are NOT in the manifest, so they need this separate pass. It
# globs every *_XZB_*.npz in the embed dir, parses (dataset, method, seed) from the tag, and scores
# each through the IDENTICAL score_final_config.py (same full_metric_suite, same 0.4/0.6 scIB), into
# its own per-tag row CSV under the SAME rows dir — so the merge in analyze picks them up alongside
# the adversarial rows. Baselines carry lam=0 and dec=base. Resume-safe (skip existing row CSV).
set -u
cd "$(dirname "$0")/.."
: "${SCVI_PY:?}" "${WCD_EMBED_OUT:?}"
export PYTHONPATH="$(pwd)/src" KMP_AFFINITY=disabled OMP_NUM_THREADS="${OMP:-2}" \
       MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore NUMBA_NUM_THREADS="${OMP:-2}"
EMB="$WCD_EMBED_OUT"
ROWDIR="${SCORE_ROWDIR:-results/final/rows}"
NWORK="${SCORE_NWORK:-12}"
mkdir -p "$ROWDIR"

score_one() {  # npz
  local npz=$1
  local tag; tag=$(basename "$npz" .npz)
  local row="$ROWDIR/${tag}.csv"
  [ -s "$row" ] && return                         # already scored (resume-safe)
  # tag = <dataset>_XZB_<method>_s<seed>
  local ds method seed
  ds=${tag%%_XZB_*}
  local rest=${tag#*_XZB_}                          # <method>_s<seed>
  method=${rest%_s*}
  seed=${rest##*_s}
  NPZ="$npz" TAG="$tag" DATASET="$ds" ADV="$method" LAM="0" DEC="base" SEED="$seed" \
    OUT_CSV="$row" "$SCVI_PY" scripts/score_final_config.py > "logs/wave/score_${tag}.log" 2>&1 \
    && echo "[scored] $tag" || { echo "[score-FAIL] $tag"; rm -f "$row"; }
}
export -f score_one; export EMB ROWDIR SCVI_PY

ls "$EMB"/*_XZB_*.npz 2>/dev/null | xargs -P "$NWORK" -I {} bash -c 'score_one "{}"'
echo "=== baseline rows scored: $(ls $ROWDIR/*_XZB_*.csv 2>/dev/null | wc -l) ==="
