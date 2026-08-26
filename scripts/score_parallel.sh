#!/bin/bash
# PARALLEL CPU SCORER — score completed latents N-wide, race-free.
#
# WHY: scoring (kNN graph + Leiden + LISI + silhouette) is the dominant per-config cost and it runs
# on CPU, independent per config. Running it in parallel on a CPU node lets the metric suite keep up
# with the GPU fit stream. Scoring needs NO GPU (silhouette has a CPU fallback), so this can run on a
# CPU-only allocation (e.g. scavenge) without stealing cores from the fit pilot's GPU node.
#
# RACE-SAFETY: score_final_config.py appends to ONE shared CSV, and a scored row (20+ metric cols)
# EXCEEDS PIPE_BUF, so concurrent appends WOULD interleave/corrupt. So here each config writes to its
# OWN per-tag CSV (OUT_CSV=$ROWDIR/<tag>.csv) — no shared file, no race — and we merge at the end.
# Resume-safe: skip a config whose per-tag row CSV already exists.
set -u
cd "$(dirname "$0")/.."
: "${SCVI_PY:?}" "${WCD_EMBED_OUT:?}"
export PYTHONPATH="$(pwd)/src" KMP_AFFINITY=disabled OMP_NUM_THREADS="${OMP:-2}" \
       MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore NUMBA_NUM_THREADS="${OMP:-2}"
MANIFEST="${1:-scripts/scvi_final_manifest.tsv}"
EMB="$WCD_EMBED_OUT"
ROWDIR="${SCORE_ROWDIR:-results/final/rows}"
MERGED="${SCORE_MERGED:-results/final/scored_final.csv}"
NWORK="${SCORE_NWORK:-12}"
mkdir -p "$ROWDIR" "$(dirname "$MERGED")"

score_one() {  # tag dataset adv lam dec seed
  local tag=$1 ds=$2 adv=$3 lam=$4 dec=$5 seed=$6
  local npz="$EMB/${tag}.npz" row="$ROWDIR/${tag}.csv"
  [ -s "$npz" ]  || { return; }          # latent not produced yet (skip; a later pass gets it)
  [ -s "$row" ]  && { return; }          # already scored (resume-safe)
  NPZ="$npz" TAG="$tag" DATASET="$ds" ADV="$adv" LAM="$lam" DEC="$dec" SEED="$seed" \
    OUT_CSV="$row" "$SCVI_PY" scripts/score_final_config.py > "logs/wave/score_${tag}.log" 2>&1 \
    && echo "[scored] $tag" || { echo "[score-FAIL] $tag"; rm -f "$row"; }
}
export -f score_one; export EMB ROWDIR SCVI_PY

# feed every manifest row that has a completed latent into the scorer pool
grep -vE '^#|^model' "$MANIFEST" | while IFS=$'\t' read -r model cond adv lam di seed ds dec tag; do
  [ -z "$tag" ] && continue
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$ds" "$adv" "$lam" "$dec" "$seed"
done | xargs -P "$NWORK" -I {} bash -c '
    IFS=$'"'"'\t'"'"' read -r t ds a l dec s <<< "{}"; score_one "$t" "$ds" "$a" "$l" "$dec" "$s"'

echo "=== merge per-tag rows -> $MERGED ==="
"$SCVI_PY" - <<PY
import glob, pandas as pd, os
rows = [pd.read_csv(f) for f in glob.glob("$ROWDIR/*.csv") if os.path.getsize(f) > 0]
if rows:
    df = pd.concat(rows, ignore_index=True).drop_duplicates("tag")
    df.to_csv("$MERGED", index=False)
    print(f"merged {len(df)} scored configs -> $MERGED")
else:
    print("no scored rows yet")
PY
