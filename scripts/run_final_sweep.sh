#!/bin/bash
# Reproducible final benchmark sweep driver.
# Reads scripts/scvi_final_manifest.tsv (model cond adv lam di seed dataset dec tag).
# Front-loads cheap arms (none + discriminator, then critic-free, then critics) so the
# discriminator-vs-baselines headline lands first. 6-wide concurrency at OMP=2 (measured optimum
# on 12 cores). Latents to DURABLE storage (ephemeral guard below). Resume-safe (skips existing npz).
set -u
WS="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$(dirname "$0")/.."
ES="${SCVI_PY:-/home/kendall/.claude-science/conda/envs/scvi-api/bin/python}"
WCD_SRC="$(pwd)/src"
EMB="${WCD_EMBED_OUT:-/home/kendall/experiment_data/wasserstein-critic-deconfounding/embeddings_final}"
MANIFEST="${1:-scripts/scvi_final_manifest.tsv}"
NWORK="${NWORK:-6}"
export KMP_AFFINITY=disabled OMP_NUM_THREADS="${OMP:-2}" NUMBA_NUM_THREADS="${OMP:-2}" \
       MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore

# --- EPHEMERAL LATENTS GUARD (load-bearing: 3 prior latent losses in this project) ---
mkdir -p "$EMB"
case "$EMB" in */workspaces/*|/tmp/*|*/scratch/*) echo "FATAL: EMB=$EMB is EPHEMERAL"; exit 2;; esac
fstype=$(df -PT "$EMB" 2>/dev/null | tail -1 | awk '{print $2}')
case "$fstype" in tmpfs|ramfs) echo "FATAL: EMB=$EMB is $fstype (RAM-backed)"; exit 2;; esac
echo "[guard] latents -> $EMB (fstype=$fstype) OK"
mkdir -p logs/wave

run_one() {
  local model=$1 cond=$2 arm=$3 lam=$4 di=$5 seed=$6 ds=$7 dec=$8 tag=$9
  local out="$EMB/${tag}.npz"
  [ -s "$out" ] && { echo "[skip] $tag"; return; }
  SCVI_DS="$ds" SCVI_MODEL="$model" ADV="$arm" DCOEF="$lam" DISC_ITER="$di" COND="$cond" \
    SEED="$seed" MAXEP=239 BATCH=512 OUT="$out" WCD_SRC="$WCD_SRC" \
    "$ES" scripts/scvi_adv_fit.py > "logs/wave/${tag}.log" 2>&1 \
    && echo "[ok] $tag" || echo "[FAIL] $tag rc=$?"
}
export -f run_one; export ES EMB WCD_SRC

# --- front-load order: none (clean control, cheap) -> discriminator -> mmd/sinkhorn -> critics ---
order_key() {  # lower = runs earlier
  case "$1" in none) echo 0;; discriminator) echo 1;; mmd|sinkhorn) echo 2;; *) echo 3;; esac
}
# emit "priority<TAB>row" then sort by priority, strip, feed to xargs
grep -vE '^#|^model' "$MANIFEST" | while IFS=$'\t' read -r model cond adv lam di seed ds dec tag; do
  [ -z "$tag" ] && continue
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(order_key "$adv")" \
    "$model" "$cond" "$adv" "$lam" "$di" "$seed" "$ds" "$dec" "$tag"
done | sort -s -k1,1n | cut -f2- \
  | xargs -P "$NWORK" -I {} bash -c 'IFS=$'"'"'\t'"'"' read -r m c a l d s ds dec t <<< "{}"; run_one "$m" "$c" "$a" "$l" "$d" "$s" "$ds" "$dec" "$t"'

echo "FINAL SWEEP DRIVER DONE: $(ls "$EMB"/*_XZ_*.npz 2>/dev/null | wc -l) npz of $(grep -vcE '^#|^model' "$MANIFEST")"
