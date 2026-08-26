#!/bin/bash
# JHPCE PILOT/PACKING driver — grab a full GPU node, HOLD it, drain the manifest.
#
# WHY: the JHPCE GPU queue is deep (36+ pending). Submitting one Slurm task per config would
# queue 2000 times. Instead this is ONE job that, once a node is allocated, runs as many configs
# as possible across ALL the node's GPUs until the manifest is drained or wall-time runs low —
# never releasing the node between configs.
#
# MULTI-ALLOCATION SAFE: configs are claimed atomically via mkdir (POSIX-atomic on the shared FS),
# so you can launch several of these pilots concurrently and they will not run the same config.
#
# Env: SCVI_PY (env python), WCD_EMBED_OUT (durable latent dir), MANIFEST, NGPU (GPUs on node),
#      LANES_PER_GPU (fits per GPU), OMP (threads/fit), WALL_STOP_MIN (stop claiming when < this many min left).
set -u
cd "$(dirname "$0")/.."
ES="${SCVI_PY:?set SCVI_PY to the env python}"
WCD_SRC="$(pwd)/src"
EMB="${WCD_EMBED_OUT:?set WCD_EMBED_OUT to a durable latent dir}"
MANIFEST="${1:-scripts/scvi_final_manifest.tsv}"
NGPU="${NGPU:-4}"
LANES_PER_GPU="${LANES_PER_GPU:-3}"
NWORK=$(( NGPU * LANES_PER_GPU ))
export KMP_AFFINITY=disabled OMP_NUM_THREADS="${OMP:-2}" NUMBA_NUM_THREADS="${OMP:-2}" \
       MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore

# --- EPHEMERAL LATENTS GUARD (fstype check, not name alone) ---
mkdir -p "$EMB"
case "$EMB" in */workspaces/*|/tmp/*) echo "FATAL: EMB=$EMB is EPHEMERAL"; exit 2;; esac
fstype=$(df -PT "$EMB" 2>/dev/null | tail -1 | awk '{print $2}')
case "$fstype" in tmpfs|ramfs) echo "FATAL: EMB=$EMB is $fstype (RAM-backed)"; exit 2;; esac
echo "[guard] latents -> $EMB (fstype=$fstype) OK  | NGPU=$NGPU lanes/gpu=$LANES_PER_GPU NWORK=$NWORK"

CLAIMDIR="$EMB/.claims"; mkdir -p "$CLAIMDIR" logs/wave
# wall-time budget: SLURM_JOB_END_TIME is epoch secs; stop claiming when < WALL_STOP_MIN left
WALL_STOP_MIN="${WALL_STOP_MIN:-30}"
end_epoch="${SLURM_JOB_END_TIME:-0}"
time_ok() {
  [ "$end_epoch" -le 0 ] && return 0
  local left=$(( end_epoch - $(date +%s) ))
  [ "$left" -gt $(( WALL_STOP_MIN * 60 )) ]
}

run_one() {  # gpu_id model cond arm lam di seed ds dec tag
  local gpu=$1 model=$2 cond=$3 arm=$4 lam=$5 di=$6 seed=$7 ds=$8 dec=$9 tag=${10}
  local out="$EMB/${tag}.npz" claim="$CLAIMDIR/${tag}"
  [ -s "$out" ] && { echo "[skip-done] $tag"; return; }
  # atomic claim: mkdir succeeds for exactly one worker across all pilots
  mkdir "$claim" 2>/dev/null || { echo "[skip-claimed] $tag"; return; }
  time_ok || { echo "[wall-stop] $tag (releasing claim)"; rmdir "$claim" 2>/dev/null; return 1; }
  CUDA_VISIBLE_DEVICES="$gpu" \
  SCVI_DS="$ds" SCVI_MODEL="$model" ADV="$arm" DCOEF="$lam" DISC_ITER="$di" COND="$cond" \
    SEED="$seed" MAXEP=239 BATCH=512 OUT="$out" WCD_SRC="$WCD_SRC" \
    "$ES" scripts/scvi_adv_fit.py > "logs/wave/${tag}.log" 2>&1
  if [ -s "$out" ]; then echo "[ok] gpu$gpu $tag"; else echo "[FAIL] gpu$gpu $tag rc=$?"; rmdir "$claim" 2>/dev/null; fi
}
export -f run_one time_ok; export ES EMB WCD_SRC CLAIMDIR end_epoch WALL_STOP_MIN

order_key() { case "$1" in none) echo 0;; discriminator) echo 1;; mmd|sinkhorn) echo 2;; *) echo 3;; esac; }

# Build the front-loaded work list once (priority sort), then PARTITION it into NGPU streams by
# round-robin line index (awk NR%NGPU). Launch one xargs -P LANES_PER_GPU per GPU, each pinned to
# that GPU id. This gives exact, race-free GPU balance (no shared counter, no hash skew) and the
# atomic mkdir-claim still guarantees no config runs twice across concurrent pilots.
WORKLIST="$CLAIMDIR/.worklist.$$"
grep -vE '^#|^model' "$MANIFEST" | while IFS=$'\t' read -r model cond adv lam di seed ds dec tag; do
  [ -z "$tag" ] && continue
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(order_key "$adv")" \
    "$model" "$cond" "$adv" "$lam" "$di" "$seed" "$ds" "$dec" "$tag"
done | sort -s -k1,1n | cut -f2- > "$WORKLIST"
echo "[dispatch] $(wc -l < "$WORKLIST") configs across $NGPU GPUs x $LANES_PER_GPU lanes"

pids=()
for gpu in $(seq 0 $(( NGPU - 1 ))); do
  ( awk -v g="$gpu" -v n="$NGPU" 'NR % n == g' "$WORKLIST" \
      | xargs -P "$LANES_PER_GPU" -I {} bash -c '
          IFS=$'"'"'\t'"'"' read -r m c a l d s ds dec t <<< "{}"
          run_one '"$gpu"' "$m" "$c" "$a" "$l" "$d" "$s" "$ds" "$dec" "$t"'
  ) &
  pids+=($!)
done
# wait on each GPU stream; surface a non-zero if any stream errored
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
rm -f "$WORKLIST"

echo "JHPCE PILOT DONE: $(ls "$EMB"/*_XZ_*.npz 2>/dev/null | wc -l) npz of $(grep -vcE '^#|^model' "$MANIFEST"); claims=$(ls "$CLAIMDIR" 2>/dev/null | grep -vc '^\.')"
