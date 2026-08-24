#!/bin/bash
# KL x formulation sweep on the scvi-native adversary plan (immune, single seed, unconditioned).
# Reads scripts/scvi_xk_jobs.tsv: model<TAB>cond<TAB>flag<TAB>arm<TAB>lam<TAB>disc_iter<TAB>max_kl<TAB>tag
# Tests whether looser/tighter KL (scvi max_kl_weight) changes the critic-vs-discriminator crossover
# on the deployment models (LinearSCVI + SCVI). Each arm at its near-peak lambda (disc 20, critics 50).
set -u
WS=/home/kendall/.claude-science/orgs/7339da5c-ddcf-4ba9-9b06-df362dd1208a/workspaces/a0b87862-8454-468c-a2f9-6326cd1433fc
cd "$WS/wcd_git"
ES=/home/kendall/.claude-science/conda/envs/scvi-env/bin/python
EMB=/home/kendall/experiment_data/wasserstein-critic-deconfounding/embeddings_scvi_adv
export WCD_SRC="$WS/wcd_git/src"
export KMP_AFFINITY=disabled OMP_NUM_THREADS=4 MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore
NWORK=${NWORK:-3}
mkdir -p logs/wave "$EMB"

run_one() {
  local model=$1 cond=$2 flag=$3 arm=$4 lam=$5 di=$6 mkl=$7 tag=$8
  local out="$EMB/${tag}.npz"
  [ -s "$out" ] && { echo "[skip] $tag"; return; }
  SCVI_DS=immune SCVI_MODEL="$model" ADV="$arm" DCOEF="$lam" DISC_ITER="$di" COND="$flag" \
    SCVI_MAX_KL="$mkl" SEED=0 MAXEP=239 BATCH=512 OUT="$out" WCD_SRC="$WCD_SRC" \
    "$ES" scripts/scvi_adv_fit.py > "logs/wave/${tag}.log" 2>&1 \
    && echo "[ok] $tag" || echo "[FAIL] $tag rc=$?"
}
export -f run_one; export ES EMB WCD_SRC

awk -F'\t' 'NF>=8{print}' scripts/scvi_xk_jobs.tsv \
  | xargs -P "$NWORK" -I {} bash -c 'IFS=$'"'"'\t'"'"' read -r m c f a l d k t <<< "{}"; run_one "$m" "$c" "$f" "$a" "$l" "$d" "$k" "$t"'
echo "XK SWEEP DONE: $(ls $EMB/immune_XK_*.npz 2>/dev/null | wc -l) npz written"
