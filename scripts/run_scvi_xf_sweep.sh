#!/bin/bash
# Finer-λ + nonlinear-decoder sweep on the scvi-native adversary plan (immune, single seed).
# Reads scripts/scvi_xf_jobs.tsv: model<TAB>cond<TAB>flag<TAB>arm<TAB>lam<TAB>disc_iter<TAB>tag
# LinearSCVI = linear decoder; SCVI = nonlinear decoder. λ ∈ {0,10,20,35,50}, both conditionings.
# Latents -> durable EMB dir; scored later through full_metric_suite in wcd-kbet.
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
  local model=$1 cond=$2 flag=$3 arm=$4 lam=$5 di=$6 tag=$7
  local out="$EMB/${tag}.npz"
  [ -s "$out" ] && { echo "[skip] $tag"; return; }
  SCVI_DS=immune SCVI_MODEL="$model" ADV="$arm" DCOEF="$lam" DISC_ITER="$di" COND="$flag" \
    SEED=0 MAXEP=239 BATCH=512 OUT="$out" WCD_SRC="$WCD_SRC" \
    "$ES" scripts/scvi_adv_fit.py > "logs/wave/${tag}.log" 2>&1 \
    && echo "[ok] $tag" || echo "[FAIL] $tag rc=$?"
}
export -f run_one; export ES EMB WCD_SRC

# feed the job rows to xargs, NWORK parallel
awk -F'\t' 'NF>=7{print}' scripts/scvi_xf_jobs.tsv \
  | xargs -P "$NWORK" -I {} bash -c 'IFS=$'"'"'\t'"'"' read -r m c f a l d t <<< "{}"; run_one "$m" "$c" "$f" "$a" "$l" "$d" "$t"'
echo "XF SWEEP DONE: $(ls $EMB/immune_XF_*.npz 2>/dev/null | wc -l) npz written"
