#!/bin/bash
# Immune adversarial sweep on the scvi-native plan: 2 backbones (cond/uncond) x 3 adversaries
# (reference, barycenter, discriminator) x lambda{0,50} x seed 0 = 12 configs. Fits in scvi-env
# (CPU), writes latent npz to durable storage. Scoring is a separate wcd-kbet pass.
#
# Env: ES (scvi-env python), WCD_SRC, EMB (durable npz dir), MAXEP (default 239), NWORK (default 3).
set -u
ES=${ES:-/home/kendall/.claude-science/conda/envs/scvi-env/bin/python}
EMB=${EMB:?set EMB to a durable npz output dir}
MAXEP=${MAXEP:-239}
NWORK=${NWORK:-3}
export KMP_AFFINITY=disabled OMP_NUM_THREADS=4 MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore
mkdir -p "$EMB" logs/wave

run_one() {
    local adv=$1 lam=$2 cond=$3
    local di=10; [ "$adv" = "discriminator" ] && di=1
    local ctag=$([ "$cond" = "1" ] && echo cond || echo uncond)
    local ls=$(echo "$lam" | sed 's/\./p/')
    local tag="immune_XA_${ctag}_${adv}_lam${ls}"
    local out="$EMB/${tag}.npz"
    if [ -s "$out" ]; then echo "[skip] $tag (npz exists)"; return; fi
    SCVI_DS=immune ADV=$adv DCOEF=$lam DISC_ITER=$di COND=$cond SEED=0 MAXEP=$MAXEP BATCH=512 \
        OUT="$out" WCD_SRC="$WCD_SRC" \
        $ES scripts/scvi_adv_fit.py > "logs/wave/${tag}.log" 2>&1 \
        && echo "[ok] $tag" || echo "[FAIL] $tag (rc=$?)"
}
export -f run_one
export ES EMB MAXEP WCD_SRC

# build the 12-config job list (adversary lambda cond)
jobs=""
for cond in 1 0; do
  for adv in reference barycenter discriminator; do
    for lam in 50 0; do
      jobs+="$adv $lam $cond\n"
    done
  done
done
echo -e "$jobs" | grep -v '^$' | xargs -P "$NWORK" -I {} bash -c 'run_one $@' _ {}
echo "SWEEP DONE: $(ls $EMB/immune_XA_*.npz 2>/dev/null | wc -l)/12 npz written"
