#!/usr/bin/env bash
# Runs every test suite of the tagged checkout (prereg-tier12-v1) one at a time, CPU only, at nice 19.
# Reads the prepped inputs read-only; writes only to pytest tmp dirs and review/outputs/.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REPO=$ROOT/repo
OUT=$ROOT/outputs/tests
mkdir -p "$OUT"
E=/home/kendall/.claude-science/conda/envs
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
echo "HEAD $(git rev-parse HEAD) $(git describe --tags) dirty=$(git status --porcelain --untracked-files=no | wc -l)" > "$OUT/HEAD.txt"
run() {  # name env args...
  local name=$1 env=$2; shift 2
  local t0=$(date +%s)
  PYTHONPATH=src WCD_SRC=src nice -n 19 "$E/$env/bin/python" -m pytest -q -p no:cacheprovider "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo -e "$name\t$env\t$*\trc=$rc\t$(( $(date +%s) - t0 ))s\t$(tail -n 1 "$OUT/$name.log")" >> "$OUT/summary.tsv"
}
: > "$OUT/summary.tsv"
run prereg wcd-gpu tests/prereg
run stage wcd-gpu tests/stage
run wcd_vae wcd-gpu tests/wcd_vae
PREPPED_DIR=$D/prepped_scib run x13_wcdgpu wcd-gpu tests/x13
PREPPED_DIR=$D/prepped_scib run x13_wcdkbet wcd-kbet tests/x13
R_HOME=$E/wcd-kbet/lib/R R_LIBS=$D/Rlib_kbet run scoring wcd-kbet tests/scoring
PREPPED_DIR=$D/prepped_scib run scvi scvi-api tests/scvi
STAGE_E2E_FIT_PY=$E/scvi-api/bin/python STAGE_E2E_SCORE_PY=$E/wcd-kbet/bin/python R_HOME=$E/wcd-kbet/lib/R R_LIBS=$D/Rlib_kbet PREPPED_DIR=$D/prepped_scib run stage_e2e wcd-gpu tests/stage/test_run_stage_e2e.py
echo DONE >> "$OUT/summary.tsv"
