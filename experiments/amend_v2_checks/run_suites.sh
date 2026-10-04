#!/usr/bin/env bash
# All test suites of amend-v2 at one commit (nice 19, one thread, CUDA hidden), one suite per call:
#   bash run_suites.sh <repo checkout> <out dir> <suite: stage|scvi|x13_scoring|other|jhpce>
set -uo pipefail
R=$1; OUT=$2; S=$3; cd "$R"
export KMP_AFFINITY=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
C=/home/kendall/.claude-science/conda/envs; P=/home/kendall/experiment_data/wasserstein-critic-deconfounding/prepped_scib
RL=/home/kendall/experiment_data/wasserstein-critic-deconfounding/Rlib_kbet
mkdir -p "$OUT"
case $S in
  stage) cmd=(env PYTHONPATH=$R/src WCD_SRC=$R/src STAGE_E2E_FIT_PY=$C/scvi-api/bin/python STAGE_E2E_SCORE_PY=$C/wcd-kbet/bin/python
              R_HOME=$C/wcd-kbet/lib/R R_LIBS=$RL PREPPED_DIR=$P nice -n 19 $C/wcd-gpu/bin/python -m pytest -q -p no:cacheprovider -rs tests/stage) ;;
  scvi) cmd=(env WCD_SRC=$R/src PREPPED_DIR=$P WCD_REQUIRE_DATA=1 nice -n 19 $C/scvi-api/bin/python -m pytest -q -p no:cacheprovider -rs tests/scvi) ;;
  x13_scoring) cmd=(env R_HOME=$C/wcd-kbet/lib/R R_LIBS=$RL PREPPED_DIR=$P PYTHONPATH=$R/src nice -n 19 $C/wcd-kbet/bin/python -m pytest -q
              -p no:cacheprovider -rs tests/x13 tests/scoring) ;;
  other) cmd=(env PYTHONPATH=$R/src nice -n 19 $C/wcd-gpu/bin/python -m pytest -q -p no:cacheprovider -rs tests --ignore=tests/stage
              --ignore=tests/scvi --ignore=tests/x13 --ignore=tests/scoring --ignore=tests/jhpce) ;;
  jhpce) cmd=(env PYTHONPATH=$R/src nice -n 19 $C/wcd-gpu/bin/python -m pytest -q -p no:cacheprovider -rs tests/jhpce) ;;
  *) echo "unknown suite $S" >&2; exit 2 ;;
esac
echo "HEAD $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) suite=$S" > "$OUT/$S.txt"
echo "${cmd[*]}" >> "$OUT/$S.txt"
s=$(date +%s); "${cmd[@]}" >> "$OUT/$S.txt" 2>&1; rc=$?
echo "rc=$rc seconds=$(( $(date +%s) - s ))" >> "$OUT/$S.txt"
tail -2 "$OUT/$S.txt"
exit $rc
