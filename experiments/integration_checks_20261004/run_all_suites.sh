#!/usr/bin/env bash
# Usage: REPO=<worktree> OUT=<dir> WORKSPACE=<unused> bash run_all_suites.sh  (env paths below are the local machine's)
# Full test suites on integrate-fixes (merged fix-runner + jhpce-production + fix-score-fit), one at a time, nice 19.
set -uo pipefail
W=${WORKSPACE:?}
cd ${REPO:?}; T=${OUT:?}; mkdir -p $T
C=/home/kendall/.claude-science/conda/envs; D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled CUDA_VISIBLE_DEVICES= PYTHONWARNINGS=ignore
echo "HEAD $(git rev-parse --short HEAD) clean=$(git status --porcelain | wc -l)" > $T/summary.txt
run() { name=$1; shift; s=$(date +%s); nice -n 19 env "$@" > $T/$name.txt 2>&1; rc=$?; echo "$name exit $rc $(( $(date +%s) - s ))s | $(tail -1 $T/$name.txt)" >> $T/summary.txt; }
run stage_e2e PYTHONPATH=src WCD_SRC=src STAGE_E2E_FIT_PY=$C/scvi-api/bin/python STAGE_E2E_SCORE_PY=$C/wcd-kbet/bin/python R_HOME=$C/wcd-kbet/lib/R R_LIBS=$D/Rlib_kbet PREPPED_DIR=$D/prepped_scib $C/wcd-gpu/bin/python -m pytest -q -rxXs -p no:cacheprovider tests/stage
run scvi WCD_SRC=src PREPPED_DIR=$D/prepped_scib WCD_REQUIRE_DATA=1 $C/scvi-api/bin/python -m pytest -q -rs -p no:cacheprovider tests/scvi
run x13_scoring PYTHONPATH=src R_HOME=$C/wcd-kbet/lib/R R_LIBS=$D/Rlib_kbet PREPPED_DIR=$D/prepped_scib $C/wcd-kbet/bin/python -m pytest -q -rs -p no:cacheprovider tests/x13 tests/scoring
run other PYTHONPATH=src $C/wcd-gpu/bin/python -m pytest -q -rs -p no:cacheprovider tests --ignore=tests/scvi --ignore=tests/x13 --ignore=tests/stage --ignore=tests/jhpce --ignore=tests/scoring
run jhpce $C/scvi-api/bin/python -m pytest -q -rs -p no:cacheprovider tests/jhpce
echo "ALL DONE $(date -Is)" >> $T/summary.txt
