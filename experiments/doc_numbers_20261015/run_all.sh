#!/usr/bin/env bash
# Runs every script of this folder from the repository root and writes results/*.txt (DOC-R04). See README.md.
# Usage: bash experiments/doc_numbers_20261015/run_all.sh        (env scvi-api, CPU only; about 1 minute)
set -euo pipefail
export KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL CUDA_VISIBLE_DEVICES=
cd "$(dirname "$0")/../.."
D=experiments/doc_numbers_20261015
mkdir -p $D/results
python $D/common.py                  > $D/results/blinding_guard.txt
python $D/filler_fit_counts.py       > $D/results/filler_fit_counts.txt
python $D/validation_minibatch_counts.py > $D/results/validation_minibatch_counts.txt
python $D/pilot_ranges.py            > $D/results/pilot_ranges.txt
python $D/p2_pilot_recount.py        > $D/results/p2_pilot_recount.txt
python $D/no_pair_expectations.py    > $D/results/no_pair_expectations.txt
python $D/no_pair_assumptions.py     > $D/results/no_pair_assumptions.txt
echo "all scripts finished; every assertion held"
