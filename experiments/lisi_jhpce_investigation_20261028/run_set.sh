#!/bin/bash
# usage: run_set.sh PYTHON LATENT_DIR OUT_DIR LABEL [NPROC]   (extra env, e.g. LD_PRELOAD, is inherited)
set -euo pipefail
PY=$1; LD=$2; OUT=$3; LABEL=$4; NP=${5:-1}
mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_AFFINITY=disabled PYTHONWARNINGS=ignore
ls $LD/*.npz | xargs -P $NP -I{} bash -c 'b=$(basename {} .npz); '"$PY"' '"$(dirname "$0")"'/probe_lisi.py '"$OUT"'/'"$LABEL"'_$b.json {} > '"$OUT"'/'"$LABEL"'_$b.log 2>&1 || echo "FAIL $b"'
# fl: allow FL111 counts the files, picks none; a failed latent leaves no json, so this count check fails the whole set
test "$(ls $OUT/${LABEL}_*.json | wc -l)" = "$(ls $LD/*.npz | wc -l)"
# fl: allow FL111 counts the files, picks none
echo "$LABEL done: $(ls $OUT/${LABEL}_*.json | wc -l) json"
