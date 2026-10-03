#!/bin/bash
# Score a list of latents with scripts/score_scib_native.py (scIB-native), one process per latent, NPROC at a time,
# one thread per process (the thread settings of scripts/run_wave.sh). Host-agnostic: the caller exports PY (the
# scoring env's python) and R_HOME / R_LIBS for rpy2 + kBET (scripts/run_wave.sh; cluster/jhpce/env.sh score_r_env).
#
# Usage: bash scripts/score_list.sh LIST_TSV OUT_DIR NPROC
#   LIST_TSV: no header; columns tag <TAB> npz path <TAB> prepped h5ad path <TAB> meta json (one line per latent)
# Out: OUT_DIR/<tag>.csv per latent, OUT_DIR/logs/<tag>.log, OUT_DIR/scores.csv (all rows, header once).
# Fails (exit 1) unless every listed tag produced its CSV; failures are listed in OUT_DIR/failed.txt.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
LIST=$(readlink -f "${1:?list tsv}")
OUT=$(mkdir -p "${2:?out dir}" && cd "$2" && pwd)
NPROC=${3:?nproc}
: "${PY:?export PY=<scoring env python>}" "${R_HOME:?export R_HOME}" "${R_LIBS:?export R_LIBS}"
mkdir -p "$OUT/logs"
: > "$OUT/failed.txt"
n=$(grep -c . "$LIST")
[ "$(cut -f1 "$LIST" | sort | uniq -d | wc -l)" -eq 0 ] || { echo "FATAL: duplicate tags in $LIST"; exit 1; }
export KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore
export PY REPO OUT
score_one() {
  local tag npz pre meta rc=0
  IFS=$'\t' read -r tag npz pre meta <<< "$1"
  rm -f "$OUT/$tag.csv"
  PREPPED="$pre" NPZ="$npz" TAG="$tag" META="$meta" OUT_CSV="$OUT/$tag.csv" \
      "$PY" "$REPO/scripts/score_scib_native.py" > "$OUT/logs/$tag.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$OUT/$tag.csv" ]; then echo "$tag rc=$rc" >> "$OUT/failed.txt"; fi
  echo "[score] $tag rc=$rc"
}
export -f score_one
xargs -a "$LIST" -d '\n' -P "$NPROC" -I{} bash -c 'score_one "$@"' _ {}
nf=$(awk 'NF' "$OUT/failed.txt" | wc -l)
ncsv=0
while IFS=$'\t' read -r tag _rest; do [ -s "$OUT/$tag.csv" ] && ncsv=$((ncsv + 1)); done < "$LIST"
echo "scored $ncsv of $n latents; failed $nf"
[ "$nf" -eq 0 ] && [ "$ncsv" -eq "$n" ] || { cat "$OUT/failed.txt"; exit 1; }
"$PY" - "$LIST" "$OUT" <<'PYEOF'
import sys, pandas as pd
tags = [l.split("\t")[0] for l in open(sys.argv[1]) if l.strip()]
d = pd.concat([pd.read_csv(f"{sys.argv[2]}/{t}.csv") for t in tags], ignore_index=True)
assert list(d.tag) == tags, "tag order/content mismatch"
d.to_csv(f"{sys.argv[2]}/scores.csv", index=False)
print(f"scores.csv: {len(d)} rows x {d.shape[1]} cols")
PYEOF
