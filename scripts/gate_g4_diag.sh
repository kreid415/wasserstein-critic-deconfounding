#!/bin/bash
# G4 follow-up (2026-10-02): the same latent + prepped files scored on local and JHPCE disagreed beyond 1e-6 only in
# kNN-graph-based metrics. Run identically on each host:
#   (1) rebuild the scorer's kNN graph for one latent under numba code-generation targets GRAPH_TARGETS (default
#       "host generic"; host = the CPU's own ISA) and save them for an entry-by-entry cross-host comparison
#       (scripts/gate_g4_diagnose.py --compare);
#   (2) score the 2 G4 latents under each pin set in PIN_SETS (default "none numba_generic"):
#       none = defaults (the scoring protocol of record); numba_generic = NUMBA_CPU_NAME=generic (no host-specific
#       SIMD in numba code: pynndescent/umap kNN + graph).
# Usage: PY=... R_HOME=... R_LIBS=... bash scripts/gate_g4_diag.sh LATENT_DIR PREPPED_H5AD OUT_DIR HOSTLABEL
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
LAT=$(readlink -f "${1:?latent dir}"); PRE=$(readlink -f "${2:?prepped h5ad}")
OUT=$(mkdir -p "${3:?out dir}" && cd "$3" && pwd); HOST=${4:?host label}
: "${PY:?}" "${R_HOME:?}" "${R_LIBS:?}"
GRAPH_TARGETS=${GRAPH_TARGETS:-host generic}
PIN_SETS=${PIN_SETS:-none numba_generic}
export KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore
grep -m1 'model name' /proc/cpuinfo | tee "$OUT/cpu_${HOST}.txt"
grep -o -w -E 'avx512f|avx2|fma|fma4|avx' /proc/cpuinfo | sort | uniq -c | tee -a "$OUT/cpu_${HOST}.txt"
for v in $GRAPH_TARGETS; do
  if [ "$v" = host ]; then
    env -u NUMBA_CPU_NAME PREPPED="$PRE" NPZ="$LAT/Q_none_0.npz" OUT="$OUT/graph_${HOST}_${v}.npz" "$PY" "$REPO/scripts/gate_g4_diagnose.py"
  else
    NUMBA_CPU_NAME=$v PREPPED="$PRE" NPZ="$LAT/Q_none_0.npz" OUT="$OUT/graph_${HOST}_${v}.npz" "$PY" "$REPO/scripts/gate_g4_diagnose.py"
  fi
done | tee "$OUT/graphs_${HOST}.jsonl"
for p in $PIN_SETS; do
  : > "$OUT/list_$p.tsv"
  for q in Q_none_0 Q_barycenter_1; do
    printf 'G4_%s\t%s\t%s\t%s\n' "$q" "$LAT/$q.npz" "$PRE" "{\"host\":\"$HOST\",\"pins\":\"$p\"}" >> "$OUT/list_$p.tsv"
  done
  case "$p" in
    none) env -u NUMBA_CPU_NAME bash "$REPO/scripts/score_list.sh" "$OUT/list_$p.tsv" "$OUT/scores_$p" 2 ;;
    numba_generic) NUMBA_CPU_NAME=generic bash "$REPO/scripts/score_list.sh" "$OUT/list_$p.tsv" "$OUT/scores_$p" 2 ;;
    *) echo "unknown pin set $p"; exit 2 ;;
  esac &
done
fail=0; for j in $(jobs -p); do wait "$j" || fail=1; done
[ "$fail" -eq 0 ] || { echo "a scoring run failed"; exit 1; }
echo "G4 DIAG DONE ($HOST)"
