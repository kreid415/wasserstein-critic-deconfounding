#!/bin/bash
# G4 follow-up (2026-10-02): the same latent + prepped files scored on local and JHPCE disagreed beyond 1e-6 only in
# kNN-graph-based metrics. This driver, run identically on both hosts, (1) rebuilds the scorer's kNN graph for one
# latent under three numba code-generation targets -- host default, NUMBA_CPU_NAME=haswell (AVX2/FMA, available on
# both CPUs), NUMBA_CPU_NAME=generic -- and saves them for an entry-by-entry cross-host comparison
# (scripts/gate_g4_diagnose.py --compare), and (2) re-scores the 2 G4 latents with the code paths pinned on every
# library that dispatches on the CPU: NUMBA_CPU_NAME=haswell, MKL_CBWR=AVX2, OPENBLAS_CORETYPE=Haswell (G4-pinned).
# Usage: PY=... R_HOME=... R_LIBS=... bash scripts/gate_g4_diag.sh LATENT_DIR PREPPED_H5AD OUT_DIR HOSTLABEL
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
LAT=$(readlink -f "${1:?latent dir}"); PRE=$(readlink -f "${2:?prepped h5ad}")
OUT=$(mkdir -p "${3:?out dir}" && cd "$3" && pwd); HOST=${4:?host label}
: "${PY:?}" "${R_HOME:?}" "${R_LIBS:?}"
export KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore
for v in host haswell generic; do
  if [ "$v" = host ]; then
    env -u NUMBA_CPU_NAME PREPPED="$PRE" NPZ="$LAT/Q_none_0.npz" OUT="$OUT/graph_${HOST}_${v}.npz" "$PY" "$REPO/scripts/gate_g4_diagnose.py"
  else
    NUMBA_CPU_NAME=$v PREPPED="$PRE" NPZ="$LAT/Q_none_0.npz" OUT="$OUT/graph_${HOST}_${v}.npz" "$PY" "$REPO/scripts/gate_g4_diagnose.py"
  fi
done | tee "$OUT/graphs_${HOST}.jsonl"
: > "$OUT/list_pinned.tsv"
for q in Q_none_0 Q_barycenter_1; do
  printf 'G4_%s\t%s\t%s\t%s\n' "$q" "$LAT/$q.npz" "$PRE" "{\"host\":\"$HOST\",\"pins\":\"numba=haswell,mkl_cbwr=AVX2,openblas=Haswell\"}" >> "$OUT/list_pinned.tsv"
done
NUMBA_CPU_NAME=haswell MKL_CBWR=AVX2 OPENBLAS_CORETYPE=Haswell \
  bash "$REPO/scripts/score_list.sh" "$OUT/list_pinned.tsv" "$OUT/pinned_scores" 2
echo "G4 DIAG DONE ($HOST)"
