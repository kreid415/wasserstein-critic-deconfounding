#!/bin/bash
# JHPCE combined: hold the GPU node through GATE -> full-sweep DRAIN, never releasing.
#
# WHY: the GPU queue is long, so once a node is allocated we must NOT release it between the
# reproducibility gate and the sweep. This script runs on the allocated node and:
#   1. Runs the bit-identity gate ON THIS NODE (adversary=none must == stock LinearSCVI, max|delta|=0).
#      This verifies THIS physical node produces bit-reproducible latents before it contributes.
#   2. Only if the gate PASSES, runs the pilot driver to drain the manifest across all the node's GPUs.
#   3. If the gate FAILS, aborts WITHOUT touching the sweep (never pollute the benchmark with a bad node).
# Multi-allocation safe: the pilot's atomic mkdir-claim lets several of these run concurrently.
set -u
cd "$(dirname "$0")/.."
: "${SCVI_PY:?}" "${WCD_EMBED_OUT:?}"
export PYTHONPATH="$(pwd)/src" KMP_AFFINITY=disabled OMP_NUM_THREADS="${OMP:-2}" \
       MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore
MANIFEST="${1:-scripts/scvi_final_manifest.tsv}"

echo "=== NODE $(hostname) GPUs ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "=== [1/2] BIT-IDENTITY GATE on this node (immune, 2 seeds) ==="
if GATE_DS=immune GATE_SEEDS=0,1 GATE_EPOCHS=15 CUDA_VISIBLE_DEVICES=0 \
     "$SCVI_PY" scripts/scvi_adv_bitidentity_gate.py; then
  echo "GATE PASSED on $(hostname) — proceeding to drain the manifest."
else
  echo "GATE FAILED on $(hostname) — ABORTING, will NOT run the sweep on this node."
  exit 1
fi

echo "=== [2/2] DRAIN the manifest across all GPUs (pilot, holds the node) ==="
exec bash scripts/run_jhpce_pilot.sh "$MANIFEST"
