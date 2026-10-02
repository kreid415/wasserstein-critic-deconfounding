#!/bin/bash
# JHPCE GPU gate job ('gpu' partition, one pinned GPU model; 2026-10-02). Waits (bounded) for the setup job's
# verified wcd-fit env and staged local reference, runs cluster/jhpce/gate_gpu.sh, then packs the 66 JHPCE latents
# (64 queue + 2 single-lane) for harvest as <=60 MB parts with the md5 of the whole tar.
# Usage (repo root, inside the job): bash cluster/jhpce/gate_job.sh WORKDIR EXPECT_GPU [WAIT_MIN]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
source cluster/jhpce/env.sh
W=$(readlink -f "${1:?workdir}")
export EXPECT_GPU=${2:?expected GPU model substring, e.g. A100}
WAIT_MIN=${3:-60}
REF="$SCRATCH/ref_local"
GATE="$SCRATCH/gate/run_${SLURM_JOB_ID:?not inside a Slurm job}"
for _ in $(seq 1 "$WAIT_MIN"); do
  if [ -f "$FIT_ENV/.verified" ] && [ -f "$REF/.staged" ]; then break; fi
  sleep 60
done
[ -f "$FIT_ENV/.verified" ] && [ -f "$REF/.staged" ] || { echo "FATAL: setup outputs not ready after $WAIT_MIN min"; exit 1; }
REF_PREPPED_DIR="$REF" bash cluster/jhpce/gate_gpu.sh "$REF" "$GATE" "$W/gate_report"
tar -cf "$GATE/gate_latents_jhpce.tar" -C "$GATE/bench_out" latents
md5sum "$GATE/gate_latents_jhpce.tar" | awk '{print $1"  gate_latents_jhpce.tar"}' > "$W/gate_latents_jhpce.tar.md5"
split -b 60M -d "$GATE/gate_latents_jhpce.tar" "$W/gate_latents_jhpce.tar.part"
ls -l "$W"/gate_latents_jhpce.tar.part* | tee "$W/gate_report/harvest_parts.txt"
echo "GATE JOB OK: $GATE"
