#!/usr/bin/env bash
# The six scenarios of the first review's pipeline simulation (limitation 1 of docs/ANALYSIS_SPEC.md section 16), four processes at a time.
# Run from anywhere (conda env scvi-api, CPU only; about 7 minutes). The raw per-world records (5 MB each) go to a temporary folder;
# the summaries printed by sim_pipeline.py are the committed outputs sim_pipeline_<scenario>.txt.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL
tmp=$(mktemp -d)
scn() { name=$1; shift; python sim_pipeline.py --scenario "$name" --worlds 20 --reps 200 "$@" --out "$tmp/$name.json" > "sim_pipeline_$name.txt"; }
scn S0_null --pool_reps 500 --seed 101 &
scn S1_het --tau_f 0.03 --tau_d 0.03 --pool_reps 500 --seed 102 &
scn S2_effect --delta 0.02 --tau_f 0.02 --tau_d 0.02 --pool_reps 500 --seed 103 &
scn S3_noisy_null --noise_scale 2.0 --pool_reps 500 --seed 104 &
wait
scn S4_curved --kappa 3.0 --pool_reps 500 --seed 105 &
scn W1_wiggle --collapse 0.25 --wiggle 1 --pool_reps 20 --seed 205 &
wait
