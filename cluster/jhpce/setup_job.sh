#!/bin/bash
# JHPCE setup job ('shared' partition, 2026-10-02):
#   0. move the staged local reference inputs off the HOME workdir to fastscratch and verify them
#      (immune__scib.h5ad = the local prepped file the local calibration fitted on; the local calibration bundle
#      with bench_queue_manifest.tsv, queue_order.txt, queue_times.tsv and the 64 local latents Q_*.npz);
#   1. download the 8 scIB files serially (background, network-bound) while both envs are built;
#   2. prep the 8 tasks with the committed prep script and compare fingerprints (G2);
#   3. JHPCE side of G4: score 2 identical latent files in wcd-score, against the staged local prepped copy
#      (G4 proper: identical inputs, only the host and env differ) and against the JHPCE-regenerated file (G4b).
# Usage (repo root, inside the job): bash cluster/jhpce/setup_job.sh WORKDIR REPORT_DIR
#   WORKDIR = job workdir with the staged inputs scib_raw_sources.tsv, immune__scib.h5ad, local_queue_latents_stock.tar.gz
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
source cluster/jhpce/env.sh
W=$(readlink -f "${1:?workdir}")
REPORT=$(mkdir -p "${2:?report dir}" && cd "$2" && pwd)
REF="$SCRATCH/ref_local"
trap 'echo "SETUP FAILED at line $LINENO"; for f in "$REPORT"/*.log; do echo "== $f"; tail -n 40 "$f"; done' ERR
echo "setup: repo $(git rev-parse HEAD) host $(hostname) cpus $(nproc) mem $(free -g | awk '/Mem:/{print $2}')G"

# ---- 0. staged reference inputs -> fastscratch ----
mkdir -p "$REF"
mv -f "$W/immune__scib.h5ad" "$REF/immune__scib.h5ad"
echo "3888d2563a7243f9854e41de973f3607  $REF/immune__scib.h5ad" | md5sum -c -
tar -xzf "$W/local_queue_latents_stock.tar.gz" -C "$REF"
rm -f "$W/local_queue_latents_stock.tar.gz"
echo "9cca9c8c38e767ebf13a98ad44bb9e95  $REF/bench_queue_manifest.tsv" | md5sum -c -
nq=$(find "$REF/bench_out_stock/latents" -name 'Q_*.npz' | wc -l)
[ "$nq" -eq 64 ] || { echo "FATAL: $nq local latents, expected 64"; exit 1; }
date -Is > "$REF/.staged"

# ---- 1. downloads || env builds ----
bash cluster/jhpce/stage_scib.sh download "$W/scib_raw_sources.tsv" "$REPORT/data" > "$REPORT/download.log" 2>&1 &
DL=$!
bash cluster/jhpce/build_envs.sh "$REPORT/env" all > "$REPORT/build_envs.log" 2>&1
tail -n 3 "$REPORT/build_envs.log"
wait "$DL"
tail -n 1 "$REPORT/download.log"

# ---- 2. prep + fingerprints (G2) ----
bash cluster/jhpce/stage_scib.sh prep "$W/scib_raw_sources.tsv" "$REPORT/data" 2 > "$REPORT/prep.log" 2>&1
tail -n 12 "$REPORT/prep.log"

# ---- 3. G4, JHPCE side ----
score_r_env
export PY="$SCORE_PY" PATH="$SCORE_ENV/bin:$PATH"
L="$REF/bench_out_stock/latents"
: > "$REPORT/g4_list.tsv"
for q in Q_none_0 Q_barycenter_1; do
  printf 'G4_%s\t%s\t%s\t%s\n' "$q" "$L/$q.npz" "$REF/immune__scib.h5ad" '{"host":"jhpce","prepped":"local_copy"}' >> "$REPORT/g4_list.tsv"
  printf 'G4b_%s\t%s\t%s\t%s\n' "$q" "$L/$q.npz" "$PREPPED_DIR/immune__scib.h5ad" '{"host":"jhpce","prepped":"jhpce_regenerated"}' >> "$REPORT/g4_list.tsv"
done
md5sum "$L/Q_none_0.npz" "$L/Q_barycenter_1.npz" "$REF/immune__scib.h5ad" "$PREPPED_DIR/immune__scib.h5ad" > "$REPORT/g4_inputs_md5.txt"
bash scripts/score_list.sh "$REPORT/g4_list.tsv" "$REPORT/g4" 4 > "$REPORT/g4.log" 2>&1
tail -n 2 "$REPORT/g4.log"
echo "SETUP JOB OK"
