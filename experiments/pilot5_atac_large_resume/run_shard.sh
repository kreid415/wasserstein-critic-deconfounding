#!/bin/bash
# One segment of an atac_large pilot resume shard: same commit (10a1ca5), same manifest and same runner settings as
# job 36551149; only the tags file differs (the 128 tags without a latent, split into 3 shards).
# Usage (inside a Slurm GPU job): bash run_shard.sh SHARD_NUMBER [--dry-run]
set -euo pipefail
N=${1:?shard number 1|2|3}; shift || true
R=/fastscratch/myscratch/kreid/wcd/atac_resume_20261028
SHA=10a1ca5874fbcdc033b148525b5e71d866306bdb
MSHA=5dd65550fdb18e5c80e73dcfa731d72bdeb6128bdeac7ffb36df6c54450396ff
TAGS=pilot5_jhpce_atac_large_resume_s${N}.tags
TSHA=$(awk -v t="$TAGS" '$2==t{print $1}' "$R/shard_tags.sha256")
[ -n "$TSHA" ] || { echo "FATAL: no checksum for $TAGS" >&2; exit 9; }
JOBDIR=/fastscratch/myscratch/$USER/wcd/prod/job_${SLURM_JOB_ID:?not in a Slurm job}
mkdir -p "$JOBDIR/x1_inputs"
cp "$R/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv" "$R/$TAGS" "$JOBDIR/x1_inputs/"
printf '%s  %s\n%s  %s\n' "$MSHA" paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv "$TSHA" "$TAGS" | (cd "$JOBDIR/x1_inputs" && sha256sum -c --quiet)
# Fail fast on a node whose file reads are very slow (2026-10-28: on compute-172 'import numpy' alone took 164 s and
# 'import torch' did not finish in 15 min, so the job would have sat on the GPU until its time limit).
T0=$(date +%s)
timeout 300 /users/kreid/envs/wcd-fit/bin/python -c 'import numpy, torch, scvi' \
  || { echo "FATAL: importing numpy, torch and scvi failed or took over 300 s on $(hostname)" >&2; exit 9; }
echo "import check: $(( $(date +%s) - T0 )) s on $(hostname)"
git clone -q --no-checkout "$R/wcd.bundle" "$JOBDIR/repo"
git -C "$JOBDIR/repo" -c advice.detachedHead=false checkout -q "$SHA"
exec bash "$JOBDIR/repo/cluster/jhpce/prod_fit_job.sh" --expected-sha "$SHA" --stage pilot5_jhpce \
  --tags-file "../x1_inputs/$TAGS" --tags-sha256 "$TSHA" \
  --manifest ../x1_inputs/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv --manifest-sha256 "$MSHA" \
  --experiments A1 --tasks atac_large --fit-lanes 8 --fit-timeout-s 28800 --max-attempts 2 --guard-margin-s 2700 --slots 5 "$@"
