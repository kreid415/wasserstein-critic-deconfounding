#!/bin/bash
# Fresh-seed replication fit shard (rescope 2026-11-02; experiments/sweep_replication on branch sweep-replication).
# Usage inside a Slurm GPU job: bash run_shard.sh <shard: T|1|2|3|4|5> [extra prod_fit_job.sh args]
# Code: clean clone of 10a1ca5 (tag prereg-tier12-v3) from wcd.bundle; run list manifests/sweep_replication_x1.tsv (sha256 below).
set -euo pipefail
N=${1:?shard T|1..5}; shift || true
R=/fastscratch/myscratch/kreid/wcd/replication_20261102
SHA=10a1ca5874fbcdc033b148525b5e71d866306bdb
MAN=sweep_replication_x1.tsv
MSHA=d8d35481cd06b28c58045216e86bf7fee7c470cb69636fe80b291d0da4f225f1
case $N in
  T) TASKS="atac_large immune_hum_mou";; 1) TASKS="atac_large";; 2) TASKS="immune_hum_mou sim1";;
  3) TASKS="lung sim2";; 4) TASKS="pancreas atac_small";; 5) TASKS="immune";; 2b) TASKS="immune_hum_mou";; 4b) TASKS="pancreas";; *) echo "bad shard $N" >&2; exit 9;;
esac
TAGS=replication_s${N}.tags
TSHA=$(awk -v t="$TAGS" '$2==t{print $1}' "$R/shard_tags.sha256")
[ -n "$TSHA" ] || { echo "FATAL: no checksum for $TAGS" >&2; exit 9; }
JOBDIR=/fastscratch/myscratch/$USER/wcd/prod/job_${SLURM_JOB_ID:?not in a Slurm job}
mkdir -p "$JOBDIR/x1_inputs"
cp "$R/$MAN" "$R/$TAGS" "$JOBDIR/x1_inputs/"
printf '%s  %s\n%s  %s\n' "$MSHA" "$MAN" "$TSHA" "$TAGS" | (cd "$JOBDIR/x1_inputs" && sha256sum -c --quiet)
T0=$(date +%s)
timeout 300 /users/kreid/envs/wcd-fit/bin/python -c 'import numpy, torch, scvi' \
  || { echo "FATAL: importing numpy, torch and scvi failed or took over 300 s on $(hostname)" >&2; exit 9; }
echo "import check: $(( $(date +%s) - T0 )) s on $(hostname)"
git clone -q --no-checkout "$R/wcd.bundle" "$JOBDIR/repo"
git -C "$JOBDIR/repo" -c advice.detachedHead=false checkout -q "$SHA"
exec bash "$JOBDIR/repo/cluster/jhpce/prod_fit_job.sh" --expected-sha "$SHA" --stage sweep_replication \
  --tags-file "../x1_inputs/$TAGS" --tags-sha256 "$TSHA" --manifest "../x1_inputs/$MAN" --manifest-sha256 "$MSHA" \
  --experiments X1 --tasks "$TASKS" --fit-lanes 8 --fit-timeout-s 28800 --max-attempts 2 --guard-margin-s 2700 --slots 6 "$@"
