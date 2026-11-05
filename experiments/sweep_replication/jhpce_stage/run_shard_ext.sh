#!/bin/bash
# Replication EXTENSION fit shard E (rule AGREEMENT.md line 9; run list manifests/sweep_replication_ext1.tsv): 25 rows, immune_hum_mou + pancreas + sim2.
# Usage inside a Slurm GPU job: bash run_shard.sh <shard: T|1|2|3|4|5> [extra prod_fit_job.sh args]
# Code: clean clone of 10a1ca5 (tag prereg-tier12-v3) from wcd.bundle; run list manifests/sweep_replication_x1.tsv (sha256 below).
set -euo pipefail
N=${1:?shard T|1..5}; shift || true
R=/fastscratch/myscratch/kreid/wcd/replication_20261102
SHA=10a1ca5874fbcdc033b148525b5e71d866306bdb
MAN=sweep_replication_ext1.tsv
MSHA=13013d7d18215784e7d8da39a316884a26b6961f35e1eee12a6d1b0c0ee16494
case $N in
  T) TASKS="atac_large immune_hum_mou";; 1) TASKS="atac_large";; 2) TASKS="immune_hum_mou sim1";;
  3) TASKS="lung sim2";; 4) TASKS="pancreas atac_small";; 5) TASKS="immune";; 2b) TASKS="immune_hum_mou";; E) TASKS="immune_hum_mou pancreas sim2";; 4b) TASKS="pancreas";; *) echo "bad shard $N" >&2; exit 9;;
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
