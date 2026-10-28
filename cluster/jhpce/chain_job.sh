#!/bin/bash
#SBATCH --job-name=wcd-chain
#SBATCH --partition=shared
#SBATCH --account=jhpce
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
# The fit-to-scoring supervisor (chain_score.py) under job-watch (cluster/vendor/jw.py): ONE 1-CPU job; jw restarts the supervisor if it
# dies or stalls (status.json not touched for 90 min) and ends with one validated report line. The supervisor only sleeps,
# checks files and calls sbatch/squeue/sacct; the scoring arrays it submits are separate jobs (<= 200 CPUs held).
#   sbatch --export=ALL,WCD_REPO=$PWD cluster/jhpce/chain_job.sh STATE_DIR -- <chain_score.py arguments, without --state-dir>
# STATE_DIR must be on fastscratch. Re-running the same command continues from STATE_DIR (submissions.jsonl is the record).
set -euo pipefail
STATE=${1:?state dir}; shift; [ "${1:-}" = "--" ] && shift
REPO=$(cd "${WCD_REPO:?set WCD_REPO to the repo checkout (Slurm runs a copy of this script)}" && pwd -P)
source "$REPO/cluster/jhpce/env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$STATE"
CMD="$SCORE_PY $REPO/cluster/jhpce/chain_score.py --state-dir $STATE $*"
if [ ! -f "$STATE/jw/state.json" ]; then
  "$SCORE_PY" "$REPO/cluster/vendor/jw.py" start --state "$STATE/jw" --kind local --workdir "$STATE" --cmd "$CMD" \
    --validate "$CMD --check-finished" --progress-file "$STATE/status.json" --stall-min 90 --retries 2
fi
exec "$SCORE_PY" "$REPO/cluster/vendor/jw.py" watch --state "$STATE/jw" --interval 60 --deadline-min 100000
