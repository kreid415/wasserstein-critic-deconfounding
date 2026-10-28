#!/usr/bin/env bash
# Workstation side of the JHPCE-scored datasets (SI-115): under job-watch, every 15 minutes copy the dataset's finished latents from JHPCE
# (as the fit shards produce them; SHA-256 verified), copy and verify its JHPCE score files (the fixed-seed rows) into the mirror's
# scores/ folder, and never score here. A tag this machine already scored keeps its workstation row (the JHPCE file goes to
# scores_jhpce_duplicate/). Ends 0 when all 378 tags of the task have a score or a recorded fit failure.
# Usage: supervise_pull_jhpce_scores.sh TASK STATE_DIR     (TASK atac_large or immune_hum_mou; rerun with the same dir to re-attach)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
TASK=${1:?task}; ST=${2:?state dir}
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
I=$D/jhpce_tier12/pilot5_inputs; M=$D/jhpce_tier12/pilot5_jhpce
REPO=$(cd "$HERE/../.." && pwd -P)
CMD="python3 $REPO/cluster/jhpce/chain_ws.py --task $TASK --tags-file $I/pilot5_jhpce_$TASK.tags --mirror $M --state-dir $ST/chain --no-score --pull-scores --batch 100 --interval 900"
if [ ! -f "$ST/jw/state.json" ]; then
  mkdir -p "$ST/chain"
  python3 /home/kendall/job-watch/jw.py start --state "$ST/jw" --kind local --workdir "$HERE" --cmd "$CMD" \
    --progress-file "$ST/chain/chain_ws.log" --stall-min 600 --retries 5 --max-attempts 8 --tag "pilot5-pull-jhpce-scores-$TASK"
fi
exec python3 /home/kendall/job-watch/jw.py watch --state "$ST/jw" --interval 300 --deadline-min 1000000
