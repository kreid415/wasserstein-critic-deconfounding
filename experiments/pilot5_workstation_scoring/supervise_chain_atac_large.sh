#!/usr/bin/env bash
# Starts (or re-attaches to) the workstation backend of the fit-to-scoring chain for atac_large under job-watch: it copies finished
# atac_large latents from JHPCE as the three fit shards produce them, checks every file's SHA-256 against a hash made on JHPCE,
# and scores them here once the four-dataset run (job-watch state $1) is done. Ends 0 when all 378 tags have a score or a failure record.
# Usage: supervise_chain_atac_large.sh STATE_DIR FOUR_DATASET_JW_STATE
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ST=${1:?chain state dir}; AFTER=${2:?job-watch state dir of the four-dataset run}
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
I=$D/jhpce_tier12/pilot5_inputs; M=$D/jhpce_tier12/pilot5_jhpce
REPO=$(cd "$HERE/../.." && pwd -P)
CMD="python3 $REPO/cluster/jhpce/chain_ws.py --task atac_large --tags-file $I/pilot5_jhpce_atac_large.tags --mirror $M --state-dir $ST/chain --after-jw $AFTER --workers 3 --batch 100 --interval 900"
if [ ! -f "$ST/jw/state.json" ]; then
  mkdir -p "$ST/chain"
  python3 /home/kendall/job-watch/jw.py start --state "$ST/jw" --kind local --workdir "$HERE" --cmd "$CMD" \
    --progress-file "$ST/chain/chain_ws.log" --stall-min 600 --retries 5 --max-attempts 8 --tag pilot5-chain-ws-atac-large
fi
exec python3 /home/kendall/job-watch/jw.py watch --state "$ST/jw" --interval 300 --deadline-min 1000000
