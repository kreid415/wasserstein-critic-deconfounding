#!/usr/bin/env bash
# Workstation scoring of the penalty-strength pilot's five new datasets (score only; no fit can start: any refit is refused
# because --expect-device L40S is required before a fit and this machine has no L40S, SI-17). Same scorer code, environment
# and thread settings as the first three datasets' scores: scripts/run_stage.py from a clean checkout of tag prereg-tier12-v3
# (10a1ca5), scorer env wcd-kbet, thread_env(1), KBET seed from the scorer's own default (PREREG section 9 item 7, case iii).
# TAGS_FILE (environment) replaces the dataset's full run list (the chain backend scores the latents that have arrived).
# Usage: run_pilot_scoring.sh TASK WORKERS [extra run_stage.py args, e.g. --dry-run]
#   TASK in atac_large immune_hum_mou lung pancreas sim2; WORKERS = scorers at once (about 6 small, 3 large)
set -euo pipefail
TASK=${1:?task}; WORKERS=${2:?scorers at once}
REPO=/home/kendall/experiment_data/wasserstein-critic-deconfounding/code/wcd_prereg-tier12-v3
[ "$(git -C "$REPO" rev-parse HEAD)" = 10a1ca5874fbcdc033b148525b5e71d866306bdb ] || { echo "scorer checkout is not at 10a1ca5" >&2; exit 2; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo "scorer checkout is dirty" >&2; exit 2; }
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
ENVS=/home/kendall/.claude-science/conda/envs
I=$D/jhpce_tier12/pilot5_inputs
sha256sum --quiet -c <<EOF
5dd65550fdb18e5c80e73dcfa731d72bdeb6128bdeac7ffb36df6c54450396ff  $I/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv
EOF
export KMP_AFFINITY=disabled
cd "$REPO"
exec nice -n 10 "$ENVS/scvi-api/bin/python" scripts/run_stage.py \
  --manifest "$I/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv" --experiments A1 --tasks "$TASK" \
  --tags-file "${TAGS_FILE:-$I/pilot5_jhpce_$TASK.tags}" \
  --out-dir "$D/jhpce_tier12/pilot5_jhpce" --prepped-dir "$D/prepped_scib" \
  --fit-python "$ENVS/scvi-api/bin/python" --score-python "$ENVS/wcd-kbet/bin/python" \
  --r-home "$ENVS/wcd-kbet/lib/R" --r-libs "$D/Rlib_kbet" --expect-device L40S \
  --fit-lanes 1 --score-workers-during "$WORKERS" --score-workers-after "$WORKERS" "${@:3}"
