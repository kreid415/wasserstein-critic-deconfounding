#!/usr/bin/env bash
# Local scoring of the 250 JHPCE filler latents (X1 none/scvi_adv + X13 scanvi/sysvi; preflight GO in this
# directory, PREFLIGHT.md). Runs scripts/run_stage.py from a clean checkout of tag prereg-tier12-v2 (c88cce3),
# score only: the dry runs plan 0 fits, and any refit is refused here because --expect-device L40S is required
# before a fit and the local GPU is an RTX 3080 (SI-17). One scorer per pass at nice 19 beside A1 (PF-18:
# measured peak RSS 24.8 GB atac_large, 19.4 GB immune_hum_mou).
# Usage: run_scoring.sh TAG_CHECKOUT A|B [extra run_stage.py args, e.g. --dry-run]
set -euo pipefail
REPO=$1; PASS=$2
[ "$(git -C "$REPO" rev-parse HEAD)" = c88cce3c569e9de47f2d65c6794ca3890637786a ] || { echo "checkout is not at c88cce3" >&2; exit 2; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "checkout is dirty" >&2; exit 2; }
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
ENVS=/home/kendall/.claude-science/conda/envs
case $PASS in
  A) TASKS="atac_large immune_hum_mou"; TAGS=cluster/jhpce/tags/fillers_x1_x13_jobA.tags ;;
  B) TASKS="lung pancreas sim2"; TAGS=cluster/jhpce/tags/fillers_x1_x13_jobB.tags ;;
  *) echo "pass must be A or B" >&2; exit 2 ;;
esac
export KMP_AFFINITY=disabled
cd "$REPO"
# shellcheck disable=SC2086  # TASKS is a word list on purpose
exec nice -n 19 "$ENVS/scvi-api/bin/python" scripts/run_stage.py \
  --manifest manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv --experiments X1 X13 --tasks $TASKS --tags-file "$TAGS" \
  --out-dir "$D/jhpce_tier12/fillers_x1_x13" --prepped-dir "$D/prepped_scib" \
  --fit-python "$ENVS/scvi-api/bin/python" --score-python "$ENVS/wcd-kbet/bin/python" \
  --r-home "$ENVS/wcd-kbet/lib/R" --r-libs "$D/Rlib_kbet" --expect-device L40S \
  --fit-lanes 1 --score-workers-during 1 --score-workers-after 1 "${@:3}"
