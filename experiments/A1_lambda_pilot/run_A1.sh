#!/usr/bin/env bash
# A1 lambda pilot (docs/PREREG.md sec 0) on the local RTX 3080. Preflight of the restart: experiments/A1_lambda_pilot_v2/
# PREFLIGHT.md (GO); experiments/A1_lambda_pilot/PREFLIGHT.md is the record of the discarded v1 launch.
# Run from a worktree pinned at tag prereg-tier12-v2, never a later commit (PREREG sec 9: every stage shares one HEAD, the
# scorer provenance; A1 restarts at the post-fix tag, SI-43); the runner refuses a
# dirty tree and mixed commits. Outputs go to tier12_v2/A1, a NEW directory: tier12/A1 holds the 202 discarded v1 fits.
# Resume after an interruption: rerun this script (valid outputs are skipped); a reaped sandbox leaves "foreign" claims,
# clear them with --clear-foreign-claims only after checking that no runner is alive (ledger mtimes).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
ENVS=/home/kendall/.claude-science/conda/envs
MAN=$REPO/manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv
export KMP_AFFINITY=disabled
exec "$ENVS/scvi-api/bin/python" "$REPO/scripts/run_stage.py" \
  --manifest "$MAN" --experiments A1 --tasks atac_small immune sim1 \
  --out-dir "$D/tier12_v2/A1" --prepped-dir "$D/prepped_scib" \
  --fit-python "$ENVS/scvi-api/bin/python" --score-python "$ENVS/wcd-kbet/bin/python" \
  --r-home "$ENVS/wcd-kbet/lib/R" --r-libs "$D/Rlib_kbet" --expect-device "RTX 3080" \
  --fit-lanes 8 --fit-threads 1 --score-workers-during 3 --score-workers-after 8 "$@"
