#!/usr/bin/env bash
# A1 extension, R4-a1 round 1 (docs/prereg/r4_a1_record.json, status "extend"; PREREG sec. 4: 1 point per round,
# <= 2 rounds per edge): 9 A1 rows = immune, conditioned decoder (c1), discriminator / pooled / reference at
# lambda 0.03 (matched point on the low edge 0.1), seeds 100-102. Same code, device and settings as A1
# (experiments/A1_lambda_pilot/run_A1.sh at tag prereg-tier12-v2): only the manifest and the output dir differ.
# After it passes its gate, rerun scripts/freeze_matched_lambda.py --stage a1 with this manifest as a second
# --manifest and all three score dirs. Preflight: experiments/A1_ext_r4_round1/PREFLIGHT.md.
# Usage: run_A1_ext.sh [extra run_stage.py args, e.g. --dry-run | --clear-foreign-claims]
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
REPO=$D/code/wcd_prereg-tier12-v2
MAN=$D/tier12_v2/manifests/a1_extension_r4_round1.tsv
MAN_SHA=f3030a6a6a4337cf28a602e379458ab68525fafb4b5dcd6d3674840abe9a59c0
ENVS=/home/kendall/.claude-science/conda/envs
[ "$(git -C "$REPO" rev-parse HEAD)" = c88cce3c569e9de47f2d65c6794ca3890637786a ] || { echo "checkout is not at c88cce3" >&2; exit 2; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "checkout is dirty" >&2; exit 2; }
[ "$(sha256sum "$MAN" | cut -d' ' -f1)" = "$MAN_SHA" ] || { echo "extension manifest differs from the committed one" >&2; exit 2; }
export KMP_AFFINITY=disabled
cd "$REPO"
exec "$ENVS/scvi-api/bin/python" scripts/run_stage.py \
  --manifest "$MAN" --experiments A1 --tasks immune \
  --out-dir "$D/tier12_v2/A1_ext_r4_round1" --prepped-dir "$D/prepped_scib" \
  --fit-python "$ENVS/scvi-api/bin/python" --score-python "$ENVS/wcd-kbet/bin/python" \
  --r-home "$ENVS/wcd-kbet/lib/R" --r-libs "$D/Rlib_kbet" --expect-device "RTX 3080" \
  --fit-lanes 8 --fit-threads 1 --score-workers-during 3 --score-workers-after 8 "$@"
