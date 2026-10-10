#!/usr/bin/env bash
# A2/A3 pilots (PREREG sec. 3; stage order sec. 0: after R4-a1, before R2/R3): the 288 new halves only, A2 = posterior-sample
# adversary input (108 rows: discriminator, pooled, reference), A3 = per-dimension standardisation on (180 rows: + mmd, sinkhorn);
# atac_small + immune, both decoders (SI-27), seeds 100-102, lambda = R4-a1 lo / matched / hi (docs/prereg/r4_a1_record.json).
# The default halves are the A1 fits. Same code, device and settings as A1 (experiments/A1_lambda_pilot/run_A1.sh at tag
# prereg-tier12-v2): only the manifest (R4-a1 resolved), the experiments and the output dir differ. After the gate passes, run
# scripts/decide_a2_a3.py (R2/R3). Preflight: experiments/A2A3_pilot/PREFLIGHT.md; switch check: check_switches.sh.
# Usage: run_A2A3.sh [extra run_stage.py args, e.g. --dry-run | --clear-foreign-claims]
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
REPO=$D/code/wcd_prereg-tier12-v2
MAN=$D/tier12_v2/manifests/paper_manifest_stock_pilot_u5_b10_v3.r4a1.tsv
MAN_SHA=ca28d991767d85f8fb7f25b8335267676846d70e8c7887ea9cb3e87900c33459
ENVS=/home/kendall/.claude-science/conda/envs
[ "$(git -C "$REPO" rev-parse HEAD)" = c88cce3c569e9de47f2d65c6794ca3890637786a ] || { echo "checkout is not at c88cce3" >&2; exit 2; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "checkout is dirty" >&2; exit 2; }
[ "$(sha256sum "$MAN" | cut -d' ' -f1)" = "$MAN_SHA" ] || { echo "resolved manifest differs from the committed one" >&2; exit 2; }
export KMP_AFFINITY=disabled
cd "$REPO"
exec "$ENVS/scvi-api/bin/python" scripts/run_stage.py \
  --manifest "$MAN" --experiments A2 A3 --tasks atac_small immune \
  --out-dir "$D/tier12_v2/A2A3" --prepped-dir "$D/prepped_scib" \
  --fit-python "$ENVS/scvi-api/bin/python" --score-python "$ENVS/wcd-kbet/bin/python" \
  --r-home "$ENVS/wcd-kbet/lib/R" --r-libs "$D/Rlib_kbet" --expect-device "RTX 3080" \
  --fit-lanes 8 --fit-threads 1 --score-workers-during 3 --score-workers-after 8 "$@"
