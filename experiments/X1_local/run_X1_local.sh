#!/usr/bin/env bash
# X1 local part (PREREG sec 0: X1 after R2/R3): the 1,125 X1 rows of the local tasks atac_small, immune and sim1 (SI-30),
# all arms incl. none / scvi_adv; lambda from R1, adv_input mean and zstd 0 from R2/R3 (docs/prereg/r2r3_record.json).
# Same code, device and settings as A1-A3 (tag prereg-tier12-v2 checkout; RTX 3080; 8 lanes, 1 thread; 3 scorers during
# fits, 8 after): only the manifest (R2/R3-resolved, pinned by SHA-256), the experiment and the output dir differ.
# Runs as a cs-local host job (user choice 2026-10-12) after experiments/X1_local identity check; preflight
# experiments/X1/PREFLIGHT.md. Usage: run_X1_local.sh [extra run_stage.py args, e.g. --dry-run | --clear-foreign-claims]
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
REPO=$D/code/wcd_prereg-tier12-v2
MAN=$D/tier12_v2/manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv
MAN_SHA=7c2fe8679631c882abf6a4d020c6c3ba7f327261903bf47fc968e6cf80feaf85
ENVS=/home/kendall/.claude-science/conda/envs
[ "$(git -C "$REPO" rev-parse HEAD)" = c88cce3c569e9de47f2d65c6794ca3890637786a ] || { echo "checkout is not at c88cce3" >&2; exit 2; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "checkout is dirty" >&2; exit 2; }
[ "$(sha256sum "$MAN" | cut -d' ' -f1)" = "$MAN_SHA" ] || { echo "resolved manifest differs from the committed one" >&2; exit 2; }
export KMP_AFFINITY=disabled
cd "$REPO"
exec "$ENVS/scvi-api/bin/python" scripts/run_stage.py \
  --manifest "$MAN" --experiments X1 --tasks atac_small immune sim1 \
  --out-dir "$D/tier12_v2/X1_local" --prepped-dir "$D/prepped_scib" \
  --fit-python "$ENVS/scvi-api/bin/python" --score-python "$ENVS/wcd-kbet/bin/python" \
  --r-home "$ENVS/wcd-kbet/lib/R" --r-libs "$D/Rlib_kbet" --expect-device "RTX 3080" \
  --fit-lanes 8 --fit-threads 1 --score-workers-during 3 --score-workers-after 8 "$@"
