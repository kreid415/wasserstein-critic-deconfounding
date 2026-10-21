#!/usr/bin/env bash
# X1 local part (PREREG sec 0: X1 after R2/R3): the 1,125 X1 rows of the local tasks atac_small, immune and sim1 (SI-30),
# all arms incl. none / scvi_adv; lambda from R1, adv_input and zstd from R2/R3 (docs/prereg/r2r3_record.json).
# Retargeted 2026-10-17 from prereg-tier12-v2 (c88cce3) to prereg-tier12-v3 (10a1ca5): runs from a clean checkout pinned at the v3 tag
# (SI-54: non-Sinkhorn outputs already in the output folder that were made at c88cce3 count as v3; Sinkhorn output at c88cce3 is refused).
# Same device and settings as A1-A3 (RTX 3080; 8 lanes, 1 thread; 3 scorers during fits, 8 after); only the manifest, the experiment
# and the output dir differ. Preflight: experiments/X1/PREFLIGHT.md (to be redone for v3 and the redone rules).
# MAN_SHA pins the resolved X1 manifest made by stage 2 of the Sinkhorn redo (R1, R4-a1, R2/R3; rules/r23/m.x1.tsv, set 2026-10-20 at
# experiments/sinkhorn_redo_stage2/PLAN.md step X); the launcher refuses any other manifest.
# Output folder: NEW, $D/tier12_v3/X1_local (decision D-S2-3). $D/tier12_v2/X1_local holds the 19 pre-redo scored rows (hashed under
# SI-79, never written to); the runner refuses outputs recorded for other settings and Sinkhorn output made at c88cce3, so it is
# not reused as the output folder. Reusable non-Sinkhorn outputs are linked in by name first (link_v2_outputs.py; PLAN.md step X2).
# Usage: run_X1_local.sh [--dry-run | --report-only | --clear-stale-claims | --clear-foreign-claims]
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
REPO=$D/code/wcd_prereg-tier12-v3
TAG_COMMIT=10a1ca5874fbcdc033b148525b5e71d866306bdb
MAN=$D/tier12_v3/manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv
MAN_SHA="034403eb206f77296bc857e5776a84e22e6a172f787913875a9ddc159e1fa6c8"
ENVS=/home/kendall/.claude-science/conda/envs
EXITLOG=$D/tier12_v3/logs/X1_local_exit.txt
die() { echo "$1" >&2; exit "${2:-2}"; }
PASS=()
for a in "$@"; do
  case "$a" in
    --dry-run|--report-only|--clear-stale-claims|--clear-foreign-claims) PASS+=("$a") ;;
    *) die "argument $a is not allowed (never --allow-dirty or --allow-mixed-sha: SI-85, SI-89)" ;;
  esac
done
[[ "$MAN_SHA" =~ ^[0-9a-f]{64}$ ]] || die "MAN_SHA is not set: build the resolved X1 manifest after stage 2 and put its sha256 here"
[ "$(git -C "$REPO" rev-parse HEAD)" = "$TAG_COMMIT" ] || die "checkout is not at $TAG_COMMIT"
[ "$(git -C "$REPO" describe --tags --exact-match)" = prereg-tier12-v3 ] || die "HEAD is not tag prereg-tier12-v3"
[ -z "$(git -C "$REPO" status --porcelain)" ] || die "checkout is dirty"
[ "$(sha256sum "$MAN" | cut -d' ' -f1)" = "$MAN_SHA" ] || die "resolved manifest differs from the pinned one"
if pgrep -f 'run_stage[.]py' >/dev/null; then die "another run_stage.py is alive" 3; fi
mkdir -p "$D/tier12_v3/logs"
export KMP_AFFINITY=disabled
cd "$REPO"
rc=0
"$ENVS/scvi-api/bin/python" scripts/run_stage.py \
  --manifest "$MAN" --experiments X1 --tasks atac_small immune sim1 \
  --out-dir "$D/tier12_v3/X1_local" --prepped-dir "$D/prepped_scib" \
  --fit-python "$ENVS/scvi-api/bin/python" --score-python "$ENVS/wcd-kbet/bin/python" \
  --r-home "$ENVS/wcd-kbet/lib/R" --r-libs "$D/Rlib_kbet" --expect-device "RTX 3080" \
  --fit-lanes 8 --fit-threads 1 --score-workers-during 3 --score-workers-after 8 ${PASS[@]+"${PASS[@]}"} || rc=$?
echo "X1_local exit $rc $(date -Is)" >> "$EXITLOG"
exit "$rc"
