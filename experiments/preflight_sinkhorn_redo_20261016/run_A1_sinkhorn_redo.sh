#!/usr/bin/env bash
# Stage 1 of the Sinkhorn redo (SI-57, SI-51): refit and rescore the 180 A1 Sinkhorn rows with the corrected Sinkhorn code.
# Preflight record: experiments/preflight_sinkhorn_redo_20261016/PREFLIGHT.md (the committed copy of this file is a
# reference; INSTALL IT OUTSIDE the pinned checkout, e.g. D/code/run_A1_sinkhorn_redo.sh, so the checkout stays clean).
# Runs from a clean checkout pinned at tag prereg-tier12-v3 (commit 10a1ca5), never a later commit (SI-54, SI-85, SI-89).
# New output directory D/tier12_v3/A1_sinkhorn: tier12_v2/A1 holds 180 pre-fix Sinkhorn outputs made at c88cce3, which the
# runner refuses (SI-54) and which must stay untouched.
# Usage (tmux, so a platform restart cannot kill it):  tmux new -d -s a1sk 'bash D/code/run_A1_sinkhorn_redo.sh'
# Extra run_stage.py arguments (e.g. --dry-run) are passed through.
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
REPO=$D/code/wcd_prereg-tier12-v3
TAG_COMMIT=10a1ca5874fbcdc033b148525b5e71d866306bdb
TAGS=$D/tier12_v3/manifests/sinkhorn_a1_tags.txt
TAGS_SHA=a19e4683640aff7922d4124afff8f04d1a492ba83fbf9444d113ddc8f41b6ebf
MAN=$REPO/manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv
MAN_SHA=6efbe422c8f09f9884001a84f7d7b363506f435803d238280e00bbb7b7e1cf24
ENVS=/home/kendall/.claude-science/conda/envs
OUT=$D/tier12_v3/A1_sinkhorn
[ "$(git -C "$REPO" rev-parse HEAD)" = "$TAG_COMMIT" ] || { echo "checkout is not at $TAG_COMMIT" >&2; exit 2; }
[ "$(git -C "$REPO" describe --tags --exact-match)" = prereg-tier12-v3 ] || { echo "HEAD is not tag prereg-tier12-v3" >&2; exit 2; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "checkout is dirty" >&2; exit 2; }
[ "$(sha256sum "$TAGS" | cut -d' ' -f1)" = "$TAGS_SHA" ] || { echo "tags file differs from the committed one" >&2; exit 2; }
[ "$(sha256sum "$MAN" | cut -d' ' -f1)" = "$MAN_SHA" ] || { echo "manifest differs from the committed one" >&2; exit 2; }
if pgrep -f scripts/run_stage.py >/dev/null; then echo "another run_stage.py is alive" >&2; exit 3; fi
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$mem" -lt 500 ] || { echo "GPU memory in use (${mem} MiB): stop the other user first" >&2; exit 3; }
mkdir -p "$D/tier12_v3/logs"
LOG=$D/tier12_v3/logs/A1_sinkhorn_runner_$(date +%Y%m%dT%H%M%S).log
echo "A1 sinkhorn redo $(date -Is) | $(git -C "$REPO" describe --tags --exact-match) | log $LOG"
export KMP_AFFINITY=disabled
cd "$REPO"
exec "$ENVS/scvi-api/bin/python" scripts/run_stage.py \
  --manifest "$MAN" --experiments A1 --tasks atac_small immune sim1 --tags-file "$TAGS" \
  --out-dir "$OUT" --prepped-dir "$D/prepped_scib" \
  --fit-python "$ENVS/scvi-api/bin/python" --score-python "$ENVS/wcd-kbet/bin/python" \
  --r-home "$ENVS/wcd-kbet/lib/R" --r-libs "$D/Rlib_kbet" --expect-device "RTX 3080" \
  --fit-lanes 8 --fit-threads 1 --score-workers-during 3 --score-workers-after 8 "$@" > "$LOG" 2>&1
