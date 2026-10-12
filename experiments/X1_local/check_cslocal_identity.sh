#!/usr/bin/env bash
# cs-local identity check (X1 local part runs as a cs-local host job, user choice 2026-10-12): the same steps run once in
# the chat sandbox (MODE inchat) and once as a cs-local host job (MODE cslocal); compare_identity.py then requires the
# two to agree. Steps, all from the tag checkout (prereg-tier12-v2, c88cce3) and the pinned R4-a1 manifest:
#   env/      env.json: interpreter, package versions, thread/CUDA/numba env vars, CPU model + flags, GPU, namespaces
#   cpu/      experiments/A2A3_pilot/check_switches.sh (7 one-epoch CPU fits, bit-exact by construction)
#   gpu/      two A1 rows (atac_small, conditioned decoder, seed 100: reference lambda 100, mmd lambda 30) trained 2 epochs on
#             the GPU, twice (rep1, rep2), sequentially
#   score/    scripts/run_stage.py re-scores the two full A1 latents of the same rows (copied in), as the A1 runner did
# Usage: check_cslocal_identity.sh OUT_DIR MODE      (fresh OUT_DIR; exit 0 = every step ran, 1 = a step failed, 2 = refused)
set -euo pipefail
OUT=${1:?usage: check_cslocal_identity.sh OUT_DIR MODE}; MODE=${2:?usage: check_cslocal_identity.sh OUT_DIR MODE}
[[ "$MODE" =~ ^(inchat|cslocal)$ ]] || { echo "MODE must be inchat or cslocal" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "OUT_DIR $OUT exists; use a fresh one" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
REPO=$D/code/wcd_prereg-tier12-v2; ENVS=/home/kendall/.claude-science/conda/envs; E=$ENVS/scvi-api/bin/python; K=$ENVS/wcd-kbet/bin/python
MAN=$D/tier12_v2/manifests/paper_manifest_stock_pilot_u5_b10_v3.r4a1.tsv
MAN_SHA=ca28d991767d85f8fb7f25b8335267676846d70e8c7887ea9cb3e87900c33459
TAGS=(A1_atac_small_reference_l100_c1_s100_697aa0ae A1_atac_small_mmd_l30_c1_s100_e88830bf)
[ "$(git -C "$REPO" rev-parse HEAD)" = c88cce3c569e9de47f2d65c6794ca3890637786a ] && [ -z "$(git -C "$REPO" status --porcelain)" ] \
  || { echo "checkout not clean at c88cce3" >&2; exit 2; }
[ "$(sha256sum "$MAN" | cut -d' ' -f1)" = "$MAN_SHA" ] || { echo "R4-a1 manifest differs from the committed one" >&2; exit 2; }
for t in "${TAGS[@]}"; do [ -f "$D/tier12_v2/A1/latents/$t.npz" ] && [ -f "$D/tier12_v2/A1/scores/$t.csv" ] || { echo "A1 latent/score of $t missing" >&2; exit 2; }; done
mkdir -p "$OUT/env" "$OUT/gpu" "$OUT/score/out/latents"
echo "$MODE" > "$OUT/mode"
: > "$OUT/steps.tsv"
step() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/steps.tsv"; echo "[$(date +%T)] $1 exit $2"; }
# ---- env ----------------------------------------------------------------------------------------------------------
rc=0; KMP_AFFINITY=disabled "$E" "$HERE/env_record.py" --fit-python "$E" --score-python "$K" --r-home "$ENVS/wcd-kbet/lib/R" \
  --r-libs "$D/Rlib_kbet" --out "$OUT/env/env.json" > "$OUT/env/env.log" 2>&1 || rc=$?; step env "$rc"
# ---- cpu: the switch check (1-epoch CPU fits, bit-exact) ---------------------------------------------------------
rc=0; bash "$HERE/../A2A3_pilot/check_switches.sh" "$OUT/cpu" > "$OUT/cpu.log" 2>&1 || rc=$?; step cpu "$rc"
# ---- gpu: two rows x 2 epochs, twice ------------------------------------------------------------------------------
rc=0
"$E" - "$MAN" "$OUT/gpu" "${TAGS[@]}" <<'PY' || rc=$?
import sys, pandas as pd
src, out, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
head = open(src).readline()
m = pd.read_csv(src, sep="\t", comment="#", dtype=str)
sel = m[m.tag.isin(tags)].copy()
assert len(sel) == len(tags), (len(sel), tags)
sel["max_epochs"] = "2"
with open(f"{out}/manifest.tsv", "w") as f:
    f.write(head); sel.to_csv(f, sep="\t", index=False)
PY
for rep in rep1 rep2; do
  for t in "${TAGS[@]}"; do
    mkdir -p "$OUT/gpu/$rep"
    r=0; ( cd "$REPO" && MANIFEST=$OUT/gpu/manifest.tsv PREPPED_DIR=$D/prepped_scib OUT_DIR=$OUT/gpu/$rep WCD_SRC=$REPO/src TAG=$t \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled PYTHONWARNINGS=ignore \
        nice -n 19 "$E" scripts/fit_paper_config.py > "$OUT/gpu/$rep/$t.log" 2>&1 ) || r=$?
    printf '%s\t%s\t%s\n' "$rep" "$t" "$r" >> "$OUT/gpu/fit_rc.tsv"; [ "$r" = 0 ] || rc=1
  done
done
step gpu "$rc"
# ---- score: the runner re-scores the two full A1 latents ------------------------------------------------------------
rc=0
for t in "${TAGS[@]}"; do cp "$D/tier12_v2/A1/latents/$t.npz" "$OUT/score/out/latents/"; echo "$t" >> "$OUT/score/tags.txt"; done
( cd "$REPO" && KMP_AFFINITY=disabled nice -n 19 "$E" scripts/run_stage.py --manifest "$MAN" --experiments A1 --tasks atac_small \
    --tags-file "$OUT/score/tags.txt" --out-dir "$OUT/score/out" --prepped-dir "$D/prepped_scib" --fit-python "$E" --score-python "$K" \
    --r-home "$ENVS/wcd-kbet/lib/R" --r-libs "$D/Rlib_kbet" --expect-device "RTX 3080" --fit-lanes 1 \
    --score-workers-during 1 --score-workers-after 2 > "$OUT/score/runner.log" 2>&1 ) || rc=$?
step score "$rc"
awk -F'\t' '$2 != 0 {bad=1} END {exit bad}' "$OUT/steps.tsv"
