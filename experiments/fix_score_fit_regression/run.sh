#!/usr/bin/env bash
# Regression of the fix-score-fit scorer against the tagged scorer (prereg-tier12-v1), lead brief 2026-10-03.
# Usage: run.sh REPO TAGGED_WORKTREE OUT
#   (i)   score two A1 atac_small latents (copied read-only from tier12/A1 into OUT/A1) with the patched scorer;
#         every metric column must equal the A1 score row exactly
#   (ii)  fit A1_immune_none_l0_c1_s100 for 3 epochs on CPU with the TAGGED fitter (OUT/ii/manifest_immune_e3.tsv),
#         score it with the tagged and the patched scorer; every metric column identical
#   (iii) X3 pancreas dose-50 cells (manifest v2 row in OUT/iii/tag.txt), uncorrected HVG-PCA latent, patched
#         scorer; its PCR_batch must equal the all-feature-reference value recomputed on the subset
# Environment as scripts/run_stage.py scores (thread_env(1), CUDA hidden), nice -n 19, one process at a time.
# STAGES (env, default "i ii iii") selects the stages to run.
set -euo pipefail
REPO=$(realpath "$1"); TAGREPO=$(realpath "$2"); OUT=$(realpath "$3")
KBET=/home/kendall/.claude-science/conda/envs/wcd-kbet; SCVI=/home/kendall/.claude-science/conda/envs/scvi-api/bin/python
P=/home/kendall/experiment_data/wasserstein-critic-deconfounding/prepped_scib
export R_HOME=$KBET/lib/R R_LIBS=/home/kendall/experiment_data/wasserstein-critic-deconfounding/Rlib_kbet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" KBET_SEED=0
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"     # patched scorer from a clean tree
test -z "$(git -C "$TAGREPO" status --porcelain --untracked-files=no)"
echo "[regress] patched $(git -C "$REPO" rev-parse --short HEAD), tagged $(git -C "$TAGREPO" rev-parse --short HEAD)"
meta() { "$KBET/bin/python" -c "import json,sys,pandas as pd; r=pd.read_csv(sys.argv[1]).iloc[0]; print(json.dumps({k: (str(r[k]) if k in ('latent_sha256','experiment','task','arm','lam') else int(r[k])) for k in ['latent_sha256','experiment','task','arm','lam','cond','seed']}))" "$1"; }
score() {  # score SCORER PREPPED NPZ TAG META OUT_CSV  (an existing OUT_CSV from an earlier run is kept: resume)
  if [ -s "$6" ]; then echo "[regress] keep existing $6"; return 0; fi
  PREPPED="$2" NPZ="$3" TAG="$4" META="$5" OUT_CSV="$6" nice -n 19 "$KBET/bin/python" "$1" > "$6.log" 2>&1
}
fail=0
STAGES=" ${STAGES:-i ii iii} "
# ---- (i)
if [[ "$STAGES" == *" i "* ]]; then
mkdir -p "$OUT/i"
for t in A1_atac_small_none_l0_c1_s100_8d38e2a3 A1_atac_small_discriminator_l3_c1_s100_c2822374; do
  score "$REPO/scripts/score_scib_native.py" "$P/atac_small__scib.h5ad" "$OUT/A1/latents/$t.npz" "$t" "$(meta "$OUT/A1/scores/$t.csv")" "$OUT/i/$t.csv"
  "$KBET/bin/python" "$REPO/scripts/compare_score_rows.py" "$OUT/A1/scores/$t.csv" "$OUT/i/$t.csv" --out "$OUT/i/$t.compare.csv" || fail=1
done
fi
# ---- (ii)
if [[ "$STAGES" == *" ii "* ]]; then
mkdir -p "$OUT/ii/scores_tagged" "$OUT/ii/scores_patched"
t=REG_immune_none_e3
if [ ! -f "$OUT/ii/fit/latents/$t.npz" ]; then
  MANIFEST="$OUT/ii/manifest_immune_e3.tsv" TAG="$t" PREPPED_DIR="$P" OUT_DIR="$OUT/ii/fit" WCD_SRC="$TAGREPO/src" \
    nice -n 19 "$SCVI" "$TAGREPO/scripts/fit_paper_config.py" > "$OUT/ii/fit.log" 2>&1
fi
LSHA=$(sha256sum "$OUT/ii/fit/latents/$t.npz" | cut -d' ' -f1)
M=$(printf '{"latent_sha256": "%s", "experiment": "REG", "task": "immune", "arm": "none", "lam": "0", "cond": 1, "seed": 100}' "$LSHA")
score "$TAGREPO/scripts/score_scib_native.py" "$P/immune__scib.h5ad" "$OUT/ii/fit/latents/$t.npz" "$t" "$M" "$OUT/ii/scores_tagged/$t.csv"
score "$REPO/scripts/score_scib_native.py" "$P/immune__scib.h5ad" "$OUT/ii/fit/latents/$t.npz" "$t" "$M" "$OUT/ii/scores_patched/$t.csv"
"$KBET/bin/python" "$REPO/scripts/compare_score_rows.py" "$OUT/ii/scores_tagged/$t.csv" "$OUT/ii/scores_patched/$t.csv" --out "$OUT/ii/$t.compare.csv" || fail=1
fi
# ---- (iii)
if [[ "$STAGES" == *" iii "* ]]; then
t=$(cat "$OUT/iii/tag.txt")
nice -n 19 "$KBET/bin/python" "$REPO/scripts/regress_pcr_reference.py" make-latent --prepped "$P/pancreas__scib.h5ad" \
  --manifest "$REPO/manifests/paper_manifest_stock_pilot_u5_b10_v2.tsv" --tag "$t" --out "$OUT/iii/$t.npz"
LSHA=$(sha256sum "$OUT/iii/$t.npz" | cut -d' ' -f1)
M=$(printf '{"latent_sha256": "%s", "experiment": "X3", "task": "pancreas", "arm": "none", "lam": "0", "cond": 1, "seed": 10}' "$LSHA")
score "$REPO/scripts/score_scib_native.py" "$P/pancreas__scib.h5ad" "$OUT/iii/$t.npz" "$t" "$M" "$OUT/iii/$t.csv"
nice -n 19 "$KBET/bin/python" "$REPO/scripts/regress_pcr_reference.py" check --prepped "$P/pancreas__scib.h5ad" \
  --npz "$OUT/iii/$t.npz" --score "$OUT/iii/$t.csv" --out "$OUT/iii/$t.pcr_reference.csv" || fail=1
fi
echo "[regress] done, fail=$fail"
exit $fail
