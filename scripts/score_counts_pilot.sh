#!/bin/bash
# Score every counts-pilot latent through the SAME scorer as the final sweep (score_final_config.py),
# one row file per tag (no shared-file appends), plus an 'unintegrated' row = PCA of the new prep,
# which is the pre-integration reference for scIB's pcr_comparison (applied post-hoc in analysis).
# Env: KBET_PY (scoring env python), PILOT (pilot root with prepped/ and embeddings/), NWORK.
set -u
cd "$(dirname "$0")/.."
PY="${KBET_PY:?}"; P="${PILOT:?}"; NWORK="${NWORK:-6}"
ROWS="$P/rows"; mkdir -p "$ROWS"
export KMP_AFFINITY=disabled OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore

# unintegrated reference latent from the new prep's X_pca (written atomically)
U="$P/embeddings/atac_small_XC_unintegrated_s0.npz"
[ -s "$U" ] || "$PY" - "$P/prepped/atac_small_prepped.h5ad" "$U" <<'EOF'
import sys, os, numpy as np, h5py
src, out = sys.argv[1:3]
with h5py.File(src, "r") as f:
    z = f["obsm"]["X_pca"][:]
    def col(k):
        g = f["obs"][k]; return np.asarray(g["categories"][:]).astype(str)[g["codes"][:]]
    b, c = col("batch"), col("celltype")
np.savez(out + ".tmp.npz", z=z.astype(np.float32), batch=b, celltype=c); os.replace(out + ".tmp.npz", out)
EOF

score_one() {
  local npz=$1 tag; tag=$(basename "$npz" .npz)
  [ -s "$ROWS/$tag.csv" ] && return
  local adv lam seed cond
  seed=${tag##*_s}
  if [[ $tag == *_XC_stock_scvi_* ]]; then adv=scvi_stock; lam=0; cond=1
  elif [[ $tag == *_XC_unintegrated_* ]]; then adv=unintegrated; lam=0; cond=1
  else
    adv=$(sed -E 's/.*_(cond|uncond)_([a-z]+)_lam.*/\2/' <<< "$tag")
    lam=$(sed -E 's/.*_lam([0-9.]+)_s.*/\1/' <<< "$tag")
    [[ $tag == *_uncond_* ]] && cond=0 || cond=1
  fi
  NPZ="$npz" TAG="$tag" DATASET=atac_small ADV="$adv" LAM="$lam" DEC="nl_c$cond" SEED="$seed" \
    OUT_CSV="$ROWS/$tag.csv.tmp" "$PY" scripts/score_final_config.py > "$ROWS/$tag.log" 2>&1 \
    && mv "$ROWS/$tag.csv.tmp" "$ROWS/$tag.csv" || echo "[FAIL] $tag"
}
export -f score_one; export PY ROWS
ls "$P"/embeddings/*.npz | xargs -P "$NWORK" -I {} bash -c 'score_one "{}"'
echo "scored $(ls "$ROWS"/*.csv | wc -l) of $(ls "$P"/embeddings/*.npz | wc -l) latents"
