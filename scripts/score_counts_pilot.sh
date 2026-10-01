#!/bin/bash
# Score the counts pilot with scIB native (score_scib_native.py), one row file per tag.
# Two input sets, each scored against ITS OWN unintegrated prep (PCR/HVG reference):
#   NEW  = pilot latents (real counts)            vs prepped/atac_small_prepped.h5ad
#   OLD  = final-sweep atac_small nl uncond latents at the same lambdas/seeds (log-normalised
#          input)                                  vs prepped/atac_small_prepped_OLDLOG.h5ad
# plus one 'unintegrated' row per set (the prep's own X_pca as the embedding).
# Env: E (wcd-kbet env root), RLIB (durable R lib with kBET), PILOT, OLD_EMB, NWORK.
set -u
cd "$(dirname "$0")/.."
P="${PILOT:?}"; OLD="${OLD_EMB:?}"; NWORK="${NWORK:-6}"
ROWS="$P/rows_scib"; mkdir -p "$ROWS"
export PY="$E/bin/python" R_HOME="$E/lib/R" R_LIBS="${RLIB:?}"
export KMP_AFFINITY=disabled OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore

mk_unint() {  # prepped -> npz holding its X_pca as the embedding
  [ -s "$2" ] || "$PY" - "$1" "$2" <<'EOF'
import sys, os, numpy as np, h5py
src, out = sys.argv[1:3]
with h5py.File(src, "r") as f:
    z = f["obsm"]["X_pca"][:]
    def col(k):
        g = f["obs"][k]; return np.asarray(g["categories"][:]).astype(str)[g["codes"][:]]
    b, c = col("batch"), col("celltype")
np.savez(out + ".tmp.npz", z=z.astype(np.float32), batch=b, celltype=c); os.replace(out + ".tmp.npz", out)
EOF
}
NEWP="$P/prepped/atac_small_prepped.h5ad"; OLDP="$P/prepped/atac_small_prepped_OLDLOG.h5ad"
mk_unint "$NEWP" "$P/embeddings/atac_small_XC_unintegrated_s0.npz"
mkdir -p "$P/old_unint"; mk_unint "$OLDP" "$P/old_unint/atac_small_XZ_unintegrated_s0.npz"

TASKS="$ROWS/.tasks"; : > "$TASKS"
for f in "$P"/embeddings/*.npz; do echo -e "new\t$NEWP\t$f" >> "$TASKS"; done
echo -e "old\t$OLDP\t$P/old_unint/atac_small_XZ_unintegrated_s0.npz" >> "$TASKS"
for f in "$OLD"/atac_small_XZ_nl_uncond_{none,discriminator,reference,pooled}_lam{0,1,5,20,100}_s{0,1,2}.npz; do
  [ -f "$f" ] && echo -e "old\t$OLDP\t$f" >> "$TASKS"; done

score_one() {
  local set=$1 prep=$2 npz=$3 tag; tag=$(basename "$npz" .npz)
  [ -s "$ROWS/$tag.csv" ] && return
  local adv lam seed cond
  seed=${tag##*_s}
  if [[ $tag == *_stock_scvi_* ]]; then adv=scvi_stock; lam=0; cond=1
  elif [[ $tag == *_unintegrated_* ]]; then adv=unintegrated; lam=0; cond=-1
  else
    adv=$(sed -E 's/.*_(cond|uncond)_([a-z]+)_lam.*/\2/' <<< "$tag")
    lam=$(sed -E 's/.*_lam([0-9.]+)_s.*/\1/' <<< "$tag")
    [[ $tag == *_uncond_* ]] && cond=0 || cond=1
  fi
  PREPPED="$prep" NPZ="$npz" TAG="$tag" ORGANISM=none OUT_CSV="$ROWS/$tag.csv" \
  META="{\"dataset\":\"atac_small_$set\",\"input\":\"$set\",\"adv\":\"$adv\",\"lam\":$lam,\"cond\":$cond,\"seed\":$seed}" \
    "$PY" scripts/score_scib_native.py > "$ROWS/$tag.log" 2>&1 || echo "[FAIL] $tag"
}
export -f score_one; export ROWS
[ -n "${ONLY:-}" ] && { grep -P "^${ONLY}\t" "$TASKS" > "$TASKS.f"; mv "$TASKS.f" "$TASKS"; }
echo "[score] $(wc -l < "$TASKS") latents, $NWORK workers"
tr '\t' ' ' < "$TASKS" | xargs -P "$NWORK" -n 3 bash -c 'score_one "$0" "$1" "$2"'
echo "scored $(ls "$ROWS"/*.csv | wc -l) of $(wc -l < "$TASKS")"
