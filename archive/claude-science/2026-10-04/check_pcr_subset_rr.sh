#!/usr/bin/env bash
# Re-review check of PREREG sec 9 (CR-01): X3 pancreas uncorrected HVG-PCA latent on nested_v1 cells of dose 0 and
# dose 50; PCR_batch with the all-feature reference vs the tagged HVG reference (regress_pcr_reference.py), and the
# review-commit scorer's PCR_batch on the dose-50 latent. Read-only on prepped files; outputs to OUT.
# Usage: check_pcr_subset_rr.sh REPO OUT TAG_DOSE0 TAG_DOSE50
set -euo pipefail
REPO=$(realpath $1); OUT=$(realpath -m $2); T0=$3; T50=$4; mkdir -p $OUT
C=/home/kendall/.claude-science/conda/envs; D=/home/kendall/experiment_data/wasserstein-critic-deconfounding; P=$D/prepped_scib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled CUDA_VISIBLE_DEVICES= PYTHONWARNINGS=ignore
export R_HOME=$C/wcd-kbet/lib/R R_LIBS=$D/Rlib_kbet
MAN=$REPO/manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv
for t in $T0 $T50; do
  nice -n 19 $C/wcd-kbet/bin/python $REPO/scripts/regress_pcr_reference.py make-latent --prepped $P/pancreas__scib.h5ad --manifest $MAN --tag $t --out $OUT/$t.npz
done
# scorer on the dose-50 latent (as run_stage would call it)
L=$(sha256sum $OUT/$T50.npz | cut -d' ' -f1)
( cd $OUT && PREPPED=$P/pancreas__scib.h5ad NPZ=$OUT/$T50.npz TAG=$T50 OUT_CSV=$OUT/$T50.score.csv \
  META="{\"latent_sha256\": \"$L\"}" nice -n 19 $C/wcd-kbet/bin/python $REPO/scripts/score_scib_native.py > $OUT/$T50.score.log 2>&1 )
nice -n 19 $C/wcd-kbet/bin/python $REPO/scripts/regress_pcr_reference.py check --prepped $P/pancreas__scib.h5ad --npz $OUT/$T50.npz --score $OUT/$T50.score.csv --out $OUT/$T50.pcr_reference.csv || echo "check exit $?"
# dose 0: reference values only (fake score row so check prints the table; its exit code is not used)
printf 'tag,PCR_batch\n%s,nan\n' $T0 > $OUT/$T0.fake_score.csv
nice -n 19 $C/wcd-kbet/bin/python $REPO/scripts/regress_pcr_reference.py check --prepped $P/pancreas__scib.h5ad --npz $OUT/$T0.npz --score $OUT/$T0.fake_score.csv --out $OUT/$T0.pcr_reference.csv || true
