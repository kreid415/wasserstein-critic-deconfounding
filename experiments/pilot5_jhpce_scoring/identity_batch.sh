#!/bin/bash
# First JHPCE batch of the fixed-seed scoring: the identity check (SI-115). Writes one Slurm array file per size class for 5 latents that
# already have workstation scores at scorer 10a1ca5 plus 1 immune_hum_mou latent (scored on the workstation for this check), through the same
# wrapper, scorer checkout, preload library and node type as the production arrays. Run on the JHPCE login node (text files only), then
# `sbatch` each printed file. Compare with experiments/pilot5_jhpce_scoring/identity_check.py.
# Usage: identity_batch.sh WORKDIR HUMMOU_LIST_LINE_FILE
#   HUMMOU_LIST_LINE_FILE: one list line (tag, latent npz, prepped h5ad, meta json; the format of cluster/jhpce/chain_score.py batches)
set -euo pipefail
W=${1:?workdir}; HM=${2:?immune_hum_mou list line file}
S=/fastscratch/myscratch/kreid/wcd; REPO=${WCD_REPO:?deployed repo (cluster/jhpce/score_sizes.json)}
SCORER=$S/scorer_10a1ca5; PLAN=$S/score_check/plan
mkdir -p "$W/logs"
TAGS="A1_atac_small_none_l0_c1_s100_8d38e2a3 A1_atac_small_discriminator_l10_c0_s101_b841ed69 A1_immune_sinkhorn_l3_c1_s101_6c0fba12 A1_sim1_sinkhorn_l30_c0_s100_ebcbe68a"
: > "$W/list_atac_small.tsv"; : > "$W/list_immune.tsv"; : > "$W/list_sim1.tsv"
for t in $TAGS; do
  case $t in A1_atac_small_*) f=atac_small;; A1_immune_*) f=immune;; A1_sim1_*) f=sim1;; esac
  line=$(grep -h "^$t"$'\t' "$PLAN"/list_$f.tsv || true)
  [ "$(printf '%s\n' "$line" | grep -c .)" = 1 ] || { echo "FATAL: $t not found exactly once in $PLAN/list_$f.tsv" >&2; exit 1; }
  printf '%s\n' "$line" >> "$W/list_$f.tsv"
done
cp "$HM" "$W/list_immune_hum_mou.tsv"
for f in atac_small immune sim1 immune_hum_mou; do
  read -r MEM WALL < <(python3 -c "import json,sys; d=json.load(open('$REPO/cluster/jhpce/score_sizes.json'))['$f']; print(d['mem_gb'], d['walltime'])")
  N=$(wc -l < "$W/list_$f.tsv")
  cat > "$W/score_$f.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=wcdident-$f
#SBATCH --partition=shared
#SBATCH --account=jhpce
#SBATCH --array=1-$N
#SBATCH --cpus-per-task=1
#SBATCH --mem=${MEM}G
#SBATCH --time=$WALL
#SBATCH --no-requeue
#SBATCH --constraint=znver2
#SBATCH --output=$W/logs/slurm-%x_%A_%a.out
set -euo pipefail
export SCORER_REPO=$SCORER FIXED_SEED_LIB=$S/lisi_fix/fixed_random_device.so
exec bash $REPO/cluster/jhpce/score_array_fixed_seed.sh 10a1ca5874fbcdc033b148525b5e71d866306bdb $W/list_$f.tsv $W/out
EOF
  echo "$W/score_$f.sbatch ($N latents, ${MEM}G, $WALL)"
done
