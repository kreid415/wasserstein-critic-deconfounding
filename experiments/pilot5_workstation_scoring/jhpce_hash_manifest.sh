#!/bin/bash
# JHPCE helper job (1 CPU, 1 GB, shared partition): list and sha256 the 4 finished pilot datasets' files (latents, models,
# attempts, ledger views, failures.csv) of /fastscratch/.../tier12/pilot5_jhpce. Output: ws_mirror_20261028/pilot5_4ds.{files,sha256}
# (manifest sha256 87bb6d74d62c3e131a34bec33c16356c4d7fee9bf962a8caa7a739816252b9c0, 4,551 files on 2026-10-28).
#SBATCH --partition=shared
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
set -euo pipefail
P=/fastscratch/myscratch/kreid/wcd/tier12/pilot5_jhpce
cd $P
OUT=/fastscratch/myscratch/kreid/wcd/ws_mirror_20261028
mkdir -p $OUT
rm -f $OUT/pilot5_4ds.sha256 $OUT/pilot5_4ds.files
for t in pancreas sim2 lung immune_hum_mou; do
  find latents models attempts -name "A1_${t}_*" -type f
  find models -path "models/A1_${t}_*" -type f
done | sort -u > $OUT/pilot5_4ds.files
find ledger -type f | sort >> $OUT/pilot5_4ds.files
echo failures.csv >> $OUT/pilot5_4ds.files
xargs -a $OUT/pilot5_4ds.files -d '\n' sha256sum > $OUT/pilot5_4ds.sha256
sha256sum $OUT/pilot5_4ds.sha256
