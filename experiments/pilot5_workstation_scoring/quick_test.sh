#!/usr/bin/env bash
# Quick test before the full scoring (same scorer launcher as the real run: scripts/run_stage.py at 10a1ca5, wcd-kbet, thread_env(1)).
#  T1 two latents that already have workstation scores (A1 atac_small) are rescored: every metric column must be identical;
#  T2 one new pilot latent (pancreas) is scored twice into two folders: every metric column must be identical.
# Compared with compare_scores.py (verbatim copy of cluster/jhpce/compare_scores.py at origin/pilot5-amend a4989ff; its tests are there) (bit-identical float64 in 19 metric columns, same latent hash, same package versions).
# Exit 0 only if both pass.
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
REPO=$D/code/wcd_prereg-tier12-v3; ENVS=/home/kendall/.claude-science/conda/envs
I=$D/jhpce_tier12/pilot5_inputs; PW=${PWS:-/home/kendall/wcd_wt/pws}
T=${1:?test dir}; mkdir -p $T; cd $T
OLD="A1_atac_small_none_l0_c1_s100_8d38e2a3 A1_atac_small_discriminator_l10_c0_s101_b841ed69"
NEW=$(grep -v -E '^(#|tag)' $I/pilot5_jhpce_pancreas.tags | cut -f1 | sed -n 5p)
echo "new latent: $NEW"
mk() {  # mk OUT TAG SRC_ROOT : out dir with symlinked latent+model
  mkdir -p $1/latents $1/models; ln -sf $3/latents/$2.npz $1/latents/$2.npz; ln -sfn $3/models/$2 $1/models/$2; }
for t in $OLD; do mk $T/t1 $t $D/tier12_v2/A1; echo $t; done > t1.tags
mk $T/t2a $NEW $D/jhpce_tier12/pilot5_jhpce; mk $T/t2b $NEW $D/jhpce_tier12/pilot5_jhpce; echo $NEW > t2.tags
run() { # run OUT MANIFEST TAGS DEVICE TASKS EXTRA
  ( cd $REPO && KMP_AFFINITY=disabled nice -n 10 $ENVS/scvi-api/bin/python scripts/run_stage.py --manifest $2 --experiments A1 --tasks $5 \
    --tags-file $3 --out-dir $1 --prepped-dir $D/prepped_scib --fit-python $ENVS/scvi-api/bin/python \
    --score-python $ENVS/wcd-kbet/bin/python --r-home $ENVS/wcd-kbet/lib/R --r-libs $D/Rlib_kbet --expect-device "$4" \
    --fit-lanes 1 --score-workers-during 2 --score-workers-after 2 $6 ) > $1.log 2>&1; }
run $T/t1  $REPO/manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv $T/t1.tags "RTX 3080" atac_small "--allow-mixed-sha" &
p1=$!
run $T/t2a $I/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv $T/t2.tags L40S pancreas "" &
p2=$!
run $T/t2b $I/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv $T/t2.tags L40S pancreas "" &
p3=$!
rc=0; wait $p1 || rc=1; wait $p2 || rc=1; wait $p3 || rc=1
echo "run_stage exits ok? rc=$rc"
printf 'tag\tref_score\n' > t1.list; for t in $OLD; do printf '%s\t%s\n' $t $D/tier12_v2/A1/scores/$t.csv; done >> t1.list
printf 'tag\tref_score\n%s\t%s\n' $NEW $T/t2a/scores/$NEW.csv > t2.list
PYX=$ENVS/scvi-api/bin/python
echo "== T1: rescored vs existing workstation scores"; $PYX $PW/experiments/pilot5_workstation_scoring/compare_scores.py --list t1.list --cand $T/t1/scores || rc=1
echo "== T2: same new latent scored twice";            $PYX $PW/experiments/pilot5_workstation_scoring/compare_scores.py --list t2.list --cand $T/t2b/scores || rc=1
echo "== provenance of the new row"; $PYX - <<PYEOF
import csv; r=next(csv.DictReader(open("$T/t2a/scores/$NEW.csv")))
print({k:r[k] for k in ("scorer_git_sha","scorer_dirty","cpu_simd","numba_cpu_name","kbet_seed","score_seconds")})
PYEOF
exit $rc
