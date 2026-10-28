#!/bin/bash
# Starts the JHPCE scoring of immune_hum_mou (378 latents) and atac_large (250 now, 128 as their fits land) with the fixed LISI seed
# (CONSTRAINTS.md SI-115): two 1-CPU / 1-GB controller jobs (the second starts after the first ends, so the chain outlives one 3-day job
# limit) that run cluster/jhpce/chain_score.py under job-watch. The controller submits the scoring arrays itself: one latent per task,
# 1 CPU each, at most 200 CPUs and 2,000 GB held, memory and time from cluster/jhpce/score_sizes.json (measured peaks x 1.3).
# Run on the JHPCE login node (submits only), after the identity check passed (identity_check.py exit 0).
# Usage: start_production_chain.sh DEPLOYED_REPO          (CONSTRAINT=znver2 by default; DRY=1 prints the controller command only)
set -euo pipefail
REPO=$(cd "${1:?deployed repo (git archive of the branch)}" && pwd -P)
S=/fastscratch/myscratch/kreid/wcd
OUT=$S/tier12/pilot5_jhpce                        # fits write here; the scores go to its scores/ folder
STATE=$S/pilot5_jhpce_scoring/chain/state
SCORER=$S/scorer_10a1ca5                          # clean checkout of 10a1ca5 (job file score_array.sh untracked there)
FITJOBS=${FITJOBS:-"36749854 36750324 36750325 36749855"}   # the three atac_large fit-shard controllers and the first shard's GPU job
ARGS=(--stage-repo "$SCORER" --score-repo "$SCORER"
      --score-array "$REPO/cluster/jhpce/score_array_fixed_seed.sh"
      --score-env "SCORER_REPO=$SCORER" "FIXED_SEED_LIB=$S/lisi_fix/fixed_random_device.so"
      --manifest "$OUT/manifests/5dd65550fdb18e5c.tsv" --manifest-sha256 5dd65550fdb18e5c80e73dcfa731d72bdeb6128bdeac7ffb36df6c54450396ff
      --experiments A1
      --tags-files "$REPO/experiments/chaining_board_20261028/expected/pilot5_jhpce_immune_hum_mou.tags" "$REPO/experiments/chaining_board_20261028/expected/pilot5_jhpce_atac_large.tags"
      --fit-out "$OUT" --score-out "$OUT" --prepped-dir "$S/prepped_scib" --expect-device L40S
      --fit-jobs $FITJOBS --constraint "${CONSTRAINT:-znver2}" --max-cpus 200 --max-mem-gb 2000 --min-batch 10 --interval 300)
[ -z "${DRY:-}" ] || { printf '%q ' "${ARGS[@]}"; echo; exit 0; }
mkdir -p "$STATE"
J1=$(sbatch --parsable --export=ALL,WCD_REPO="$REPO" "$REPO/cluster/jhpce/chain_job.sh" "$STATE" -- "${ARGS[@]}")
J2=$(sbatch --parsable --dependency=afterany:"$J1" --export=ALL,WCD_REPO="$REPO" "$REPO/cluster/jhpce/chain_job.sh" "$STATE" -- "${ARGS[@]}")
echo "controllers $J1 and $J2 (state $STATE)"
