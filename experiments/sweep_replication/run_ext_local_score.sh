#!/bin/bash
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding; REPO=$D/code/wcd_prereg-tier12-v3; ENVS=/home/kendall/.claude-science/conda/envs
cd $REPO
[ "$(git rev-parse HEAD)" = 10a1ca5874fbcdc033b148525b5e71d866306bdb ] || { echo bad sha; exit 2; }
[ -z "$(git status --porcelain)" ] || { echo dirty; exit 2; }
export KMP_AFFINITY=disabled
exec $ENVS/scvi-api/bin/python scripts/run_stage.py --manifest /home/kendall/experiment_data/wasserstein-critic-deconfounding/code/wt_sweep_replication/manifests/sweep_replication_ext1.tsv --experiments X1 --tasks atac_small sim1 \
  --tags-file /home/kendall/experiment_data/wasserstein-critic-deconfounding/code/wt_sweep_replication/experiments/sweep_replication/ext1_local.tags --out-dir $D/tier12_v3/sweep_replication --prepped-dir $D/prepped_scib \
  --fit-python $ENVS/scvi-api/bin/python --score-python $ENVS/wcd-kbet/bin/python --r-home $ENVS/wcd-kbet/lib/R --r-libs $D/Rlib_kbet \
  --expect-device "RTX 3080" --fit-lanes 1 --fit-threads 1 --score-workers-during 4 --score-workers-after 4 "$@"
