#!/bin/bash
# Rebalanced launcher for the 6-light-dataset wave.
#
# WHY SHARDED, NOT ONE-DATASET-PER-WORKER: at one dataset per worker the machine idles a
# third of its capacity -- atac_small finishes in ~4 h while pancreas runs ~27 h, so the
# wall-clock is set by the slowest dataset. Sharding at (dataset, experiment, head) and
# packing longest-first balances the six workers to ~98%, taking the wall-clock from
# ~23 h to ~15.4 h for the same 92 worker-hours.
#
# WHY THESE SHARD BOUNDARIES: each shard is ONE process that loops its own config grid, so
# the ~81 s process startup is paid once per shard (58 times), not once per config (924
# times = 21 h of pure startup). Only splits run_experiment.py supports natively are used
# (--experiment / --dataset / --head / --seed-only); shards over ~5 h are split by seed.
#
# EPOCHS: 500 is a CEILING, not a target -- early stopping decides. Measured: the critic
# stops at 160-200 epochs on the three datasets that were budget-limited at 150, and the
# discriminator at 110-130. Do NOT set --epochs 150: that silently truncated pancreas,
# lung and sim2 on the critic head, where early stopping never fired.
set -uo pipefail
cd "$(dirname "$0")/.."

export KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
       MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore
export PYTHONPATH=$(pwd)/src
PY=${WCD_PYTHON:-/home/kendall/.claude-science/conda/envs/wcd-gpu/bin/python}
R=${WCD_DATA:-$(pwd)/../data}
export PY R
mkdir -p results/wave logs/wave

MANIFEST=${1:-scripts/wave_manifest.tsv}
WORKERS=${WCD_WORKERS:-6}

# TAB-delimited, not CSV: the command field is free text and a future flag with a
# comma-separated value (--override a,b) would split mid-command under IFS=','.

# One background shell per worker lane; each runs its shards SEQUENTIALLY. Concurrency is
# therefore exactly $WORKERS regardless of shard count -- important because 6 concurrent
# light-dataset workers peak at ~7.7 GiB of the 8 GiB card and a 7th would not fit.
for w in $(seq 1 "$WORKERS"); do
  (
    while IFS=$'\t' read -r worker tag est cmd; do
      [ "$worker" = "worker" ] && continue
      [ "$worker" != "$w" ] && continue
      log="logs/wave/${tag}.log"
      if [ -f "results/wave/${tag}.done" ]; then
        echo "[w$w] skip $tag (done)"; continue
      fi
      echo "[w$w] start $tag (est ${est} h)"
      # --resume makes a re-run of an interrupted shard skip completed configs.
      if eval "$cmd" > "$log" 2>&1; then
        touch "results/wave/${tag}.done"
        echo "[w$w] ok    $tag"
      else
        echo "[w$w] FAIL  $tag -- see $log" >&2
      fi
    done < <(tail -n +1 "$MANIFEST")
  ) &
done
wait
echo "wave complete: $(ls results/wave/*.csv 2>/dev/null | wc -l) shard CSVs"
