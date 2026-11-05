#!/usr/bin/env bash
# Copies the unchanged inputs of Supplementary Tables S6-S7 and Figures S1-S6 into paper/figure_data and records their
# SHA-256 in paper/figure_data/MANIFEST.sha256 (appended once). Run from the repository root on the workstation.
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
SW=$D/sweep_analysis_final_20261105/tables
cd "$(dirname "$0")/../figure_data"
for f in ranking_pairs low_signal_pairs best_of_k label_transfer label_transfer_curve; do cp "$SW/$f.csv" "$f.csv"; done
cp $D/code/wt_sweep_paper/experiments/sweep_checks/out/placement_summary.csv placement_summary.csv
for f in ranking_pairs low_signal_pairs best_of_k label_transfer label_transfer_curve placement_summary; do
  grep -q " $f.csv$" MANIFEST.sha256 || sha256sum "$f.csv" >> MANIFEST.sha256
done
sha256sum -c MANIFEST.sha256 | grep -v ': OK$' || true
