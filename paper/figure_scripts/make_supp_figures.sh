#!/usr/bin/env bash
# Rebuild Supplementary Tables S6-S7 and Figures S1-S6 (PNG 300 dpi + PDF in paper/figures/, LaTeX in paper/supp/). Run from the repository root on the workstation
# (steps 1-3 read the sweep analysis directory D/sweep_analysis_final_20261105 and the saved latents; steps 4-6 read paper/figure_data only).
# Environments: SCVI_PY = the scvi-api conda environment (scikit-learn 1.9.0, pandas 2.3.3; label transfer only); PY = Python with matplotlib, pandas, scipy, numpy
# (versions in figure_scripts/ENVIRONMENT_supp.txt). No random numbers are used except a fixed-seed jitter of dots in Figure S5a.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
SCVI_PY=${SCVI_PY:-/home/kendall/.claude-science/conda/envs/scvi-api/bin/python}
bash paper/figure_scripts/copy_supp_inputs.sh                                              # 1 unchanged copies of sweep tables + sha256
$PY paper/figure_scripts/prep_supp_figure_data.py                                          # 2 sensitivity_ranks.csv, metric_scores.csv
$SCVI_PY paper/figure_scripts/prep_label_transfer_types.py --jobs 8                        # 3 label_transfer_types.csv (about 15 min on 8 cores)
for f in sensitivity_ranks metric_scores label_transfer_types; do                          # derived files: checksum recorded once
  grep -q " $f.csv$" paper/figure_data/MANIFEST.sha256 || (cd paper/figure_data && sha256sum "$f.csv" >> MANIFEST.sha256)
done
for n in 1 2 3 4 5 6; do $PY paper/figure_scripts/figS$n.py; done                          # 4 figures (S5 also writes scaling_check_ranks.csv)
grep -q " scaling_check_ranks.csv$" paper/figure_data/MANIFEST.sha256 || (cd paper/figure_data && sha256sum scaling_check_ranks.csv >> MANIFEST.sha256)
(cd paper/figure_data && sha256sum -c MANIFEST.sha256)
$PY paper/tools/make_supp_tables.py --check-table4                                         # 5 tables S6, S7 (+ positive control against Table 4)
$PY paper/tools/make_supp_supp-analysis.py                                                 # 5b Table S8 and the label-transfer table of Note 9
(cd paper/figures && grep -v ' figS' MANIFEST.sha256 > MANIFEST.tmp && sha256sum figS*.png >> MANIFEST.tmp && mv MANIFEST.tmp MANIFEST.sha256)   # 6 (PNG only: the PDF files carry a creation time stamp, so their bytes change on every run)
