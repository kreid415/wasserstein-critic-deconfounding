#!/usr/bin/env bash
# Rebuild Figures 1-5 (PNG 300 dpi + PDF in paper/figures/; Figure 1 also SVG) from paper/figure_data/. Run from the repository root.
# Environment: Python, matplotlib, pandas, numpy (versions in figure_scripts/ENVIRONMENT.txt). No random numbers are used.
set -euo pipefail
cd "$(dirname "$0")/../.."
(cd paper/figure_data && sha256sum -c MANIFEST.sha256)
for n in 1 2 3 4 5; do python paper/figure_scripts/fig$n.py; done
(cd paper/figures && sha256sum fig*.png fig*.pdf fig1.svg > MANIFEST.sha256)
