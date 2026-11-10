#!/usr/bin/env bash
# Rebuild Figures 1-4 (PNG 300 dpi + PDF in paper/figures/; Figure 1 also SVG) from paper/figure_data/. Run from the repository root.
# Pass 27: the collapse-rate figure (former Figure 4) is Table 1, paper/tools/make_table1.py; the placement figure (former Figure 5) is Figure 4.
# Environment: Python, matplotlib, pandas, numpy (versions in figure_scripts/ENVIRONMENT.txt). No random numbers are used.
set -euo pipefail
cd "$(dirname "$0")/../.."
(cd paper/figure_data && sha256sum -c MANIFEST.sha256)
for n in 1 2 3 4; do python paper/figure_scripts/fig$n.py; done
(cd paper/figures && sha256sum fig*.png fig*.pdf fig1.svg > MANIFEST.sha256)
