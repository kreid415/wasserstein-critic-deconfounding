#!/usr/bin/env bash
# Rebuild Figures 2-6 (PNG 300 dpi + PDF in paper/figures/) from paper/figure_data/. Run from the repository root.
# Environment: Python, matplotlib, pandas, numpy (versions in figure_scripts/ENVIRONMENT.txt). No random numbers are used.
set -euo pipefail
cd "$(dirname "$0")/../.."
(cd paper/figure_data && sha256sum -c MANIFEST.sha256)
for n in 2 3 4 5 6; do python paper/figure_scripts/fig$n.py; done
(cd paper/figures && sha256sum fig*.png fig*.pdf > MANIFEST.sha256)
