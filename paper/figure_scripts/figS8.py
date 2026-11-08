"""Supplementary Figure S8 (Pass 23: the local-versus-global panel of the former Figure 4; main Figure 6 for one step, then moved
to the supplement by user choice because the penalties barely differ, Res-5 keeps one sentence): local minus
global batch removal at b* per penalty, one panel per decoder, one dot per real task, black tick = median over the ranked
real tasks (the low-signal immune pair is ringed and left out of the median, as in the Results text).
Reads only figure_data/paper_numbers.json."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *

apply_style()
pn = load_pn()
val = lambda a, t, c: pnv(pn, f"local_minus_global.{a}.{t}.cond{c}")
assert round(val("discriminator", "lung", 1), 3) == -0.356  # paper_numbers.json check
fig, axes = decoder_dot_panels(val, "local minus global batch removal\n(positive = stronger locally)", low_in_median=False, bottom=0.25,
                               zero_line=True)
check_and_save(fig, "figS8")
