"""Figure 5 (Pass 23: the collapse panel of the former Figure 4, split from local versus global, which is now Figure 6; user
choice): collapse rate up to the highest common batch score per penalty, one panel per decoder, one dot per real task with its
nominal exact 95% interval, black tick = median over the six real tasks (the low-signal immune pair, ringed, included as in
the Results text). Reads figure_data/collapse_common_range.csv (paper/tools/collapse_common_range.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *

apply_style()
cc = pd.read_csv(DATA / "collapse_common_range.csv").set_index(["task", "cond", "arm"])
assert round(cc.loc[("immune_hum_mou", 1, "discriminator"), "rate_common"], 3) == 0.556  # Res-4 text
fig, axes = decoder_dot_panels(lambda a, t, c: cc.loc[(t, c, a), "rate_common"],
                               "collapse rate, percentage points (lower = better)", low_in_median=True,
                               ci=lambda a, t, c: cc.loc[(t, c, a), ["lo_common", "hi_common"]].tolist(), scale=100)
axes[0].set_xlim(-3, 103)
check_and_save(fig, "fig5")
