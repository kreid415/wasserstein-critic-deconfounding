"""Figure 6, batch-conditioned decoder: batch against biology score per real task (one panel per task, 2 x 3) at the single
strength (or setting) chosen on other task families (figure_data/transfer_strength.csv: penalties, Harmony, Scanorama, sysVI)
and the fixed settings in figure_data/single_methods.csv (plain scVI none|0|10, scVI adversary, scANVI, PCA pca|0|10).
Pass 23: the former panel a, a score curve against the number of settings tried, is cut with the whole best-of-k analysis (paper/pass23_changelog.md)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.lines import Line2D

apply_style()
COND = 1
ts = pd.read_csv(DATA / "transfer_strength.csv"); ts = ts[(ts.cond == COND) & ts.task.isin(REAL_TASKS)]
sm = pd.read_csv(DATA / "single_methods.csv"); sm = sm[(sm.cond == COND) & sm.task.isin(REAL_TASKS)]
TOOL_MK = {"harmony": ("s", "Harmony"), "scanorama": ("D", "Scanorama"), "sysvi": ("^", "sysVI"),
           "scanvi": ("v", "scANVI"), "pca": ("X", "PCA"), "scvi_adv": ("o", "scVI adversary")}

fig = plt.figure(figsize=(WIDTH_IN, 4.0))
gs = fig.add_gridspec(2, 3, wspace=0.28, hspace=0.5, left=0.07, right=0.995, top=0.92, bottom=0.2)

star = sm[sm.name == "none|0|10"].set_index("task")
for n, t in enumerate(REAL_TASKS):
    ax = fig.add_subplot(gs[n // 3, n % 3])
    for f in ARMS:
        r = ts[(ts.family == f) & (ts.task == t)].iloc[0]
        ax.plot(r.B, r.C, "o", color=ARM_COLOR[f], ms=3.4, mew=0, zorder=3)
    for f in ("harmony", "scanorama", "sysvi"):
        r = ts[(ts.family == f) & (ts.task == t)].iloc[0]
        ax.plot(r.B, r.C, TOOL_MK[f][0], mfc="none", mec=GREY, ms=4, mew=0.9, zorder=2)
    for name, f in (("scanvi|0|10", "scanvi"), ("scvi_adv|0|10", "scvi_adv"), ("pca|0|10", "pca")):
        r = sm[(sm.name == name) & (sm.task == t)].iloc[0]
        ax.plot(r.B, r.C, TOOL_MK[f][0], mfc="none", mec=GREY, ms=4, mew=0.9, zorder=2)
    s_ = star.loc[t]
    ax.plot(s_.B, s_.C, "*", color="black", ms=9, mec="white", mew=0.4, zorder=5)
    ax.set_title(TASK_LABEL[t], loc="left", pad=3)
    ax.margins(0.1); ax.locator_params(nbins=3)
    if n // 3 == 1: ax.set_xlabel("batch score")
    if n % 3 == 0: ax.set_ylabel("biology score")
hb = [Line2D([], [], color="black", marker="*", ms=8, lw=0, label="plain scVI")]
hb += [Line2D([], [], color=ARM_COLOR[a], marker="o", ms=3.5, lw=0, label=ARM_LABEL[a]) for a in ARMS]
hb += [Line2D([], [], color=GREY, marker=m, mfc="none", ms=4, lw=0, label=l) for m, l in TOOL_MK.values()]
fig.legend(handles=hb, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.0), columnspacing=0.9, handletextpad=0.2, fontsize=6.5)
fig.text(0.995, 0.995, "higher = better on both axes", ha="right", va="top", fontsize=7, color=GREY)
check_and_save(fig, "fig6")
print(ts.B.isna().sum(), ts.groupby("family").size().to_dict())
