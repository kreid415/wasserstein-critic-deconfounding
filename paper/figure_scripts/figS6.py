"""Supplementary Figure S6: best of k settings per task. Expected best selection score (0.4 batch + 0.6 biology) among k settings drawn at random,
k = 1 to 6, one panel per real task and decoder (figure_data/best_of_k.csv, per task; Figure 8a is the mean of these curves over the six real tasks,
batch-conditioned decoder). Penalties continue their curves only to k = 6, the number of settings of Harmony, Scanorama and sysVI. Plain scVI, the
scVI adversary and scANVI have one setting (k = 1)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.lines import Line2D

apply_style()
bk = pd.read_csv(DATA / "best_of_k.csv")
bk = bk[bk.task.isin(REAL_TASKS) & (bk.k <= 6)]
TOOL_LS = {"harmony": ("--", "Harmony"), "scanorama": (":", "Scanorama"), "sysvi": ("-.", "sysVI")}
fig, axes = plt.subplots(2, 6, figsize=(WIDTH_IN, 4.1), sharex=True, gridspec_kw=dict(wspace=0.42, hspace=0.14, left=0.11, right=0.995, top=0.915, bottom=0.2))
for r, c in enumerate(COND_ORDER):
    for j, t in enumerate(REAL_TASKS):
        ax = axes[r, j]
        d = bk[(bk.task == t) & (bk.cond == c)]
        plain = d[(d.family == "none") & (d.k == 1)].E.iloc[0]
        ax.axhline(plain, color="black", lw=1.0, zorder=1)
        for a in ARMS:
            e = d[d.family == a].sort_values("k")
            assert list(e.k) == list(range(1, 7)), (t, c, a, list(e.k))
            ax.plot(e.k, e.E, "-o", color=ARM_COLOR[a], ms=2.2, lw=1.0, mew=0, zorder=3)
        for f, (ls, lab) in TOOL_LS.items():
            e = d[d.family == f].sort_values("k")
            ax.plot(e.k, e.E, ls, color=GREY, lw=1.0, zorder=2)
            if len(e) < 6:   # Scanorama on the large chromatin-accessibility task: the 80-neighbour setting could not be fitted
                ax.plot(e.k.iloc[-1], e.E.iloc[-1], "o", mfc="white", mec=GREY, ms=3.2, mew=0.8, zorder=4)
        for f, mk in (("scanvi", "v"), ("scvi_adv", "o")):
            if f == "scvi_adv" and c == 0: continue      # the scvi-tools adversary is run with the batch-conditioned decoder only
            e = d[d.family == f]
            ax.plot(1, e.E.iloc[0], mk, mfc="none", mec=GREY, ms=4.2, mew=0.9, zorder=4)
        if r == 0: ax.set_title(TASK_LABEL[t], loc="left", pad=3)
        if r == 1: ax.set_xlabel("settings tried, $k$")
        if j == 0: ax.set_ylabel(COND_LABEL[c] + "\nexpected best score")
        ax.set_xticks([1, 3, 5]); ax.margins(y=0.08); ax.locator_params(axis="y", nbins=4)
    fig.text(-0.012, axes[r, 0].get_position().y1 + 0.005, "ab"[r], fontsize=10, fontweight="bold", va="bottom")
h = [Line2D([], [], color=ARM_COLOR[a], marker="o", ms=3, lw=1.2, label=ARM_LABEL[a]) for a in ARMS]
h += [Line2D([], [], color="black", lw=1.0, label="plain scVI")]
h += [Line2D([], [], color=GREY, ls=ls, lw=1.0, label=lab) for ls, lab in TOOL_LS.values()]
h += [Line2D([], [], color=GREY, marker="v", mfc="none", ms=4.2, lw=0, label="scANVI (uses labels)"), Line2D([], [], color=GREY, marker="o", mfc="none", ms=4.2, lw=0, label="scVI adversary")]
fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.4)
check_and_save(fig, "figS6")
print("rows used:", len(bk))
