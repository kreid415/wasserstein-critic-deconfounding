"""Figure 3: batch score against biology score for each penalty's sweep strengths, six real tasks x two decoders.
Curves: figure_data/curves.csv (seed means, joined from plain scVI at strength 0 upward);
dashed line: b* from paper_numbers.json; grey points: standard tools at their best setting
(highest selection score S = 0.4 batch + 0.6 biology among the settings run), figure_data/baseline_points.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.lines import Line2D

apply_style()
pn = load_pn()
cv = pd.read_csv(DATA / "curves.csv")
bp = pd.read_csv(DATA / "baseline_points.csv")
assert cv[cv.task.isin(REAL_TASKS)].n_fail.sum() == 0      # no real-task curve ends early because of failed fits
TOOLS = {"scvi_adv": ("o", "scVI adversary"), "harmony": ("s", "Harmony"), "scanorama": ("D", "Scanorama"),
         "sysvi": ("^", "sysVI"), "scanvi": ("v", "scANVI"), "pca": ("X", "PCA")}
bp["S"] = 0.4 * bp.B + 0.6 * bp.C
best = bp[bp.arm.isin(TOOLS)].sort_values("S").groupby(["task", "cond", "arm"]).tail(1)

fig, axes = plt.subplots(2, 6, figsize=(WIDTH_IN, 4.1), sharex="col", sharey="col",
                         gridspec_kw=dict(wspace=0.38, hspace=0.12, left=0.105, right=0.995, top=0.915, bottom=0.2))
for r, c in enumerate(COND_ORDER):
    for j, t in enumerate(REAL_TASKS):
        ax = axes[r, j]
        d = cv[(cv.task == t) & (cv.cond == c)]
        for a in ARMS:
            e = d[d.arm == a].sort_values("lam")
            ax.plot(e.B, e.C, "-o", color=ARM_COLOR[a], ms=2.2, lw=1.0, mew=0, zorder=2)
        p0 = d[d.lam == 0].iloc[0]
        ax.plot(p0.B, p0.C, "*", color="black", ms=8, zorder=5, mec="white", mew=0.4)
        ax.axvline(pnv(pn, f"pair.{t}.cond{c}.bstar"), color="#444444", ls="--", lw=0.7, zorder=1)
        for _, row in best[(best.task == t) & (best.cond == c)].iterrows():
            m = TOOLS[row.arm][0]
            ax.plot(row.B, row.C, m, mfc="none", mec=GREY, ms=4, mew=0.9, zorder=3)
        if r == 0: ax.set_title(TASK_LABEL[t], loc="left", pad=3)
        if r == 1: ax.set_xlabel("batch score")
        if j == 0: ax.set_ylabel(COND_LABEL[c] + "\nbiology score")
        ax.margins(0.06)
        ax.locator_params(nbins=4)
    fig.text(0.003, axes[r, 0].get_position().y1 + 0.005, "ab"[r], fontsize=10, fontweight="bold", va="bottom")
axes[0, 0].text(pnv(pn, "pair.pancreas.cond1.bstar"), 1.0, r"$b^{*}$", transform=axes[0, 0].get_xaxis_transform(),
                ha="left", va="bottom", fontsize=7, color="#444444")
h = [Line2D([], [], color=ARM_COLOR[a], marker="o", ms=3, lw=1.2, label=ARM_LABEL[a]) for a in ARMS]
h += [Line2D([], [], color="black", marker="*", ms=8, lw=0, label="plain scVI")]
h += [Line2D([], [], color=GREY, marker=m, mfc="none", ms=4, lw=0, label=l) for m, l in TOOLS.values()]
fig.legend(handles=h, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.3)
fig.text(0.995, 0.999, "higher = better on both axes", ha="right", va="top", fontsize=7, color=GREY)
check_and_save(fig, "fig3")
print(best.groupby("arm").size().to_dict(), len(cv[cv.task.isin(REAL_TASKS)]))
