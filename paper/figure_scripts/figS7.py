"""Supplementary Figure S7 (the main-text Figure 3 until Pass 23): batch score against biology score for each penalty's sweep strengths, six real tasks x two decoders.
Curves: figure_data/curves.csv (seed means, joined from plain scVI at strength 0 upward);
dashed line: b* from paper_numbers.json; open circle: each penalty's strongest fitted setting; grey panel: low-signal pair.
Pass 23: the standard tools are no longer drawn (Figure 8b places them; user rule SI-115, no repeats in the main text)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.lines import Line2D

apply_style()
pn = load_pn()
cv = pd.read_csv(DATA / "curves.csv")
low = {c: set(pnv(pn, f"low_signal_tasks.cond{c}")) for c in (0, 1)}
assert cv[cv.task.isin(REAL_TASKS)].n_fail.sum() == 0      # no real-task curve ends early because of failed fits
fig, axes = plt.subplots(2, 6, figsize=(WIDTH_IN, 4.1), sharex="col", sharey="col",
                         gridspec_kw=dict(wspace=0.38, hspace=0.12, left=0.105, right=0.995, top=0.915, bottom=0.2))
for r, c in enumerate(COND_ORDER):
    for j, t in enumerate(REAL_TASKS):
        ax = axes[r, j]
        d = cv[(cv.task == t) & (cv.cond == c)]
        for a in ARMS:
            e = d[d.arm == a].sort_values("lam")
            ax.plot(e.B, e.C, "-o", color=ARM_COLOR[a], ms=2.2, lw=1.0, mew=0, zorder=2)
            ax.plot(e.B.iloc[-1], e.C.iloc[-1], "o", mfc="white", mec=ARM_COLOR[a], ms=4.2, mew=1.0, zorder=4)  # strongest setting
        p0 = d[d.lam == 0].iloc[0]
        ax.plot(p0.B, p0.C, "*", color="black", ms=8, zorder=5, mec="white", mew=0.4)
        ax.axvline(pnv(pn, f"pair.{t}.cond{c}.bstar"), color="#444444", ls="--", lw=0.7, zorder=1)
        if t in low[c]:  # low-signal pair: shown, not ranked
            ax.set_facecolor("#f0f0f0")
            ax.text(0.97, 0.03, "low-signal,\nnot ranked", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=GREY)
        if r == 0: ax.set_title(TASK_LABEL[t], loc="left", pad=3)
        if r == 1: ax.set_xlabel("batch score")
        if j == 0: ax.set_ylabel(COND_LABEL[c] + "\nbiology score")
        ax.margins(0.06)
        ax.locator_params(nbins=4)
    fig.text(0.003, axes[r, 0].get_position().y1 + 0.005, "ab"[r], fontsize=10, fontweight="bold", va="bottom")
axes[0, 0].text(pnv(pn, "pair.pancreas.cond1.bstar") + 0.005, 0.03, r"$b^{*}$",  # inside the panel, away from the title
                transform=axes[0, 0].get_xaxis_transform(),
                ha="left", va="bottom", fontsize=7, color="#444444")
h = [Line2D([], [], color=ARM_COLOR[a], marker="o", ms=3, lw=1.2, label=ARM_LABEL[a]) for a in ARMS]
h += [Line2D([], [], color="black", marker="*", ms=8, lw=0, label="plain scVI")]
h += [Line2D([], [], color=GREY, marker="o", mfc="white", ms=4.2, mew=1.0, lw=0, label="strongest setting"),
      Line2D([], [], color="#444444", ls="--", lw=0.7, label=r"matched batch removal $b^{*}$")]
fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.3)
fig.text(0.995, 0.999, "higher = better on both axes", ha="right", va="top", fontsize=7, color=GREY)
check_and_save(fig, "figS7")
print(len(cv[cv.task.isin(REAL_TASKS)]), sorted((t, c) for c in low for t in low[c] if t in REAL_TASKS))
