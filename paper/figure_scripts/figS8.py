"""Supplementary Figure S8 (Pass 24: the hypervolume half of the former Figure 4; its frontier half is Figure 3, fig3.py):
hypervolume up to the highest common batch score per task (dots, one per ranked real task) and its median (black tick): the
area under that task's frontier from x = 0 to 1 above 0.10 below plain scVI, so 0.10 (dashed) means plain scVI's biology kept
throughout. Batch-conditioned decoder (a), unconditioned decoder (b); the low-signal immune pair is excluded.
The x limit is set from the data (Pass 24: the former fixed limit of 0.13 hid points up to 0.20).
Reads figure_data/pareto_frontier.csv, written by paper/tools/pareto_frontier.py from curves.csv, paper_numbers.json and secondary.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *

HEIGHT = 3.0
apply_style()
per = pd.read_csv(DATA / "pareto_frontier.csv")
per = per[per.task.isin(REAL_TASKS)]
n_tasks = per.groupby("cond").task.nunique()
assert dict(n_tasks) == {0: 6, 1: 5}
xmax = np.ceil(1.05 * per.hypervolume_to_common.max() * 100) / 100    # every point inside the axis, 0.01 steps

fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, HEIGHT), sharex=True, sharey=True,
                         gridspec_kw=dict(wspace=0.08, left=0.13, right=0.98, top=0.88, bottom=0.24))
n_plotted = 0
for ax, c in zip(axes, COND_ORDER):
    for i, a in enumerate(ARMS):
        v = per[(per.cond == c) & (per.arm == a)].hypervolume_to_common
        ax.scatter(v, [i] * len(v), s=14, color=ARM_COLOR[a], lw=0, zorder=3)
        ax.plot([v.median()] * 2, [i - 0.3, i + 0.3], color="black", lw=1.6, zorder=4, solid_capstyle="butt")
        if i:
            ax.axhline(i - .5, color="#e6e6e6", lw=0.5, zorder=0)
        n_plotted += len(v)
    ax.axvline(0.10, color="#999999", ls="--", lw=0.6, zorder=0)
    ax.set_title(f"{COND_LABEL[c]} ({n_tasks[c]} ranked tasks)", loc="left", pad=3)
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)
ax = axes[0]
ax.set_xlim(0, xmax)
ax.set_ylim(5.5, -0.5)
ax.set_yticks(range(6))
ax.set_yticklabels([ARM_LABEL[a] for a in ARMS])
lo, hi = ax.get_xlim()
assert lo <= per.hypervolume_to_common.min() and per.hypervolume_to_common.max() <= hi     # no point clipped
assert n_plotted == len(per)                                                               # every real-task row is drawn
fig.text(0.555, ax.get_position().y0 - 0.085, "hypervolume up to the highest common batch score (higher = better; "
         "0.10 = plain scVI's biology kept throughout); dots, tasks; tick, median", ha="center", va="top", fontsize=7)
for L, ax in zip("ab", axes):
    p = ax.get_position()
    fig.text(p.x0 - (0.1 if ax is axes[0] else 0.03), p.y1 + 0.175 / HEIGHT, L, fontsize=10, fontweight="bold")
check_and_save(fig, "figS8")
