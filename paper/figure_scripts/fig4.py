"""Figure 4 (Pass 23, new; user choice, version A): each penalty's Pareto frontier of batch removal against biology.
(a, b) median over the ranked real tasks of the best biology reachable, minus plain scVI's, at batch removal x or beyond,
with batch removal scaled per task as x = (B - b0) / (b_common - b0) (0 = plain scVI, 0.5 = b*, 1 = the highest batch score
every penalty reaches); (c, d) hypervolume up to b_common per task (dots) and its median (tick): the area under that task's
frontier from x = 0 to 1 above 0.10 below plain scVI. Batch-conditioned decoder (a, c), unconditioned decoder (b, d); the
low-signal immune pair is excluded. Reads figure_data/pareto_frontier_median.csv and pareto_frontier.csv, written by
paper/tools/pareto_frontier.py from curves.csv, paper_numbers.json and secondary.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.lines import Line2D

apply_style()
med = pd.read_csv(DATA / "pareto_frontier_median.csv")
per = pd.read_csv(DATA / "pareto_frontier.csv")
per = per[per.task.isin(REAL_TASKS)]
n_tasks = med.groupby("cond").n_tasks.max()
assert dict(n_tasks) == {0: 6, 1: 5} and set(per.groupby("cond").task.nunique()) == {5, 6}

fig = plt.figure(figsize=(WIDTH_IN, 5.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.08, left=0.13, right=0.98, top=0.93, bottom=0.16)
top = [fig.add_subplot(gs[0, 0])]
top.append(fig.add_subplot(gs[0, 1], sharey=top[0]))
low = [fig.add_subplot(gs[1, 0])]
low.append(fig.add_subplot(gs[1, 1], sharex=low[0], sharey=low[0]))
for k, (ax, c) in enumerate(zip(top, COND_ORDER)):
    for a in ARMS:
        m = med[(med.cond == c) & (med.arm == a)]
        ax.plot(m.x, m["median"], color=ARM_COLOR[a], lw=1.5, drawstyle="steps-post")
    ax.axhline(0, color="#999999", lw=0.6, zorder=0)
    ax.axvline(0.5, color="#444444", ls="--", lw=0.7, zorder=0)
    ax.text(0.515, 0.97, r"$b^{*}$", transform=ax.get_xaxis_transform(), fontsize=7, va="top", color="#444444")
    ax.set_xlim(0, 1)
    ax.set_title(f"{COND_LABEL[c]} ({n_tasks[c]} ranked tasks)", loc="left", pad=3)
    if k == 0:
        ax.set_ylabel("best biology reachable\nminus plain scVI's (median)")
    else:
        ax.tick_params(labelleft=False)
fig.text(0.555, top[0].get_position().y0 - 0.06, r"batch removal, scaled per task (0 = plain scVI, 0.5 = $b^{*}$, "
         "1 = the highest batch score every penalty reaches)", ha="center", va="top", fontsize=7)
for k, (ax, c) in enumerate(zip(low, COND_ORDER)):
    for i, a in enumerate(ARMS):
        v = per[(per.cond == c) & (per.arm == a)].hypervolume_to_common
        ax.scatter(v, [i] * len(v), s=14, color=ARM_COLOR[a], lw=0, zorder=3)
        ax.plot([v.median()] * 2, [i - 0.3, i + 0.3], color="black", lw=1.6, zorder=4, solid_capstyle="butt")
        if i:
            ax.axhline(i - .5, color="#e6e6e6", lw=0.5, zorder=0)
    ax.axvline(0.10, color="#999999", ls="--", lw=0.6, zorder=0)
    ax.set_ylim(5.5, -0.5)
    ax.set_yticks(range(6))
    ax.set_yticklabels([ARM_LABEL[a] for a in ARMS])
    ax.tick_params(left=False, labelleft=(k == 0))
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 0.13)
fig.text(0.555, low[0].get_position().y0 - 0.06, "hypervolume up to the highest common batch score (higher = better; "
         "0.10 = plain scVI's biology kept throughout); dots, tasks; tick, median", ha="center", va="top", fontsize=7)
for L, ax in zip("abcd", top + low):
    p = ax.get_position()
    fig.text(p.x0 - (0.1 if ax in (top[0], low[0]) else 0.03), p.y1 + 0.035, L, fontsize=10, fontweight="bold")
fig.legend(handles=[Line2D([], [], color=ARM_COLOR[a], lw=1.8, label=ARM_LABEL[a]) for a in ARMS],
           loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0), frameon=False)
check_and_save(fig, "fig4")
