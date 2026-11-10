"""Figure 3 (Pass 24: the frontier half of the former Figure 4; its hypervolume half is Supplementary Figure S8, figS8.py):
each penalty's Pareto frontier of batch removal against biology. Median over the ranked real tasks of the best biology
reachable, minus plain scVI's, at batch removal x or beyond, with batch removal scaled per task as
x = (B - b0) / (b_common - b0) (0 = plain scVI, 0.5 = b*, 1 = the highest batch score every penalty reaches).
Batch-conditioned decoder (a), unconditioned decoder (b); the low-signal immune pair is excluded.
Reads figure_data/pareto_frontier_median.csv, written by paper/tools/pareto_frontier.py from curves.csv,
paper_numbers.json and secondary.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.lines import Line2D

HEIGHT = 3.0
apply_style()
med = pd.read_csv(DATA / "pareto_frontier_median.csv")
n_tasks = med.groupby("cond").n_tasks.max()
assert dict(n_tasks) == {0: 6, 1: 5}

fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, HEIGHT), sharey=True,
                         gridspec_kw=dict(wspace=0.08, left=0.13, right=0.98, top=0.88, bottom=0.24))
for ax, c in zip(axes, COND_ORDER):
    for a in ARMS:
        m = med[(med.cond == c) & (med.arm == a)]
        ax.plot(m.x, m["median"], color=ARM_COLOR[a], lw=1.5, drawstyle="steps-post")
    ax.axhline(0, color="#999999", lw=0.6, zorder=0)
    ax.axvline(0.5, color="#444444", ls="--", lw=0.7, zorder=0)
    ax.text(0.515, 0.97, r"$b^{*}$", transform=ax.get_xaxis_transform(), fontsize=7, va="top", color="#444444")
    ax.set_xlim(0, 1)
    ax.set_title(f"{COND_LABEL[c]} ({n_tasks[c]} ranked tasks)", loc="left", pad=3)
axes[0].set_ylabel("best biology reachable\nminus plain scVI's (median)")
fig.text(0.555, axes[0].get_position().y0 - 0.085, r"batch removal, scaled per task (0 = plain scVI, 0.5 = $b^{*}$, "
         "1 = the highest batch score every penalty reaches)", ha="center", va="top", fontsize=7)
for L, ax in zip("ab", axes):
    p = ax.get_position()
    fig.text(p.x0 - (0.1 if ax is axes[0] else 0.03), p.y1 + 0.175 / HEIGHT, L, fontsize=10, fontweight="bold")
fig.legend(handles=[Line2D([], [], color=ARM_COLOR[a], lw=1.8, label=ARM_LABEL[a]) for a in ARMS],
           loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0), frameon=False)
check_and_save(fig, "fig3")
