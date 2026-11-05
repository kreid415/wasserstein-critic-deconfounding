"""Figure 2: ranking of the six penalties by biology kept at matched batch removal b*.
(a) biology score at b* (colour) and rank within the task column (number) per real task and decoder;
(b) mean rank over the ranked real tasks with the range over tasks.
Reads only figure_data/paper_numbers.json (sweep analysis 8595a16)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *

apply_style()
pn = load_pn()
mr = mean_rank_table(pn)
low = {c: set(pnv(pn, f"low_signal_tasks.cond{c}")) for c in (0, 1)}

bio = {c: np.array([[pnv(pn, f"bio_at_bstar.{a}.{t}.cond{c}") for t in REAL_TASKS] for a in ARMS]) for c in (0, 1)}
rank = {c: np.array([[pnv(pn, f"rank.{a}.{t}.cond{c}") if f"rank.{a}.{t}.cond{c}" in pn else np.nan
                      for t in REAL_TASKS] for a in ARMS]) for c in (0, 1)}
assert not np.isnan(bio[0]).any() and not np.isnan(bio[1]).any()
for c in (0, 1):  # a cell is unranked exactly when its task is low-signal
    for j, t in enumerate(REAL_TASKS):
        assert np.isnan(rank[c][:, j]).all() == (t in low[c]), (c, t)
vmin, vmax = min(bio[0].min(), bio[1].min()), max(bio[0].max(), bio[1].max())

fig = plt.figure(figsize=(WIDTH_IN, 3.6))
gs = fig.add_gridspec(1, 5, width_ratios=[6, 6, 0.28, 4.6, 4.6], wspace=0.07, left=0.12, right=0.995, top=0.86, bottom=0.33)
axh = {1: fig.add_subplot(gs[0]), 0: fig.add_subplot(gs[1])}
axc = fig.add_subplot(gs[2])
axb = fig.add_subplot(gs[4], sharey=axh[1])
cmap = plt.get_cmap("viridis")
for c, ax in axh.items():
    im = ax.pcolormesh(np.arange(-0.5, 6), np.arange(-0.5, 6), bio[c], cmap=cmap, vmin=vmin, vmax=vmax,
                       edgecolors="white", linewidth=1.0)
    for i in range(6):
        for j in range(6):
            if np.isnan(rank[c][i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#e6e6e6", edgecolor="white",
                                           hatch="////", linewidth=1.0, zorder=3))
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="#8a8a8a", hatch="////", lw=0, zorder=3))
            else:
                v = (bio[c][i, j] - vmin) / (vmax - vmin)
                r = rank[c][i, j]
                ax.text(j, i, f"{r:g}", ha="center", va="center", fontsize=7, zorder=4,
                        color="black" if v > 0.45 else "white")
    ax.set_xlim(-0.5, 5.5)
    ax.set_xticks(range(6)); ax.set_xticklabels([TASK_LABEL[t] for t in REAL_TASKS], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(COND_LABEL[c], loc="left", pad=3)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=2)
axh[1].set_ylim(5.5, -0.5)
axh[1].set_yticks(range(6)); axh[1].set_yticklabels([ARM_LABEL[a] for a in ARMS])
axh[0].tick_params(labelleft=False, left=False)
cb = fig.colorbar(im, cax=axc); cb.outline.set_visible(False)
cb.set_label(r"biology score at $b^{*}$", fontsize=8, labelpad=3); cb.ax.tick_params(labelsize=6, length=2)

for i, a in enumerate(ARMS):
    for c, dy, mk in ((1, -0.17, "o"), (0, 0.17, "s")):
        row = mr[(mr.cond == c) & (mr.arm == a)].iloc[0]
        axb.plot([row.lo, row.hi], [i + dy] * 2, color=ARM_COLOR[a], lw=1.4, solid_capstyle="butt")
        axb.plot(row["mean"], i + dy, mk, color=ARM_COLOR[a], ms=4.5, mec="white", mew=0.5, zorder=3)
    if i: axb.axhline(i - .5, color="#dddddd", lw=0.5, zorder=0)
axb.set_xlim(0.7, 6.3); axb.set_xticks(range(1, 7))
axb.tick_params(left=False)  # row labels repeat panel a (shared y axis)
plt.setp(axb.get_yticklabels(), visible=True)
axb.spines["left"].set_visible(False)
axb.yaxis.set_tick_params(labelleft=True)
axb.set_xlabel("mean rank over ranked real tasks\n\u2190 better (1 = best)")
axb.set_title("mean rank; bars, range over tasks", loc="left", pad=3)
axb.plot([], [], "o", color=GREY, ms=4.5, label=COND_LABEL[1] + " (upper of each pair)")
axb.plot([], [], "s", color=GREY, ms=4.5, label=COND_LABEL[0] + " (lower)")
axb.legend(loc="upper center", bbox_to_anchor=(0.35, -0.42), handletextpad=0.2, borderaxespad=0)
# hatched key beneath panel a
key = fig.add_axes([0.12, 0.015, 0.02, 0.05]); key.axis("off")
key.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#e6e6e6", edgecolor="#8a8a8a", hatch="////", lw=0))
fig.text(0.15, 0.04, "not ranked (low-signal pair)", fontsize=7, va="center")
panel_letter(axh[1], "a", x=-0.2, y=1.14); panel_letter(axb, "b", x=-0.5, y=1.14)
check_and_save(fig, "fig2")
print(mr.round(3).to_string())
