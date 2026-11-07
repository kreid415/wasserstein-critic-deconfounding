"""Figure 2: ranking of the six penalties by biology kept at matched batch removal b*.
(a) biology score at b* (colour) and rank within the task column (number) per real task and decoder;
(b) mean rank over the ranked real tasks with the range over tasks, one column per decoder, penalties sorted by mean rank (best first), rank axis reversed (1 = best at the right).
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

fig = plt.figure(figsize=(WIDTH_IN, 5.9))
gs = fig.add_gridspec(1, 3, width_ratios=[6, 6, 0.28], wspace=0.05, left=0.13, right=0.9, top=0.93, bottom=0.6)
axh = {1: fig.add_subplot(gs[0]), 0: fig.add_subplot(gs[1])}
axc = fig.add_subplot(gs[2])
gsb = fig.add_gridspec(1, 2, wspace=0.62, left=0.2, right=0.97, top=0.3, bottom=0.08)
axb = {c: fig.add_subplot(gsb[k]) for k, c in enumerate(COND_ORDER)}  # (b) one column per decoder, own row order
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

for c, ax in axb.items():
    sub = mr[mr.cond == c].sort_values("mean").reset_index(drop=True)  # best mean rank at the top
    for i, row in sub.iterrows():
        col = ARM_COLOR[row.arm]
        ax.plot([row.lo, row.hi], [i, i], color=col, lw=2.2, alpha=0.35, solid_capstyle="round")
        ax.plot(row["mean"], i, "o", color=col, ms=5, mec="white", mew=0.5, zorder=3)
        if i: ax.axhline(i - .5, color="#dddddd", lw=0.5, zorder=0)
    ax.set_ylim(5.5, -0.5); ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f"{ARM_LABEL[a]} {m:.1f}" for a, m in zip(sub.arm, sub["mean"])])
    for t, a in zip(ax.get_yticklabels(), sub.arm): t.set_color(ARM_COLOR[a])
    ax.set_xlim(6.4, 0.6); ax.set_xticks(range(1, 7))  # reversed: best rank (1) at the right
    ax.tick_params(left=False); ax.spines["left"].set_visible(False)
    title = COND_LABEL[c].replace(" decoder", "")
    ax.set_title(f"{title} decoder ({int(sub.n.iloc[0])} tasks), best first", loc="center", pad=4)
fig.text(0.5 * (axb[1].get_position().x0 + axb[0].get_position().x1), axb[1].get_position().y0 - 0.045,
         "rank over ranked real tasks (1 = best): dot, mean; bar, range", ha="center", va="top")
# hatched key beneath panel a
key = fig.add_axes([0.13, 0.385, 0.02, 0.028]); key.axis("off")
key.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#e6e6e6", edgecolor="#8a8a8a", hatch="////", lw=0))
fig.text(0.16, 0.399, "not ranked (low-signal pair)", fontsize=7, va="center")
panel_letter(axh[1], "a", x=-0.2, y=1.1); panel_letter(axb[1], "b", x=-0.62, y=1.2)
check_and_save(fig, "fig2")
print(mr.round(3).to_string())
