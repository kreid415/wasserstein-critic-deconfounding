"""Figure 2: ranking of the six penalties by biology kept at matched batch removal b*.
(a) biology score at b* (colour) and rank among the penalties (number), tasks as rows and penalties as columns, one block per
decoder (batch-conditioned above, unconditioned below); colour bar under the panel;
(b) mean rank over the ranked real tasks with the range over tasks, one block per decoder beside its block in (a), penalties
sorted by mean rank (best first), rank axis reversed (1 = best at the right).
Reads only figure_data/paper_numbers.json (sweep analysis 8595a16)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *

apply_style()
pn = load_pn()
mr = mean_rank_table(pn)
low = {c: set(pnv(pn, f"low_signal_tasks.cond{c}")) for c in (0, 1)}

# rows = tasks, columns = penalties
bio = {c: np.array([[pnv(pn, f"bio_at_bstar.{a}.{t}.cond{c}") for a in ARMS] for t in REAL_TASKS]) for c in (0, 1)}
rank = {c: np.array([[pnv(pn, f"rank.{a}.{t}.cond{c}") if f"rank.{a}.{t}.cond{c}" in pn else np.nan
                      for a in ARMS] for t in REAL_TASKS]) for c in (0, 1)}
assert not np.isnan(bio[0]).any() and not np.isnan(bio[1]).any()
for c in (0, 1):  # a row is unranked exactly when its task is low-signal
    for i, t in enumerate(REAL_TASKS):
        assert np.isnan(rank[c][i, :]).all() == (t in low[c]), (c, t)
vmin, vmax = min(bio[0].min(), bio[1].min()), max(bio[0].max(), bio[1].max())

fig = plt.figure(figsize=(WIDTH_IN, 5.6))
TOP, BOTTOM, GAP = 0.93, 0.22, 0.065           # the two decoder blocks share these rows in (a) and (b)
BH = (TOP - BOTTOM - GAP) / 2
ROWS = {1: TOP - BH, 0: BOTTOM}               # batch-conditioned block above, unconditioned below
A_X, A_W, B_X, B_W = 0.19, 0.31, 0.68, 0.305
cmap = plt.get_cmap("viridis")
axa, axb = {}, {}
for c in COND_ORDER:
    ax = axa[c] = fig.add_axes([A_X, ROWS[c], A_W, BH])
    im = ax.pcolormesh(np.arange(-0.5, 6), np.arange(-0.5, 6), bio[c], cmap=cmap, vmin=vmin, vmax=vmax,
                       edgecolors="white", linewidth=1.0)
    for i in range(6):
        for j in range(6):
            if np.isnan(rank[c][i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#e6e6e6", edgecolor="#8a8a8a",
                                           hatch="////", lw=0, zorder=3))
            else:
                v = (bio[c][i, j] - vmin) / (vmax - vmin)
                ax.text(j, i, f"{rank[c][i, j]:g}", ha="center", va="center", fontsize=7, zorder=4,
                        color="black" if v > 0.45 else "white")
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(5.5, -0.5)
    ax.set_yticks(range(6)); ax.set_yticklabels([TASK_LABEL[t] for t in REAL_TASKS])
    ax.set_xticks(range(6))
    if c == COND_ORDER[-1]:
        ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=2)
    ax.set_title(COND_LABEL[c], loc="left", pad=4, fontsize=10)

    bx = axb[c] = fig.add_axes([B_X, ROWS[c], B_W, BH])
    sub = mr[mr.cond == c].sort_values("mean", kind="stable").reset_index(drop=True)  # best mean rank at the top
    for i, row in sub.iterrows():
        col = ARM_COLOR[row.arm]
        bx.plot([row.lo, row.hi], [i, i], color=col, lw=2.2, alpha=0.35, solid_capstyle="round")
        bx.plot(row["mean"], i, "o", color=col, ms=5, mec="white", mew=0.5, zorder=3)
        if i: bx.axhline(i - .5, color="#dddddd", lw=0.5, zorder=0)
    bx.set_ylim(5.5, -0.5); bx.set_yticks(range(len(sub)))
    bx.set_yticklabels([f"{ARM_LABEL[a]} {m:.1f}" for a, m in zip(sub.arm, sub["mean"])])
    for t, a in zip(bx.get_yticklabels(), sub.arm): t.set_color(ARM_COLOR[a])
    bx.set_xlim(6.4, 0.6); bx.set_xticks(range(1, 7))  # reversed: best rank (1) at the right
    if c != COND_ORDER[-1]: bx.set_xticklabels([])
    bx.tick_params(left=False); bx.spines["left"].set_visible(False)
    bx.set_title(f"{COND_LABEL[c]} ({int(sub.n.iloc[0])} tasks)", loc="right", pad=4, fontsize=10)
axb[COND_ORDER[-1]].set_xlabel("rank over tasks (1 = best); best penalty first\ndot, mean; bar, range over tasks",
                               fontsize=7, ha="right", x=1.0)

cax = fig.add_axes([A_X + 0.03, 0.075, A_W - 0.06, 0.014])
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_visible(False)
cb.set_label(r"biology score at $b^{*}$", fontsize=7, labelpad=2); cb.ax.tick_params(labelsize=6, length=2)
key = fig.add_axes([0.02, 0.012, 0.02, 0.022]); key.axis("off")
key.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#e6e6e6", edgecolor="#8a8a8a", hatch="////", lw=0))
fig.text(0.05, 0.023, "not ranked (low-signal pair)", fontsize=7, va="center")
fig.text(0.015, 0.965, "a", fontsize=10, fontweight="bold", va="bottom")
fig.text(0.535, 0.965, "b", fontsize=10, fontweight="bold", va="bottom")
check_and_save(fig, "fig2")
print(mr.round(3).to_string())
