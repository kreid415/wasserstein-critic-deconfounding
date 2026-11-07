"""Figure 6: width of each penalty's range of good strengths (decades), one point per real task and decoder.
Good strength: mean selection score >= lower 95% bound of the best strength; width = span of good strengths.
Upper row of each band batch-conditioned decoder (circles), lower row unconditioned (squares); tasks top to
bottom in the order of Figure 2a; ringed point = low-signal immune pair (batch-conditioned decoder), included
as in the Results text. Reads only figure_data/paper_numbers.json."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.lines import Line2D

apply_style()
pn = load_pn()
W = {(a, t, c): pnv(pn, f"width_decades.{a}.{t}.cond{c}") for a in ARMS for t in REAL_TASKS for c in (0, 1)}
assert all(v is not None for v in W.values())
rng = lambda a, c: (round(min(W[a, t, c] for t in REAL_TASKS), 1), round(max(W[a, t, c] for t in REAL_TASKS), 1))
assert rng("discriminator", 1) == (0.0, 4.5) and rng("sinkhorn", 1) == (2.0, 5.5) and rng("pooled", 0) == (0.5, 4.5)  # Results text
mean = {(a, c): np.mean([W[a, t, c] for t in REAL_TASKS]) for a in ARMS for c in (0, 1)}

fig, ax = plt.subplots(figsize=(WIDTH_IN * 0.62, 3.6))
fig.subplots_adjust(left=0.24, right=0.97, top=0.95, bottom=0.28)
off = np.linspace(-0.12, 0.12, 6)
for i, a in enumerate(ARMS):
    if i: ax.axhline(i - .5, color="#dddddd", lw=0.5, zorder=0)
    for c, base, mk in ((1, -0.22, "o"), (0, 0.22, "s")):
        for j, t in enumerate(REAL_TASKS):
            ringed = (t, c) == ("immune", 1)
            ax.plot(W[a, t, c], i + base + off[j], mk, color=ARM_COLOR[a], ms=3.8,
                    mec="white", mew=0.3, zorder=3)
            if ringed:
                ax.plot(W[a, t, c], i + base + off[j], "o", mfc="none", mec="black", mew=0.8, ms=7, zorder=4)
        ax.plot([mean[a, c]] * 2, [i + base - 0.15, i + base + 0.15], color="black", lw=1.6, zorder=4, solid_capstyle="butt")
ax.set_ylim(5.5, -0.5); ax.set_yticks(range(6)); ax.set_yticklabels([ARM_LABEL[a] for a in ARMS])
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.3, 6); ax.set_xticks(range(0, 7))
ax.set_xlabel("width of the range of good strengths (decades)\n(wider = more strengths close to the best)")
h = [Line2D([], [], marker="o", color=GREY, lw=0, ms=4, label=COND_LABEL[1] + " (upper row)"),
     Line2D([], [], marker="s", color=GREY, lw=0, ms=4, label=COND_LABEL[0] + " (lower row)"),
     Line2D([], [], color="black", lw=1.6, label="mean over tasks"),
     Line2D([], [], marker="o", mfc="none", color="white", mec="black", mew=0.8, lw=0, ms=7, label="low-signal immune pair")]
fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.6, 0.0), columnspacing=1.2, handletextpad=0.3)
check_and_save(fig, "fig6")
