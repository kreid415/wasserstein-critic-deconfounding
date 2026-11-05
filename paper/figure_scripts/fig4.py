"""Figure 4: per-penalty dot plots, one point per real task (n = 6 per decoder) and the mean (black tick).
(a) hypervolume; (b) collapse rate with exact 95% intervals; (c) local minus global batch removal at b*.
Within each penalty band: upper row batch-conditioned decoder (circles), lower row unconditioned decoder (squares);
tasks run top to bottom in the order of Figure 3. Ringed point = the low-signal immune pair with the
batch-conditioned decoder: kept in the mean in (b) (as in the Results text), excluded from the mean in (c);
hypervolume is not evaluable (n.e.) for it, so it is absent from (a).
Reads only figure_data/paper_numbers.json."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.lines import Line2D

apply_style()
pn = load_pn()
LOW = {(t, c) for c in (0, 1) for t in pnv(pn, f"low_signal_tasks.cond{c}") if t in REAL_TASKS}
assert LOW == {("immune", 1)}

def val(kind, a, t, c):
    return pnv(pn, f"{kind}.{a}.{t}.cond{c}")

# checks against numbers quoted in the Results text (Pass 8 outline)
hv = [val("hypervolume", "discriminator", t, 1) for t in REAL_TASKS if val("hypervolume", "discriminator", t, 1) is not None]
assert (round(min(hv), 3), round(max(hv), 3)) == (0.029, 0.255)
assert round(val("collapse_rate", "discriminator", "immune_hum_mou", 1), 2) == 0.73
assert round(val("local_minus_global", "discriminator", "lung", 1), 3) == -0.356

PANELS = [("hypervolume", "hypervolume", "higher = better", True),
          ("collapse_rate", "collapse rate (percentage points)", "lower = better", True),
          ("local_minus_global", "local minus global batch removal", "positive = stronger locally", False)]
titles = ["hypervolume", "collapse rate, exact 95% interval", r"local minus global removal at $b^{*}$"]
fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 4.6), sharey=True,
                         gridspec_kw=dict(wspace=0.08, left=0.125, right=0.995, top=0.92, bottom=0.2))
off = np.linspace(-0.12, 0.12, 6)
for ax, (kind, xlabel, cue, low_in_mean), title in zip(axes, PANELS, titles):
    for i, a in enumerate(ARMS):
        if i: ax.axhline(i - .5, color="#dddddd", lw=0.5, zorder=0)
        for c, base, mk in ((1, -0.22, "o"), (0, 0.22, "s")):
            ys, xs = [], []
            for j, t in enumerate(REAL_TASKS):
                y = i + base + off[j]
                x = val(kind, a, t, c)
                if x is None:  # hypervolume is null (not evaluable, n.e.) in low-signal pairs
                    assert kind == "hypervolume" and (t, c) in LOW
                    ax.text(0.015, y, "n.e.", transform=ax.get_yaxis_transform(), fontsize=5.5, color=GREY, va="center", ha="left")
                    continue
                scale = 100 if kind == "collapse_rate" else 1
                x *= scale
                if kind == "collapse_rate":
                    lo, hi = [v * 100 for v in val("collapse_ci", a, t, c)]
                    ax.plot([lo, hi], [y, y], color=ARM_COLOR[a], lw=0.6, alpha=0.5, zorder=1)
                ringed = (t, c) in LOW
                ax.plot(x, y, mk, color=ARM_COLOR[a], ms=3.6, mec="black" if ringed else "white",
                        mew=0.9 if ringed else 0.3, zorder=3)
                if low_in_mean or not ringed:
                    xs.append(x)
            ax.plot([np.mean(xs)] * 2, [i + base - 0.15, i + base + 0.15], color="black", lw=1.6, zorder=4,
                    solid_capstyle="butt")
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel(f"{xlabel}\n({cue})")
    ax.margins(x=0.05)
axes[0].set_ylim(5.5, -0.5)
axes[0].set_yticks(range(6)); axes[0].set_yticklabels([ARM_LABEL[a] for a in ARMS])
axes[0].tick_params(axis="y", length=0)
for ax in axes[1:]: ax.tick_params(left=False)
axes[1].set_xlim(-3, 103)
axes[0].set_xlim(-0.075, 0.46)
axes[0].spines["bottom"].set_bounds(0, 0.45)
axes[0].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
axes[2].axvline(0, color="#444444", ls="--", lw=0.7, zorder=0)
for ax, L in zip(axes, "abc"): panel_letter(ax, L, x=-0.02 if L != "a" else -0.02, y=1.04)
h = [Line2D([], [], marker="o", color=GREY, lw=0, ms=4, label=COND_LABEL[1] + " (upper row)"),
     Line2D([], [], marker="s", color=GREY, lw=0, ms=4, label=COND_LABEL[0] + " (lower row)"),
     Line2D([], [], color="black", lw=1.6, label="mean over tasks"),
     Line2D([], [], marker="o", mfc="none", color="white", mec="black", mew=0.8, lw=0, ms=7, label="low-signal immune pair")]
fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.55, 0.0), columnspacing=1.2, handletextpad=0.3)
check_and_save(fig, "fig4")
