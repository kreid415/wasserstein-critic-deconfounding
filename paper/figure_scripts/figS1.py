"""Supplementary Figure S1: the two simulation tasks, each on its own, both decoders.
Top row: batch score against biology score for each penalty's sweep strengths (figure_data/curves.csv, seed means, plain scVI as the star,
b* dashed from ranking_pairs.csv / low_signal_pairs.csv, open grey markers: each standard tool's best setting by selection score,
figure_data/baseline_points.csv). Bottom row: biology score at b* per penalty with the leave-one-seed-out standard error and, where the pair is
ranked, the rank (number above the point); with the batch-conditioned decoder both simulations are low-signal: the reading is shown (open
marker) but not ranked and has no standard error."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.lines import Line2D

apply_style()
SIMS = ["sim1", "sim2"]
SIM_LABEL = {"sim1": "Simulation 1", "sim2": "Simulation 2"}
cv = pd.read_csv(DATA / "curves.csv"); bp = pd.read_csv(DATA / "baseline_points.csv")
rk = pd.concat([pd.read_csv(DATA / "ranking_pairs.csv"), pd.read_csv(DATA / "low_signal_pairs.csv")], ignore_index=True)
assert set(rk[rk.task.isin(SIMS)].groupby(["task", "cond"]).size()) == {6}
assert rk[rk.task.isin(SIMS)].groupby(["task", "cond"]).ngroups == 4
TOOLS = {"scvi_adv": ("o", "scVI adversary"), "harmony": ("s", "Harmony"), "scanorama": ("D", "Scanorama"),
         "sysvi": ("^", "sysVI"), "scanvi": ("v", "scANVI"), "pca": ("X", "PCA")}
bp["S"] = 0.4 * bp.B + 0.6 * bp.C
best = bp[bp.arm.isin(TOOLS)].sort_values("S").groupby(["task", "cond", "arm"]).tail(1)
cols = [(t, c) for t in SIMS for c in COND_ORDER]

fig, axes = plt.subplots(2, 4, figsize=(WIDTH_IN, 4.6), gridspec_kw=dict(wspace=0.45, hspace=0.62, left=0.075, right=0.995, top=0.92, bottom=0.33))
for j, (t, c) in enumerate(cols):
    ax = axes[0, j]
    d = cv[(cv.task == t) & (cv.cond == c)]
    for a in ARMS:
        e = d[d.arm == a].sort_values("lam")
        ax.plot(e.B, e.C, "-o", color=ARM_COLOR[a], ms=2.2, lw=1.0, mew=0, zorder=2)
    p0 = d[d.lam == 0].iloc[0]
    ax.plot(p0.B, p0.C, "*", color="black", ms=8, zorder=5, mec="white", mew=0.4)
    g = rk[(rk.task == t) & (rk.cond == c)]
    ax.axvline(g.bstar.iloc[0], color="#444444", ls="--", lw=0.7, zorder=1)
    for _, row in best[(best.task == t) & (best.cond == c)].iterrows():
        ax.plot(row.B, row.C, TOOLS[row.arm][0], mfc="none", mec=GREY, ms=4, mew=0.9, zorder=3)
    ax.set_title(f"{SIM_LABEL[t]}, {'conditioned' if c else 'unconditioned'}", loc="left", pad=3)
    ax.set_xlabel("batch score"); ax.margins(0.06); ax.locator_params(nbins=4)
    if j == 0: ax.set_ylabel("biology score")
    ax2 = axes[1, j]
    for i, a in enumerate(ARMS):
        r = g[g.arm == a].iloc[0]
        ranked = not r.low_signal
        ax2.errorbar(i, r.bio_at_bstar, yerr=(r.loso_se if ranked else None), fmt="o", color=ARM_COLOR[a], mfc=(ARM_COLOR[a] if ranked else "white"),
                     ms=4, lw=0.9, capsize=1.5, zorder=3)
        if ranked:
            ax2.annotate(f"{r['rank']:g}", (i, r.bio_at_bstar + r.loso_se), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=6)
    ax2.set_xticks(range(6)); ax2.set_xticklabels([ARM_LABEL[a] for a in ARMS], rotation=60, ha="right", rotation_mode="anchor")
    ax2.set_xlim(-0.6, 5.6)
    lo_, hi_ = g.bio_at_bstar.min(), g.bio_at_bstar.max(); mid, half = (lo_ + hi_) / 2, max(0.03, 0.75 * (hi_ - lo_) + 0.01)
    ax2.set_ylim(mid - half, mid + half)          # at least 0.06 wide, so that readings 0.001 apart are not drawn as large differences
    if j == 0: ax2.set_ylabel("biology at $b^{*}$")
    if not g.low_signal.iloc[0]: ax2.set_title("ranked", loc="left", pad=3, fontsize=7, color=GREY)
    else: ax2.set_title("low-signal, not ranked", loc="left", pad=3, fontsize=7, color=GREY)
for r_, lab in enumerate("ab"):
    fig.text(0.003, axes[r_, 0].get_position().y1 + 0.03, lab, fontsize=10, fontweight="bold", va="bottom")
h = [Line2D([], [], color=ARM_COLOR[a], marker="o", ms=3, lw=1.2, label=ARM_LABEL[a]) for a in ARMS]
h += [Line2D([], [], color="black", marker="*", ms=8, lw=0, label="plain scVI")]
h += [Line2D([], [], color=GREY, marker=m, mfc="none", ms=4, lw=0, label=l) for m, l in TOOLS.values()]
fig.legend(handles=h, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.01), columnspacing=1.0, handletextpad=0.3)
check_and_save(fig, "figS1")
