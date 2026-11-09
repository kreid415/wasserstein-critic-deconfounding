"""Figure 3: biology kept by each penalty relative to plain scVI, at the matched batch removal b* (filled dots, ranked real
tasks) and at the penalty's strongest fitted setting (open dots, all six real tasks), one panel per decoder.
Reads figure_data/curves.csv (seed means per strength; plain scVI is strength 0) and figure_data/ranking_pairs.csv (biology at b*).
The full trade-off curves are Supplementary Figure S6 (figS6.py). Pass 23, user choice among two drafts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.lines import Line2D

apply_style()
cv = pd.read_csv(DATA / "curves.csv")
rp = pd.read_csv(DATA / "ranking_pairs.csv")
cv, rp = cv[cv.task.isin(REAL_TASKS)], rp[rp.task.isin(REAL_TASKS)]
assert cv.n_fail.sum() == 0
plain = cv[cv.lam == 0].groupby(["task", "cond"]).C.first()
rp = rp[rp["rank"].notna()].assign(d=lambda x: [r.bio_at_bstar - plain[(r.task, r.cond)] for r in x.itertuples()])
st = cv[cv.lam > 0].sort_values("lam").groupby(["task", "cond", "arm"]).tail(1)
st = st.assign(d=[r.C - plain[(r.task, r.cond)] for r in st.itertuples()])

fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.9), sharey=True, sharex=True,
                         gridspec_kw=dict(wspace=0.08, left=0.15, right=0.99, top=0.8, bottom=0.17))
for ax, c in zip(axes, COND_ORDER):
    ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=1)
    for i, a in enumerate(ARMS):
        b, s = rp[(rp.cond == c) & (rp.arm == a)], st[(st.cond == c) & (st.arm == a)]
        assert len(s) == 6 and len(b) in (5, 6)
        ax.scatter(b.d, [i - 0.15] * len(b), s=14, color=ARM_COLOR[a], lw=0, zorder=3)
        ax.scatter(s.d, [i + 0.15] * len(s), s=14, facecolor="white", edgecolor=ARM_COLOR[a], lw=0.9, zorder=3)
        if i: ax.axhline(i - .5, color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_ylim(5.5, -0.5); ax.set_yticks(range(6)); ax.set_yticklabels([ARM_LABEL[a] for a in ARMS])
    ax.set_title(f"{COND_LABEL[c]} ({rp[rp.cond == c].task.nunique()} ranked tasks)", loc="left", pad=3)
    ax.set_xlabel("biology score minus plain scVI's (higher = better)")
    ax.spines["left"].set_visible(False); ax.tick_params(left=False)
    ax.margins(x=0.04)
for k, ax in enumerate(axes):
    fig.text(ax.get_position().x0 - (0.13 if k == 0 else 0.01), 0.86, "ab"[k], fontsize=10, fontweight="bold", va="bottom")
fig.legend(handles=[Line2D([], [], marker="o", color=GREY, lw=0, ms=4, label=r"at matched batch removal $b^{*}$"),
                    Line2D([], [], marker="o", mfc="white", mec=GREY, lw=0, ms=4, label="at the strongest fitted setting")],
           loc="upper center", bbox_to_anchor=(0.57, 0.995), ncol=2, frameon=False)
check_and_save(fig, "fig3")
print(rp.groupby("cond").d.agg(["min", "max"]).round(3).to_dict(), st.groupby("cond").d.agg(["min", "max"]).round(3).to_dict())
