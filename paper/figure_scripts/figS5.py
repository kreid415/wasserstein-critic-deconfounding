"""Supplementary Figure S5: scaling checks (exploratory; PAPER_PLAN section 5, decision SI-82). The six penalties are re-ranked at matched batch removal after each of
four changes of how the metrics are scaled or combined (figure_scripts/scaling_checks.py, from figure_data/metric_scores.csv): scIB min-max scaling over one comparison
set per task (every scored fit of the task), the same with each of the 13 method families left out of the set, z-score standardisation, and a composite without the three
silhouette metrics. The primary reading (unscaled metrics) is reproduced from the same file before any variant is run (asserted). Ranked pairs only (13).
(a) Kendall's tau_b between the primary ranks and each variant's ranks, one point per task and decoder pair (leave-one-family-out: one point per pair and family);
(b, c) mean rank over the ranked real tasks per penalty and decoder, primary and variants; the grey bar is the range over the 13 leave-one-family-out runs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
import scaling_checks as SC

apply_style()
rp = pd.read_csv(DATA / "ranking_pairs.csv")
SC.primary_check(rp)
pairs = sorted({(t, c) for t, c in zip(rp.task, rp.cond)})
V = {"primary": ("raw", None), "minmax": ("minmax", None), "zscore": ("zscore", None), "no_silhouette": ("no_silhouette", None)}
V.update({f"lomo_{f}": ("lomo", f) for f in SC.FAMILIES})
res = SC.run(pairs, V)
res.to_csv(DATA / "scaling_check_ranks.csv", index=False)         # derived by this script; its checksum is in MANIFEST.sha256
base = res[res.variant == "primary"].set_index(["task", "cond", "arm"])["rank"]
tau = pd.DataFrame([dict(variant=v, task=t, cond=c, tau=SC.kendall([base[(t, c, a)] for a in ARMS],
                                                                  res[(res.variant == v) & (res.task == t) & (res.cond == c)].set_index("arm").loc[ARMS, "rank"].values))
                    for v in V if v != "primary" for t, c in pairs])
tau["group"] = tau.variant.where(~tau.variant.str.startswith("lomo_"), "lomo")
print(tau.groupby("group").tau.agg(["min", "median", "mean", "max", "count"]).round(3).to_string())
real = res[res.task.isin(REAL_TASKS)]
mr = real.groupby(["variant", "cond", "arm"])["rank"].mean().reset_index()
GROUPS = ["minmax", "lomo", "zscore", "no_silhouette"]
GLAB = {"minmax": "min-max", "lomo": "min-max, family\nleft out (13)", "zscore": "z-score", "no_silhouette": "no silhouette\nmetrics"}
GCOL = {"primary": "black", "minmax": "#D55E00", "zscore": "#0072B2", "no_silhouette": "#009E73"}
GMK = {"primary": ("o", 4.2, True), "minmax": ("s", 3.4, False), "zscore": ("D", 3.4, False), "no_silhouette": ("^", 3.4, False)}

fig = plt.figure(figsize=(WIDTH_IN, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1, 1], wspace=0.34, left=0.075, right=0.995, top=0.89, bottom=0.42)
axa = fig.add_subplot(gs[0])
rng = np.random.default_rng(0)                                      # jitter only
for jg, g in enumerate(GROUPS):
    d = tau[tau.group == g]
    for c, mk in ((1, "o"), (0, "s")):
        e = d[d.cond == c]
        xs = jg + rng.uniform(-0.28, 0.28, len(e))
        axa.plot(xs, e.tau, mk, color=GCOL.get(g, "#444444"), mfc=(GCOL.get(g, "#444444") if c else "none"), ms=2.4 if g == "lomo" else 3.2, mew=0.7, alpha=0.55 if g == "lomo" else 1, zorder=3)
axa.axhline(1, color=GREY, lw=0.6, ls=":")
axa.set_xticks(range(4)); axa.set_xticklabels([GLAB[g] for g in GROUPS], rotation=60, ha="right", rotation_mode="anchor")
axa.set_ylabel(r"Kendall $\tau_b$ with primary ranks"); axa.set_xlim(-0.6, 3.6); axa.set_ylim(-0.7, 1.08)
axa.set_title("per task and decoder pair", loc="left", pad=3); panel_letter(axa, "a", x=-0.2)
for k, c in enumerate(COND_ORDER):
    ax = fig.add_subplot(gs[k + 1])
    for i, a in enumerate(ARMS):
        lo = mr[(mr.variant.str.startswith("lomo_")) & (mr.cond == c) & (mr.arm == a)]["rank"]
        ax.plot([i + 0.0] * 2, [lo.min(), lo.max()], "-", color="#bbbbbb", lw=3.2, solid_capstyle="butt", zorder=1)
        for jv, v in enumerate(["primary", "minmax", "zscore", "no_silhouette"]):
            r = mr[(mr.variant == v) & (mr.cond == c) & (mr.arm == a)]["rank"].iloc[0]
            mk, ms_, filled = GMK[v]
            ax.plot(i + (jv - 1.5) * 0.14, r, mk, color=GCOL[v], mfc=GCOL[v] if filled else "none", ms=ms_, mew=0.9, zorder=3)
    ax.set_xticks(range(6)); ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], rotation=60, ha="right", rotation_mode="anchor")
    ax.set_ylim(6.3, 0.7); ax.set_yticks(range(1, 7)); ax.set_xlim(-0.6, 5.6)
    ax.set_title(COND_LABEL[c], loc="left", pad=3)
    if k == 0: ax.set_ylabel("mean rank over ranked real tasks\n(1 = best)")
    panel_letter(ax, "bc"[k], x=-0.2 if k == 0 else -0.14)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=GCOL[v], marker=GMK[v][0], ms=GMK[v][1], lw=0, mfc=GCOL[v] if GMK[v][2] else "none", label={"primary": "primary (unscaled)", "minmax": "min-max", "zscore": "z-score", "no_silhouette": "no silhouette metrics"}[v]) for v in GMK]
h += [Line2D([], [], color="#bbbbbb", lw=3.2, label="range, leave-one-family-out"), Line2D([], [], color=GREY, marker="o", ms=3, lw=0, label="(a) batch-conditioned"), Line2D([], [], color=GREY, marker="s", mfc="none", ms=3, lw=0, label="(a) unconditioned")]
fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.3)
check_and_save(fig, "figS5")
