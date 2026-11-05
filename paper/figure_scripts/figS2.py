"""Supplementary Figure S2: sensitivity of the ranking to the reading rules (figure_data/sensitivity_ranks.csv, from the sweep analysis' sensitivity block).
Variants: per-run reading (each seed read on its own curve, the five-to-three per-seed ranks averaged), biology without trajectory conservation on the two
immune tasks, points with a failed fit removed (the sweep has none, so this reproduces the primary ranking), against the primary reading.
(a, b) mean rank over the real tasks that are ranked, per penalty and decoder; (c) Kendall's tau_b between the primary ranks and each variant's ranks in each
of the 13 ranked task and decoder pairs (simulations included)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from scipy.stats import kendalltau

apply_style()
sr = pd.read_csv(DATA / "sensitivity_ranks.csv")
sr = sr[~sr.low_signal]
pn = load_pn()
VARS = ["primary", "per_run", "no_trajectory", "drop_failed"]
VLAB = {"primary": "primary", "per_run": "per-run reading", "no_trajectory": "no trajectory score", "drop_failed": "failed fits removed"}
VMK = {"primary": ("o", 4.2, True), "per_run": ("s", 3.4, False), "no_trajectory": ("D", 3.4, False), "drop_failed": ("^", 3.4, False)}
VCOL = {"primary": "black", "per_run": "#D55E00", "no_trajectory": "#0072B2", "drop_failed": "#009E73"}
real = sr[sr.task.isin(REAL_TASKS)]
mr = real.groupby(["variant", "cond", "arm"])["rank"].agg(["mean", "count"]).reset_index()
for c in (0, 1):                                  # the primary variant must equal the paper's mean ranks
    for a in ARMS:
        r = mr[(mr.variant == "primary") & (mr.cond == c) & (mr.arm == a)].iloc[0]
        assert abs(r["mean"] - pnv(pn, f"mean_rank.cond{c}.{a}")) < 1e-9 and r["count"] == pnv(pn, f"mean_rank_n_tasks.cond{c}.{a}")
rp = pd.read_csv(DATA / "ranking_pairs.csv")
m = sr[sr.variant == "primary"].merge(rp[["task", "cond", "arm", "rank"]], on=["task", "cond", "arm"], suffixes=("", "_ref"))
assert len(m) == 78 and (abs(m["rank"] - m.rank_ref) < 1e-9).all()
tau = []
for (t, c), g in sr[sr.variant == "primary"].groupby(["task", "cond"]):
    base = g.set_index("arm").loc[ARMS, "rank"].values
    for v in VARS[1:]:
        o = sr[(sr.variant == v) & (sr.task == t) & (sr.cond == c)].set_index("arm").loc[ARMS, "rank"].values
        tau.append(dict(variant=v, task=t, cond=c, tau=float(kendalltau(base, o, variant="b").statistic)))
tau = pd.DataFrame(tau)
assert len(tau) == 39
print(tau.groupby("variant").tau.agg(["min", "median", "mean", "max", "count"]).round(3).to_string())
print("pairs where a variant changes the rank-1 penalty:", {v: int(sum(
    sr[(sr.variant == v) & (sr.task == t) & (sr.cond == c)].set_index("arm").loc[ARMS, "rank"].idxmin() != g.set_index("arm").loc[ARMS, "rank"].idxmin()
    for (t, c), g in sr[sr.variant == "primary"].groupby(["task", "cond"]))) for v in VARS[1:]})

fig = plt.figure(figsize=(WIDTH_IN, 3.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.34, left=0.075, right=0.995, top=0.88, bottom=0.34)
for k, c in enumerate(COND_ORDER):
    ax = fig.add_subplot(gs[k])
    for i, a in enumerate(ARMS):
        for jv, v in enumerate(VARS):
            r = mr[(mr.variant == v) & (mr.cond == c) & (mr.arm == a)].iloc[0]
            mk, ms, filled = VMK[v]
            ax.plot(i + (jv - 1.5) * 0.14, r["mean"], mk, color=VCOL[v], mfc=VCOL[v] if filled else "none", ms=ms, mew=0.9, zorder=3)
    ax.set_xticks(range(6)); ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], rotation=60, ha="right", rotation_mode="anchor")
    ax.set_ylim(6.3, 0.7); ax.set_yticks(range(1, 7)); ax.set_xlim(-0.6, 5.6)
    ax.set_title(COND_LABEL[c], loc="left", pad=3)
    if k == 0: ax.set_ylabel("mean rank over ranked real tasks\n(1 = best)")
    panel_letter(ax, "ab"[k], x=-0.2 if k == 0 else -0.14)
axc = fig.add_subplot(gs[2])
for jv, v in enumerate(VARS[1:]):
    d = tau[tau.variant == v]
    for c, mk in ((1, "o"), (0, "s")):
        e = d[d.cond == c]
        xs = jv + (np.arange(len(e)) - (len(e) - 1) / 2) * 0.07 + (0.0 if c else 0.0)
        axc.plot(xs, e.tau, mk, color=VCOL[v], mfc=VCOL[v] if c else "none", ms=3, mew=0.8, zorder=3)
axc.axhline(1, color=GREY, lw=0.6, ls=":")
axc.set_xticks(range(3)); axc.set_xticklabels([VLAB[v] for v in VARS[1:]], rotation=60, ha="right", rotation_mode="anchor")
axc.set_ylabel(r"Kendall $\tau_b$ with primary ranks"); axc.set_xlim(-0.6, 2.6); axc.set_ylim(-0.45, 1.08)
axc.set_title("per task and decoder pair", loc="left", pad=3)
panel_letter(axc, "c", x=-0.22)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=VCOL[v], marker=VMK[v][0], ms=VMK[v][1], lw=0, mfc=VCOL[v] if VMK[v][2] else "none", label=VLAB[v]) for v in VARS]
h += [Line2D([], [], color=GREY, marker="o", ms=3, lw=0, label="(c) batch-conditioned"), Line2D([], [], color=GREY, marker="s", mfc="none", ms=3, lw=0, label="(c) unconditioned")]
fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.3)
check_and_save(fig, "figS2")
