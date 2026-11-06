"""Supplementary Figure S4: each scIB batch and biology metric against penalty strength, per penalty, task and decoder (S4a batch-conditioned, S4b unconditioned).
Seed means over the three sweep seeds of the raw, unscaled metrics of every fit (figure_data/metric_scores.csv; the same files as the sweep analysis); the horizontal
dotted line is plain scVI (strength 0). Rows: the five batch metrics, then the biology metrics (cell-cycle conservation exists for the four RNA real tasks,
trajectory conservation for the two immune tasks; other panels are empty). One y range per row, so that tasks can be compared."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *

apply_style()
ms = pd.read_csv(DATA / "metric_scores.csv")
ms = ms[ms.experiment == "A1"]
cv = pd.read_csv(DATA / "curves.csv")
ROWS = [("PCR_batch", "PCR\nbatch"), ("ASW_label/batch", "batch\nASW"), ("graph_conn", "graph\nconnectivity"), ("kBET", "kBET"), ("iLISI", "iLISI"),
        ("NMI_cluster/label", "NMI"), ("ARI_cluster/label", "ARI"), ("ASW_label", "cell-type\nASW"), ("isolated_label_F1", "isolated-\nlabel F1"),
        ("isolated_label_silhouette", "isolated-\nlabel ASW"), ("cLISI", "cLISI"), ("cell_cycle_conservation", "cell\ncycle"), ("trajectory", "trajectory")]
TASKS = REAL_TASKS + ["sim1", "sim2"]
TL = dict(TASK_LABEL, sim1="simulation 1", sim2="simulation 2", immune_hum_mou="human-mouse\nimmune", atac_small="ATAC\nsmall", atac_large="ATAC\nlarge")
for c, part in ((1, "a"), (0, "b")):
    d = ms[ms.cond == c]
    fig, axes = plt.subplots(len(ROWS), len(TASKS), figsize=(WIDTH_IN, 8.9), sharex=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.5, hspace=0.25, left=0.1, right=0.995, top=0.945, bottom=0.075))
    for i, (m, lab) in enumerate(ROWS):
        lo, hi = d[m].min(), d[m].max()
        pad = 0.08 * (hi - lo + 1e-9)
        first = next(j for j, t in enumerate(TASKS) if d[d.task == t][m].notna().any())  # row label on the first drawn panel
        for j, t in enumerate(TASKS):
            ax = axes[i, j]
            g = d[d.task == t]
            if g[m].isna().all():
                ax.set_axis_off(); continue
            plain = g[g.arm == "none"][m].mean()
            ax.axhline(plain, color="black", lw=0.8, ls=":", zorder=1)
            for a in ARMS:
                e = g[g.arm == a].groupby("lam")[m].mean().sort_index()
                ax.plot(e.index, e.values, "-o", color=ARM_COLOR[a], ms=1.4, lw=0.8, mew=0, zorder=2)
            ax.set_xscale("log"); ax.set_ylim(lo - pad, hi + pad); ax.locator_params(axis="y", nbins=3)
            ax.tick_params(labelsize=5, length=2)
            if j == first: ax.set_ylabel(lab, fontsize=6.5, labelpad=2)
            if i == 0: ax.set_title(TL[t], loc="left", pad=2, fontsize=6.5)
            last = max(k for k, (mm, _) in enumerate(ROWS) if g[mm].notna().any())
            if i == last:                           # x axis under the lowest drawn panel of each column
                ax.tick_params(labelbottom=True); ax.set_xlabel("strength", fontsize=6.5)
    h = [plt.Line2D([], [], color=ARM_COLOR[a], marker="o", ms=3, lw=1.1, label=ARM_LABEL[a]) for a in ARMS] + [plt.Line2D([], [], color="black", ls=":", lw=0.9, label="plain scVI")]
    fig.legend(handles=h, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.3)
    fig.text(0.003, 0.985, f"{part}  {COND_LABEL[c]}", fontsize=8, fontweight="bold", va="top")
    check_and_save(fig, f"figS4{part}")
# positive control: the seed-mean metrics reproduce the sweep's curve points (B = mean of the five batch metrics)
b = ms[(ms.task == "pancreas") & (ms.cond == 1) & (ms.arm == "mmd")].groupby("lam")[[m for m, _ in ROWS[:5]]].mean().mean(axis=1)
c_ = cv[(cv.task == "pancreas") & (cv.cond == 1) & (cv.arm == "mmd") & (cv.lam > 0)].set_index("lam").B
assert np.allclose(b.loc[c_.index].values, c_.values, atol=1e-9), "metric means do not reproduce curves.csv"
print("per-metric means reproduce curves.csv B (pancreas, conditioned, MMD)")
