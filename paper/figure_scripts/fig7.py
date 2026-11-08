"""Figure 7, batch-conditioned decoder.
(a) expected best selection score among k settings drawn at random, mean over the six real tasks of each method's
    per-task curve (figure_data/best_of_k_real_mean.csv); only k where all six tasks have k settings is drawn,
    except Scanorama k = 6 (5 of 6 tasks; open marker) because Scanorama atac_large strength 80 could not be fitted.
(b) batch against biology score per real task at the single strength (or setting) chosen on other task families
    (figure_data/transfer_strength.csv: penalties, Harmony, Scanorama, sysVI) and the fixed settings in
    figure_data/single_methods.csv (plain scVI none|0|10, scVI adversary, scANVI, PCA pca|0|10)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.lines import Line2D

apply_style()
COND = 1
bk = pd.read_csv(DATA / "best_of_k_real_mean.csv"); bk = bk[bk.cond == COND]
ts = pd.read_csv(DATA / "transfer_strength.csv"); ts = ts[(ts.cond == COND) & ts.task.isin(REAL_TASKS)]
sm = pd.read_csv(DATA / "single_methods.csv"); sm = sm[(sm.cond == COND) & sm.task.isin(REAL_TASKS)]
TOOL_LS = {"harmony": ("--", "Harmony"), "scanorama": (":", "Scanorama"), "sysvi": ("-.", "sysVI")}
TOOL_MK = {"harmony": ("s", "Harmony"), "scanorama": ("D", "Scanorama"), "sysvi": ("^", "sysVI"),
           "scanvi": ("v", "scANVI"), "pca": ("X", "PCA"), "scvi_adv": ("o", "scVI adversary")}

fig = plt.figure(figsize=(WIDTH_IN, 4.1))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.17, left=0.07, right=0.995, top=0.92, bottom=0.2)
axa = fig.add_subplot(gs[0])
gsb = gs[1].subgridspec(2, 3, wspace=0.5, hspace=0.45)

# ---- (a)
plain = bk[(bk.family == "none") & (bk.k == 1)].mean_E.iloc[0]
axa.axhline(plain, color="black", lw=1.2, zorder=1)
axa.text(13.4, plain + 0.0008, "plain scVI", va="bottom", ha="right", fontsize=7)
for a in ARMS:
    d = bk[(bk.family == a) & (bk.n_tasks == 6)].sort_values("k")
    axa.plot(d.k, d.mean_E, "-o", color=ARM_COLOR[a], ms=3, lw=1.1, zorder=3)
for f, (ls, lab) in TOOL_LS.items():
    d = bk[bk.family == f].sort_values("k"); full = d[d.n_tasks == 6]
    axa.plot(full.k, full.mean_E, ls, color=GREY, lw=1.1, zorder=2)
    end = full.iloc[-1]; y_end = end.mean_E
    part = d[d.n_tasks < 6]
    if len(part):
        axa.plot([end.k, part.k.iloc[0]], [end.mean_E, part.mean_E.iloc[0]], ls, color=GREY, lw=0.8, zorder=2)
        axa.plot(part.k, part.mean_E, "o", mfc="white", mec=GREY, ms=3.5, mew=0.9, zorder=3)
        end, y_end = part.iloc[0], part.mean_E.iloc[0]
    axa.text(end.k + 0.25, y_end, lab, fontsize=6.5, color="#444444", va="center")
sv = bk[bk.family == "scanvi"].mean_E.iloc[0]
av = bk[bk.family == "scvi_adv"].mean_E.iloc[0]
axa.plot(1, sv, "v", mfc="none", mec=GREY, ms=5, mew=1.0, zorder=3)
axa.plot(1, av, "o", mfc="none", mec=GREY, ms=4, mew=1.0, zorder=3)
axa.set_xlim(0.5, 13.5); axa.set_xticks([1, 2, 4, 6, 8, 10, 12])
axa.set_xlabel("number of settings tried, $k$")
axa.set_ylabel("expected best selection score\n(higher = better)")
axa.set_title("best score against number of tries", loc="left", pad=4)
hl = [Line2D([], [], color=ARM_COLOR[a], marker="o", ms=3, lw=1.1, label=ARM_LABEL[a]) for a in ARMS]
hl += [Line2D([], [], color=GREY, marker="v", mfc="none", ms=5, lw=0, label="scANVI (uses labels)"),
       Line2D([], [], color=GREY, marker="o", mfc="none", ms=4, lw=0, label="scVI adversary")]
axa.legend(handles=hl, loc="upper right", ncol=2, fontsize=6.3, handlelength=1.6, handletextpad=0.4, labelspacing=0.25, columnspacing=1.0)

# ---- (b): one small panel per real task
star = sm[sm.name == "none|0|10"].set_index("task")
for n, t in enumerate(REAL_TASKS):
    axb = fig.add_subplot(gsb[n // 3, n % 3])
    for f in ARMS:
        r = ts[(ts.family == f) & (ts.task == t)].iloc[0]
        axb.plot(r.B, r.C, "o", color=ARM_COLOR[f], ms=3.4, mew=0, zorder=3)
    for f in ("harmony", "scanorama", "sysvi"):
        r = ts[(ts.family == f) & (ts.task == t)].iloc[0]
        axb.plot(r.B, r.C, TOOL_MK[f][0], mfc="none", mec=GREY, ms=4, mew=0.9, zorder=2)
    for name, f in (("scanvi|0|10", "scanvi"), ("scvi_adv|0|10", "scvi_adv"), ("pca|0|10", "pca")):
        r = sm[(sm.name == name) & (sm.task == t)].iloc[0]
        axb.plot(r.B, r.C, TOOL_MK[f][0], mfc="none", mec=GREY, ms=4, mew=0.9, zorder=2)
    s_ = star.loc[t]
    axb.plot(s_.B, s_.C, "*", color="black", ms=9, mec="white", mew=0.4, zorder=5)
    axb.set_title(TASK_LABEL[t], loc="left", pad=3)
    axb.margins(0.1); axb.locator_params(nbins=3)
    if n // 3 == 1: axb.set_xlabel("batch score")
    if n % 3 == 0: axb.set_ylabel("biology score")
    if n == 0: bax = axb
hb = [Line2D([], [], color="black", marker="*", ms=8, lw=0, label="plain scVI")]
hb += [Line2D([], [], color=ARM_COLOR[a], marker="o", ms=3.5, lw=0, label=ARM_LABEL[a]) for a in ARMS]
hb += [Line2D([], [], color=GREY, marker=m, mfc="none", ms=4, lw=0, label=l) for m, l in TOOL_MK.values()]
fig.legend(handles=hb, loc="lower center", ncol=7, bbox_to_anchor=(0.55, 0.0), columnspacing=0.9, handletextpad=0.2, fontsize=6.5)
fig.text(0.995, 0.995, "(b) higher = better on both axes", ha="right", va="top", fontsize=7, color=GREY)
panel_letter(axa, "a", x=-0.2, y=1.04)
fig.text(gsb[0, 0].get_position(fig).x0 - 0.065, axa.get_position().y1 + 0.01, "b", fontsize=10, fontweight="bold", va="bottom")
check_and_save(fig, "fig7")
print(ts.B.isna().sum(), ts.groupby("family").size().to_dict())
