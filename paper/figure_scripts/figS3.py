"""Supplementary Figure S3 (redesigned in Pass 25): label transfer at the first-crossing strength, as F1 minus plain scVI's with the same decoder.
Reads figure_data/label_transfer.csv (macro F1) and figure_data/label_transfer_types.csv (F1 per cell type, made by prep_label_transfer_types.py
from the same embeddings and classifier); both hold the fits at the first-crossing strengths (a 15-nearest-neighbour classifier predicts each held-out
batch from the other batches; a type's F1 is averaged over the held-out batches that contain it and over the three seeds, five for the scvi-tools
adversary). Real tasks only: the simulations are not drawn (their values stay in label_transfer_types.csv).
Panels a, b: macro F1 minus plain scVI's per penalty, one dot per real task, black tick = median over the ranked tasks (immune with the
batch-conditioned decoder is low-signal: open dot, not in the median). Panel c: per cell type and task, plain scVI's own F1 (grey column, printed) and
each method's F1 minus plain scVI's (colour, blue above plain scVI). Writes figure_data/figS3_values.csv (exactly the plotted values)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

apply_style()
lt = pd.read_csv(DATA / "label_transfer_types.csv")
mac = pd.read_csv(DATA / "label_transfer.csv")
low = pd.read_csv(DATA / "low_signal_pairs.csv")
LOWSET = {(r.task, int(r.cond)) for r in low[low.low_signal].itertuples() if r.task in REAL_TASKS}
assert LOWSET == {("immune", 1)}, LOWSET                  # the titles and the median rule below assume this
assert len(mac) == lt.groupby(["cond", "task", "arm", "lam"], dropna=False).ngroups == 120

ADV = "scvi_adv"                                          # scvi-tools adversary: batch-conditioned decoder only
LABEL, COLOR = dict(ARM_LABEL, **{ADV: "scvi-tools adversary"}), dict(ARM_COLOR, **{ADV: GREY})
SHORT = {"discriminator": "disc.", "reference": "ref.", "pooled": "pooled", "barycenter": "bary.", "mmd": "MMD", "sinkhorn": "Sinkh.", ADV: "adv."}
GROUPS = [(1, ARMS + [ADV]), (0, ARMS)]                   # decoder, methods drawn (plain scVI is the reference)
MACRO, LIM = "macro F1", 0.15

# ---- values: one row per (decoder, real task, method, cell type) and per (decoder, real task, method) macro F1
cols = ["cond", "task", "arm", "celltype", "f1"]
types = lt[lt.task.isin(REAL_TASKS)].rename(columns={"f1_mean": "f1"})[cols]
macro = mac[mac.task.isin(REAL_TASKS)].rename(columns={"f1_mean": "f1"}).assign(celltype=MACRO)[cols]
allv = pd.concat([macro, types], ignore_index=True)
plain = allv[allv.arm == "none"].set_index(["cond", "task", "celltype"]).f1
vals = allv[allv.arm != "none"].copy()
vals["f1_plain"] = plain.reindex(pd.MultiIndex.from_frame(vals[["cond", "task", "celltype"]])).to_numpy()
assert vals.f1_plain.notna().all() and not vals.duplicated(["cond", "task", "arm", "celltype"]).any()
vals["diff"] = vals.f1 - vals.f1_plain
key = [(COND_ORDER.index(r.cond), REAL_TASKS.index(r.task), (ARMS + [ADV]).index(r.arm), r.celltype != MACRO, r.celltype) for r in vals.itertuples()]
vals = vals.iloc[sorted(range(len(vals)), key=key.__getitem__)]
vals[cols + ["f1_plain", "diff"]].to_csv(DATA / "figS3_values.csv", index=False)
assert vals[vals.celltype != MACRO]["diff"].abs().max() <= LIM          # no per-type difference is clipped by the colour scale
D = vals.set_index(["cond", "task", "arm", "celltype"])["diff"].to_dict()
P = plain.to_dict()

# ---- layout (inches from the top left of the figure)
W_IN, LEFT, LAB, GAP, CELL, HDR, GRP = WIDTH_IN, 0.17, 1.05, 0.12, 0.088, 0.32, 0.13
PAIRS = [("lung", "immune_hum_mou"), ("immune", "pancreas"), ("atac_large", "atac_small")]      # tasks of similar size share a row
ntypes = {t: lt[lt.task == t].celltype.nunique() for t in REAL_TASKS}
TOP_H, CB_H, ROW_GAP = 1.28, 0.24, 0.03
H = TOP_H + sum(max(ntypes[a], ntypes[b]) * CELL + HDR + GRP + ROW_GAP for a, b in PAIRS) + CB_H
fig = plt.figure(figsize=(W_IN, H))


def add_axes(x, y_top, w, h, **kw):
    return fig.add_axes([x / W_IN, (H - y_top - h) / H, w / W_IN, h / H], **kw)


# ---- panels a, b: macro F1 minus plain scVI's, one row per method, one dot per real task
AH, AX = 7 * 0.108, 0.98
aw = (W_IN - AX - 0.05 - GAP) / 2
axes = [add_axes(AX, 0.2, aw, AH)]
axes.append(add_axes(AX + aw + GAP, 0.2, aw, AH, sharex=axes[0], sharey=axes[0]))
off = np.linspace(-0.27, 0.27, len(REAL_TASKS))
macro_d = {k[:3]: v for k, v in D.items() if k[3] == MACRO}
for ax, (c, arms) in zip(axes, GROUPS):
    ranked = [t for t in REAL_TASKS if (t, c) not in LOWSET]
    ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=1)
    for i, a in enumerate(arms):
        if i:
            ax.axhline(i - .5, color="#e6e6e6", lw=0.5, zorder=0)
        for j, t in enumerate(REAL_TASKS):
            ax.plot(macro_d[(c, t, a)], i + off[j], "o", color=COLOR[a], ms=3.4, zorder=3, **(
                dict(mfc="white", mec=COLOR[a], mew=0.9) if (t, c) in LOWSET else dict(mec="white", mew=0.3)))
        ax.plot([np.median([macro_d[(c, t, a)] for t in ranked])] * 2, [i - 0.38, i + 0.38], color="black", lw=1.6, zorder=4, solid_capstyle="butt")
    ax.set_title(f"{COND_LABEL[c]} ({len(ranked)} ranked tasks)", pad=3)
    ax.set_xlabel("label-transfer macro F1 minus plain scVI's (higher = better)", fontsize=6.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
axes[0].set_ylim(6.5, -0.5)
axes[0].set_yticks(range(7))
axes[0].set_yticklabels([LABEL[a] for a in ARMS + [ADV]])
lo, hi = min(macro_d.values()), max(macro_d.values())
axes[0].set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))           # every dot inside the axis
axes[1].tick_params(labelleft=False)
axes[1].legend(handles=[Line2D([], [], color="black", lw=1.6, label="median over ranked tasks"),
                        Line2D([], [], marker="o", mfc="white", mec=GREY, mew=0.9, lw=0, ms=3.4, label="low-signal pair, not in the median")],
               loc="center left", bbox_to_anchor=(-0.01, 0.5 / 7), ncol=2, fontsize=6, handlelength=1.0, handletextpad=0.4, columnspacing=1.0,
               borderpad=0.0, handleheight=0.6)  # centred in the empty seventh row (no adversary with this decoder)
for L, x in zip("ab", (0.0, AX + aw + GAP - 0.17)):
    fig.text(x / W_IN, (H - 0.03) / H, L, fontsize=10, fontweight="bold", va="top")

# ---- panel c: per cell type, one small heatmap per real task
body = (W_IN - LEFT - 0.02 - GAP) / 2 - LAB                      # heat-map width without its row labels
PW, GG = 0.26, 0.09                                              # plain-F1 column, gap between the two decoder groups
CW = (body - 2 * PW - GG) / sum(len(arms) for _, arms in GROUPS)
cmap = ListedColormap(["#ececec"])
y = TOP_H
fig.text(0.0, (H - y - 0.01) / H, "c", fontsize=10, fontweight="bold", va="top")
im = None
for pair in PAIRS:
    n_max = max(ntypes[t] for t in pair)
    y += GRP + HDR
    for k, t in enumerate(pair):
        mp = {ty: np.mean([P[(c, t, ty)] for c, _ in GROUPS]) for ty in lt[lt.task == t].celltype.unique()}     # plain scVI's F1, mean of the two decoders
        order = sorted(mp, key=lambda ty: (-mp[ty], ty))
        n, x0 = len(order), LEFT + k * (LAB + body + GAP)
        ax = add_axes(x0 + LAB, y, body, n * CELL)
        ax.set_xlim(0, body)
        ax.set_ylim(n, 0)
        x, centres, heads = 0.0, [], []
        for c, arms in GROUPS:
            p_edges = [x, x + PW]
            edges = p_edges[1] + CW * np.arange(len(arms) + 1)
            ax.pcolormesh(p_edges, np.arange(n + 1), np.zeros((n, 1)), cmap=cmap, vmin=0, vmax=1, edgecolors="white", linewidth=0.3)
            im = ax.pcolormesh(edges, np.arange(n + 1), [[D[(c, t, a, ty)] for a in arms] for ty in order],
                               cmap="RdBu", vmin=-LIM, vmax=LIM, edgecolors="white", linewidth=0.3)
            for i, ty in enumerate(order):
                ax.text(x + PW / 2, i + 0.5, f"{P[(c, t, ty)]:.2f}", ha="center", va="center", fontsize=6)
            centres += [x + PW / 2] + list(edges[:-1] + CW / 2)
            heads += ["plain\nscVI F1"] + [SHORT[a] for a in arms]
            lab = COND_LABEL[c].replace(" decoder", "") + (" (low-signal)" if (t, c) in LOWSET else "")
            xt = ax.get_xaxis_transform()
            ax.text((x + edges[-1]) / 2, 1 + (HDR + 0.03) / (n * CELL), lab, transform=xt, ha="center", va="bottom", fontsize=6)
            ax.plot([x, edges[-1]], [1 + HDR / (n * CELL)] * 2, transform=xt, color="black", lw=0.6, clip_on=False)
            x = edges[-1] + GG
        ax.text(-LAB, 1 + (HDR + 0.03) / (n * CELL), TASK_LABEL[t], transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=6.5, fontweight="bold")
        ax.set_xticks(centres)
        ax.set_xticklabels(heads, rotation=90, fontsize=6, linespacing=0.9)
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_yticklabels([o if len(o) <= 23 else o[:22].rstrip() + "." for o in order], fontsize=6)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, pad=1.5)
    y += n_max * CELL + ROW_GAP
cax = add_axes(1.5, y + 0.06, 2.0, 0.08)
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
fig.text(3.6 / W_IN, (H - y - 0.1) / H, "F1 minus plain scVI's (same decoder)", fontsize=6.5, va="center")
cb.ax.tick_params(labelsize=6, length=2)
check_and_save(fig, "figS3")
