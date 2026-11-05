"""Supplementary Figure S3: label transfer, F1 per cell type, method and task at the matched strength (figure_data/label_transfer_types.csv, made by
prep_label_transfer_types.py from the same embeddings and classifier as figure_data/label_transfer.csv; the macro F1 recomputed from the per-type values equals the stored
one for all 120 rows). Each held-out batch is predicted from the other batches by a 15-nearest-neighbour classifier; a type's F1 is averaged over the held-out batches that
contain it and over the three seeds (five for the scvi-tools adversary). Penalties are at the strength whose batch score is nearest b* (never chosen by accuracy); plain
scVI and the scvi-tools adversary are unmatched. The number under each column is the distance between that column's seed-mean batch score and b*."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *

apply_style()
lt = pd.read_csv(DATA / "label_transfer_types.csv")
mac = pd.read_csv(DATA / "label_transfer.csv")
low = pd.read_csv(DATA / "low_signal_pairs.csv")
LOWSET = {(r.task, int(r.cond)) for r in low.itertuples()}
TASKS = REAL_TASKS + ["sim1", "sim2"]
TL = dict(TASK_LABEL, sim1="simulation 1", sim2="simulation 2")
SHORT = {"none": "plain", "discriminator": "disc.", "reference": "ref.", "pooled": "pool.", "barycenter": "bary.", "mmd": "MMD", "sinkhorn": "Sink.", "scvi_adv": "adv."}
COLS = [(1, a) for a in ["none"] + ARMS + ["scvi_adv"]] + [(0, a) for a in ["none"] + ARMS]
# the macro F1 of a column is the mean of its per-type values over held-out batches; check the plotted table against the stored macro F1
for (c, a), g in lt.groupby(["cond", "arm"]):
    assert (c, a) in COLS, (c, a)
assert len(mac) == lt.groupby(["cond", "task", "arm", "lam"], dropna=False).ngroups == 120

ntypes = {t: lt[lt.task == t].celltype.nunique() for t in TASKS}
PAIRS = [("lung", "immune_hum_mou"), ("immune", "pancreas"), ("atac_large", "sim1"), ("atac_small", "sim2")]     # panels of similar size share a row
W_IN, CELL, LAB, TOP, GAP_ROW, LEFT, MID = WIDTH_IN, 0.10, 1.05, 0.62, 0.18, 1.1, 1.1
panel_w = (W_IN - LEFT - MID - 0.02) / 2
H = sum(max(ntypes[a], ntypes[b]) * CELL + TOP + GAP_ROW for a, b in PAIRS) + 0.55
fig = plt.figure(figsize=(W_IN, H))
y_top = H - 0.12
im = None
for pair in PAIRS:
    row_h = max(ntypes[pair[0]], ntypes[pair[1]]) * CELL
    for k, t in enumerate(pair):
        d = lt[lt.task == t]
        order = d.groupby("celltype").f1_mean.mean().sort_values(ascending=False).index.tolist()
        h = len(order) * CELL
        x0 = LEFT + k * (panel_w + MID)
        ax = fig.add_axes([x0 / W_IN, (y_top - TOP - h) / H, panel_w / W_IN, h / H])
        M = np.full((len(order), len(COLS)), np.nan); gap = []
        for j, (c, a) in enumerate(COLS):
            g = d[(d.cond == c) & (d.arm == a)].set_index("celltype")
            for i, ty in enumerate(order):
                if ty in g.index: M[i, j] = g.loc[ty, "f1_mean"]
            gap.append(g.distance.iloc[0] if len(g) else np.nan)
        im = ax.imshow(M, aspect="auto", vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        ax.axvline(7.5, color="white", lw=2.0)                                   # the two decoders
        ax.set_xticks(range(len(COLS)))
        ax.set_xticklabels([f"{SHORT[a]}  {gp:.2f}" for (_, a), gp in zip(COLS, gap)], rotation=90, fontsize=4.8)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(order))); ax.set_yticklabels([o if len(o) <= 24 else o[:23] + "." for o in order], fontsize=5)
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0, pad=1.5)
        lab = TL[t] + ("  (batch-conditioned decoder: low-signal)" if (t, 1) in LOWSET else "")
        fig.text(x0 / W_IN, (y_top - 0.02) / H, lab, fontsize=6.5, ha="left", va="top")
    y_top -= row_h + TOP + GAP_ROW
cax = fig.add_axes([0.36, 0.2 / H, 0.3, 0.1 / H])
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("label-transfer F1 of the cell type (column label: method and distance of its batch score from $b^{*}$)", fontsize=6); cb.ax.tick_params(labelsize=5.5, length=2)
check_and_save(fig, "figS3")
