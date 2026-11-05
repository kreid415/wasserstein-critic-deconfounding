"""Figure 1: (a) model and where the penalties enter; (b) how biology is read at the matched batch level b*.
Panel b is one real task x decoder pair of the sweep (ATAC small, unconditioned decoder), three penalties, read from
figure_data/curves.csv (seed means, strength 0 = plain scVI). The reading is the first crossing of b*, linear
interpolation between neighbouring strengths; the script recomputes it and asserts it equals paper_numbers.json.
Panel a is drawn with matplotlib shapes (no icons); SVG keeps the text as text (svg.fonttype none)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

TASK, COND = "atac_small", 0
SHOWN = ["discriminator", "reference", "mmd"]            # one penalty per family: JS adversarial, Wasserstein critic, non-adversarial
SHOWN_LABEL = {"discriminator": "JS discriminator", "reference": "reference critic", "mmd": "MMD"}

apply_style()
mpl.rcParams["svg.hashsalt"] = "wcd-fig1"                  # fixed element ids: the SVG is byte-reproducible
mpl.rcParams["svg.fonttype"] = "none"                     # text stays editable text in the SVG
pn = load_pn()
cv = pd.read_csv(DATA / "curves.csv")

# ---- panel b data and assertions against paper_numbers.json ----
d = cv[(cv.task == TASK) & (cv.cond == COND)]
bstar = pnv(pn, f"pair.{TASK}.cond{COND}.bstar")
b0 = pnv(pn, f"pair.{TASK}.cond{COND}.b0")
assert not pnv(pn, f"pair.{TASK}.cond{COND}.low_signal") and d.n_fail.sum() == 0
curve, reading, cross_pt = {}, {}, {}
for a in SHOWN:
    e = d[d.arm == a].sort_values("lam").reset_index(drop=True)
    assert e.lam.iloc[0] == 0 and abs(e.B.iloc[0] - b0) < 1e-12        # every curve starts at plain scVI
    assert len(e) - 1 == pnv(pn, f"n_strengths.{a}.{TASK}.cond{COND}")  # plain scVI + the swept strengths
    assert pnv(pn, f"reading_status.{a}.{TASK}.cond{COND}") == "bracketed"
    B, C = e.B.values, e.C.values
    i = next(i for i in range(len(B) - 1) if B[i] < bstar <= B[i + 1])  # first crossing
    cross_pt[a] = C[i] + (C[i + 1] - C[i]) * (bstar - B[i]) / (B[i + 1] - B[i])
    reading[a] = pnv(pn, f"bio_at_bstar.{a}.{TASK}.cond{COND}")
    assert abs(cross_pt[a] - reading[a]) < 1e-9, (a, cross_pt[a], reading[a])
    assert int(((B[:-1] < bstar) & (B[1:] >= bstar) | (B[:-1] >= bstar) & (B[1:] < bstar)).sum()) >= 1
    curve[a] = e
plain_c = {a: curve[a].C.iloc[0] for a in SHOWN}
assert len({round(v, 12) for v in plain_c.values()}) == 1              # one plain scVI point shared by the curves
assert len({r for r in reading.values()}) == 3 and len({f"{r:.3f}" for r in reading.values()}) == 3  # printed readings differ

# ---- layout ----
H = 4.25
fig = plt.figure(figsize=(WIDTH_IN, H))
WA = 4.3                                                  # panel a width in inches (axes in inch units)
axa = fig.add_axes([0.0, 0.0, WA / WIDTH_IN, 1.0]); axa.set_xlim(0, WA); axa.set_ylim(0, H); axa.axis("off")
axb = fig.add_axes([(WA + 0.62) / WIDTH_IN, 1.62 / H, (WIDTH_IN - WA - 0.70) / WIDTH_IN, 2.30 / H])

INK, PALE, BLUE = "#333333", "#f2f2f2", "#dbe7f3"
def box(x, y, w, h, text, fc=PALE, ec=INK, ls="-", fs=7, lw=1.0):
    axa.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=lw, ls=ls))
    axa.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="black", linespacing=1.15)
def arrow(p, q, color=INK, ls="-"):
    axa.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=color, lw=1.0, ls=ls, shrinkA=0, shrinkB=0))

# penalty box
PX, PY, PW, PH = 0.12, 2.62, WA - 0.20, 1.42
axa.add_patch(FancyBboxPatch((PX, PY), PW, PH, boxstyle="round,pad=0,rounding_size=0.06", fc="#fbf1e6", ec="#b5651d", lw=1.0))
axa.text(PX + 0.1, PY + PH - 0.13, "alignment penalty on the latent embedding (one of six per model)", fontsize=7, va="center")
def swatch(x, y, a, text, fs=7):
    axa.plot([x], [y], "o", color=ARM_COLOR[a], ms=4.6, mec="none")
    axa.text(x + 0.1, y, text, fontsize=fs, va="center", ha="left")
LX, RX, y0, dy = PX + 0.14, PX + 2.42, PY + PH - 0.42, 0.20
axa.text(LX - 0.04, y0, "adversarial", fontsize=7, fontweight="bold", va="center", color=INK)
swatch(LX, y0 - dy, "discriminator", "JS discriminator (1 update step)")
axa.text(LX - 0.04, y0 - 2 * dy, "Wasserstein critics (5 update steps,", fontsize=7, va="center")
axa.text(LX - 0.04, y0 - 3 * dy + 0.01, "gradient penalty):", fontsize=7, va="center")
swatch(LX + 0.08, y0 - 4 * dy, "reference", "reference")
swatch(LX + 0.86, y0 - 4 * dy, "pooled", "pooled")
swatch(LX + 1.50, y0 - 4 * dy, "barycenter", "barycenter")
axa.text(RX - 0.04, y0, "non-adversarial", fontsize=7, fontweight="bold", va="center", color=INK)
swatch(RX, y0 - dy, "mmd", "MMD")
axa.text(RX + 0.1, y0 - 1.75 * dy, "fixed multi-scale Gaussian kernel", fontsize=6, va="center", color="#555555")
swatch(RX, y0 - 3 * dy + 0.0, "sinkhorn", "Sinkhorn divergence")
axa.plot([PX + 2.32] * 2, [PY + 0.12, PY + PH - 0.28], color="#b5651d", lw=0.5)

# model row
BY, BH = 1.55, 0.62
xs = [(0.12, 0.62, "counts\n$x$"), (1.00, 0.76, "encoder"), (2.14, 0.80, "latent $z$\n(10-d)"), (3.20, 0.78, "decoder")]
for x, w, t in xs:
    box(x, BY, w, BH, t, fc=BLUE if "latent" in t else PALE, ec="#0072B2" if "latent" in t else INK)
for (x, w, _), (x2, _, _) in zip(xs[:-1], xs[1:]):
    arrow((x + w + 0.03, BY + BH / 2), (x2 - 0.03, BY + BH / 2))
axa.text(4.17, BY + BH / 2, "$\\hat{x}$", fontsize=8, va="center", ha="left")
arrow((3.98 + 0.03, BY + BH / 2), (4.13, BY + BH / 2))
zc = 2.14 + 0.40
arrow((zc, BY + BH + 0.03), (zc, PY - 0.03), color="#b5651d")
axa.text(zc + 0.08, (BY + BH + PY) / 2, "penalty $\\times$ strength", fontsize=7, color="#8a4a12", va="center")
# batch label
box(2.88, 0.62, 1.10, 0.46, "batch label $s$\n(conditioned only)", fc="white", ls=(0, (3, 2)), fs=7)
arrow((3.59, 1.09), (3.59, BY - 0.03), ls=(0, (3, 2)))
axa.text(0.14, 1.08, "decoder input: with the batch label $s$\n(conditioned) or without it (unconditioned);\nthe penalty acts on $z$ in both cases", fontsize=7, color="#444444", va="center", ha="left", linespacing=1.25)
axa.text(0.14, 0.32, "loss = scVI loss + strength $\\times$ penalty\n(strength 0 = plain scVI)", fontsize=7, va="center", ha="left", linespacing=1.3)
axa.text(0.02, H - 0.05, "a", fontsize=10, fontweight="bold", va="top", ha="left")

# ---- panel b ----
ax = axb
for a in SHOWN:
    e = curve[a]
    ax.plot(e.B, e.C, "-o", color=ARM_COLOR[a], ms=2.4, lw=1.0, mew=0, zorder=2)
ax.axvline(bstar, color="#444444", ls="--", lw=0.7, zorder=1)
ax.plot(b0, plain_c[SHOWN[0]], "*", color="black", ms=9, mec="white", mew=0.4, zorder=6)
XL = 0.36
for a in SHOWN:
    ax.plot(bstar, reading[a], "o", mfc="white", mec=ARM_COLOR[a], ms=4.6, mew=1.2, zorder=5)
    ax.annotate("", xy=(b0 + 0.045, reading[a]), xytext=(bstar - 0.012, reading[a]),
                arrowprops=dict(arrowstyle="-|>", color=ARM_COLOR[a], lw=0.8, ls=(0, (2, 1.5)), shrinkA=0, shrinkB=0, mutation_scale=6), zorder=4)
ax.set_xlim(XL, 0.92); ax.set_ylim(0.36, 0.76)
ax.text(bstar + 0.008, 0.755, "$b^{*}$: matched\nbatch removal", fontsize=7, color="#444444", va="top", ha="left")
ax.set_xlabel("batch score\n(higher = more batch removal)"); ax.set_ylabel("biology score (higher = better)")
ax.set_title("Biology is read where each curve\nfirst reaches $b^{*}$", pad=4, fontsize=8, linespacing=1.2)
h = [Line2D([], [], color=ARM_COLOR[a], marker="o", ms=3, lw=1.2, label=f"{SHOWN_LABEL[a]}:  {reading[a]:.3f}") for a in SHOWN]
h += [Line2D([], [], color="black", marker="*", ms=8, lw=0, label="plain scVI"),
      Line2D([], [], color=GREY, marker="o", mfc="white", mec=GREY, ms=4.6, lw=0, label="first crossing of $b^{*}$")]
leg = fig.legend(handles=h, loc="upper left", bbox_to_anchor=((WA + 0.50) / WIDTH_IN, 0.26), ncol=1, handletextpad=0.4,
                 labelspacing=0.35, title="ATAC small, unconditioned decoder:\nbiology score at $b^{*}$", alignment="left", fontsize=7, title_fontsize=7)
fig.text((WA + 0.12) / WIDTH_IN, 0.99, "b", fontsize=10, fontweight="bold", va="top", ha="left")

bad = check_and_save(fig, "fig1", exts=("png", "pdf", "svg"))
assert not bad, bad
print(f"example pair: {TASK}, cond{COND}; b*={bstar!r}; b0={b0!r}")
for a in SHOWN: print(a, "reading", repr(reading[a]), "n_strengths", pnv(pn, f"n_strengths.{a}.{TASK}.cond{COND}"),
                      "crossings", pnv(pn, f"crossings.{a}.{TASK}.cond{COND}"), "rank", pnv(pn, f"rank.{a}.{TASK}.cond{COND}"))
