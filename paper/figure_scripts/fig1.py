"""Figure 1: the model and where the penalties enter (scVI encoder and decoder, with or without the batch label, one alignment penalty
on the 10-dimensional latent embedding).
Drawn with matplotlib shapes (no icons), in inch units on a canvas of the full text width (7.0 in) with no data read; the PNG is the
figure itself (no cropping), so its width is exactly 7.0 in. SVG keeps the text as text (svg.fonttype none).
Pass 26: the reading panel that was panel b is Supplementary Figure S9 (figS9.py), and the panel letter is gone."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

apply_style()
mpl.rcParams["svg.hashsalt"] = "wcd-fig1"                  # fixed element ids: the SVG is byte-reproducible
mpl.rcParams["svg.fonttype"] = "none"                     # text stays editable text in the SVG
mpl.rcParams["savefig.bbox"] = "standard"                 # save the canvas as drawn: 7.0 in wide, no cropping

W, H = 7.0, 3.62
fig = plt.figure(figsize=(W, H))
axa = fig.add_axes([0.0, 0.0, 1.0, 1.0]); axa.set_xlim(0, W); axa.set_ylim(0, H); axa.axis("off")

INK, PALE, BLUE = "#333333", "#f2f2f2", "#dbe7f3"
def box(x, y, w, h, text, fc=PALE, ec=INK, ls="-", fs=7, lw=1.0):
    axa.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=lw, ls=ls))
    axa.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="black", linespacing=1.15)
def arrow(p, q, color=INK, ls="-"):
    axa.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=color, lw=1.0, ls=ls, shrinkA=0, shrinkB=0))

# penalty box
PX, PY, PW, PH = 0.05, 2.27, W - 0.10, 1.30
axa.add_patch(FancyBboxPatch((PX, PY), PW, PH, boxstyle="round,pad=0,rounding_size=0.06", fc="#fbf1e6", ec="#b5651d", lw=1.0))
axa.text(PX + 0.12, PY + PH - 0.14, "alignment penalty on the latent embedding (one of six per model)", fontsize=7, va="center")
def swatch(x, y, a, text, fs=7):
    axa.plot([x], [y], "o", color=ARM_COLOR[a], ms=4.6, mec="none")
    axa.text(x + 0.1, y, text, fontsize=fs, va="center", ha="left")
LX, RX, y0, dy = PX + 0.20, PX + 4.30, PY + PH - 0.45, 0.22
axa.text(LX - 0.04, y0, "adversarial", fontsize=7, fontweight="bold", va="center", color=INK)
swatch(LX, y0 - dy, "discriminator", "JS discriminator (1 update step)")
axa.text(LX - 0.04, y0 - 2 * dy, "Wasserstein critics (5 update steps, gradient penalty):", fontsize=7, va="center")
swatch(LX + 0.08, y0 - 3 * dy, "reference", "reference")
swatch(LX + 1.10, y0 - 3 * dy, "pooled", "pooled")
swatch(LX + 2.05, y0 - 3 * dy, "barycenter", "barycenter")
axa.text(RX - 0.04, y0, "non-adversarial", fontsize=7, fontweight="bold", va="center", color=INK)
swatch(RX, y0 - dy, "mmd", "MMD")
axa.text(RX + 0.1, y0 - 2 * dy, "fixed multi-scale Gaussian kernel", fontsize=6, va="center", color="#555555")
swatch(RX, y0 - 3 * dy, "sinkhorn", "Sinkhorn divergence")
axa.plot([PX + 4.20] * 2, [PY + 0.12, PY + PH - 0.30], color="#b5651d", lw=0.5)

# model row
BY, BH = 1.17, 0.62
xs = [(0.05, 1.00, "counts\n$x$"), (1.60, 1.25, "encoder"), (3.30, 1.35, "latent $z$\n(10-d)"), (5.10, 1.30, "decoder")]
for x, w, t in xs:
    box(x, BY, w, BH, t, fc=BLUE if "latent" in t else PALE, ec="#0072B2" if "latent" in t else INK)
for (x, w, _), (x2, _, _) in zip(xs[:-1], xs[1:]):
    arrow((x + w + 0.03, BY + BH / 2), (x2 - 0.03, BY + BH / 2))
arrow((6.40 + 0.03, BY + BH / 2), (6.70, BY + BH / 2))
axa.text(6.74, BY + BH / 2, "$\\hat{x}$", fontsize=8, va="center", ha="left")
zc = 3.30 + 1.35 / 2
arrow((zc, BY + BH + 0.03), (zc, PY - 0.03), color="#b5651d")
axa.text(zc + 0.10, (BY + BH + PY) / 2, "penalty $\\times$ strength", fontsize=7, color="#8a4a12", va="center")
# batch label
box(5.00, 0.40, 1.50, 0.46, "batch label $s$\n(conditioned only)", fc="white", ls=(0, (3, 2)), fs=7)
arrow((5.75, 0.87), (5.75, BY - 0.03), ls=(0, (3, 2)))
axa.text(0.10, 0.63, "decoder input: with the batch label $s$ (conditioned) or without it (unconditioned);\nthe penalty acts on $z$ in both cases", fontsize=7, color="#444444", va="center", ha="left", linespacing=1.25)
axa.text(0.10, 0.17, "loss = scVI loss + strength $\\times$ penalty (strength 0 = plain scVI)", fontsize=7, va="center", ha="left")

bad = check_and_save(fig, "fig1", exts=("png", "pdf", "svg"))
assert not bad, bad
