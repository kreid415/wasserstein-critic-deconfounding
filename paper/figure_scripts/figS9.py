"""Supplementary Figure S9 (Pass 26: the former panel b of Figure 1, which now shows the model only, fig1.py): how biology is read at the
matched batch level b*. One real task x decoder pair of the sweep (ATAC small, unconditioned decoder), three penalties, read from
figure_data/curves.csv (seed means, strength 0 = plain scVI). The reading is the first crossing of b*, linear interpolation between
neighbouring strengths; the script recomputes it and asserts it equals paper_numbers.json.
Batch score (top) and biology score (bottom) are drawn against penalty strength on one log axis (plain scVI left of the break); it
writes the drawn rows to figure_data/figS9_points.csv and the readings to figure_data/figS9_reading.csv (the same values on every run)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import *
from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D

TASK, COND = "atac_small", 0
SHOWN = ["discriminator", "reference", "mmd"]            # one penalty per family: JS adversarial, Wasserstein critic, non-adversarial
SHOWN_LABEL = {"discriminator": "JS discriminator", "reference": "reference critic", "mmd": "MMD"}

apply_style()
pn = load_pn()
cv = pd.read_csv(DATA / "curves.csv")

# ---- data and assertions against paper_numbers.json ----
d = cv[(cv.task == TASK) & (cv.cond == COND)]
bstar = pnv(pn, f"pair.{TASK}.cond{COND}.bstar")
b0 = pnv(pn, f"pair.{TASK}.cond{COND}.b0")
assert not pnv(pn, f"pair.{TASK}.cond{COND}.low_signal") and d.n_fail.sum() == 0
curve, reading, cross_pt, cross_lam, bracket, later = {}, {}, {}, {}, {}, {}
for a in SHOWN:
    e = d[d.arm == a].sort_values("lam").reset_index(drop=True)
    assert e.lam.iloc[0] == 0 and abs(e.B.iloc[0] - b0) < 1e-12        # every curve starts at plain scVI
    assert len(e) - 1 == pnv(pn, f"n_strengths.{a}.{TASK}.cond{COND}")  # plain scVI + the swept strengths
    assert pnv(pn, f"reading_status.{a}.{TASK}.cond{COND}") == "bracketed"
    B, C, L = e.B.values, e.C.values, e.lam.values
    i = next(i for i in range(len(B) - 1) if B[i] < bstar <= B[i + 1])  # first crossing
    t = lambda j: (bstar - B[j]) / (B[j + 1] - B[j])                    # where segment j -> j+1 meets b*
    strength = lambda j: float(np.exp(np.log(L[j]) + t(j) * (np.log(L[j + 1]) - np.log(L[j]))))  # log interpolation, same t
    cross_pt[a] = C[i] + t(i) * (C[i + 1] - C[i])
    assert L[i] > 0, (a, "first crossing starts at plain scVI: its strength has no place on a log axis")
    bracket[a], cross_lam[a] = (L[i], L[i + 1]), strength(i)
    reading[a] = pnv(pn, f"bio_at_bstar.{a}.{TASK}.cond{COND}")
    assert abs(cross_pt[a] - reading[a]) < 1e-9, (a, cross_pt[a], reading[a])
    assert int(((B[:-1] < bstar) & (B[1:] >= bstar) | (B[:-1] >= bstar) & (B[1:] < bstar)).sum()) >= 1
    later[a] = [strength(j) for j in range(i + 1, len(B) - 1) if (B[j] < bstar) != (B[j + 1] < bstar)]  # crossings after the first (not read)
    assert len(later[a]) + 1 == pnv(pn, f"crossings.{a}.{TASK}.cond{COND}")
    curve[a] = e
plain_c = {a: curve[a].C.iloc[0] for a in SHOWN}
assert len({round(v, 12) for v in plain_c.values()}) == 1              # one plain scVI point shared by the curves
assert len({r for r in reading.values()}) == 3 and len({f"{r:.3f}" for r in reading.values()}) == 3  # printed readings differ

# ---- layout: sizes in inches ----
W, H = 4.2, 4.05
rect = lambda x, y, w, h: [x / W, y / H, w / W, h / H]
AX, AW = 0.80, 3.30                                       # left edge and width of both axes
fig = plt.figure(figsize=(W, H))
axb = fig.add_axes(rect(AX, 1.27, AW, 1.25))
axt = fig.add_axes(rect(AX, 2.64, AW, 1.00), sharex=axb)  # both axes share the log strength axis

swept = np.concatenate([curve[a].lam.values[1:] for a in SHOWN])  # row 0 of each curve is plain scVI
lo, hi = int(np.floor(np.log10(swept.min()))), int(np.floor(np.log10(swept.max())))  # decades with a tick
X0 = 10.0 ** (lo - 1)                                      # plain scVI (strength 0) cannot sit on a log axis: drawn one decade left of the first tick
xpos = lambda lam: np.where(np.asarray(lam) > 0, lam, X0)
axb.set_xscale("log")                                      # axt shares it
axb.set_xlim(10 ** (lo - 1.35), 10 ** (hi + 0.7))
axb.set_xticks([X0] + [10.0 ** k for k in range(lo, hi + 1)], ["0"] + [f"{10.0 ** k:g}" for k in range(lo, hi + 1)])
axb.xaxis.set_minor_locator(mpl.ticker.NullLocator())
axt.tick_params(labelbottom=False)
for ax, col, y0, ylim in ((axt, "B", b0, (0.36, 0.94)), (axb, "C", plain_c[SHOWN[0]], (0.35, 0.74))):
    for a in SHOWN:
        e = curve[a]
        ax.plot(xpos(e.lam), e[col], "-o", color=ARM_COLOR[a], ms=2.4, lw=1.0, mew=0, zorder=2)
    ax.plot(X0, y0, "*", color="black", ms=9, mec="white", mew=0.4, zorder=6)
    ax.set_ylim(*ylim)
    xm = 10.0 ** (lo - 0.5)                                # axis break between plain scVI and the smallest strength
    ax.plot([xm / 2, xm * 2], [0, 0], transform=ax.get_xaxis_transform(), color="white", lw=2, clip_on=False, zorder=3)
    for x in (xm / 1.5, xm * 1.5):
        ax.plot([x / 1.3, x * 1.3], [-0.04, 0.04], transform=ax.get_xaxis_transform(), color="black", lw=0.6, clip_on=False, zorder=4)
axt.axhline(bstar, color="#444444", ls="--", lw=0.7, zorder=1)
axt.text(10 ** (lo - 1.25), bstar + 0.02, "$b^{*}$: matched\nbatch\nremoval", fontsize=7, color="#444444", va="bottom", ha="left", linespacing=1.1)
for a in SHOWN:
    s = cross_lam[a]
    for ax, y in ((axt, bstar), (axb, reading[a])):
        ax.plot(s, y, "o", mfc="white", mec=ARM_COLOR[a], ms=4.6, mew=1.2, zorder=5)
    fig.add_artist(ConnectionPatch((s, bstar), (s, reading[a]), "data", "data", axesA=axt, axesB=axb, color=ARM_COLOR[a],
                                   ls=":", lw=0.9, shrinkA=3, shrinkB=3, zorder=3))
    axt.plot(later[a], [bstar] * len(later[a]), "x", color=GREY, ms=3.5, mew=0.8, zorder=4)
axt.set_ylabel("batch score\n(higher = more removal)", fontsize=7); axb.set_ylabel("biology score\n(higher = better)", fontsize=7)
axb.set_xlabel("penalty strength (log scale)")
axt.set_title("Biology is read where batch removal\nfirst reaches $b^{*}$", pad=4, fontsize=8, linespacing=1.2)
h = [Line2D([], [], color=ARM_COLOR[a], marker="o", ms=3, lw=1.2, label=f"{SHOWN_LABEL[a]}: {reading[a]:.3f}") for a in SHOWN]
h += [Line2D([], [], color="black", marker="*", ms=8, lw=0, label="plain scVI"),
      Line2D([], [], color=GREY, marker="o", mfc="white", mec=GREY, ms=4.6, lw=0, label="first reaches $b^{*}$"),
      Line2D([], [], color=GREY, marker="x", ms=3.5, mew=0.8, lw=0, label="unused crossing")]
fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.12 / W, 0.04 / H), ncol=2, handletextpad=0.3, columnspacing=0.8,
                 handlelength=1.4, labelspacing=0.35, title="ATAC small, unconditioned decoder:\nbiology score at $b^{*}$", alignment="left", fontsize=7, title_fontsize=7)

# ---- values: the rows drawn and the readings (same arithmetic as above) ----
pd.concat([curve[a][["arm", "lam", "B", "C"]] for a in SHOWN]).rename(columns={"B": "batch", "C": "biology"}).sort_values(
    ["arm", "lam"]).to_csv(DATA / "figS9_points.csv", index=False)
pd.DataFrame([dict(arm=a, strength=cross_lam[a], batch=bstar, biology=cross_pt[a], lam_below=bracket[a][0], lam_above=bracket[a][1])
              for a in sorted(SHOWN)]).to_csv(DATA / "figS9_reading.csv", index=False)

bad = check_and_save(fig, "figS9")
assert not bad, bad
print(f"example pair: {TASK}, cond{COND}; b*={bstar!r}; b0={b0!r}")
for a in SHOWN: print(a, "reading", repr(reading[a]), "n_strengths", pnv(pn, f"n_strengths.{a}.{TASK}.cond{COND}"),
                      "crossings", pnv(pn, f"crossings.{a}.{TASK}.cond{COND}"), "rank", pnv(pn, f"rank.{a}.{TASK}.cond{COND}"))
