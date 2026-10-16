
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
apply_figure_style()
C_CRIT, C_DISC, C_HARM, C_UNINT, C_SCAN = "#c0392b","#2c6fbb","#27ae60","#7f8c8d","#16a085"
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
lamX = X.groupby(["method","d_coef"])[["batch_correction","bio_conservation","scib_overall"]].mean()
cx, dxx = lamX.loc["critic"], lamX.loc["discriminator"]
prev = lam.loc["critic"]

ax = axes[0]
ax.plot(prev.batch_correction, prev.bio_conservation, "o-", color=C_CRIT, ms=3.4, lw=1.1,
        alpha=0.45, label="critic, NB (λ≤1)")
ax.plot(cx.batch_correction, cx.bio_conservation, "o-", color=C_CRIT, ms=5.2, lw=1.8,
        label="critic, LDVAE_uncond (λ 1→5)")
for l in (1.0, 5.0):
    ax.annotate(f"λ={l:g}", (cx.loc[l,"batch_correction"], cx.loc[l,"bio_conservation"]),
                textcoords="offset points", xytext=(6,-2), fontsize=6, color=C_CRIT)
ax.plot(dxx.batch_correction, dxx.bio_conservation, "s-", color=C_DISC, ms=4.4, lw=1.4,
        label="discriminator, LDVAE_uncond")
for nm,(col,mk,off) in {"harmony":(C_HARM,"D",(-7,5)),"scanorama":(C_SCAN,"^",(-7,5)),
                        "unintegrated":(C_UNINT,"P",(7,-9))}.items():
    g = ref[ref.method==nm]
    ax.scatter(g.batch_correction.mean(), g.bio_conservation.mean(), s=58, color=col,
               marker=mk, zorder=4, edgecolor="white", linewidth=0.6)
    ax.annotate(nm, (g.batch_correction.mean(), g.bio_conservation.mean()),
                textcoords="offset points", xytext=off,
                ha="right" if off[0]<0 else "left", fontsize=6, color=col)
ax.set_xlabel("batch correction"); ax.set_ylabel("bio conservation")
ax.set_title("Tuning moves the critic up the\nfront, toward the baselines", loc="left")
ax.legend(frameon=False, fontsize=5.6, loc="lower left")
ax.margins(0.18)

ax = axes[1]
ax.plot(prev.index, prev.scib_overall, "o--", color=C_CRIT, ms=3.2, lw=1.0, alpha=0.45,
        label="NB (published grid)")
ax.plot(cx.index, cx.scib_overall, "o-", color=C_CRIT, ms=5, lw=1.8, label="LDVAE_uncond")
ax.plot(dxx.index, dxx.scib_overall, "s-", color=C_DISC, ms=4, lw=1.4, label="disc, LDVAE_uncond")
for v,c_,ls,nm in ((ref[ref.method=="harmony"].scib_overall.mean(), C_HARM, ":", "harmony"),
                   (ref[ref.method=="unintegrated"].scib_overall.mean(), C_UNINT, "--", "unintegrated")):
    ax.axhline(v, color=c_, ls=ls, lw=1.1)
    ax.annotate(nm, (5.0, v), textcoords="offset points", xytext=(-3,3), ha="right",
                fontsize=6, color=c_)
ax.annotate("still climbing\nat grid edge", (5.0, cx.loc[5.0,"scib_overall"]),
            textcoords="offset points", xytext=(-6,-22), ha="right", fontsize=6, color=C_CRIT)
ax.set_xscale("symlog", linthresh=0.1)
ax.set_xlim(-0.02, 7)
ax.set_xlabel("λ (adversarial weight)"); ax.set_ylabel("scIB overall")
ax.set_title("λ=5 recovers 63% of the gap;\noptimum still not bracketed", loc="left")
ax.legend(frameon=False, fontsize=5.6, loc="center left", bbox_to_anchor=(0.02, 0.62))

ax = axes[2]
crit = [old[old.method=="critic"].scib_overall.mean(), cx.loc[5.0,"scib_overall"]]
disc = [old[old.method=="discriminator"].scib_overall.mean(), dxx.loc[5.0,"scib_overall"]]
xp = np.arange(2)
ax.bar(xp-0.19, crit, 0.36, color=C_CRIT, label="critic")
ax.bar(xp+0.19, disc, 0.36, color=C_DISC, label="discriminator")
for i,(a_,b_) in enumerate(zip(crit,disc)):
    ax.annotate(f"Δ={a_-b_:+.3f}", (i, max(a_,b_)), textcoords="offset points",
                xytext=(0,14 if i==1 else 4), ha="center", fontsize=6,
                color=C_CRIT if a_>b_ else "#555")
uv = ref[ref.method=="unintegrated"].scib_overall.mean()
ax.axhline(uv, color=C_UNINT, ls="--", lw=1.1)
ax.annotate("unintegrated", (1.42, uv), textcoords="offset points", xytext=(0,3),
            ha="right", fontsize=6, color=C_UNINT)
ax.set_xticks(xp); ax.set_xticklabels(["reported\nNB, λ=0.2","tuned\nLDVAE, λ=5"], fontsize=6.4)
ax.set_ylabel("scIB overall"); ax.set_ylim(0, 0.52)
ax.set_title(f"Head effect is {(crit[1]-disc[1])/(crit[0]-disc[0]):.0f}x larger at the\ntuned point (p=8e-06)", loc="left")
ax.legend(frameon=False, fontsize=6, loc="upper left")

for a,L in zip(axes,"abc"): panel_letter(a,L)
for a in axes: a.yaxis.set_label_coords(-0.135, 0.5)
fig.tight_layout(pad=1.25, w_pad=2.3)
fig.savefig("crossed_sweep.png", dpi=200)
r_=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tl={a:set(a.get_xticklabels(which="both")+a.get_yticklabels(which="both")) for a in fig.axes}
sp=[(s,s.get_window_extent(r_)) for a in fig.axes for s in a.spines.values() if s.get_visible()]
ov=[(a_.get_text()[:12],b_.get_text()[:12]) for i,(a_,ba) in enumerate(tx) for b_,bb_ in tx[i+1:] if ba.overlaps(bb_)]
ov+=[(t.get_text()[:12],"SPINE") for t,bt in tx for s_,bs in sp if bt.overlaps(bs) and t not in tl[s_.axes]]
print("overlaps:", ov)
print("head delta reported", round(crit[0]-disc[0],4), "tuned", round(crit[1]-disc[1],4),
      "ratio", round((crit[1]-disc[1])/(crit[0]-disc[0]),1))
