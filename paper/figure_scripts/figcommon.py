"""Shared settings for Figures 2-6 (paper/figure_scripts/figN.py).

Style = the `figure-style` skill's apply_figure_style(frame='open', sizes=(8, 7, 6)), copied here so
the scripts run from the repository alone. Inputs are read only from paper/figure_data/.
"""
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # .../paper
DATA = ROOT / "figure_data"
OUT = ROOT / "figures"
WIDTH_IN = 17.4 / 2.54                              # double-column width used by the mock figures

ARMS = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]
ARM_LABEL = {"discriminator": "discriminator", "reference": "reference critic", "pooled": "pooled critic",
             "barycenter": "barycenter critic", "mmd": "MMD", "sinkhorn": "Sinkhorn"}
ARM_COLOR = {"discriminator": "#D55E00", "reference": "#0072B2", "pooled": "#56B4E9",
             "barycenter": "#009E73", "mmd": "#CC79A7", "sinkhorn": "#E69F00"}
REAL_TASKS = ["pancreas", "lung", "immune", "atac_small", "atac_large", "immune_hum_mou"]
TASK_LABEL = {"pancreas": "pancreas", "lung": "lung", "immune": "immune", "atac_small": "ATAC small",
              "atac_large": "ATAC large", "immune_hum_mou": "human-mouse immune"}
# paper_numbers.json: cond1 = batch-conditioned decoder, cond0 = unconditioned decoder (checked in load_pn)
COND_LABEL = {1: "batch-conditioned decoder", 0: "unconditioned decoder"}
COND_ORDER = [1, 0]
GREY = "#6b6b6b"


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "legend.frameon": False, "figure.dpi": 200,
        "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titleweight": "normal",
        "axes.titlelocation": "left", "axes.labelweight": "normal", "lines.linewidth": 1.2,
        "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def load_pn():
    pn = json.load(open(DATA / "paper_numbers.json"))
    assert "unconditioned" in pn["bio_at_bstar.barycenter.atac_large.cond0"]["description"]
    assert "conditioned decoder" in pn["bio_at_bstar.barycenter.atac_large.cond1"]["description"]
    assert not pn["_meta.git_dirty"]["value"]
    return pn


def pnv(pn, key):
    """Value of a paper_numbers.json key; None where the file stores null."""
    return pn[key]["value"]


def panel_letter(ax, letter, dx=-0.0, dy=0.0, x=-0.12, y=1.08):
    ax.text(x + dx, y + dy, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="right")


def check_and_save(fig, stem, exts=("png", "pdf")):
    """Overlap check (figure-style 9.1), then PNG (300 dpi) and PDF (Figure 1 also writes SVG); no creation time in PDF or SVG."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ticks = {t for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels()}
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t not in ticks]
    bad = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
           if ba.overlaps(bb)]
    bad += [(t.get_text(), "spine") for t, bt in texts for ax in fig.axes for s in ax.spines.values()
            if s.get_visible() and bt.overlaps(s.get_window_extent(r)) and t.axes is not ax]
    OUT.mkdir(exist_ok=True)
    no_stamp = {"svg": {"Date": None}, "pdf": {"CreationDate": None}}   # no creation time: the files are byte-reproducible
    for e in exts:
        fig.savefig(OUT / f"{stem}.{e}", dpi=300, **({"metadata": no_stamp[e]} if e in no_stamp else {}))
    print(f"{stem}: saved; text overlaps: {bad}")
    return bad


def decoder_dot_panels(value, xlabel, low_in_median, ci=None, scale=1.0, zero_line=False, height=3.4, bottom=0.2):
    """One panel per decoder (batch-conditioned a, unconditioned b), one row per penalty, one dot per real task (tasks in
    the order of Figure 2a, top to bottom within a row) and a black median tick; value(a, t, c) and ci(a, t, c) -> (lo, hi).
    The low-signal immune pair is ringed and enters the median only if low_in_median. Pass 23 (Figures 5 and 6)."""
    from matplotlib.lines import Line2D
    pn = load_pn()
    low = {(t, c) for c in (0, 1) for t in pnv(pn, f"low_signal_tasks.cond{c}") if t in REAL_TASKS}
    assert low == {("immune", 1)}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, height), sharey=True, sharex=True,
                             gridspec_kw=dict(wspace=0.08, left=0.15, right=0.98, top=0.86, bottom=bottom))
    off = np.linspace(-0.27, 0.27, len(REAL_TASKS))
    for ax, c in zip(axes, COND_ORDER):
        for i, a in enumerate(ARMS):
            if i:
                ax.axhline(i - .5, color="#e6e6e6", lw=0.5, zorder=0)
            xs = []
            for j, t in enumerate(REAL_TASKS):
                y, x, ringed = i + off[j], scale * value(a, t, c), (t, c) in low
                if ci is not None:
                    lo, hi = ci(a, t, c)
                    ax.plot([scale * lo, scale * hi], [y, y], color=ARM_COLOR[a], lw=0.6, alpha=0.5, zorder=1)
                ax.plot(x, y, "o", color=ARM_COLOR[a], ms=3.4, mec="black" if ringed else "white", mew=0.9 if ringed else 0.3,
                        zorder=3)
                if low_in_median or not ringed:
                    xs.append(x)
            ax.plot([np.median(xs)] * 2, [i - 0.38, i + 0.38], color="black", lw=1.6, zorder=4, solid_capstyle="butt")
        if zero_line:
            ax.axvline(0, color="#444444", ls="--", lw=0.7, zorder=0)
        ax.set_title(COND_LABEL[c], loc="left", pad=3)
        ax.set_xlabel(xlabel)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    axes[0].set_ylim(5.5, -0.5)
    axes[0].set_yticks(range(6))
    axes[0].set_yticklabels([ARM_LABEL[a] for a in ARMS])
    for k, ax in enumerate(axes):
        fig.text(ax.get_position().x0 - (0.13 if k == 0 else 0.01), 0.9, "ab"[k], fontsize=10, fontweight="bold", va="bottom")
    h = [Line2D([], [], color="black", lw=1.6, label="median over tasks"),
         Line2D([], [], marker="o", mfc="none", color="white", mec="black", mew=0.8, lw=0, ms=6,
                label="low-signal immune pair" + ("" if low_in_median else " (not in the median)"))]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.57, 0.0), frameon=False)
    return fig, axes


def mean_rank_table(pn):
    """Mean rank and range over the ranked real tasks, per (cond, arm), recomputed from per-task ranks."""
    rows = []
    for c in (0, 1):
        for a in ARMS:
            rk = [pnv(pn, f"rank.{a}.{t}.cond{c}") for t in REAL_TASKS if f"rank.{a}.{t}.cond{c}" in pn]
            rk = [x for x in rk if x is not None]
            m = pnv(pn, f"mean_rank.cond{c}.{a}")
            assert abs(np.mean(rk) - m) < 1e-9 and len(rk) == pnv(pn, f"mean_rank_n_tasks.cond{c}.{a}")
            rows.append(dict(cond=c, arm=a, mean=m, lo=min(rk), hi=max(rk), n=len(rk)))
    return pd.DataFrame(rows)
