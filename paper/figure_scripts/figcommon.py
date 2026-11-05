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
    """Overlap check (figure-style 9.1), then PNG (300 dpi) and PDF (Figure 1 also writes SVG)."""
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
    for e in exts:
        fig.savefig(OUT / f"{stem}.{e}", dpi=300, **({"metadata": {"Date": None}} if e == "svg" else {}))
    print(f"{stem}: saved; text overlaps: {bad}")
    return bad


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
