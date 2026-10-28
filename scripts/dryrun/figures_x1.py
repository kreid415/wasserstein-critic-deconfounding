#!/usr/bin/env python
"""Tables and the ranking figure from an analysis_x1.py record (docs/ANALYSIS_SPEC.md sections 3-6).

Writes to <out>: ranking_pairs.csv (one row per task, decoder, penalty), mean_rank.csv, secondary.csv (frontier area, collapse rate with
its exact interval, good-strength width), fig_ranking.png (biology at b* with rank per task, and mean rank per penalty).
Reads only the record; values are copied from it, not recomputed. Only the ranking figure exists so far: Figures 3-6 of the outline are
still mock-ups (paper/drafts_20261022/make_mock_figures.py), see docs/ANALYSIS_DRYRUN.md.

Usage: python scripts/dryrun/figures_x1.py --record <analysis.json> --out <dir>
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from figure_helpers import apply_figure_style, panel_letter  # noqa: E402

ARMS = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]
DECODERS = {"1": "conditioned decoder", "0": "unconditioned decoder"}


def tables(rec):
    pairs, sec = [], []
    for key, c in rec["cells"].items():
        for arm in ARMS:
            r, w, col = c["penalties"][arm], c["window"][arm], c["collapse"][arm]
            ranked = "rank" in r
            pairs.append(dict(task=c["task"], cond=c["cond"], arm=arm, low_signal=c["low_signal"], b0=c["b0"], bstar=c["bstar"],
                              bio_at_bstar=r.get("bio"), status=r.get("status"), rank=r.get("rank") if ranked else None,
                              rank_lo=r.get("rank_lo") if ranked else None, rank_hi=r.get("rank_hi") if ranked else None,
                              loso_se=r.get("s") if ranked else None, n_loso=r.get("m") if ranked else None))
            sec.append(dict(task=c["task"], cond=c["cond"], arm=arm, area=None if c["area"] is None else c["area"][arm],
                            collapse_rate=col["rate"], collapse_lo=col["lo"], collapse_hi=col["hi"], n_runs=col["n"],
                            window_status=w["status"], window_width_decades=w["width"]))
    mean = [dict(cond=int(k), arm=a, mean_rank=v["mean"], n_tasks=v["n_tasks"]) for k, d in rec["ranking"].items()
            for a, v in d["mean_rank"].items()]
    return pd.DataFrame(pairs), pd.DataFrame(mean), pd.DataFrame(sec)


def figure(rec, pairs, mean, path):
    real = sorted({t for t in pairs.task if not t.startswith("sim")})
    apply_figure_style(sizes=(8, 7, 6))
    fig, axs = plt.subplots(1, 3, figsize=(17.4 / 2.54, 8.5 / 2.54), gridspec_kw=dict(width_ratios=[1, 1, 0.9], wspace=0.12, bottom=0.38, left=0.1, right=0.98))
    cmap, vals = plt.cm.viridis, pairs.bio_at_bstar.dropna()
    for ax, cond in zip(axs[:2], (1, 0)):
        d = pairs[pairs.cond == cond]
        grid = np.full((len(ARMS), len(real)), np.nan)
        for i, a in enumerate(ARMS):
            for j, t in enumerate(real):
                row = d[(d.arm == a) & (d.task == t)].iloc[0]
                grid[i, j] = np.nan if row.low_signal or row.bio_at_bstar is None else row.bio_at_bstar
        ax.imshow(grid, cmap=cmap, vmin=vals.min(), vmax=vals.max(), aspect="auto")
        for i, a in enumerate(ARMS):
            for j, t in enumerate(real):
                row = d[(d.arm == a) & (d.task == t)].iloc[0]
                if np.isnan(grid[i, j]):     # low-signal pair, or the penalty never reaches b*
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#F2F2F2", ec="#7F7F7F", hatch="////", lw=0))
                if not row.low_signal:
                    ax.text(j, i, f"{row['rank']:g}", ha="center", va="center", fontsize=6, color="white" if grid[i, j] < vals.mean() else "black")
        ax.set_xticks(range(len(real)), real, rotation=55, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(ARMS)), ARMS if cond == 1 else [])
        ax.set_title(DECODERS[str(cond)], loc="left")
    panel_letter(axs[0], "a", dx=-0.5)
    for cond, mk in ((1, "o"), (0, "s")):
        m = mean[mean.cond == cond].set_index("arm").reindex(ARMS)
        axs[2].plot(m.mean_rank, np.arange(len(ARMS)) + (-0.15 if cond == 1 else 0.15), mk, color="k", ms=3.5, label=DECODERS[str(cond)])
    axs[2].set_ylim(5.6, -0.6); axs[2].set_yticks(range(len(ARMS)), []); axs[2].set_xlim(0.8, 6.2); axs[2].grid(axis="y", color="#E5E5E5")
    axs[2].set_xlabel("mean rank (1 = best)"); axs[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=5.5, handletextpad=0.2)
    axs[2].set_title("mean rank", loc="left")
    panel_letter(axs[2], "b", dx=-0.1)
    fig.text(0.01, 0.01, "a: colour = biology at b*; number = rank in the task (1 = best).\nHatched: low-signal pair (not ranked), or the curve never reaches b* (bottom places).", fontsize=6)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rec = json.load(open(a.record))["primary"]
    os.makedirs(a.out, exist_ok=True)
    pairs, mean, sec = tables(rec)
    for name, df in (("ranking_pairs", pairs), ("mean_rank", mean), ("secondary", sec)):
        df.to_csv(os.path.join(a.out, name + ".csv"), index=False)
    figure(rec, pairs, mean, os.path.join(a.out, "fig_ranking.png"))
    print(f"tables {len(pairs)} + {len(mean)} + {len(sec)} rows, fig_ranking.png -> {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
