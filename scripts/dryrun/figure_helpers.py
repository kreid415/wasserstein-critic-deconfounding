"""Minimal stand-ins for the figure-style helpers that paper/drafts_20261022/make_mock_figures.py expects in scope.

The Claude Science figure-style skill defines apply_figure_style and panel_letter; they are not in the repo, so a run from the command
line has none. These versions set the three-size font ladder (base, legend, ticks) and a bold panel letter, nothing else.
"""
import matplotlib as mpl


def apply_figure_style(sizes=(8, 7, 6)):
    base, small, tick = sizes
    mpl.rcParams.update({"font.size": base, "axes.titlesize": base, "axes.labelsize": base, "legend.fontsize": small,
                         "xtick.labelsize": tick, "ytick.labelsize": tick, "axes.spines.top": False, "axes.spines.right": False})


def panel_letter(ax, letter, dx=-0.1, dy=1.05, case="lower"):
    ax.text(dx, dy, letter.upper() if case == "upper" else letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="bottom")
