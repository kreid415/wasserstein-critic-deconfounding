"""Tests of the Pass 24 figure scripts: Figures 3-4 (Pass 27 numbering) and Supplementary Figures S7-S8 (run from the repository root:
python -m pytest paper/tools/test_pass24_figures.py)."""
import runpy
import sys
from pathlib import Path

import matplotlib as mpl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import figcommon  # noqa: E402


def run_script(stem, tmp_path, monkeypatch):
    """Run figure_scripts/<stem>.py into tmp_path; return the figure, the sorted names of the CSV files read and the real read_csv."""
    read, real_read_csv = [], pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda p, *a, **k: read.append(Path(p).name) or real_read_csv(p, *a, **k))
    monkeypatch.setattr(figcommon, "OUT", tmp_path)
    with mpl.rc_context():  # the scripts apply the paper's plot style
        fig = runpy.run_path(str(ROOT / "figure_scripts" / f"{stem}.py"), run_name="__main__")["fig"]
    assert (tmp_path / f"{stem}.png").stat().st_size > 0 and (tmp_path / f"{stem}.pdf").stat().st_size > 0
    return fig, sorted(read), real_read_csv


def test_fig3_is_the_two_frontier_panels_only(tmp_path, monkeypatch):  # test-reason: runs fig3.py, which no other test runs; catches the hypervolume panels (moved to Supplementary Figure S8) or a panel letter c or d coming back
    fig, read, _ = run_script("fig3", tmp_path, monkeypatch)
    assert read == ["pareto_frontier_median.csv"]
    assert len(fig.axes) == 2 and fig.get_size_inches()[1] < 3.3
    assert [t.get_text() for t in fig.texts if len(t.get_text()) == 1] == ["a", "b"]


def test_fig4_is_the_per_task_placement_panels(tmp_path, monkeypatch):  # test-reason: runs fig4.py, which no other test runs; catches a lost or extra task panel after the renumbering from Figure 6 (Pass 24) and Figure 5 (Pass 27, when the collapse-rate figure became Table 1: its percent scale is tested in test_table1.py)
    fig, read, _ = run_script("fig4", tmp_path, monkeypatch)
    assert read == ["single_methods.csv", "transfer_strength.csv"]
    assert len(fig.axes) == len(figcommon.REAL_TASKS)


def test_figS7_is_the_dot_summary_of_biology_kept(tmp_path, monkeypatch):  # test-reason: runs figS7.py, which no other test runs; catches an input file lost in the move from main Figure 3
    fig, read, _ = run_script("figS7", tmp_path, monkeypatch)
    assert read == ["curves.csv", "ranking_pairs.csv"]
    assert len(fig.axes) == 2


def test_figS8_shows_every_hypervolume_point(tmp_path, monkeypatch):  # test-reason: runs figS8.py, which no other test runs; catches points clipped by the x limit (the old fixed limit of 0.13 hid values up to 0.20)
    fig, read, real_read_csv = run_script("figS8", tmp_path, monkeypatch)
    assert read == ["pareto_frontier.csv"]
    assert len(fig.axes) == 2
    hv = real_read_csv(figcommon.DATA / "pareto_frontier.csv")
    hv = hv[hv.task.isin(figcommon.REAL_TASKS)].hypervolume_to_common
    assert hv.max() > 0.13 and hv.max() < fig.axes[0].get_xlim()[1]
    assert sum(len(c.get_offsets()) for ax in fig.axes for c in ax.collections) == len(hv)
