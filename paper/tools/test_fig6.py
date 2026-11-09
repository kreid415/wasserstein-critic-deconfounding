"""Test of the Figure 6 script (run from the repository root: python -m pytest paper/tools/test_fig6.py)."""
import runpy
import sys
from pathlib import Path

import matplotlib as mpl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import figcommon  # noqa: E402


def test_fig6_is_the_per_task_placement_panels_only(tmp_path, monkeypatch):  # test-reason: runs fig6.py, which no other test runs; catches the best-of-k curve panel (moved to Supplementary Figure S6) or a panel letter coming back
    read, real_read_csv = [], pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda p, *a, **k: read.append(Path(p).name) or real_read_csv(p, *a, **k))
    monkeypatch.setattr(figcommon, "OUT", tmp_path)
    with mpl.rc_context():  # fig6.py applies the paper's plot style
        fig = runpy.run_path(str(ROOT / "figure_scripts" / "fig6.py"), run_name="__main__")["fig"]
    assert sorted(read) == ["single_methods.csv", "transfer_strength.csv"]
    assert len(fig.axes) == len(figcommon.REAL_TASKS)
    assert [t.get_text() for t in fig.texts] == ["higher = better on both axes"]
    assert (tmp_path / "fig6.png").stat().st_size > 0 and (tmp_path / "fig6.pdf").stat().st_size > 0
