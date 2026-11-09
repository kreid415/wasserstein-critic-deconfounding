"""Test of the Supplementary Figure S6 script (run from the repository root: python -m pytest paper/tools/test_figS6.py)."""
import runpy
import sys
from pathlib import Path

import matplotlib as mpl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import figcommon  # noqa: E402


def test_figS6_is_the_trade_off_curves_not_the_best_of_k_figure(tmp_path, monkeypatch):  # test-reason: runs figS6.py, which no other test runs; catches the best-of-k curves (removed in Pass 23) coming back under this number
    read, real_read_csv = [], pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda p, *a, **k: read.append(Path(p).name) or real_read_csv(p, *a, **k))
    monkeypatch.setattr(figcommon, "OUT", tmp_path)
    with mpl.rc_context():  # figS6.py applies the paper's plot style
        fig = runpy.run_path(str(ROOT / "figure_scripts" / "figS6.py"), run_name="__main__")["fig"]
    assert read == ["curves.csv"]
    assert len(fig.axes) == 2 * len(figcommon.REAL_TASKS)
    assert (tmp_path / "figS6.png").stat().st_size > 0 and (tmp_path / "figS6.pdf").stat().st_size > 0
