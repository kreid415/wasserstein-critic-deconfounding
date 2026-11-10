"""Test of the Supplementary Figure S9 script, the former panel b of Figure 1 (run from the repository root: python -m pytest paper/tools/test_figS9.py)."""
import runpy
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import figcommon  # noqa: E402


def test_figS9_draws_and_writes_the_first_crossing_reading(tmp_path, monkeypatch):  # test-reason: runs figS9.py, which no other test runs; catches a reading, a crossing strength or a bracket that departs from the first-crossing rule applied to curves.csv, a points file that is not the rows drawn, a circle off its line, a figure outside the supplement size, or a committed csv that is stale
    data = tmp_path / "figure_data"                     # the script writes its two csv files next to its inputs: keep the repository copies untouched
    data.mkdir()
    for name in ("curves.csv", "paper_numbers.json"):
        shutil.copyfile(figcommon.DATA / name, data / name)
    monkeypatch.setattr(figcommon, "OUT", tmp_path)
    monkeypatch.setattr(figcommon, "DATA", data)
    with mpl.rc_context():                              # figS9.py applies the paper's plot style
        g = runpy.run_path(str(ROOT / "figure_scripts" / "figS9.py"), run_name="__main__")
    for ext in ("png", "pdf"):
        assert (tmp_path / f"figS9.{ext}").stat().st_size > 0
    cv = pd.read_csv(figcommon.DATA / "curves.csv")
    cv = cv[(cv.task == "atac_small") & (cv.cond == 0) & cv.arm.isin(["discriminator", "reference", "mmd"])].sort_values(["arm", "lam"])
    bstar = g["bstar"]
    pts = pd.read_csv(data / "figS9_points.csv")
    assert list(pts.columns) == ["arm", "lam", "batch", "biology"]
    pd.testing.assert_frame_equal(pts, cv[["arm", "lam", "B", "C"]].rename(columns={"B": "batch", "C": "biology"}).reset_index(drop=True))
    assert (pts[pts.lam == 0].groupby("arm").size() == 1).all() and pts.arm.nunique() == 3     # plain scVI once per arm
    rd = pd.read_csv(data / "figS9_reading.csv")
    assert list(rd.columns) == ["arm", "strength", "batch", "biology", "lam_below", "lam_above"] and list(rd.arm) == sorted(rd.arm)
    for _, r in rd.iterrows():
        e = cv[cv.arm == r.arm]
        B, C, L = e.B.values, e.C.values, e.lam.values
        i = next(i for i in range(len(B) - 1) if B[i] < bstar <= B[i + 1])
        t = (bstar - B[i]) / (B[i + 1] - B[i])
        assert (r.lam_below, r.lam_above) == (L[i], L[i + 1]) and abs(r.batch - bstar) < 1e-12
        assert abs(r.biology - (C[i] + t * (C[i + 1] - C[i]))) < 1e-12
        assert abs(np.log(r.strength) - (np.log(L[i]) + t * (np.log(L[i + 1]) - np.log(L[i])))) < 1e-12
        assert r.lam_below < r.strength < r.lam_above
    axt, axb = g["axt"], g["axb"]
    assert axb.get_xscale() == "log" and axt.get_xscale() == "log" and axt.get_shared_x_axes().joined(axt, axb)
    ring = lambda ax: sorted((l.get_xdata()[0], l.get_ydata()[0]) for l in ax.lines if l.get_marker() == "o" and l.get_markerfacecolor() == "white")
    np.testing.assert_allclose(ring(axt), sorted(zip(rd.strength, [bstar] * 3)), rtol=1e-9)    # open circles at the reading, in both axes
    np.testing.assert_allclose(ring(axb), sorted(zip(rd.strength, rd.biology)), rtol=1e-9)
    star = [l for ax in (axt, axb) for l in ax.lines if l.get_marker() == "*"]
    assert len(star) == 2 and all(l.get_xdata()[0] < cv.lam[cv.lam > 0].min() for l in star)   # plain scVI left of the smallest strength
    assert sum(len(l.get_xdata()) for l in axt.lines if l.get_marker() == "x") == 2             # the later crossings: discriminator and reference critic
    fig = g["fig"]
    assert 3.6 <= fig.get_figwidth() <= 4.5                                                     # supplement size
    fig.canvas.draw()
    ticks = [t for t in axb.get_xticklabels() if t.get_text()]
    boxes = [t.get_window_extent() for t in ticks]
    assert [t.get_text() for t in ticks] == ["0", "0.01", "0.1", "1", "10", "100", "1000"] and not any(a.overlaps(b) for a, b in zip(boxes, boxes[1:]))
    assert min(t.get_fontsize() for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()) >= 6    # every label at least 6 pt
    for name in ("figS9_points.csv", "figS9_reading.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(data / name), pd.read_csv(figcommon.DATA / name))   # committed copies are what the script writes
