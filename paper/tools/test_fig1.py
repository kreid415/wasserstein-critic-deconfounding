"""Test of the Figure 1 script, the model schematic (run from the repository root: python -m pytest paper/tools/test_fig1.py)."""
import runpy
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import figcommon  # noqa: E402


def test_fig1_is_the_schematic_on_a_full_width_canvas(tmp_path, monkeypatch):  # test-reason: runs fig1.py, which no other test runs; catches the reading panel or a panel letter coming back, data being read or csv files written, a canvas that is not 7.0 in wide or has an empty margin, clipped or small text, and a time stamp in the PDF or SVG
    monkeypatch.setattr(figcommon, "OUT", tmp_path)
    monkeypatch.setattr(figcommon, "DATA", tmp_path / "no_data")        # fig1.py reads no data: reading any file would fail
    with mpl.rc_context():                                              # fig1.py applies the paper's plot style
        g = runpy.run_path(str(ROOT / "figure_scripts" / "fig1.py"), run_name="__main__")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fig1.pdf", "fig1.png", "fig1.svg"]     # no csv file
    fig = g["fig"]
    assert fig.get_figwidth() == 7.0 and len(fig.axes) == 1             # full text width; the schematic alone
    assert mpimg.imread(tmp_path / "fig1.png").shape[:2] == (round(fig.get_figheight() * 300), 2100)   # saved as drawn, 300 dpi
    for name, stamp in (("fig1.pdf", b"CreationDate"), ("fig1.svg", b"<dc:date>")):
        assert stamp not in (tmp_path / name).read_bytes()              # byte-reproducible files
    fig.canvas.draw()
    ax = fig.axes[0]
    texts = [t for t in ax.texts if t.get_text().strip()]
    assert {t.get_text() for t in texts}.isdisjoint({"a", "b"}) and "read-out" not in " ".join(t.get_text() for t in texts)   # no panel letters
    assert min(t.get_fontsize() for t in texts) >= 6                    # every label at least 6 pt
    boxes = [t.get_window_extent() for t in texts] + [p.get_window_extent() for p in ax.patches]
    canvas = fig.bbox
    assert all(canvas.x0 <= b.x0 and b.x1 <= canvas.x1 and canvas.y0 <= b.y0 and b.y1 <= canvas.y1 for b in boxes)   # nothing clipped
    assert (max(b.x1 for b in boxes) - min(b.x0 for b in boxes)) / canvas.width > 0.95            # no empty margin
    assert (max(b.y1 for b in boxes) - min(b.y0 for b in boxes)) / canvas.height > 0.95
