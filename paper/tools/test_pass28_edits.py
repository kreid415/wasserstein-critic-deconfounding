"""Pass 28: the edit script re-run in a temporary copy on the text it started from (the Pass 27 result, commit 6282edb) reproduces the
committed paper/outline.tex (never edited by hand), and Figure 2 panel a is coloured by the biology score at b* minus the median of the
six penalties in the same row (run from the repository root: python -m pytest paper/tools/test_pass28_edits.py)."""
import difflib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
from accept_changes import accept  # noqa: E402
from test_pass24_figures import figcommon, run_script  # noqa: E402

START = "6282edb"
OLD = [r"(a) Biology score at $b^*$ (colour) and rank within each task row (number) for each real task,",
       r"(a) heatmap of biology score at $b^*$ (colour) and rank (number) for the six penalties in each real task,"]
NEW = [r"(a) Rank within each task row (number) for each real task, and biology score at $b^*$ minus the median of the six penalties "
       r"in that row (colour),",
       r"(a) heatmap of rank (number) and of biology score at $b^*$ minus the row's median over the six penalties (colour) in each "
       r"real task,"]


def test_pass28_script_reproduces_the_committed_outline(tmp_path, monkeypatch, capsys):  # test-reason: runs pass28_edits.py, which no other test runs; catches a hand edit of outline.tex, a file written besides it, a printed change outside the Pass 28 author line, the Figure 2 caption and its list item, an author line out of place, and a wording that differs from the first author's
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{START}:paper/outline.tex"], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"{START} not available")  # pragma: no cover  # shallow copies lack the history
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper/outline.tex").write_text(r.stdout)
    monkeypatch.setenv("PASS28_ROOT", str(tmp_path))
    runpy.run_path(str(TOOLS / "pass28_edits.py"), run_name="__main__")
    raw = (REPO / "paper/outline.tex").read_text()
    assert (tmp_path / "paper/outline.tex").read_text() == raw
    assert [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()] == ["paper/outline.tex"]
    assert "Table 1 included, 422 words (limit 450)" in capsys.readouterr().out
    # the printed text differs from 6282edb in the new author line, the caption and the list item, nowhere else
    changed = [ln for ln in difflib.ndiff(accept(r.stdout).split("\n"), accept(raw).split("\n")) if ln[:2] in ("+ ", "- ")]
    assert len(changed) == 5 and changed[0].startswith(r"+ \definechangesauthor[name={Pass 28}") and "{P28}" in changed[0]
    lines = raw.split("\n")
    assert lines[lines.index(changed[0][2:]) + 1].startswith(r"\definechangesauthor[name={Pass 27}")
    text = " ".join(accept(raw).split())
    for old, new in zip(OLD, NEW):
        assert old not in text and text.count(new) == 1, new


def contrast(a, b):
    """WCAG contrast ratio of two RGB colours (relative luminance of sRGB)."""
    lum = lambda rgb: np.dot([0.2126, 0.7152, 0.0722], [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb[:3]])
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_fig2a_colour_is_biology_minus_the_row_median(tmp_path, monkeypatch):  # test-reason: runs fig2.py, which no other test runs; catches a colour that is not biology at b* minus the row median, colour limits that are not symmetric or not rounded up to two decimals, a lost tick at 0 or colour-bar label, a rank number changed, moved or drawn on the hatched row, and a number that contrasts less with its cell than the other of black and white, or less than 4.5 to 1
    fig, read, _ = run_script("fig2", tmp_path, monkeypatch)
    assert read == []                                   # reads paper_numbers.json only, no data file
    pn = json.loads((figcommon.DATA / "paper_numbers.json").read_text())
    arms, tasks = figcommon.ARMS, figcommon.REAL_TASKS
    heat = {c: next(ax for ax in fig.axes if ax.get_title(loc="left") == figcommon.COND_LABEL[c]) for c in (0, 1)}
    mesh = {c: heat[c].collections[0] for c in (0, 1)}
    val = {c: np.asarray(mesh[c].get_array()).reshape(6, 6) for c in (0, 1)}
    # known values, worked out apart from fig2.py
    assert val[0][tasks.index("pancreas"), arms.index("mmd")] == pytest.approx(0.006809, abs=5e-7)
    assert val[1][tasks.index("lung"), arms.index("sinkhorn")] == pytest.approx(0.008105, abs=5e-7)
    assert val[1][tasks.index("atac_small"), arms.index("discriminator")] == pytest.approx(-0.039846, abs=5e-7)
    ranked = {c: [i for i, t in enumerate(tasks) if t not in pn[f"low_signal_tasks.cond{c}"]["value"]] for c in (0, 1)}
    top = max(np.abs(val[c][ranked[c]]).max() for c in (0, 1))
    assert top == pytest.approx(0.039846, abs=5e-7)
    for c in (0, 1):
        assert mesh[c].cmap.name == "PuOr_r" and mesh[c].get_clim() == pytest.approx((-0.04, 0.04), abs=1e-12)
        bio = np.array([[pn[f"bio_at_bstar.{a}.{t}.cond{c}"]["value"] for a in arms] for t in tasks])
        assert np.allclose(val[c], bio - np.median(bio, axis=1, keepdims=True), rtol=0, atol=1e-12)
        assert np.allclose(np.median(val[c][ranked[c]], axis=1), 0, rtol=0, atol=1e-12)
        got = {(round(t.get_position()[1]), round(t.get_position()[0])): t for t in heat[c].texts}
        want = {(i, j): f"{pn[f'rank.{a}.{tasks[i]}.cond{c}']['value']:g}" for i in ranked[c] for j, a in enumerate(arms)}
        assert {k: t.get_text() for k, t in got.items()} == want
        for (i, j), t in got.items():
            cell = mesh[c].to_rgba(val[c][i, j])
            other = {"black": "white", "white": "black"}[t.get_color()]
            best = contrast(cell, mpl.colors.to_rgb(t.get_color()))
            assert best >= 4.5 and best >= contrast(cell, mpl.colors.to_rgb(other)), (c, i, j)
    cbar = next(ax for ax in fig.axes if ax.get_xlabel().startswith("biology score"))
    assert cbar.get_xlabel() == r"biology score at $b^{*}$ minus the row median" and 0 in cbar.get_xticks()
