"""Pass 26 edit script re-run in a temporary copy on the text it started from (the Pass 25 result, commit a33ccc6); the files it
writes must equal the committed files (paper/outline.tex, paper/supp/figures.tex and paper/supp_items.tex are never edited by hand)."""
import difflib
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
from accept_changes import accept  # noqa: E402

START = "a33ccc6"
RESULT = "9db2382"  # the Pass 26 result; Pass 27 edits paper/outline.tex, so the script is compared with this commit
EDITED = ["paper/outline.tex", "paper/supp/figures.tex", "paper/supp_items.tex"]


def committed(rel):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{RESULT}:{rel}"], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"{RESULT} not available")  # pragma: no cover  # shallow copies lack the history
    return r.stdout


def test_pass26_script_reproduces_the_committed_files(tmp_path, monkeypatch, capsys):  # test-reason: runs pass26_edits.py, which no other test runs; catches a hand edit of a file it writes, a change outside the Figure 1 caption, the Figure 1a and 1b references, the Supplementary material heading and the list, a tracked mark in a supplement file, a Figure 1 caption that keeps a panel (b) sentence, and a Figure S9 caption that loses a point of the reading
    start = {}
    for rel in EDITED:
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{START}:{rel}"], capture_output=True, text=True)
        if r.returncode:
            pytest.skip(f"{START} not available")  # pragma: no cover  # shallow copies lack the history
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(r.stdout)
        start[rel] = r.stdout
    monkeypatch.setenv("PASS26_ROOT", str(tmp_path))
    runpy.run_path(str(TOOLS / "pass26_edits.py"), run_name="__main__")
    for rel in EDITED:
        assert (tmp_path / rel).read_text() == committed(rel), rel
    assert "captions 374 words (limit 450)" in capsys.readouterr().out
    # outline.tex: the printed text differs from a33ccc6 in the new author line, the caption, three Methods lines, the two Supplementary material lines and one list item, nowhere else
    raw = committed("paper/outline.tex")
    changed = [ln for ln in difflib.ndiff(accept(start["paper/outline.tex"]).split("\n"), accept(raw).split("\n")) if ln[:2] in ("+ ", "- ")]
    assert len(changed) == 14 and changed[0].startswith(r"+ \definechangesauthor[name={Pass 26}") and "{P26}" in changed[0]
    text = " ".join(accept(raw).split())
    assert "Figure 1a" not in text and "Figure 1b" not in text           # the references say Figure 1 or Supplementary Figure S9
    for part in ("(Figure 1). Hyperparameters", "(Supplementary Figure S9; Supplementary Note 5). A task", "(Supplementary Figure S9). A leave-one-seed-out",
                 "Figures S1--S9}", "Figures S1--S9, follow the main figures", r"\item Figure S9, how biology is read at matched batch removal"):
        assert part in text, part
    caption = text[text.index(r"\caption{Model."):text.index(r"\label{fig:1}")]
    assert "(a)" not in caption and "(b)" not in caption and "$b^*$" not in caption and "dashed arrows" not in caption   # the model only
    assert "dashed arrows" in raw                                         # the old panel (b) sentences stay as struck text
    # supplement files: Figure S9 added after S8, S1-S8 untouched, no tracked markup
    figures, items = committed("paper/supp/figures.tex"), committed("paper/supp_items.tex")
    assert figures.split("\n", 1)[1].startswith(start["paper/supp/figures.tex"].split("\n", 1)[1]) and figures.count(r"\begin{figure}") == 10
    assert not any(mark in figures + items for mark in (r"\replaced", r"\added", r"\deleted"))
    assert items.count("Figures S1--S9") == 1 and "S1--S8" not in items
    s9 = " ".join(figures[figures.index(r"\renewcommand{\thefigure}{S9}"):].split())
    for part in (r"\label{fig:s9}", "figures/figS9.png", "(top)", "(bottom)", "log scale", "ATAC small with the unconditioned decoder",
                 "three of the six penalties", "three seeds", "plain scVI (star) is drawn at strength 0, on the left",
                 "first reaches $b^*$ (dashed line)", "interpolating between neighbouring strengths (open circles, joined by dotted lines); later crossings (grey x) are not used"):
        assert part in s9, part
