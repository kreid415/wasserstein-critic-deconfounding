"""Pass 27 edit script re-run in a temporary copy on the text it started from (the Pass 26 result, commit 9db2382); the files it writes
must equal the committed files (paper/outline.tex, the supplement files and the two generators of the supplement tables are never edited
by hand), and the figure files follow the renumbering (the collapse rate is Table 1, the placement figure is Figure 4)."""
import difflib
import hashlib
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
from accept_changes import accept  # noqa: E402

START = "9db2382"
EDITED = ["paper/outline.tex", "paper/supp/note7.tex", "paper/supp/tableS7.tex", "paper/supp/tableS8.tex",
          "paper/tools/make_supp_tables.py", "paper/tools/make_supp_supp-analysis.py"]


def start_file(rel):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{START}:{rel}"], capture_output=True)
    if r.returncode:
        pytest.skip(f"{START} not available")  # pragma: no cover  # shallow copies lack the history
    return r.stdout


def test_pass27_script_reproduces_the_committed_files(tmp_path, monkeypatch, capsys):  # test-reason: runs pass27_edits.py, which no other test runs; catches a hand edit of a file it writes, a printed change outside the Res-4, Res-6, figure-list and history-sentence references and the collapse float, tracked markup in a supplement file or on the \input line, a leftover Figure 5 or old Figure 4 reference, and a Table 1 caption that breaks the caption limit
    start = {}
    for rel in EDITED:
        start[rel] = start_file(rel).decode()
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(start[rel])
    monkeypatch.setenv("PASS27_ROOT", str(tmp_path))
    runpy.run_path(str(TOOLS / "pass27_edits.py"), run_name="__main__")
    for rel in EDITED:
        assert (tmp_path / rel).read_text() == (REPO / rel).read_text(), rel
    assert "Table 1 included, 412 words (limit 450)" in capsys.readouterr().out
    # outline.tex: the printed text differs from 9db2382 in the new author line, the history sentence, Res-4, Res-6, the collapse float,
    # the placement float's three lines and two figure-list items, nowhere else
    raw = (REPO / "paper/outline.tex").read_text()
    changed = [ln for ln in difflib.ndiff(accept(start["paper/outline.tex"]).split("\n"), accept(raw).split("\n")) if ln[:2] in ("+ ", "- ")]
    assert len(changed) == 34 and changed[0].startswith(r"+ \definechangesauthor[name={Pass 27}") and "{P27}" in changed[0]
    live = re.sub(r"(?m)(?<!\\)%.*$", "", accept(raw))        # printed text without TeX comments
    text = " ".join(live.split())
    assert "Figure 5" not in text and "fig5" not in live and text.count("Figure 4") == 2 and text.count("Table 1") == 2
    for part in ("(Table 1; the share of non-dominated strengths", "(Figure 4; Supplementary Table", "Table 1: collapse rate up to the highest common",
                 "Figure 4: batch against biology score at single strength"):
        assert part in text, part
    assert raw.count("input{table1}") == 1 and raw.count("\n\\input{table1}\n") == 1     # no markup on the \input line
    assert raw.count("\n\\includegraphics[width=\\textwidth]{figures/fig4.png}\n") == 1 and "%\\caption{Collapse rate per" in raw  # old float commented out
    # supplement files and generators: references follow the numbering, no tracked markup
    supp = "".join((REPO / rel).read_text() for rel in EDITED[1:])
    assert not re.search(r"\\(replaced|added|deleted)", supp) and not re.search(r"Figure[ ~]5|Figures 3 and 4|rate, Figure 4", supp)
    for part in ("Figure 3, Table 1 and Supplementary Figure S8; the placement is in Figure 4 and", "The main text and Table 1 count only",
                 r"and of Figure~4)", "collapse rate, Table 1)"):
        assert part in supp, part


def test_placement_figure_files_are_figure_4():  # test-reason: runs no code; catches a leftover fig5 file, a fig4 that is not the Pass 26 placement figure byte for byte, and a figures manifest that still lists fig5 or the retired collapse figure
    figures = REPO / "paper/figures"
    sha = lambda b: hashlib.sha256(b).hexdigest()
    assert not (REPO / "paper/figure_scripts/fig5.py").exists() and not list(figures.glob("fig5.*"))
    # the PNG is byte-identical to the former fig5.png; PDFs are now written without a creation time, so the PDF is checked via the manifest
    assert sha((figures / "fig4.png").read_bytes()) == sha(start_file("paper/figures/fig5.png"))
    manifest = dict(ln.split()[::-1] for ln in (figures / "MANIFEST.sha256").read_text().splitlines())
    assert not {"fig5.png", "fig5.pdf"} & manifest.keys()
    assert all(manifest[f"fig4.{ext}"] == sha((figures / f"fig4.{ext}").read_bytes()) for ext in ("png", "pdf"))
    assert "for n in 1 2 3 4; do" in (REPO / "paper/figure_scripts/make_figures.sh").read_text()
