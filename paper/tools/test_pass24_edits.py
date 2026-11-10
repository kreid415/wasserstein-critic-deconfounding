"""Pass 24 tests (run from the repository root: python -m pytest paper/tools/test_pass24_edits.py): the edit script re-run in a
temporary copy on the text it started from (24f2665) reproduces every file it writes; the generated Supplementary Tables S15 and
S11; the abstract counter on the three-paragraph abstract."""
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
import abstract_words as aw  # noqa: E402
import make_supp_tables as mt  # noqa: E402

START = "24f2665"
EDITED = ["paper/outline.tex", "paper/supp_items.tex", "paper/supp/figures.tex", "paper/supp/note1.tex", "paper/supp/note5.tex",
          "paper/supp/note6.tex", "paper/supp/note7.tex", "paper/supp/note8.tex", "paper/supp/note9.tex", "paper/supp/note10.tex",
          "paper/supp/note13.tex", "paper/supp/tableS13.tex"]
spec = importlib.util.spec_from_file_location("make_supp_sa", TOOLS / "make_supp_supp-analysis.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)


def test_pass24_script_reproduces_the_committed_files(tmp_path, monkeypatch, capsys):  # test-reason: runs pass24_edits.py, which no other test runs; catches a hand edit of outline.tex or of a supplement file the script writes, and a text over the word limits
    for rel in EDITED:
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{START}:{rel}"], capture_output=True, text=True)
        if r.returncode:
            pytest.skip(f"{START} not available")  # pragma: no cover  # shallow copies lack the history
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(r.stdout)
    monkeypatch.setenv("PASS24_ROOT", str(tmp_path))
    runpy.run_path(str(TOOLS / "pass24_edits.py"), run_name="__main__")
    for rel in EDITED:
        assert (tmp_path / rel).read_text() == (REPO / rel).read_text(), rel
    assert "limit 5700" in capsys.readouterr().out
    words, _ = aw.count((tmp_path / "paper/outline.tex").read_text())
    assert 0 < len(words) <= 330 and words[0] == "Background:" and "Question:" not in words   # three prose parts


# test-reason: the abstract counter on the Pass 24 form (struck numbered list, then paragraphs); the other tests use the list form
def test_abstract_counts_the_paragraphs_after_a_struck_list():
    tex = ("\\hd{Abstract}{budget}\n\\begin{enumerate}[leftmargin=1.5em]\n\\item \\deleted[id=P24]{old words}\n\\end{enumerate}\n"
           "\\noindent \\added[id=P24]{\\textbf{Background:} new words}\\par\n\\noindent \\added[id=P24]{\\textbf{Results:} 27\\% "
           "more}\n\\par\\vspace{3pt}\\noindent\\textbf{Keywords:} not counted\n")
    assert aw.count(tex)[0] == ["Background:", "new", "words", "Results:", "27%", "more"]


def test_sensitivity_table_and_label_transfer_table_without_cross_task_means(tmp_path, monkeypatch):  # test-reason: runs make_s15, which no other test runs; catches the removed cross-task mean block of Table S11 coming back
    monkeypatch.setattr(mt, "OUT", tmp_path)
    assert mt.make_s15() == {"none|0|10": 0, "harmony|2|50": 3, "scanorama|20|100": 1, "pca|0|50": 1}   # Res-6 and Note 8
    s15 = (tmp_path / "tableS15.tex").read_text()
    assert r"\label{tab:s15}" in s15 and r"Supplementary Table~\ref{tab:4}" in s15 and s15.count(r" \\") == 5
    monkeypatch.setattr(sa, "OUT", tmp_path)
    sa.make_s8()
    sa.make_note9()
    n9, s8 = (tmp_path / "note9_table.tex").read_text(), (tmp_path / "tableS8.tex").read_text()
    assert n9.count("Plain scVI &") == 16 and "Mean macro F1" not in n9 and "Last block" not in n9
    assert "Supplementary Figure S8" in s8 and "range of Figure 4" not in s8
