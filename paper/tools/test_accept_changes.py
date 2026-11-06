"""Tests of paper/tools/accept_changes.py (clean copy with tracked changes accepted; run: python -m pytest paper/tools/test_accept_changes.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstract_words as aw  # noqa: E402
import accept_changes as ac  # noqa: E402

PRE = "\\documentclass{article}\n\\newcommand{\\x}[1]{\\deleted[id=P8]{#1}}% preamble kept as is\n\\begin{document}\n"
BODY = r"""A \replaced[id=P9]{\replaced[id=P11]{new}{mid}}{old {braced}} B \added{x \emph{y}} 5\% \deleted{gone % comment with a } brace
still gone} C % \added{in a comment}
\begin{itemize}
\item \deleted{only text}
\item  % a comment is all that is left
\item kept
\end{itemize}
\pp{\deleted{T}}{\deleted{lead}}
\begin{enumerate}[leftmargin=1em]
\item \deleted{a}
\droppeditem{b}
\end{enumerate}
\end{document}
"""


# test-reason: exact output for nested markup, braces in comments, emptied bullets, lists and headings (the real-outline test checks no text)
def test_clean_accepts_nested_markup_and_removes_what_deletion_empties():
    out = ac.clean(PRE + BODY)
    head, body = out.split("\\begin{document}\n")
    assert head == ac.HEADER + PRE.split("\\begin{document}")[0]  # preamble untouched, markup in it kept
    assert body == ("A new B x \\emph{y} 5\\%  C % \\added{in a comment}\n"
                    "\\begin{itemize}\n\\item kept\n\\end{itemize}\n\\end{document}\n")


@pytest.mark.parametrize("bad", [r"\begin{document} \added{open", r"\begin{document} \replaced{a} {b}", r"\begin{document} \added[id=P3{x}",
                                 "no document"])
def test_clean_fails_loud(bad):  # unbalanced group, no old-text group, malformed markup left over, no document body
    with pytest.raises(ValueError):
        ac.clean(bad)


def test_main_writes_and_checks(tmp_path, capsys):
    src, out = tmp_path / "o.tex", tmp_path / "c.tex"
    src.write_text(PRE + BODY)
    assert ac.main(["--src", str(src), "--out", str(out), "--check"]) == 1  # not written yet
    assert ac.main(["--src", str(src), "--out", str(out)]) == 0
    assert ac.main(["--src", str(src), "--out", str(out), "--check"]) == 0 and "up to date" in capsys.readouterr().out
    out.write_text(out.read_text() + "hand edit")
    assert ac.main(["--src", str(src), "--out", str(out), "--check"]) == 1


# test-reason: result validity: committed clean copy is regenerated and prints the abstract the word counter reads
def test_real_outline_clean_copy_matches_abstract_counter_and_committed_file():
    # result validity: the clean copy prints the same abstract the word counter reads from the tracked file
    tex = (ac.ROOT / "paper/outline.tex").read_text()
    clean = ac.clean(tex)
    assert aw.count(clean)[0] == aw.count(tex)[0]
    assert clean == (ac.ROOT / "paper/outline_clean.tex").read_text()  # committed copy regenerated, never hand-edited
