"""Tests of paper/tools/abstract_words.py (word count of the printed abstract) and of the abstract-anchor and `supersedes`
features of check_ledger.py (run: python -m pytest paper/tools/test_abstract_words.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstract_words as aw  # noqa: E402
import check_ledger as cl  # noqa: E402

TEX = r"""intro \hd{Abstract}{x}
\begin{enumerate}[leftmargin=1.5em]
\item \added[id=P3]{\textbf{Background:}} \replaced[id=P13]{new text of 27\% here (a--b)}{\replaced{old}{older} struck words}% comment words
\item \res{placeholder words not counted} \deleted{gone words}
\end{enumerate}
\textbf{Keywords:} not counted
"""


def test_count_final_text_and_exit_code(tmp_path, capsys):
    words, dropped = aw.count(TEX)
    assert " ".join(words) == r"Background: new text of 27% here (a-b)" and dropped == ["placeholder words not counted"]
    f = tmp_path / "o.tex"
    f.write_text(TEX)
    assert aw.main(["--tex", str(f), "--limit", "7", "--show"]) == 0                  # 7 words: at the limit
    out = capsys.readouterr().out
    assert out.startswith("Background: new text") and "abstract words" in out and "hyphen-split count 8" in out
    assert aw.main(["--tex", str(f), "--limit", "6"]) == 1 and "limit 6" in capsys.readouterr().out


def test_ledger_abstract_anchor_reads_printed_text_and_supersedes_moves_the_old_location(tmp_path):
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper/outline.tex").write_text(TEX)
    (tmp_path / "paper/n.json").write_text('{"v": 0.27}')
    cols = "id,location,written,source,selector,fmt,note,supersedes\n"
    row = lambda i, loc, sup="": f'{i},"{loc}",27\\%,paper/n.json,key=v,.0%,,{sup}\n'
    (tmp_path / "paper/a.csv").write_text(cols[:-12] + "\n" + row("old", "paper/outline.tex::rate 99"))      # raw text has no such anchor: fails unless superseded
    (tmp_path / "paper/b.csv").write_text(cols + row("old", "abstract:paper/outline.tex::new text of 27\\% here", "old")
                                          + row("n2", "abstract:paper/outline.tex::old struck words", ""))
    sup = cl.superseded_ids(["paper/a.csv", "paper/b.csv"], tmp_path)
    assert sup == {"old"}
    assert len(cl.check_ledger("paper/a.csv", tmp_path, "")[2]) == 1                  # without the supersede the old anchor fails
    assert cl.check_ledger("paper/a.csv", tmp_path, "", sup)[2] == []                 # location of the superseded row is skipped
    errs = cl.check_ledger("paper/b.csv", tmp_path, "", sup)[2]
    assert len(errs) == 1 and errs[0].startswith("n2: anchor not found")             # old text is struck, so not in the printed abstract
